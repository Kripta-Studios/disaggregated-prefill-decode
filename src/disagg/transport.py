"""KV transport abstraction: how a prefill worker hands a KV cache to a decode worker.

The interface is deliberately shaped like the production libraries (NIXL, Mooncake
Transfer Engine, UCX active messages) so a real RDMA backend could be dropped in:

* The **receiver owns the destination memory** (``ShmTransport`` pre-allocates a slot
  arena, exactly like an SGLang/vLLM decode instance pre-allocates KV blocks and hands
  their addresses to the prefill side).
* The **sender writes, then notifies** with a small metadata message (NIXL calls this a
  notification; with RDMA it is a WRITE followed by a SEND/immediate).
* The receiver **releases** the buffer once it has consumed it, returning credit to the
  sender -- this is the back-pressure that keeps a fast prefill pool from overrunning a
  decode pool that is out of KV memory.

Backends here:
  ``TcpTransport``  -- length-prefixed frames over a loopback socket (works across hosts;
                       models a TCP/RoCE path without zero-copy).
  ``ShmTransport``  -- one-sided copy into a shared-memory arena + a notification queue
                       (models a same-node path such as NVLink/CUDA-IPC, but via host RAM).
An ``RdmaTransport`` would implement the same three calls with registered GPU memory and
GPUDirect RDMA, removing both the device->host and host->device copies.

Specs are small picklable dataclasses so they can be passed to ``multiprocessing`` children
(Windows uses *spawn*, so everything crossing the process boundary must pickle).
"""

from __future__ import annotations

import json
import queue
import socket
import struct
import threading
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from multiprocessing import shared_memory
from typing import Any

_HDR = struct.Struct("!I")  # 4-byte big-endian length of the JSON metadata


class KVSender(ABC):
    @abstractmethod
    def send(self, meta: dict, payload: memoryview | bytes) -> None:
        """Deliver ``payload`` and ``meta`` to the receiver.  Blocks until the bytes have left
        this process (TCP) or been written into receiver-owned memory (SHM)."""

    def close(self) -> None:  # noqa: B027 - optional hook
        pass


class KVReceiver(ABC):
    @abstractmethod
    def recv(self, timeout: float | None = None) -> tuple[dict, memoryview] | None:
        """Return the next (meta, payload) or ``None`` on timeout.  The payload view is only
        valid until :meth:`release` is called for that ``meta``."""

    def release(self, meta: dict) -> None:  # noqa: B027 - TCP buffers are owned by Python
        pass

    def close(self) -> None:  # noqa: B027
        pass


# --------------------------------------------------------------------------- TCP backend
def _recv_exact(sock: socket.socket, n: int) -> bytearray:
    buf = bytearray(n)
    view, got = memoryview(buf), 0
    while got < n:
        r = sock.recv_into(view[got:], min(n - got, 1 << 22))
        if r == 0:
            raise ConnectionError("peer closed mid-frame")
        got += r
    return buf


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


@dataclass
class TcpSpec:
    host: str = "127.0.0.1"
    port: int = field(default_factory=free_port)
    kind: str = "tcp"

    def make_receiver(self) -> KVReceiver:
        return TcpReceiver(self)

    def make_sender(self) -> KVSender:
        return TcpSender(self)


class TcpReceiver(KVReceiver):
    """Listens on ``spec.port``; one reader thread per connected sender pushes frames to a queue."""

    def __init__(self, spec: TcpSpec):
        self._q: queue.Queue = queue.Queue()
        self._srv = socket.create_server((spec.host, spec.port))
        threading.Thread(target=self._accept_loop, daemon=True).start()

    def _accept_loop(self) -> None:
        while True:
            try:
                conn, _ = self._srv.accept()
            except OSError:
                return  # server socket closed
            conn.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 8 << 20)
            threading.Thread(target=self._read_loop, args=(conn,), daemon=True).start()

    def _read_loop(self, conn: socket.socket) -> None:
        with conn:
            try:
                while True:
                    (mlen,) = _HDR.unpack(_recv_exact(conn, _HDR.size))
                    meta = json.loads(_recv_exact(conn, mlen))
                    payload = _recv_exact(conn, meta.get("nbytes", 0))
                    meta["t_recv_done"] = time.perf_counter()
                    self._q.put((meta, memoryview(payload)))
            except (ConnectionError, OSError):
                return

    def recv(self, timeout: float | None = None):
        try:
            return self._q.get(timeout=timeout)
        except queue.Empty:
            return None

    def close(self) -> None:
        self._srv.close()


class TcpSender(KVSender):
    def __init__(self, spec: TcpSpec, connect_timeout_s: float = 120.0):
        deadline = time.perf_counter() + connect_timeout_s
        while True:  # the receiver may still be loading its model
            try:
                self._sock = socket.create_connection((spec.host, spec.port))
                break
            except OSError:
                if time.perf_counter() > deadline:
                    raise
                time.sleep(0.1)
        self._sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        self._sock.setsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF, 8 << 20)

    def send(self, meta: dict, payload) -> None:
        meta = dict(meta, nbytes=len(payload), t_send_start=time.perf_counter())
        m = json.dumps(meta).encode()
        self._sock.sendall(_HDR.pack(len(m)) + m)
        if len(payload):
            self._sock.sendall(payload)

    def close(self) -> None:
        self._sock.close()


# ------------------------------------------------------------------ shared-memory backend
@dataclass
class ShmSpec:
    """A receiver-owned arena of ``n_slots`` fixed-size slots plus two queues:
    ``free_q`` carries slot credits receiver->sender, ``ready_q`` carries notifications."""

    n_slots: int
    slot_bytes: int
    free_q: Any  # multiprocessing.Queue (or queue.Queue for in-process use)
    ready_q: Any
    name: str = field(default_factory=lambda: f"kv_{time.time_ns() % 10**12}")
    kind: str = "shm"

    def make_receiver(self) -> KVReceiver:
        return ShmReceiver(self)

    def make_sender(self) -> KVSender:
        return ShmSender(self)


class ShmReceiver(KVReceiver):
    def __init__(self, spec: ShmSpec):
        self.spec = spec
        self._shm = shared_memory.SharedMemory(name=spec.name, create=True, size=spec.n_slots * spec.slot_bytes)
        for s in range(spec.n_slots):  # hand every slot to the sender as an initial credit
            spec.free_q.put(s)

    def recv(self, timeout: float | None = None):
        try:
            meta = self.spec.ready_q.get(timeout=timeout)
        except queue.Empty:
            return None
        meta["t_recv_done"] = time.perf_counter()
        if meta.get("slot") is None:
            return meta, memoryview(b"")
        off = meta["slot"] * self.spec.slot_bytes
        return meta, self._shm.buf[off : off + meta["nbytes"]]

    def release(self, meta: dict) -> None:
        if meta.get("slot") is not None:
            self.spec.free_q.put(meta["slot"])  # credit back to the sender

    def close(self) -> None:
        self._shm.close()
        self._shm.unlink()


class ShmSender(KVSender):
    def __init__(self, spec: ShmSpec, attach_timeout_s: float = 120.0):
        self.spec = spec
        deadline = time.perf_counter() + attach_timeout_s
        while True:
            try:
                self._shm = shared_memory.SharedMemory(name=spec.name)
                break
            except FileNotFoundError:  # receiver has not created the arena yet
                if time.perf_counter() > deadline:
                    raise
                time.sleep(0.1)

    def send(self, meta: dict, payload) -> None:
        n = len(payload)
        if n > self.spec.slot_bytes:
            raise ValueError(f"KV of {n} B exceeds slot size {self.spec.slot_bytes} B")
        t0 = time.perf_counter()
        slot = self.spec.free_q.get() if n else None  # blocks when decode is out of buffers
        if slot is not None:
            off = slot * self.spec.slot_bytes
            self._shm.buf[off : off + n] = payload  # the "one-sided write"
        self.spec.ready_q.put(dict(meta, slot=slot, nbytes=n, t_send_start=t0))

    def close(self) -> None:
        self._shm.close()
