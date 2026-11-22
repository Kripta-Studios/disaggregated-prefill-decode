"""Light tests: CPU tensors only, no model, no GPU."""

import queue
import threading

import pytest
import torch

from disagg.kvcodec import cache_to_layers, kv_bytes_per_token, layers_to_cache, pack_kv, unpack_kv
from disagg.transport import ShmSpec, TcpSpec


def _layers(n_layers=3, heads=2, T=5, D=4):
    g = torch.Generator().manual_seed(0)
    return [(torch.randn(1, heads, T, D, generator=g, dtype=torch.float32).to(torch.bfloat16),
             torch.randn(1, heads, T, D, generator=g, dtype=torch.float32).to(torch.bfloat16)) for _ in range(n_layers)]


def test_kv_bytes_per_token_qwen3():
    # Qwen3-0.6B: 28 layers, 8 KV heads, head_dim 128, bf16 -> 112 KiB per token
    assert kv_bytes_per_token(28, 8, 128) == 114_688


def test_pack_unpack_roundtrip_is_bit_exact():
    layers = _layers()
    meta, payload, _ = pack_kv(layers)
    assert meta["nbytes"] == payload.numel() == 3 * 2 * 2 * 5 * 4 * 2
    back = unpack_kv(meta, bytearray(payload.numpy().tobytes()))
    for (k, v), (k2, v2) in zip(layers, back):
        assert torch.equal(k, k2) and torch.equal(v, v2)


def test_rebuilt_dynamic_cache_has_same_tensors():
    layers = _layers()
    cache = layers_to_cache(layers)
    assert cache.get_seq_length() == 5
    for (k, v), (k2, v2) in zip(layers, cache_to_layers(cache)):
        assert torch.equal(k, k2) and torch.equal(v, v2)


@pytest.mark.parametrize("kind", ["tcp", "shm"])
def test_transport_roundtrip(kind):
    spec = TcpSpec() if kind == "tcp" else ShmSpec(2, 1 << 16, queue.Queue(), queue.Queue())
    rx = spec.make_receiver()
    tx = spec.make_sender()
    try:
        meta, payload, _ = pack_kv(_layers())
        tx.send(dict(meta, rid=7), memoryview(payload.numpy()))
        tx.send({"stop": True}, b"")
        got_meta, buf = rx.recv(timeout=5)
        assert got_meta["rid"] == 7 and got_meta["t_recv_done"] >= got_meta["t_send_start"]
        back = unpack_kv(got_meta, buf)
        del buf
        rx.release(got_meta)
        assert torch.equal(back[1][0], _layers()[1][0])
        stop_meta, _ = rx.recv(timeout=5)
        assert stop_meta["stop"] is True
    finally:
        tx.close()
        rx.close()


def test_shm_backpressure_and_oversize():
    spec = ShmSpec(1, 64, queue.Queue(), queue.Queue())
    rx, tx = spec.make_receiver(), spec.make_sender()
    try:
        with pytest.raises(ValueError):
            tx.send({}, b"x" * 65)
        tx.send({"i": 0}, b"a" * 10)  # takes the only slot
        sent = threading.Event()
        t = threading.Thread(target=lambda: (tx.send({"i": 1}, b"b" * 10), sent.set()), daemon=True)
        t.start()
        assert not sent.wait(0.3)  # blocked: no credit until the receiver releases
        meta, buf = rx.recv(timeout=5)
        assert bytes(buf) == b"a" * 10
        del buf
        rx.release(meta)
        assert sent.wait(5)
        meta, buf = rx.recv(timeout=5)
        assert bytes(buf) == b"b" * 10
        del buf
    finally:
        tx.close()
        rx.close()
