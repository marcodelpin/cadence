"""The torch records against the NumPy records: the same code, the same reads, the same writes."""

from __future__ import annotations

import numpy as np
import pytest

import cadence as cd

torch = pytest.importorskip("torch")

TOLERANCE = 1e-9


def pair(**kw):
    """One NumPy cortex and one torch cortex built from the same seed."""
    host = cd.Records(**kw)
    device = cd.Records(**kw, backend="torch", device="cpu")
    return host, device


def readings(rows: int, inputs: int, seed: int = 0) -> np.ndarray:
    return np.random.default_rng(seed).standard_normal((rows, inputs))


def test_code_and_read_agree() -> None:
    host, device = pair(inputs=40, fields={"next": 7, "reward": 1}, cells=500, active=12, seed=3)
    x = readings(9, 40)
    a, b = host.code(x, adapt=False), device.code(x, adapt=False)
    assert a.shape == b.shape
    assert np.array_equal(a != 0, b != 0)  # the same winners
    assert np.abs(a - b).max() < TOLERANCE
    for name in ("next", "reward"):
        assert np.abs(host.read(a)[name] - device.read(b)[name]).max() < TOLERANCE


def test_writes_agree_after_many() -> None:
    host, device = pair(inputs=32, fields={"next": 5, "reward": 1}, cells=400, active=10,
                        valued=["reward"], rate=0.3, valued_rate=1.0, seed=11)
    rng = np.random.default_rng(5)
    for _ in range(200):
        x = rng.standard_normal((1, 32))
        a, b = host.code(x, adapt=True), device.code(x, adapt=True)
        targets = {"next": rng.random(5), "reward": rng.random(1)}
        known = {"next": np.ones(5, bool), "reward": np.ones(1, bool)}
        assert host.write(a[:, 0], targets, known) == device.write(b[:, 0], targets, known)
    assert host.seen == device.seen and host.writes == device.writes
    assert np.abs(host.mean - device.mean).max() < TOLERANCE
    for name in ("next", "reward"):
        assert np.abs(host.tables[name] - device.tables[name]).max() < TOLERANCE


def test_pathways_and_habituation_agree() -> None:
    blocks = [np.arange(0, 10), np.arange(10, 24), np.arange(24, 30)]
    host, device = pair(inputs=30, fields={"next": 4, "reward": 1}, cells=300, active=8,
                        valued=["reward"], pathways=blocks, pathway_rate=0.05,
                        habituation=1e-3, seed=7)
    rng = np.random.default_rng(2)
    for _ in range(60):
        x = rng.standard_normal((2, 30)) * 3.0
        a, b = host.code(x, adapt=True), device.code(x, adapt=True)
        assert np.abs(a - b).max() < TOLERANCE
    assert np.abs(host.pathway_norm - device.pathway_norm).max() < TOLERANCE
    assert np.abs(host.mean - device.mean).max() < TOLERANCE


def test_task_sets_agree() -> None:
    tasks = np.array([0, 1, 2])
    host, device = pair(inputs=20, fields={"next": 3, "reward": 1}, cells=300, active=6,
                        valued=["reward"], tasks=tasks, seed=13)
    rng = np.random.default_rng(4)
    x = rng.random((7, 20))
    x[:, :3] = 0.0
    x[np.arange(7), rng.integers(0, 3, 7)] = 1.0
    a, b = host.code(x, adapt=False), device.code(x, adapt=False)
    assert np.array_equal(a != 0, b != 0)
    assert np.abs(a - b).max() < TOLERANCE


def test_unvalued_code_is_nan_on_both() -> None:
    host, device = pair(inputs=16, fields={"next": 3}, cells=200, active=5, seed=1)
    x = readings(4, 16, seed=9)
    a, b = host.code(x, adapt=False, valued=False), device.code(x, adapt=False, valued=False)
    assert np.isnan(a[1]).all() and np.isnan(b[1]).all()
    assert np.abs(a[0] - b[0]).max() < TOLERANCE


def test_state_is_readable_and_writable_through_the_device() -> None:
    host, device = pair(inputs=12, fields={"next": 3}, cells=100, active=4, seed=6)
    table = np.random.default_rng(0).standard_normal((100, 3))
    device.tables["next"] = table
    host.tables["next"][...] = table
    assert np.abs(device.tables["next"] - table).max() == 0.0
    mean = np.random.default_rng(1).standard_normal(12)
    device.mean, host.mean = mean, mean
    x = readings(3, 12, seed=2)
    assert np.abs(host.code(x, adapt=False) - device.code(x, adapt=False)).max() < TOLERANCE
    assert device.to_dict()["backend"] == "torch"
    assert host.to_dict()["backend"] == "cpu"
    assert device.parameters() == host.parameters()


def test_fan_in_and_bias_agree() -> None:
    blocks = [np.arange(0, 8), np.arange(8, 16), np.arange(16, 24)]
    host, device = pair(inputs=24, fields={"next": 4}, cells=240, active=6,
                        pathways=blocks, fan_in=2, bias=0.5, seed=21)
    x = readings(5, 24, seed=8)
    assert np.abs(host.code(x, adapt=False) - device.code(x, adapt=False)).max() < TOLERANCE


def test_unknown_backend_is_refused() -> None:
    with pytest.raises(ValueError):
        cd.Records(4, {"a": 2}, cells=20, active=2, backend="quantum")


def test_a_copy_of_a_device_cortex_is_independent() -> None:
    import copy

    cortex = cd.Records(12, {"next": 3}, cells=200, active=6, backend="torch", device="cpu", seed=4)
    x = readings(3, 12, seed=1)
    code = cortex.code(x, adapt=True)
    twin = copy.deepcopy(cortex)
    assert np.abs(twin.code(x, adapt=False) - cortex.code(x, adapt=False)).max() == 0.0
    cortex.write(code[:, 0], {"next": np.ones(3)}, None)
    assert np.abs(twin.tables["next"]).max() == 0.0
    assert twin.writes == 0 and cortex.writes == 1
