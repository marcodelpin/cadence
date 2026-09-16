"""Records: what followed each reading, kept where the reading lands.

A reading is the drive of the neurons a patch reads: sensory fields with their missing
flags, the goal, the action, the reads of declared stores. Each unit's running mean is
subtracted, so the code follows what deviates from the usual reading and a small cue can
move it. A fixed random projection maps the reading onto many cells; the most active cells
stay and the rest are inhibited. Each predicted field keeps a table with one record per
cell. The read of a reading is the sum of its active cells' records, weighted by their
activity; the witnessed outcome is written into exactly those records by the delta rule.
A reading touches few records, so one outcome is written almost exactly and a reading
elsewhere leaves it intact: records learn from one stream without replay.

Two codes come from one reading. The plain code serves the consequence fields. The valued
code first divides every declared input pathway by its running norm, so the goal and a
remembered cue have the same say as the senses; the valued fields (reward, terminal value)
read and write through it at their own rate.

The running mean settles: it is the plain average of the readings seen until one over their
count falls to the habituation rate, and follows at that rate after. A settled mean keeps the
code of a reading where its records were written; a mean that keeps moving would carry old
readings onto cells that were never written.

Task sets keep the values of one task away from the values of another. When ``tasks`` names
the reading's task units (a goal port with one active unit), the cells are divided into one
group per task unit, and a reading's valued code draws its winners from the group of the
active task; the plain code and the consequence records stay shared by every task. A reward
earned under one goal is then written into cells no other goal reads, and a cue learned for
one task survives the rewards of the tasks that follow.

The projection is drawn from ``Mulberry32``, the generator of ``brain_scan.js``, so a page
rebuilds the same cells from the seed.

``backend="torch"`` keeps the projection, the offsets, the running mean and the record tables
on a torch device and runs the same arithmetic there: the drive product, the k winners, the
reads and the delta-rule writes. The signatures and the dtypes do not change, readings arrive
and reads leave as NumPy float64 arrays, and the host state stays readable through ``mean``,
``pathway_norm`` and ``tables``. Reduction order is the device library's, so a read and a
written record agree with the NumPy path to rounding and not bit for bit; ``docs/memory.md``
states the measured deviation.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from typing import Any

import numpy as np

__all__ = ["Mulberry32", "Records"]

M32 = 0xFFFFFFFF
_TORCH: Any = None


def _torch() -> Any:
    """The torch module, imported once. A kernel holds no module attribute, so a life that
    copies itself (an evaluation on a frozen copy) copies its records with it."""
    global _TORCH
    if _TORCH is None:
        try:
            import torch
        except ImportError as exc:  # pragma: no cover - the extra is not installed
            raise ImportError("the torch backend needs cadence-net[accel]") from exc
        _TORCH = torch
    return _TORCH


class Mulberry32:
    """The 32-bit generator of ``brain_scan.js`` (``mulberry``): uniform draws in [0, 1)."""

    def __init__(self, seed: int) -> None:
        self.state = int(seed) & M32

    def random(self) -> float:
        a = (self.state + 0x6D2B79F5) & M32
        self.state = a
        t = ((a ^ (a >> 15)) * (a | 1)) & M32
        t = (t ^ ((t + (((t ^ (t >> 7)) * (t | 61)) & M32)) & M32)) & M32
        return ((t ^ (t >> 14)) & M32) / 4294967296.0

    def batch(self, n: int) -> np.ndarray:
        """``n`` draws at once, equal to ``n`` calls of ``random``."""
        if n < 0:
            raise ValueError("n must be nonnegative")
        m32 = np.uint64(M32)
        step = np.arange(1, n + 1, dtype=np.uint64) * np.uint64(0x6D2B79F5)
        a = (np.uint64(self.state) + step) & m32
        t = ((a ^ (a >> np.uint64(15))) * (a | np.uint64(1))) & m32
        t = (t ^ ((t + (((t ^ (t >> np.uint64(7))) * (t | np.uint64(61))) & m32)) & m32)) & m32
        out: np.ndarray = ((t ^ (t >> np.uint64(14))) & m32).astype(np.float64) / 4294967296.0
        self.state = int((self.state + n * 0x6D2B79F5) & M32)
        return out

    def normals(self, n: int) -> np.ndarray:
        """``n`` standard normal draws by the Box-Muller transform of ``batch`` pairs."""
        pairs = (n + 1) // 2
        u = self.batch(2 * pairs)
        radius = np.sqrt(-2.0 * np.log(np.maximum(u[:pairs], 1e-12)))
        angle = 2.0 * np.pi * u[pairs:]
        out: np.ndarray = np.concatenate([radius * np.cos(angle), radius * np.sin(angle)])
        return out[:n]


def _shuffle(n: int, generator: Mulberry32) -> np.ndarray:
    """A permutation of ``range(n)`` by Fisher-Yates from ``n - 1`` draws of the generator."""
    order = np.arange(n)
    draws = generator.batch(max(n - 1, 0))
    for i in range(n - 1, 0, -1):
        j = int(draws[n - 1 - i] * (i + 1))
        order[i], order[j] = order[j], order[i]
    return order


def _positive_int(name: str, value: int) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return int(value)


def _rate(name: str, value: float, *, upper: float) -> float:
    rate = float(value)
    if not np.isfinite(rate) or not 0.0 <= rate <= upper:
        raise ValueError(f"{name} lies in [0, {upper:g}]")
    return rate


class Records:
    """A records cortex over readings of ``inputs`` units.

    ``fields`` maps each predicted field to its width. Of ``cells`` code cells, ``active``
    stay per reading. ``rate`` is the write rate of the consequence fields; ``valued`` names
    the fields that read and write through the valued code, at ``valued_rate``.
    ``habituation`` is the slowest rate of each unit's running mean (0 subtracts nothing).
    ``pathways`` are index arrays into the reading whose running norms, at ``pathway_rate``,
    equalise their say in the valued code. ``tasks`` indexes the reading's task units; the
    cells are then divided into one group per task unit, and the valued code of a reading
    draws its winners from the group of the unit with the largest value (every cell when no
    task unit is positive). ``fan_in`` restricts each cell to that many of the pathways,
    drawn from the generator; the cell reads those pathways and every input outside the
    pathways (0 reads every input). ``bias`` scales the cells' fixed offsets. ``seed``
    starts the generator of the projection, the offsets, the division into groups and the
    pathways each cell reads.
    """

    def __init__(
        self,
        inputs: int,
        fields: Mapping[str, int],
        *,
        cells: int = 8000,
        active: int = 40,
        rate: float = 0.2,
        valued: Iterable[str] = (),
        valued_rate: float = 1.0,
        habituation: float = 1e-5,
        bias: float = 0.3,
        pathways: Sequence[Sequence[int] | np.ndarray] = (),
        pathway_rate: float = 0.002,
        tasks: Sequence[int] | np.ndarray = (),
        fan_in: int = 0,
        seed: int = 0,
        backend: str = "cpu",
        device: str | None = None,
        precision: str | None = None,
    ) -> None:
        self.inputs = _positive_int("inputs", inputs)
        self.cells = _positive_int("cells", cells)
        self.active = _positive_int("active", active)
        if self.active > self.cells:
            raise ValueError("active must not exceed cells")
        if not fields:
            raise ValueError("records need at least one field")
        self.fields = {str(k): _positive_int(f"field {k!r}", w) for k, w in fields.items()}
        self.valued = frozenset(str(name) for name in valued)
        if not self.valued <= set(self.fields):
            missing = sorted(self.valued - set(self.fields))
            raise ValueError(f"valued fields {missing} are not fields")
        self.rate = _rate("rate", rate, upper=2.0)
        self.valued_rate = _rate("valued_rate", valued_rate, upper=2.0)
        self.habituation = _rate("habituation", habituation, upper=1.0)
        self.pathway_rate = _rate("pathway_rate", pathway_rate, upper=1.0)
        self.bias = float(bias)
        if not np.isfinite(self.bias) or self.bias < 0:
            raise ValueError("bias must be finite and nonnegative")
        self.seed = int(seed) & M32
        generator = Mulberry32(self.seed)
        draws = generator.normals(self.inputs * self.cells)
        self.projection = draws.reshape(self.inputs, self.cells) / np.sqrt(self.inputs)
        self.offset = generator.normals(self.cells) * self.bias
        self.pathways = [np.asarray(p, dtype=np.int64) for p in pathways if len(p)]
        for p in self.pathways:
            if p.ndim != 1 or p.min() < 0 or p.max() >= self.inputs:
                raise ValueError("a pathway is a one-dimensional index array into the reading")
        self.tasks = np.asarray(tasks, dtype=np.int64).reshape(-1)
        if len(self.tasks) and (self.tasks.min() < 0 or self.tasks.max() >= self.inputs):
            raise ValueError("tasks is an index array into the reading")
        if len(self.tasks) and self.cells // len(self.tasks) < self.active:
            raise ValueError("every task group needs at least active cells")
        self.task_of_cell = np.zeros(self.cells, dtype=np.int64)
        if len(self.tasks):
            order = _shuffle(self.cells, generator)
            for k, group in enumerate(np.array_split(order, len(self.tasks))):
                self.task_of_cell[group] = k
        self.fan_in = int(fan_in)
        if self.fan_in < 0 or (self.fan_in and not self.pathways):
            raise ValueError("fan_in is nonnegative and needs pathways")
        if self.fan_in > len(self.pathways):
            raise ValueError("fan_in must not exceed the number of pathways")
        if self.fan_in:
            reads = np.zeros((len(self.pathways), self.cells), dtype=bool)
            for cell in range(self.cells):
                reads[_shuffle(len(self.pathways), generator)[: self.fan_in], cell] = True
            mask = np.ones((self.inputs, self.cells))
            for k, pathway in enumerate(self.pathways):
                mask[np.ix_(pathway, ~reads[k])] = 0.0
            self.projection *= mask * np.sqrt(self.inputs / mask.sum(axis=0))
        self._mean = np.zeros(self.inputs)
        self.seen = 0
        self._pathway_norm = np.ones(len(self.pathways))
        self._tables = {name: np.zeros((self.cells, width)) for name, width in self.fields.items()}
        self.writes = 0
        self.backend = str(backend)
        self._kernel: Any = None
        if self.backend == "torch":
            self._kernel = _TorchRecords(self, device, precision)
        elif self.backend != "cpu":
            raise ValueError(f"unknown backend {backend!r}")

    # -- state: the host arrays, or the device's when a kernel owns them

    @property
    def mean(self) -> np.ndarray:
        """Each unit's running mean."""
        if self._kernel is None:
            return self._mean
        out: np.ndarray = self._kernel.mean
        return out

    @mean.setter
    def mean(self, value: np.ndarray) -> None:
        array = np.asarray(value, dtype=float).reshape(self.inputs)
        if self._kernel is None:
            self._mean = array
        else:
            self._kernel.mean = array

    @property
    def pathway_norm(self) -> np.ndarray:
        """The running norm of each declared pathway."""
        if self._kernel is None:
            return self._pathway_norm
        out: np.ndarray = self._kernel.pathway_norm
        return out

    @pathway_norm.setter
    def pathway_norm(self, value: np.ndarray) -> None:
        array = np.asarray(value, dtype=float).reshape(len(self.pathways))
        if self._kernel is None:
            self._pathway_norm = array
        else:
            self._kernel.pathway_norm = array

    @property
    def tables(self) -> Any:
        """The records of each field: one row per cell. A device kernel owns them there."""
        if self._kernel is None:
            return self._tables
        proxy: Any = self._kernel.tables
        return proxy

    @tables.setter
    def tables(self, value: Mapping[str, np.ndarray]) -> None:
        if set(value) != set(self.fields):
            raise ValueError(f"the records need one table per field: {sorted(self.fields)}")
        if self._kernel is None:
            self._tables = {name: np.asarray(value[name], dtype=float) for name in self.fields}
            for name, width in self.fields.items():
                if self._tables[name].shape != (self.cells, width):
                    raise ValueError(f"the records of {name!r} have shape ({self.cells}, {width})")
            return
        for name in self.fields:
            self._kernel.set_table(name, value[name])

    def _allowed(self, readings: np.ndarray) -> np.ndarray | None:
        """The cells each reading's active task allows for the valued code, or None."""
        if not len(self.tasks):
            return None
        units = readings[:, self.tasks]
        allowed: np.ndarray = self.task_of_cell[None, :] == units.argmax(axis=1)[:, None]
        allowed[units.max(axis=1) <= 0.0] = True
        return allowed

    def _winners(self, x: np.ndarray, allowed: np.ndarray | None, out: np.ndarray) -> None:
        """Writes the k-winner code of the drives ``x @ projection + offset`` into ``out``."""
        drive = x @ self.projection + self.offset
        if allowed is not None:
            drive = np.where(allowed, drive, -np.inf)
        k = self.active
        index = np.argpartition(drive, self.cells - k, axis=1)[:, self.cells - k :]
        rows = np.arange(drive.shape[0])[:, None]
        values = np.maximum(drive[rows, index], 0.0)
        norm = np.linalg.norm(values, axis=1, keepdims=True)
        out[...] = 0.0
        out[rows, index] = np.where(norm > 0, values / np.maximum(norm, 1e-12), values)

    def code(self, readings: np.ndarray, *, adapt: bool = False, valued: bool = True) -> np.ndarray:
        """The codes of ``(batch, inputs)`` readings: ``(2, batch, cells)``, plain then valued.

        ``adapt`` first moves the running mean and the pathway norms by these readings:
        witnessed readings adapt, imagined readings do not. ``valued=False`` leaves the
        valued code unset (NaN), for imagined readings whose consequence fields alone are
        read; a read of a valued field from such a code is NaN."""
        x = np.atleast_2d(np.asarray(readings, dtype=float))
        if x.ndim != 2 or x.shape[1] != self.inputs or not np.isfinite(x).all():
            raise ValueError(f"readings must be a finite (batch, {self.inputs}) array")
        if self._kernel is not None:
            device_code: np.ndarray = self._kernel.code(x, adapt=adapt, valued=valued)
            return device_code
        allowed = self._allowed(x)
        if self.habituation > 0:
            if adapt:
                for row in x:
                    self.seen += 1
                    self._mean += max(self.habituation, 1.0 / self.seen) * (row - self._mean)
            x = x - self._mean
        codes = np.empty((2, x.shape[0], self.cells))
        self._winners(x, None, codes[0])
        if self.pathways and adapt:
            for k, pathway in enumerate(self.pathways):
                for value in np.linalg.norm(x[:, pathway], axis=1):
                    self._pathway_norm[k] += self.pathway_rate * (value - self._pathway_norm[k])
        if not valued:
            codes[1] = np.nan
        elif self.pathways:
            v = x.copy()
            for k, pathway in enumerate(self.pathways):
                v[:, pathway] /= self._pathway_norm[k] + 1e-3
            self._winners(v, allowed, codes[1])
        elif allowed is not None:
            self._winners(x, allowed, codes[1])
        else:
            codes[1] = codes[0]
        return codes

    def _code_for(self, code: np.ndarray, name: str) -> np.ndarray:
        if code.shape[0] != 2 or code.shape[-1] != self.cells:
            raise ValueError(f"a code has shape (2, {self.cells}) or (2, batch, {self.cells})")
        selected: np.ndarray = code[1] if name in self.valued else code[0]
        return selected

    def read(self, code: np.ndarray) -> dict[str, np.ndarray]:
        """Each field's read: ``(width,)`` for one code, ``(batch, width)`` for a batch."""
        if self._kernel is not None:
            device_read: dict[str, np.ndarray] = self._kernel.read(code)
            return device_read
        code = np.asarray(code, dtype=float)
        return {name: self._code_for(code, name) @ table for name, table in self._tables.items()}

    def write(
        self,
        code: np.ndarray,
        targets: Mapping[str, np.ndarray],
        known: Mapping[str, np.ndarray] | None = None,
    ) -> int:
        """Write the witnessed outcome of one reading.

        Each named field's records move toward its target by the delta rule, through the
        active cells only. ``known`` masks the entries of a target that were observed.
        Returns the number of fields written."""
        if self._kernel is not None:
            return int(self._kernel.write(code, targets, known))
        code = np.asarray(code, dtype=float)
        if code.shape != (2, self.cells):
            raise ValueError(f"write takes one code of shape (2, {self.cells})")
        written = 0
        for name, width in self.fields.items():
            if name not in targets:
                continue
            target = np.asarray(targets[name], dtype=float)
            if target.shape != (width,) or not np.isfinite(target).all():
                raise ValueError(f"the target of {name!r} must be a finite vector of width {width}")
            c = self._code_for(code, name)
            active = np.flatnonzero(c)  # the write touches the records of the active cells only
            error = target - c[active] @ self._tables[name][active]
            if known is not None and name in known:
                error = error * np.asarray(known[name], dtype=bool)
            rate = self.valued_rate if name in self.valued else self.rate
            self._tables[name][active] += rate * np.outer(c[active], error)
            written += 1
        self.writes += written
        return written

    def parameters(self) -> int:
        """Record entries, the learned state; the projection and the offsets are fixed."""
        return int(sum(self.cells * width for width in self.fields.values()))

    def to_dict(self) -> dict[str, Any]:
        """The configuration that rebuilds the fixed cells (the tables are the learned state)."""
        return {
            "inputs": self.inputs,
            "fields": dict(self.fields),
            "cells": self.cells,
            "active": self.active,
            "rate": self.rate,
            "valued": sorted(self.valued),
            "valued_rate": self.valued_rate,
            "habituation": self.habituation,
            "bias": self.bias,
            "pathways": [p.tolist() for p in self.pathways],
            "pathway_rate": self.pathway_rate,
            "tasks": self.tasks.tolist(),
            "fan_in": self.fan_in,
            "seed": self.seed,
            "backend": self.backend,
        }


# -- the device kernel


class _Tables:
    """The record tables of a device kernel, read and written as NumPy arrays."""

    def __init__(self, kernel: _TorchRecords) -> None:
        self._kernel = kernel

    def __getitem__(self, name: str) -> np.ndarray:
        return self._kernel.table(name)

    def __setitem__(self, name: str, value: np.ndarray) -> None:
        self._kernel.set_table(name, value)

    def __contains__(self, name: object) -> bool:
        return name in self._kernel.records.fields

    def __iter__(self) -> Any:
        return iter(self._kernel.records.fields)

    def __len__(self) -> int:
        return len(self._kernel.records.fields)

    def keys(self) -> Any:
        return self._kernel.records.fields.keys()

    def values(self) -> Any:
        return (self[name] for name in self._kernel.records.fields)

    def items(self) -> Any:
        return ((name, self[name]) for name in self._kernel.records.fields)


class _TorchRecords:
    """``Records`` on a torch device: the same equations, the device's reduction order.

    The projection, the offsets, the running mean, the pathway norms and every record table
    stay on the device for the life of the cortex. A reading arrives as a NumPy array and is
    uploaded; a read and a code leave as NumPy arrays. The winners come from ``topk``, which
    selects the same set as the NumPy ``argpartition`` whenever the drive at the boundary is
    not tied; the code is scattered into a zero row, so the order inside the set never matters.
    """

    def __init__(self, records: Records, device: str | None, precision: str | None) -> None:
        torch = _torch()
        self.records = records
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)
        if precision not in (None, "float64", "float32"):
            raise ValueError("precision is 'float64' or 'float32'")
        default = "float32" if self.device.type == "mps" else "float64"
        self.precision = precision or default
        if self.device.type == "mps" and self.precision == "float64":
            raise ValueError("mps has no float64; pass precision='float32'")
        self.dtype = torch.float64 if self.precision == "float64" else torch.float32
        t = self._to
        self.projection = t(records.projection)
        self.offset = t(records.offset)
        self._mean = t(records._mean)
        self._pathway_norm = t(records._pathway_norm)
        self._tables = {name: t(table) for name, table in records._tables.items()}
        self.pathways = [torch.as_tensor(p, device=self.device) for p in records.pathways]
        self.tasks = torch.as_tensor(records.tasks, device=self.device)
        self.task_of_cell = torch.as_tensor(records.task_of_cell, device=self.device)
        self.proxy = _Tables(self)

    def _to(self, array: np.ndarray) -> Any:
        return _torch().as_tensor(np.ascontiguousarray(array), dtype=self.dtype, device=self.device)

    def _host(self, tensor: Any) -> np.ndarray:
        return np.asarray(tensor.detach().to("cpu", _torch().float64).numpy(), dtype=float)

    # -- the host's view of the state

    @property
    def mean(self) -> np.ndarray:
        return self._host(self._mean)

    @mean.setter
    def mean(self, value: np.ndarray) -> None:
        self._mean = self._to(np.asarray(value, dtype=float))

    @property
    def pathway_norm(self) -> np.ndarray:
        return self._host(self._pathway_norm)

    @pathway_norm.setter
    def pathway_norm(self, value: np.ndarray) -> None:
        self._pathway_norm = self._to(np.asarray(value, dtype=float))

    @property
    def tables(self) -> _Tables:
        return self.proxy

    def table(self, name: str) -> np.ndarray:
        return self._host(self._tables[name])

    def set_table(self, name: str, value: np.ndarray) -> None:
        array = np.asarray(value, dtype=float)
        shape = (self.records.cells, self.records.fields[name])
        if array.shape != shape:
            raise ValueError(f"the records of {name!r} have shape {shape}")
        self._tables[name] = self._to(array)

    # -- the equations

    def _allowed(self, x: Any) -> Any | None:
        r = self.records
        if not len(r.tasks):
            return None
        units = x[:, self.tasks]
        allowed = self.task_of_cell[None, :] == units.argmax(dim=1)[:, None]
        allowed[units.amax(dim=1) <= 0.0] = True
        return allowed

    def _winners(self, x: Any, allowed: Any | None, out: Any) -> None:
        torch = _torch()
        r = self.records
        drive = x @ self.projection + self.offset
        if allowed is not None:
            drive = torch.where(allowed, drive, torch.full_like(drive, float("-inf")))
        values, index = torch.topk(drive, r.active, dim=1, largest=True, sorted=False)
        values = torch.clamp_min(values, 0.0)
        norm = torch.linalg.vector_norm(values, dim=1, keepdim=True)
        scaled = torch.where(norm > 0, values / torch.clamp_min(norm, 1e-12), values)
        out.zero_()
        out.scatter_(1, index, scaled)

    def code(self, x: np.ndarray, *, adapt: bool, valued: bool) -> np.ndarray:
        torch = _torch()
        r = self.records
        values = self._to(x)
        allowed = self._allowed(values)
        if r.habituation > 0:
            if adapt:
                for row in values:
                    r.seen += 1
                    self._mean += max(r.habituation, 1.0 / r.seen) * (row - self._mean)
            values = values - self._mean
        codes = torch.empty((2, values.shape[0], r.cells), dtype=self.dtype, device=self.device)
        self._winners(values, None, codes[0])
        if self.pathways and adapt:
            for k, pathway in enumerate(self.pathways):
                for value in torch.linalg.vector_norm(values[:, pathway], dim=1):
                    self._pathway_norm[k] += r.pathway_rate * (value - self._pathway_norm[k])
        if not valued:
            codes[1] = float("nan")
        elif self.pathways:
            v = values.clone()
            for k, pathway in enumerate(self.pathways):
                v[:, pathway] /= self._pathway_norm[k] + 1e-3
            self._winners(v, allowed, codes[1])
        elif allowed is not None:
            self._winners(values, allowed, codes[1])
        else:
            codes[1] = codes[0]
        return self._host(codes)

    def read(self, code: np.ndarray) -> dict[str, np.ndarray]:
        r = self.records
        array = np.asarray(code, dtype=float)
        if array.shape[0] != 2 or array.shape[-1] != r.cells:
            raise ValueError(f"a code has shape (2, {r.cells}) or (2, batch, {r.cells})")
        tensor = self._to(array)
        out = {}
        for name, table in self._tables.items():
            selected = tensor[1] if name in r.valued else tensor[0]
            out[name] = self._host(selected @ table)
        return out

    def write(
        self,
        code: np.ndarray,
        targets: Mapping[str, np.ndarray],
        known: Mapping[str, np.ndarray] | None,
    ) -> int:
        torch = _torch()
        r = self.records
        array = np.asarray(code, dtype=float)
        if array.shape != (2, r.cells):
            raise ValueError(f"write takes one code of shape (2, {r.cells})")
        tensor = self._to(array)
        written = 0
        for name, width in r.fields.items():
            if name not in targets:
                continue
            target = np.asarray(targets[name], dtype=float)
            if target.shape != (width,) or not np.isfinite(target).all():
                raise ValueError(f"the target of {name!r} must be a finite vector of width {width}")
            c = tensor[1] if name in r.valued else tensor[0]
            active = torch.nonzero(c, as_tuple=False).reshape(-1)  # the active cells alone
            table = self._tables[name]
            rows = table.index_select(0, active)
            weights = c.index_select(0, active)
            error = self._to(target) - weights @ rows
            if known is not None and name in known:
                error = error * self._to(np.asarray(known[name], dtype=bool).astype(float))
            rate = r.valued_rate if name in r.valued else r.rate
            table.index_copy_(0, active, rows + rate * torch.outer(weights, error))
            written += 1
        r.writes += written
        return written
