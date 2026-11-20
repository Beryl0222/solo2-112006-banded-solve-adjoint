"""变长带状剖面的训练侧求解与反传接口。

把交接协议中的带状方程、边界版本、通道 RHS 与物理系数槽位绑定成一次
具体前向，并在同一份 :class:`BatchSolver` 状态上提供：

* 按声明的带状存储逐段装配（不按长度补齐、不引入额外方程）并分解；
* 低精度优先，证书不达标时在 ``scratch_bytes`` 预算内晋级；
* 每段、每路的残差证书与采用精度；伴随使用同一门限的转置证书；
* 票据绑定旧矩阵、解、未知量范围、边界版本与槽位快照，事后发布不影响它；
* 对 RHS 与可训练槽位反传，多段/多路共享的物理系数汇合梯度；
* 分解复用和晋级暂存计入预算，并报告占用、峰值与无法继续的原因。

数值口径见 ``交接协议.md``。
"""

from __future__ import annotations

import hashlib
import math
from collections import OrderedDict
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np

PRECISIONS: tuple[str, str] = ("fp32", "fp64")
_DTYPES = {"fp32": np.float32, "fp64": np.float64}
_ITEMSIZE = {name: np.dtype(dtype).itemsize for name, dtype in _DTYPES.items()}

VALID = "valid"
REJECTED = "rejected"
OK = "ok"
PARTIAL = "partial"


class CaptureError(ValueError):
    """采集资料或公开 API 输入不符合交接协议。"""


def _finite(value: Any, what: str) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise CaptureError(f"{what} 不是实数") from exc
    if not math.isfinite(number):
        raise CaptureError(f"{what} 不是有限值")
    return number


# ---------------------------------------------------------------------------
# 采集解析
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class _BoundaryPart:
    row: tuple[tuple[int, str, float], ...] | None  # (带状列, 槽位, 乘数)
    add_rhs: tuple[str, float] | None


@dataclass(frozen=True)
class _BoundarySet:
    left: _BoundaryPart
    right: _BoundaryPart
    single: _BoundaryPart


_EMPTY_PART = _BoundaryPart(None, None)


def _parse_row(raw: Any, where: str) -> tuple[tuple[int, str, float], ...] | None:
    if raw is None:
        return None
    if not isinstance(raw, (list, tuple)):
        raise CaptureError(f"{where} 必须是行列表或 null")
    terms: list[tuple[int, str, float]] = []
    for column, entry in enumerate(raw):
        if entry is None:
            continue
        if not isinstance(entry, (list, tuple)) or len(entry) != 2 or not isinstance(entry[0], str):
            raise CaptureError(f"{where} 第 {column} 列必须是 [slot, multiplier] 或 null")
        terms.append(
            (column, entry[0], _finite(entry[1], f"{where} 第 {column} 列乘数"))
        )
    return tuple(terms)


def _parse_add(raw: Any, where: str) -> tuple[str, float] | None:
    if raw is None:
        return None
    if not isinstance(raw, (list, tuple)) or len(raw) != 2 or not isinstance(raw[0], str):
        raise CaptureError(f"{where} 的 add_rhs 必须是 [slot, gain] 或 null")
    return raw[0], _finite(raw[1], f"{where} 的 add_rhs 乘数")


def _parse_boundary(raw: Any, where: str) -> _BoundarySet:
    if raw is None:
        return _BoundarySet(_EMPTY_PART, _EMPTY_PART, _EMPTY_PART)
    if not isinstance(raw, Mapping):
        raise CaptureError(f"{where} 必须是对象")
    parts: dict[str, _BoundaryPart] = {}
    for name in ("left", "right", "single"):
        part = raw.get(name)
        if part is None:
            parts[name] = _EMPTY_PART
        elif not isinstance(part, Mapping):
            raise CaptureError(f"{where}.{name} 必须是对象或 null")
        else:
            parts[name] = _BoundaryPart(
                _parse_row(part.get("row"), f"{where}.{name}.row"),
                _parse_add(part.get("add_rhs"), f"{where}.{name}"),
            )
    return _BoundarySet(**parts)


def _boundary_snapshot(boundary: _BoundarySet) -> dict[str, Any]:
    def part(value: _BoundaryPart) -> dict[str, Any] | None:
        if value.row is None and value.add_rhs is None:
            return None
        row: list[Any] | None = None
        if value.row is not None:
            width = max(column for column, _, _ in value.row) + 1 if value.row else 0
            row = [None] * width
            for column, slot, mult in value.row:
                row[column] = [slot, mult]
        return {
            "row": row,
            "add_rhs": None if value.add_rhs is None else list(value.add_rhs),
        }

    return {"left": part(boundary.left), "right": part(boundary.right), "single": part(boundary.single)}


@dataclass
class _Profile:
    name: str
    active: tuple[bool, ...]
    bands: tuple[tuple[tuple[int, str, float], ...], ...]
    boundary: _BoundarySet


def _parse_profile(name: str, raw: Mapping[str, Any], offsets: Sequence[int]) -> _Profile:
    where = f"profiles.{name}"
    active = raw.get("active")
    bands = raw.get("bands")
    if not isinstance(active, list) or not active or any(not isinstance(flag, bool) for flag in active):
        raise CaptureError(f"{where}.active 必须是非空布尔列表")
    if not isinstance(bands, list) or len(bands) != len(active):
        raise CaptureError(f"{where}.bands 必须与 active 等长")
    parsed: list[tuple[tuple[int, str, float], ...]] = []
    for i, row in enumerate(bands):
        if not isinstance(row, list):
            raise CaptureError(f"{where}.bands[{i}] 必须是行列表")
        if len(row) != len(offsets):
            raise CaptureError(f"{where}.bands[{i}] 的列数必须等于 offsets 的长度")
        terms: list[tuple[int, str, float]] = []
        for column, entry in enumerate(row):
            if entry is None:
                continue
            if not isinstance(entry, (list, tuple)) or len(entry) != 2 or not isinstance(entry[0], str):
                raise CaptureError(f"{where}.bands[{i}][{column}] 必须是 [slot, multiplier] 或 null")
            terms.append((column, entry[0], _finite(entry[1], f"{where}.bands[{i}][{column}] 乘数")))
        parsed.append(tuple(terms))
    boundary = _parse_boundary(raw.get("boundary"), f"{where}.boundary")
    return _Profile(name, tuple(active), tuple(parsed), boundary)


# ---------------------------------------------------------------------------
# 按带状存储装配：分段、边界行、边界常量
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class _Segment:
    index: int
    points: tuple[int, ...]
    start: int
    stop: int
    terms: tuple[tuple[int, int, str, float], ...]  # (行, 列, 槽位, 乘数)
    constants: tuple[tuple[int, str, float], ...]   # (行, 槽位, 增益)

    @property
    def n(self) -> int:
        return len(self.points)

    @property
    def range(self) -> tuple[int, int]:
        return self.start, self.stop + 1


def _map_boundary(
    row: tuple[tuple[int, str, float], ...],
    base: int,
    start: int,
    n: int,
    offsets: Sequence[int],
    local_row: int,
) -> list[tuple[int, int, str, float]]:
    """把边界行从点 base 向 offsets 展开；只保留本段内的未知量。"""
    terms: list[tuple[int, int, str, float]] = []
    for column, slot, multiplier in row:
        if column >= len(offsets):
            break
        local = base + offsets[column] - start
        if 0 <= local < n:
            terms.append((local_row, local, slot, multiplier))
    return terms


def _compile_segments(profile: _Profile, offsets: Sequence[int], boundary: _BoundarySet) -> list[_Segment]:
    """把内点掩码的每段连续有效点编译成独立带状系统。

    断开处的耦合不属于任何段；单点段优先使用 ``single``，否则左侧约束或
    右侧约束替代该点唯一的方程。补齐空间不生成方程。
    """
    active = profile.active
    segments: list[_Segment] = []
    index = 0
    start = 0
    while start < len(active):
        if not active[start]:
            start += 1
            continue
        stop = start
        while stop + 1 < len(active) and active[stop + 1]:
            stop += 1
        n = stop - start + 1
        rows: list[list[tuple[int, int, str, float]]] = [[] for _ in range(n)]
        for i in range(start, stop + 1):
            local_row = i - start
            for column, slot, multiplier in profile.bands[i]:
                local_col = i + offsets[column] - start
                if 0 <= local_col < n:
                    rows[local_row].append((local_row, local_col, slot, multiplier))
        constants: list[tuple[int, str, float]] = []
        if n == 1:
            # 单点段只采用 single；未给出 single 时保留点自身的方程，
            # 不借用 left/right，以免补齐空间引入额外方程。
            if boundary.single.row is not None:
                rows[0] = _map_boundary(boundary.single.row, start, start, n, offsets, 0)
                if boundary.single.add_rhs is not None:
                    slot, gain = boundary.single.add_rhs
                    constants.append((0, slot, gain))
        else:
            if boundary.left.row is not None:
                rows[0] = _map_boundary(boundary.left.row, start, start, n, offsets, 0)
                if boundary.left.add_rhs is not None:
                    slot, gain = boundary.left.add_rhs
                    constants.append((0, slot, gain))
            if boundary.right.row is not None:
                local_row = n - 1
                rows[local_row] = _map_boundary(boundary.right.row, stop, start, n, offsets, local_row)
                if boundary.right.add_rhs is not None:
                    slot, gain = boundary.right.add_rhs
                    constants.append((local_row, slot, gain))
        segments.append(
            _Segment(
                index=index,
                points=tuple(range(start, stop + 1)),
                start=start,
                stop=stop,
                terms=tuple(term for row in rows for term in row),
                constants=tuple(constants),
            )
        )
        index += 1
        start = stop + 1
    return segments


def _assemble(
    segment: _Segment, slots: Mapping[str, tuple[float, bool]]
) -> tuple[_Storage, np.ndarray]:
    """按本段真实耦合装配方程与边界常量；补齐位置不生成列或方程。"""
    n = segment.n
    evaluated: list[tuple[int, int, float]] = []
    const = np.zeros(n, dtype=np.float64)
    with np.errstate(all="ignore"):
        for row, column, slot, multiplier in segment.terms:
            if slot not in slots:
                raise CaptureError(f"引用了未声明的槽位 {slot!r}")
            evaluated.append((row, column, slots[slot][0] * multiplier))
        for row, slot, gain in segment.constants:
            if slot not in slots:
                raise CaptureError(f"引用了未声明的槽位 {slot!r}")
            const[row] += slots[slot][0] * gain
    storage = _storage_for(n, evaluated)
    for row, column, value in evaluated:
        if storage.kind == "band":
            storage.data[storage._band_index(row, column), column] += value
        else:
            storage.data[row, column] += value
    return storage, const


# ---------------------------------------------------------------------------
# 方程存储与部分主元 LU
# ---------------------------------------------------------------------------


class _Singular(Exception):
    pass


@dataclass(frozen=True)
class _Storage:
    """一个段的实际方程存储，两种布局都只存真实存在的系数。

    * ``"band"``：按列带状，列数等于段长，行数 ``2*kl+ku+1`` 覆盖 LU 填充；
    * ``"dense"``：窄段按列紧凑，避免布局开销大于方程本身。

    剖面补齐位置从不生成列，因此补齐空间不进入任何布局，也不生成方程。
    """

    kind: str
    n: int
    kl: int
    ku: int
    data: np.ndarray

    @property
    def nbytes(self) -> int:
        return int(self.data.nbytes)

    def _band_index(self, i: int, j: int) -> int:
        return self.kl + self.ku + i - j

    def coefficients(self):
        """逐个产出可能非零项 ``(行, 列, 系数)``，跳过未存储的填充位。"""
        if self.kind == "band":
            for j in range(self.n):
                column = self.data[:, j]
                for offset in np.flatnonzero(column):
                    row = j + int(offset) - (self.kl + self.ku)
                    if 0 <= row < self.n:
                        yield row, j, float(column[offset])
        else:
            for j in range(self.n):
                column = self.data[:, j]
                for i in np.flatnonzero(column):
                    yield int(i), j, float(column[i])

    def get(self, i: int, j: int) -> float:
        if not (0 <= i < self.n and 0 <= j < self.n):
            return 0.0
        if self.kind == "band":
            if i < j - self.ku or i > j + self.kl:
                return 0.0
            return float(self.data[self._band_index(i, j), j])
        return float(self.data[i, j])

    def matvec(self, vector: np.ndarray) -> np.ndarray:
        out = np.zeros(self.n, dtype=np.float64)
        with np.errstate(all="ignore"):
            for i, j, value in self.coefficients():
                out[i] += value * vector[j]
        return out

    def transpose_matvec(self, vector: np.ndarray) -> np.ndarray:
        out = np.zeros(self.n, dtype=np.float64)
        with np.errstate(all="ignore"):
            for i, j, value in self.coefficients():
                out[j] += value * vector[i]
        return out

    def max_row_sum(self) -> float:
        """矩阵无穷范数：最大绝对行和。"""
        sums = np.zeros(self.n, dtype=np.float64)
        for i, _, value in self.coefficients():
            sums[i] += abs(value)
        return float(sums.max()) if sums.size else 0.0


def _storage_for(n: int, terms: Sequence[tuple[int, int, float]]) -> _Storage:
    """按本段真实耦合选择布局：长段用带状存储，窄段用紧凑按列存储。"""
    if n == 0:
        return _Storage("dense", 0, 0, 0, np.zeros((0, 0), dtype=np.float64))
    kl = 0
    ku = 0
    for row, column, _ in terms:
        if row > column:
            kl = max(kl, row - column)
        elif column > row:
            ku = max(ku, column - row)
    band_rows = 2 * kl + ku + 1
    if band_rows >= n:
        return _Storage("dense", n, kl, ku, np.zeros((n, n), dtype=np.float64))
    return _Storage("band", n, kl, ku, np.zeros((band_rows, n), dtype=np.float64))


@dataclass
class _Factor:
    """部分主元 LU。LU 与主元向量即分解所占的全部暂存。"""

    storage: _Storage
    lu: np.ndarray
    pivots: np.ndarray
    min_pivot: float

    @property
    def n(self) -> int:
        return self.storage.n

    @property
    def nbytes(self) -> int:
        return int(self.lu.nbytes + self.pivots.nbytes)


def _factor(storage: _Storage, dtype: Any) -> _Factor:
    n = storage.n
    if n == 0:
        raise _Singular
    if storage.kind == "dense":
        lu = storage.data.astype(dtype, copy=True)
        pivots = np.zeros(n, dtype=np.int64)
        min_pivot = math.inf
        with np.errstate(all="ignore"):
            for k in range(n):
                column = np.abs(lu[k:, k])
                offset = int(np.argmax(column))
                pivot = float(column[offset])
                if not math.isfinite(pivot) or pivot == 0.0:
                    raise _Singular
                min_pivot = min(min_pivot, pivot)
                p = k + offset
                pivots[k] = p
                if p != k:
                    lu[[k, p], :] = lu[[p, k], :]
                if k + 1 < n:
                    lu[k + 1 :, k] /= lu[k, k]
                    lu[k + 1 :, k + 1 :] -= np.outer(lu[k + 1 :, k], lu[k, k + 1 :])
        return _Factor(storage, lu, pivots, min_pivot)

    kl, ku = storage.kl, storage.ku
    lu = storage.data.astype(dtype, copy=True)
    index = storage._band_index
    pivots = np.zeros(n, dtype=np.int64)
    min_pivot = math.inf
    with np.errstate(all="ignore"):
        for k in range(n):
            last_row = min(n - 1, k + kl)
            rows = np.arange(k, last_row + 1)
            column = np.abs(lu[index(rows, k), k])
            offset = int(np.argmax(column))
            pivot = float(column[offset])
            if not math.isfinite(pivot) or pivot == 0.0:
                raise _Singular
            min_pivot = min(min_pivot, pivot)
            p = k + offset
            pivots[k] = p
            if p != k:
                stop = min(n - 1, k + kl + ku)
                for j in range(k, stop + 1):
                    a = lu[index(k, j), j]
                    lu[index(k, j), j] = lu[index(p, j), j]
                    lu[index(p, j), j] = a
            pivot_value = lu[index(k, k), k]
            for i in range(k + 1, last_row + 1):
                multiplier = lu[index(i, k), k] / pivot_value
                lu[index(i, k), k] = multiplier
                if multiplier == 0.0:
                    continue
                stop = min(n - 1, k + kl + ku)
                if k + 1 > stop:
                    continue
                cols = np.arange(k + 1, stop + 1)
                lu[index(i, cols), cols] -= multiplier * lu[index(k, cols), cols]
    return _Factor(storage, lu, pivots, min_pivot)


def _factor_solve(factor: _Factor, rhs: np.ndarray) -> np.ndarray:
    """解 A x = b：按记录的换行顺序应用 L 前代与 U 回代。"""
    n = factor.n
    lu = factor.lu
    pivots = factor.pivots
    dtype = lu.dtype
    with np.errstate(all="ignore"):
        y = np.array(rhs, dtype=dtype, copy=True)
        if factor.storage.kind == "dense":
            # 稠密布局按 P A = L U 收集全部行交换：先整体换行，再前代/回代。
            for k in range(n):
                p = int(pivots[k])
                if p != k:
                    y[k], y[p] = y[p], y[k]
            for k in range(n):
                if k + 1 < n:
                    y[k + 1 :] -= lu[k + 1 :, k] * y[k]
            for i in range(n - 1, -1, -1):
                y[i] = (y[i] - lu[i, i + 1 :] @ y[i + 1 :]) / lu[i, i]
            return y
        kl, ku = factor.storage.kl, factor.storage.ku
        index = factor.storage._band_index
        for k in range(n):
            p = int(pivots[k])
            if p != k:
                y[k], y[p] = y[p], y[k]
            last = min(n - 1, k + kl)
            if k + 1 <= last:
                rows = np.arange(k + 1, last + 1)
                y[rows] -= lu[index(rows, k), k] * y[k]
        for i in range(n - 1, -1, -1):
            stop = min(n - 1, i + kl + ku)
            if i + 1 <= stop:
                rows = np.arange(i + 1, stop + 1)
                y[i] -= float(lu[index(i, rows), rows] @ y[rows])
            y[i] /= lu[index(i, i), i]
    return y


def _factor_solve_transpose(factor: _Factor, rhs: np.ndarray) -> np.ndarray:
    """解 A^T y = c，复用同一份 LU 与主元记录。"""
    n = factor.n
    lu = factor.lu
    pivots = factor.pivots
    dtype = lu.dtype
    with np.errstate(all="ignore"):
        w = np.array(rhs, dtype=dtype, copy=True)
        if factor.storage.kind == "dense":
            # A = P^T L U，故 A^T = U^T L^T P：先解 U^T，再解 L^T，
            # 最后按 S_0 … S_{n-1} 的顺序还原行交换（P^T）。
            for i in range(n):
                if i:
                    w[i] = (w[i] - lu[:i, i] @ w[:i]) / lu[i, i]
                else:
                    w[i] = w[i] / lu[i, i]
            for i in range(n - 1, -1, -1):
                if i + 1 < n:
                    w[i] -= lu[i + 1 :, i] @ w[i + 1 :]
            for k in range(n):
                p = int(pivots[k])
                if p != k:
                    w[k], w[p] = w[p], w[k]
            return w
        kl, ku = factor.storage.kl, factor.storage.ku
        index = factor.storage._band_index
        for i in range(n):
            start = max(0, i - kl - ku)
            if start < i:
                rows = np.arange(start, i)
                w[i] -= float(lu[index(rows, i), i] @ w[rows])
            w[i] /= lu[index(i, i), i]
        for k in range(n - 1, -1, -1):
            last = min(n - 1, k + kl)
            if k + 1 <= last:
                rows = np.arange(k + 1, last + 1)
                w[k] -= float(lu[index(rows, k), k] @ w[rows])
            p = int(pivots[k])
            if p != k:
                w[k], w[p] = w[p], w[k]
    return w


def _certificate(storage: _Storage, x: np.ndarray, b: np.ndarray) -> float:
    """||Ax-b||∞ / (||A||∞ ||x||∞ + ||b||∞)；分母为零时残差为零。"""
    with np.errstate(all="ignore"):
        residual = float(np.linalg.norm(storage.matvec(x) - b, np.inf))
        denominator = float(
            storage.max_row_sum() * np.linalg.norm(x, np.inf) + np.linalg.norm(b, np.inf)
        )
    if denominator == 0.0:
        return 0.0
    return residual / denominator


def _absolute_residual(storage: _Storage, x: np.ndarray, b: np.ndarray) -> float:
    """对原方程的无穷范数残差 ||Ax-b||∞（训练端报告用）。"""
    with np.errstate(all="ignore"):
        return float(np.linalg.norm(storage.matvec(x) - b, np.inf))


def _transpose_certificate(storage: _Storage, y: np.ndarray, c: np.ndarray) -> float:
    """转置方程 A^T y = c 的同一口径证书。

    分母按协议使用 A 的无穷范数（最大绝对行和），不是 A^T 的行和。
    """
    with np.errstate(all="ignore"):
        residual = float(np.linalg.norm(storage.transpose_matvec(y) - c, np.inf))
        denominator = float(
            storage.max_row_sum() * np.linalg.norm(y, np.inf) + np.linalg.norm(c, np.inf)
        )
    if denominator == 0.0:
        return 0.0
    return residual / denominator


# ---------------------------------------------------------------------------
# 携带预算记账的分解缓存
# ---------------------------------------------------------------------------


def _factor_bytes(storage: "_Storage", precision: str) -> int:
    """按本段实际布局与目标精度估算一次分解的暂存（LU + 主元向量）。

    与 ``_Factor.nbytes`` 一致：区分 ``dense``/``band`` 的真实数据量。
    """
    if storage.n <= 0:
        return 0
    itemsize = _ITEMSIZE[precision]
    if storage.kind == "dense":
        cells = storage.n * storage.n
    else:
        cells = (2 * storage.kl + storage.ku + 1) * storage.n
    return int(cells * itemsize + storage.n * 8)


def _work_bytes(n: int, precision: str) -> int:
    return _ITEMSIZE[precision] * max(n, 1)


@dataclass
class _CacheEntry:
    digest: str
    precision: str
    factor: _Factor
    pivot_ratio: float
    n: int
    refs: int = 0

    @property
    def nbytes(self) -> int:
        return int(self.factor.nbytes)


@dataclass
class _Attempt:
    entry: _CacheEntry
    solution: np.ndarray
    certificate: float
    residual: float
    precision: str
    reused: bool
    peak: int
    attempts: tuple[str, ...]


def _budget_reason(needed: int, budget: int) -> str:
    return f"暂存预算不足：需要 {needed} 字节，预算 {budget} 字节"


# ---------------------------------------------------------------------------
# 票据记录
# ---------------------------------------------------------------------------


@dataclass
class _SegmentRecord:
    segment: _Segment
    storage: "_Storage"
    const: np.ndarray
    rhs: np.ndarray
    accepted: bool
    reason: str | None
    precision: str | None
    certificate: float | None
    pivot_ratio: float | None
    retried: tuple[str, ...]
    entry: _CacheEntry | None
    solution: np.ndarray | None
    residual: float | None = None


@dataclass
class _ChannelRecord:
    key: str
    segments: tuple[_SegmentRecord, ...]
    accepted: bool
    reason: str | None
    precision: str | None
    precisions: tuple[str, ...]
    certificate: float | None
    retried: tuple[str, ...]
    slots_used: tuple[str, ...]


class _Ticket:
    """一次已发出前向的不可变快照；发布新边界不会改动它。"""

    def __init__(
        self,
        label: str,
        profile: str,
        version: str,
        boundary: Mapping[str, Any],
        slots: Mapping[str, tuple[float, bool]],
        channels: Mapping[str, _ChannelRecord],
    ):
        self.label = label
        self.profile = profile
        self.version = version
        self.boundary = {
            "left": _copy_part(boundary.get("left")),
            "right": _copy_part(boundary.get("right")),
            "single": _copy_part(boundary.get("single")),
        }
        self.slots = dict(slots)
        self.channels = dict(channels)
        self.entries: list[_CacheEntry] = [
            segment.entry
            for record in channels.values()
            for segment in record.segments
            if segment.entry is not None
        ]


def _copy_part(part: Any) -> Any:
    if not isinstance(part, Mapping):
        return None
    row = part.get("row")
    return {
        "row": None if row is None else [None if entry is None else list(entry) for entry in row],
        "add_rhs": None if part.get("add_rhs") is None else list(part["add_rhs"]),
    }


# ---------------------------------------------------------------------------
# 公开结果
# ---------------------------------------------------------------------------


@dataclass
class ChannelResult:
    key: str
    status: str
    accepted: bool
    reason: str | None
    precision: str | None
    certificate: float | None
    retried: tuple[str, ...]
    segments: tuple[dict[str, Any], ...]
    boundary_version: str
    precisions: tuple[str, ...]
    trainable_slots: tuple[str, ...]
    fixed_slots: tuple[str, ...]
    ranges: tuple[tuple[int, int], ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "key": self.key,
            "status": self.status,
            "accepted": self.accepted,
            "reason": self.reason,
            "precision": self.precision,
            "precisions": list(self.precisions),
            "certificate": self.certificate,
            "retried": list(self.retried),
            "boundary_version": self.boundary_version,
            "trainable_slots": list(self.trainable_slots),
            "fixed_slots": list(self.fixed_slots),
            "ranges": [list(item) for item in self.ranges],
            "segments": list(self.segments),
        }


@dataclass
class SolveReport:
    label: str
    profile: str
    version: str
    channels: dict[str, ChannelResult]
    budget: dict[str, Any]
    numeric_tolerance: float

    def __getitem__(self, key: str) -> ChannelResult:
        return self.channels[key]

    @property
    def accepted_keys(self) -> tuple[str, ...]:
        return tuple(key for key, item in self.channels.items() if item.accepted)

    def results(self) -> list[dict[str, Any]]:
        """训练端查询：每路证书、绑定的边界版本与可更新系数。"""
        return [item.to_dict() for item in self.channels.values()]

    def continuation(self) -> dict[str, Any]:
        """能继续训练的范围与剩余资源原因。

        ``accepted`` 列出可继续（含可继续的段范围）的通道；``rejected``
        保留被拒绝通道的原因；``budget`` 给出分解复用与晋级暂存的占用、
        峰值与剩余额度，``reason`` 指出是否有资源导致的拒绝。
        """
        accepted: list[dict[str, Any]] = []
        rejected: list[dict[str, Any]] = []
        for item in self.channels.values():
            row = {
                "key": item.key,
                "ranges": [list(region) for region in item.ranges],
                "precision": item.precision,
                "reason": item.reason,
                "boundary_version": item.boundary_version,
                "trainable_slots": list(item.trainable_slots),
            }
            (accepted if item.accepted else rejected).append(row)
        return {
            "label": self.label,
            "profile": self.profile,
            "boundary_version": self.version,
            "numeric_tolerance": self.numeric_tolerance,
            "accepted": accepted,
            "rejected": rejected,
            "can_continue": bool(accepted),
            "budget": self.budget,
        }

    def to_dict(self) -> dict[str, Any]:
        return {
            "label": self.label,
            "profile": self.profile,
            "version": self.version,
            "channels": {key: item.to_dict() for key, item in self.channels.items()},
            "budget": self.budget,
            "continuation": self.continuation(),
        }


@dataclass
class BackwardReport:
    label: str
    keys: tuple[str, ...]
    status: str
    accepted_keys: tuple[str, ...]
    rejected_keys: tuple[str, ...]
    reason: str | None
    boundary_version: str
    rhs_grads: dict[str, np.ndarray]
    coefficient_grads: dict[str, float]
    coefficient_grads_by_key: dict[str, dict[str, float]]
    trainable_slots: tuple[str, ...]
    segments: dict[str, tuple[dict[str, Any], ...]]
    budget: dict[str, Any]

    @property
    def rhs_grad(self) -> np.ndarray | None:
        if len(self.keys) == 1:
            return self.rhs_grads.get(self.keys[0])
        return None

    def to_dict(self) -> dict[str, Any]:
        return {
            "label": self.label,
            "keys": list(self.keys),
            "status": self.status,
            "accepted_keys": list(self.accepted_keys),
            "rejected_keys": list(self.rejected_keys),
            "reason": self.reason,
            "boundary_version": self.boundary_version,
            "rhs_grads": {key: value.tolist() for key, value in self.rhs_grads.items()},
            "coefficient_grads": dict(self.coefficient_grads),
            "coefficient_grads_by_key": {key: dict(value) for key, value in self.coefficient_grads_by_key.items()},
            "trainable_slots": list(self.trainable_slots),
            "segments": {key: list(value) for key, value in self.segments.items()},
            "budget": self.budget,
        }


# ---------------------------------------------------------------------------
# 求解器
# ---------------------------------------------------------------------------


class BatchSolver:
    """一批带状剖面的具体前向、证书与反传。"""

    def __init__(self, capture: Mapping[str, Any], *, budget_bytes: int | None = None):
        if not isinstance(capture, Mapping):
            raise CaptureError("采集资料必须是对象")
        offsets = capture.get("offsets")
        if (
            not isinstance(offsets, list)
            or not offsets
            or any(not isinstance(value, int) or isinstance(value, bool) for value in offsets)
        ):
            raise CaptureError("offsets 必须是非空整数列表")
        if len(set(offsets)) != len(offsets):
            raise CaptureError("offsets 不能重复")
        self.offsets = tuple(offsets)
        self.tol = _finite(capture.get("tol"), "tol")
        self.pivot_floor = _finite(capture.get("pivot_floor"), "pivot_floor")
        if self.tol < 0 or self.pivot_floor < 0:
            raise CaptureError("tol 与 pivot_floor 不能为负")
        declared = _finite(capture.get("scratch_bytes"), "scratch_bytes")
        self.budget_bytes = int(declared) if budget_bytes is None else int(budget_bytes)
        if self.budget_bytes < 0:
            raise CaptureError("scratch_bytes 不能为负")

        slots = capture.get("slots")
        if not isinstance(slots, Mapping) or not slots:
            raise CaptureError("slots 必须是非空对象")
        self.slots: dict[str, tuple[float, bool]] = {}
        for name, value in slots.items():
            if not isinstance(name, str) or not isinstance(value, (list, tuple)) or len(value) != 2:
                raise CaptureError(f"槽位 {name!r} 必须是 [value, trainable]")
            if not isinstance(value[1], bool):
                raise CaptureError(f"槽位 {name!r} 的 trainable 必须是布尔值")
            self.slots[name] = (_finite(value[0], f"slots.{name}"), value[1])

        profiles = capture.get("profiles")
        if not isinstance(profiles, Mapping) or not profiles:
            raise CaptureError("profiles 必须是非空对象")
        self.profiles: dict[str, _Profile] = {}
        for name, raw in profiles.items():
            if not isinstance(name, str) or not isinstance(raw, Mapping):
                raise CaptureError(f"profiles.{name} 必须是对象")
            self.profiles[name] = _parse_profile(name, raw, self.offsets)

        rhs = capture.get("rhs")
        if not isinstance(rhs, Mapping):
            raise CaptureError("rhs 必须是对象")
        self._rhs: dict[str, np.ndarray] = {}
        for key, values in rhs.items():
            if not isinstance(key, str) or not isinstance(values, list):
                raise CaptureError(f"rhs.{key} 必须是数值列表")
            self._rhs[key] = np.array([_finite(v, f"rhs.{key}") for v in values], dtype=np.float64)

        self._cache: "OrderedDict[tuple[str, str], _CacheEntry]" = OrderedDict()
        self._used = 0
        self._peak = 0
        self._evictions = 0
        self._hits = 0
        self._misses = 0
        self._promotions: list[dict[str, Any]] = []
        self._last_budget_reason: str | None = None
        self._tickets: dict[str, _Ticket] = {}
        initial = {name: profile.boundary for name, profile in self.profiles.items()}
        self._versions: dict[str, dict[str, _BoundarySet]] = {"__initial__": initial}
        self._current_version = "__initial__"

    # -- 预算 -------------------------------------------------------------

    def _evict_until(self, limit: int) -> None:
        """按 LRU 淘汰无票据引用的分解，直到占用不超过 limit。"""
        for key in list(self._cache):
            if self._used <= limit:
                return
            entry = self._cache[key]
            if entry.refs == 0:
                self._used -= entry.nbytes
                del self._cache[key]
                self._evictions += 1

    def _reserve(self, extra_bytes: int, transient: int) -> tuple[bool, int]:
        """为一次分解与晋级暂存腾地方。

        ``transient`` 是同一段在低精度失败后仍要保留的晋级暂存；被票据
        引用的分解不可淘汰，必要时直接判定预算不足，把原因交还给训练端。
        """
        limit = self.budget_bytes - transient - extra_bytes
        if limit >= 0:
            self._evict_until(limit)
        needed = self._used + transient + extra_bytes
        return needed <= self.budget_bytes, needed

    def budget_report(self) -> dict[str, Any]:
        return {
            "budget_bytes": self.budget_bytes,
            "used_bytes": self._used,
            "remaining_bytes": self.budget_bytes - self._used,
            "peak_bytes": self._peak,
            "evictions": self._evictions,
            "hits": self._hits,
            "misses": self._misses,
            "promotions": list(self._promotions),
            "reason": self._last_budget_reason,
        }

    def cache_report(self) -> dict[str, Any]:
        """分解复用的当前内容；命中键是矩阵指纹而非剖面长度。"""
        return {
            **self.budget_report(),
            "entries": [
                {
                    "digest": entry.digest[:16],
                    "n": entry.n,
                    "precision": entry.precision,
                    "refs": entry.refs,
                    "bytes": entry.nbytes,
                }
                for entry in self._cache.values()
            ],
        }

    def _cached_better(self, digest: str, index: int) -> tuple[_CacheEntry, int] | None:
        for offset, precision in enumerate(PRECISIONS[index + 1 :], start=index + 1):
            entry = self._cache.get((digest, precision))
            if entry is not None:
                self._cache.move_to_end((digest, precision))
                return entry, offset
        return None

    def update_rhs(self, key: str, values: Sequence[float]) -> None:
        """为随后的前向给某通道换上新的 RHS。

        系数矩阵不变时，缓存仍按矩阵指纹命中；旧票据继续持有旧右端项，
        不会因为这次更新而改变。
        """
        if not isinstance(key, str) or not key:
            raise CaptureError("通道名必须是非空字符串")
        array = np.array([_finite(v, f"rhs.{key}") for v in _finite_seq(values, "rhs")])
        self._rhs[key] = array

    def release(self, label: str) -> None:
        """显式释放票据；其分解若无其他票据引用则立即归还暂存。"""
        ticket = self._tickets.pop(label, None)
        if ticket is None:
            raise CaptureError(f"未知票据 {label!r}")
        released = {id(entry) for entry in ticket.entries}
        for entry in ticket.entries:
            entry.refs = max(0, entry.refs - 1)
        for key in list(self._cache):
            entry = self._cache[key]
            if entry.refs == 0 and id(entry) in released:
                self._used -= entry.nbytes
                del self._cache[key]
        self._evict_until(self.budget_bytes)

    # -- 版本 -------------------------------------------------------------

    def publish(self, version: str, definitions: Mapping[str, Any]) -> None:
        """发布随后受理的边界版本；已发出的票据保持旧定义。"""
        if not isinstance(version, str) or not version:
            raise CaptureError("版本标签必须是非空字符串")
        if version in self._versions:
            raise CaptureError(f"版本 {version!r} 已存在")
        if not isinstance(definitions, Mapping) or not definitions:
            raise CaptureError("publish 必须给出至少一个剖面的边界")
        updated = dict(self._versions[self._current_version])
        for profile, raw in definitions.items():
            if profile not in self.profiles:
                raise CaptureError(f"publish 指向未知剖面 {profile!r}")
            updated[profile] = _parse_boundary(raw, f"publish.{profile}")
        self._versions[version] = updated
        self._current_version = version

    def version(self) -> str:
        return self._current_version

    def boundary_of(self, profile: str, version: str | None = None) -> dict[str, Any]:
        name = self._current_version if version is None else version
        if name not in self._versions:
            raise CaptureError(f"未知版本 {name!r}")
        if profile not in self.profiles:
            raise CaptureError(f"未知剖面 {profile!r}")
        return _boundary_snapshot(self._versions[name][profile])

    # -- 前向 -------------------------------------------------------------

    @staticmethod
    def _digest(storage: "_Storage") -> str:
        """缓存键取实际方程指纹（布局 + 系数），不取剖面长度或边界版本。"""
        hasher = hashlib.sha256()
        hasher.update(
            np.asarray(
                [storage.n, storage.kl, storage.ku, 0 if storage.kind == "dense" else 1],
                dtype=np.int64,
            ).tobytes()
        )
        hasher.update(np.ascontiguousarray(storage.data, dtype=np.float64).tobytes())
        return hasher.hexdigest()

    def _validate(
        self, entry: _CacheEntry, storage: "_Storage", rhs: np.ndarray
    ) -> tuple[np.ndarray | None, float | None, str | None]:
        dtype = _DTYPES[entry.precision]
        with np.errstate(all="ignore"):
            try:
                raw = _factor_solve(entry.factor, rhs.astype(dtype))
            except (ValueError, FloatingPointError):
                raw = None
        if raw is None or not np.all(np.isfinite(raw)):
            return None, None, "解非有限"
        solution = raw.astype(np.float64)
        with np.errstate(all="ignore"):
            certificate = _certificate(storage, solution, rhs)
            residual = _absolute_residual(storage, solution, rhs)
        if not math.isfinite(certificate) or not math.isfinite(residual):
            return None, None, "证书非有限"
        if entry.pivot_ratio < self.pivot_floor:
            return None, None, (
                f"主元比率 {entry.pivot_ratio:.3e} 低于门限 {self.pivot_floor:.3e}"
            )
        if certificate > self.tol:
            return None, None, f"残差证书 {certificate:.3e} 大于门限 {self.tol:.3e}"
        return solution, certificate, None

    def _obtain(
        self, storage: "_Storage", rhs: np.ndarray, trace: list[dict[str, Any]]
    ) -> _Attempt | None:
        """低精度优先地分解并验证；必要时在预算内晋级。失败返回 None 并写 trace。"""
        if not np.all(np.isfinite(storage.data)) or not np.all(np.isfinite(rhs)):
            trace.append({"precision": None, "rejected": "输入含非有限值"})
            return None
        n = storage.n
        if n == 0:
            trace.append({"precision": None, "rejected": "剖面没有可用段"})
            return None
        with np.errstate(all="ignore"):
            norm = storage.max_row_sum()
        digest = self._digest(storage)
        transient = 0
        attempts: list[str] = []
        # 晋级标明来源与目标：同时为近奇异样本晋级并保留旧票据时，训练端
        # 需要知道冻结了哪个分解、临时占用了多少，以及谁被迫停下。
        promotions: list[dict[str, Any]] = []
        index = 0
        last_reason = "没有可尝试的精度"
        while index < len(PRECISIONS):
            precision = PRECISIONS[index]
            work = _work_bytes(n, precision)
            key = (digest, precision)
            entry = self._cache.get(key)
            reused = entry is not None
            if reused:
                self._hits += 1
                self._cache.move_to_end(key)
                factor = 0
                ok, needed = self._reserve(work, transient)
            else:
                self._misses += 1
                factor = _factor_bytes(storage, precision)
                ok, needed = self._reserve(factor + work, transient)
            if not ok:
                fallback = self._cached_better(digest, index)
                if fallback is None:
                    reason = _budget_reason(needed, self.budget_bytes)
                    self._last_budget_reason = reason
                    trace.append(
                        {
                            "precision": precision,
                            "rejected": reason,
                            "needed": needed,
                            "transient": transient,
                        }
                    )
                    self._promotions.extend(promotions)
                    return None
                entry, index = fallback
                precision = entry.precision
                reused = True
                factor = 0
                work = _work_bytes(n, precision)
                ok, needed = self._reserve(work, transient)
                if not ok:
                    reason = _budget_reason(needed, self.budget_bytes)
                    self._last_budget_reason = reason
                    trace.append(
                        {
                            "precision": precision,
                            "rejected": reason,
                            "needed": needed,
                            "transient": transient,
                        }
                    )
                    self._promotions.extend(promotions)
                    return None
            attempts.append(precision)
            if not reused:
                with np.errstate(all="ignore"):
                    try:
                        factor_data = _factor(storage, _DTYPES[precision])
                    except _Singular:
                        factor_data = None
                if factor_data is None:
                    last_reason = "奇异矩阵（零主元）"
                    trace.append({"precision": precision, "rejected": last_reason})
                    transient += factor + work
                    index += 1
                    continue
                ratio = 0.0 if norm == 0.0 else float(factor_data.min_pivot) / norm
                entry = _CacheEntry(digest, precision, factor_data, ratio, n)

            peak = self._used + transient + (0 if reused else factor) + work
            self._peak = max(self._peak, peak)
            solution, certificate, failure = self._validate(entry, storage, rhs)
            if failure is not None:
                last_reason = failure
                trace.append({"precision": entry.precision, "rejected": failure})
                if reused:
                    transient += work
                else:
                    transient += factor + work
                if index + 1 < len(PRECISIONS):
                    promotions.append(
                        {
                            "n": n,
                            "from": entry.precision,
                            "to": PRECISIONS[index + 1],
                            "reason": failure,
                            "transient_bytes": transient,
                        }
                    )
                index = PRECISIONS.index(entry.precision) + 1
                continue

            assert solution is not None and certificate is not None
            if not reused:
                if self._used + entry.nbytes > self.budget_bytes:
                    self._evict_until(self.budget_bytes - entry.nbytes)
                if self._used + entry.nbytes > self.budget_bytes:
                    reason = _budget_reason(self._used + entry.nbytes, self.budget_bytes)
                    self._last_budget_reason = reason
                    trace.append({"precision": entry.precision, "rejected": reason})
                    return None
                self._cache[(digest, entry.precision)] = entry
                self._used += entry.nbytes
            entry.refs += 1
            self._promotions.extend(promotions)
            with np.errstate(all="ignore"):
                residual = _absolute_residual(storage, solution, rhs)
            return _Attempt(
                entry,
                solution,
                certificate,
                residual,
                entry.precision,
                reused,
                peak,
                tuple(attempts),
            )
        trace.append({"precision": None, "rejected": last_reason, "retried": attempts})
        self._promotions.extend(promotions)
        return None

    def _resolve(self, profile_name: str) -> tuple[list[_Segment], _BoundarySet]:
        profile = self.profiles[profile_name]
        boundary = self._versions[self._current_version][profile_name]
        return _compile_segments(profile, self.offsets, boundary), boundary

    def solve(self, label: str, profile: str, rhs_keys: Sequence[str]) -> SolveReport:
        """为通道取连续有效系统并发出票据。

        无法认证的通道保留原因，同一票据内其他通道仍可继续。``label``
        不能重复：旧票据永远表示当时的前向，不按当前条件重建。
        """
        if not isinstance(label, str) or not label:
            raise CaptureError("票据标签必须是非空字符串")
        if label in self._tickets:
            raise CaptureError(f"票据 {label!r} 已存在，不能按当前条件重建")
        if profile not in self.profiles:
            raise CaptureError(f"未知剖面 {profile!r}")
        if isinstance(rhs_keys, str) or not isinstance(rhs_keys, (list, tuple)) or not rhs_keys:
            raise CaptureError("rhs_keys 必须是非空列表")
        if any(not isinstance(key, str) for key in rhs_keys):
            raise CaptureError("rhs_keys 必须都是字符串")
        if len(set(rhs_keys)) != len(rhs_keys):
            raise CaptureError("同一前向不能重复引用同一通道")

        segments, boundary = self._resolve(profile)
        boundary_json = _boundary_snapshot(boundary)
        point_count = len(self.profiles[profile].active)
        resolved = [(segment, *_assemble(segment, self.slots)) for segment in segments]

        self._promotions = []
        self._last_budget_reason = None
        records: dict[str, _ChannelRecord] = {}
        for key in rhs_keys:
            if key not in self._rhs:
                raise CaptureError(f"未知通道 {key!r}")
            values = self._rhs[key]
            if values.shape[0] != point_count:
                raise CaptureError(
                    f"通道 {key!r} 的长度 {values.shape[0]}"
                    f" 与剖面 {profile!r} 的点数 {point_count} 不一致"
                )
            with np.errstate(all="ignore"):
                rhs_by_segment = [values[list(segment.points)] + const for segment, _, const in resolved]
            segment_records: list[_SegmentRecord] = []
            slots_used: set[str] = set()
            accepted_precisions: list[str] = []
            certificates: list[float] = []
            retried: list[str] = []
            reasons: list[str] = []
            for segment, storage, const in (
                (item[0], item[1], item[2]) for item in resolved
            ):
                rhs = rhs_by_segment[segment.index]
                trace: list[dict[str, Any]] = []
                if rhs.shape[0] == 0:
                    attempt = None
                    trace.append({"precision": None, "rejected": "剖面没有可用段"})
                else:
                    attempt = self._obtain(storage, rhs, trace)
                slots_used.update(slot for _, _, slot, _ in segment.terms)
                slots_used.update(slot for _, slot, _ in segment.constants)
                if attempt is None:
                    failed = trace[-1].get("rejected") if trace else "未知原因"
                    tried = (
                        trace[-1].get("retried")
                        if trace and trace[-1].get("retried")
                        else [item["precision"] for item in trace if item.get("precision")]
                    )
                    retried.extend(str(item) for item in tried)
                    reasons.append(f"段 {segment.index} [{segment.start},{segment.stop}]: {failed}")
                    segment_records.append(
                        _SegmentRecord(
                            segment, storage, const, rhs, False, str(failed), None, None, None,
                            tuple(str(item) for item in tried), None, None,
                        )
                    )
                    continue
                accepted_precisions.append(attempt.precision)
                certificates.append(attempt.certificate)
                retried.extend(attempt.attempts)
                segment_records.append(
                    _SegmentRecord(
                        segment, storage, const, rhs, True, None, attempt.precision,
                        attempt.certificate, attempt.entry.pivot_ratio, attempt.attempts,
                        attempt.entry, attempt.solution, attempt.residual,
                    )
                )
            accepted = bool(segments) and len(accepted_precisions) == len(segments)
            precision = None
            if accepted:
                # 通道级精度取各段使用的最高精度（最保守者）。
                precision = max(accepted_precisions, key=PRECISIONS.index)
            certificate = max(certificates) if certificates else None
            if not segments:
                reasons.append("剖面没有可用段")
            if not accepted:
                # 被拒绝的前向不参与任何反传：它已成功段的分解不能借给
                # 后续乘用，也不能占着暂存让可认证样本无处分级。
                for segment_record in segment_records:
                    if segment_record.entry is not None:
                        segment_record.entry.refs = max(0, segment_record.entry.refs - 1)
                        segment_record.entry = None
                    segment_record.solution = None
                self._evict_until(self.budget_bytes)
            records[key] = _ChannelRecord(
                key=key,
                segments=tuple(segment_records),
                accepted=accepted,
                reason=None if accepted else "；".join(reasons),
                precision=precision,
                precisions=tuple(dict.fromkeys(accepted_precisions)),
                certificate=certificate if accepted else None,
                retried=tuple(dict.fromkeys(retried)),
                slots_used=tuple(sorted(slots_used)),
            )

        ticket = _Ticket(label, profile, self._current_version, boundary_json, self.slots, records)
        self._tickets[label] = ticket
        channel_results = {key: _channel_result(key, record, ticket) for key, record in records.items()}
        return SolveReport(
            label,
            profile,
            self._current_version,
            channel_results,
            self.budget_report(),
            self.tol,
        )

    # -- 反传 -------------------------------------------------------------

    def back(
        self,
        label: str,
        keys: str | Sequence[str],
        cotangent: Sequence[float] | Mapping[str, Sequence[float]],
    ) -> BackwardReport:
        """在旧票据上对 RHS 与可训练槽位反传。

        ``cotangent`` 对应通道与有效点：可为整条剖面长度的数组（非有效点
        忽略），或有效点个数的数组；多路时用 ``{key: cotangent}``。只有前向
        证书与转置证书都通过的通道才交还梯度；被拒绝的前向不借用任何已
        缓存分解。
        """
        if label not in self._tickets:
            raise CaptureError(f"未知票据 {label!r}")
        ticket = self._tickets[label]
        if isinstance(keys, str):
            key_list = (keys,)
        elif isinstance(keys, (list, tuple)) and keys and all(isinstance(key, str) for key in keys):
            key_list = tuple(dict.fromkeys(keys))
        else:
            raise CaptureError("keys 必须是非空字符串或字符串列表")
        if isinstance(cotangent, Mapping):
            if set(cotangent) != set(key_list):
                raise CaptureError("cotangent 的键必须与 keys 一致")
            cotangents = {key: _finite_seq(cotangent[key], f"cotangent.{key}") for key in key_list}
        else:
            if len(key_list) != 1:
                raise CaptureError("多路反传必须用 {key: cotangent} 给出各自余切")
            cotangents = {key_list[0]: _finite_seq(cotangent, "cotangent")}

        profile = self.profiles[ticket.profile]
        point_count = len(profile.active)
        active = np.array(profile.active, dtype=bool)
        active_count = int(active.sum())
        self._promotions = []
        self._last_budget_reason = None

        rhs_grads: dict[str, np.ndarray] = {}
        per_key_coefficients: dict[str, dict[str, float]] = {}
        merged_coefficients: dict[str, float] = {}
        segment_reports: dict[str, tuple[dict[str, Any], ...]] = {}
        accepted_keys: list[str] = []
        rejected_keys: list[str] = []
        reasons: list[str] = []
        trainable_used: set[str] = set()

        for key in key_list:
            record = ticket.channels.get(key)
            if record is None:
                raise CaptureError(f"票据 {label!r} 不含通道 {key!r}")
            if not record.accepted:
                rejected_keys.append(key)
                reasons.append(f"通道 {key!r} 的前向未通过证书：{record.reason}")
                segment_reports[key] = ()
                continue
            values = cotangents[key]
            local_cotangents = _split_cotangent(
                values, record, point_count, active_count, active, label, key
            )

            pending: list[tuple[_SegmentRecord, np.ndarray]] = []
            key_failed = False
            rows: list[dict[str, Any]] = []
            for segment_record, c_local in zip(record.segments, local_cotangents):
                if segment_record.entry is None or segment_record.solution is None:
                    key_failed = True
                    rows.append(
                        {
                            "range": list(segment_record.segment.range),
                            "accepted": False,
                            "reason": "缺少前向分解",
                        }
                    )
                    continue
                result, reason, attempts = self._adjoint(segment_record, c_local)
                if result is None:
                    key_failed = True
                    rows.append(
                        {
                            "range": list(segment_record.segment.range),
                            "accepted": False,
                            "reason": reason,
                            "forward_precision": segment_record.precision,
                            "retried": list(attempts),
                        }
                    )
                    continue
                y, precision, certificate, peak = result
                pending.append((segment_record, y))
                rows.append(
                    {
                        "range": list(segment_record.segment.range),
                        "accepted": True,
                        "precision": precision,
                        "certificate": certificate,
                        "forward_precision": segment_record.precision,
                        "retried": list(attempts),
                        "peak_bytes": peak,
                    }
                )
            segment_reports[key] = tuple(rows)
            if key_failed:
                rejected_keys.append(key)
                reasons.append(f"通道 {key!r} 的伴随未通过同一门限的转置证书")
                continue

            key_coefficients: dict[str, float] = {}
            global_grad = np.zeros(point_count, dtype=np.float64)
            for segment_record, y in pending:
                segment = segment_record.segment
                x_local = segment_record.solution
                with np.errstate(all="ignore"):
                    # 系数梯度按 (槽位, 乘数) 分组向量化：
                    # dL/dslot = -sum y[row] * mult * x[column]。
                    terms = segment.terms
                    active_terms = [
                        (row, column, slot, multiplier)
                        for row, column, slot, multiplier in terms
                        if ticket.slots[slot][1]
                    ]
                    if active_terms:
                        rows = np.fromiter((t[0] for t in active_terms), dtype=np.int64)
                        columns = np.fromiter((t[1] for t in active_terms), dtype=np.int64)
                        multipliers = np.fromiter((t[3] for t in active_terms), dtype=np.float64)
                        products = -(y[rows] * multipliers * x_local[columns])
                        slots_of_terms = [t[2] for t in active_terms]
                        for slot in dict.fromkeys(slots_of_terms):
                            mask = np.fromiter(
                                (name == slot for name in slots_of_terms),
                                dtype=bool,
                                count=len(slots_of_terms),
                            )
                            delta = float(products[mask].sum())
                            key_coefficients[slot] = key_coefficients.get(slot, 0.0) + delta
                            merged_coefficients[slot] = merged_coefficients.get(slot, 0.0) + delta
                    for row, slot, gain in segment.constants:
                        if not ticket.slots[slot][1]:
                            continue
                        delta = float(y[row] * gain)
                        key_coefficients[slot] = key_coefficients.get(slot, 0.0) + delta
                        merged_coefficients[slot] = merged_coefficients.get(slot, 0.0) + delta
                trainable_used.update(key_coefficients.keys())
                global_grad[list(segment.points)] += y
            rhs_grads[key] = global_grad
            per_key_coefficients[key] = key_coefficients
            accepted_keys.append(key)

        if accepted_keys and rejected_keys:
            status = PARTIAL
        elif accepted_keys:
            status = OK
        else:
            status = REJECTED
        return BackwardReport(
            label=label,
            keys=key_list,
            status=status,
            accepted_keys=tuple(accepted_keys),
            rejected_keys=tuple(rejected_keys),
            reason=None if not reasons else "；".join(reasons),
            boundary_version=ticket.version,
            rhs_grads=rhs_grads,
            coefficient_grads=merged_coefficients,
            coefficient_grads_by_key=per_key_coefficients,
            trainable_slots=tuple(sorted(trainable_used)),
            segments=segment_reports,
            budget=self.budget_report(),
        )

    def _adjoint(
        self, record: _SegmentRecord, cotangent: np.ndarray
    ) -> tuple[
        tuple[np.ndarray, str, float, int] | None, str | None, tuple[str, ...]
    ]:
        """同一门限下的 A^T y = c；必要时在预算内晋级。"""
        assert record.entry is not None
        storage = record.storage
        entry = record.entry
        n = storage.n
        start = PRECISIONS.index(entry.precision)
        order = PRECISIONS[start:]
        attempts: list[str] = []
        last_reason: str | None = "没有可尝试的精度"
        for position, precision in enumerate(order):
            attempts.append(precision)
            work = _work_bytes(n, precision)
            if precision == entry.precision:
                source = entry.factor
                factor = 0
                reused = True
            else:
                cached = self._cache.get((entry.digest, precision))
                reused = cached is not None
                factor = 0 if reused else _factor_bytes(storage, precision)
                if cached is not None:
                    self._cache.move_to_end((entry.digest, precision))
                    source = cached.factor
                else:
                    source = None
            ok, needed = self._reserve(factor + work, 0)
            if not ok:
                last_reason = _budget_reason(needed, self.budget_bytes)
                self._last_budget_reason = last_reason
                return None, last_reason, tuple(attempts)
            self._peak = max(self._peak, self._used + factor + work)
            if source is None:
                with np.errstate(all="ignore"):
                    try:
                        source = _factor(storage, _DTYPES[precision])
                    except _Singular:
                        source = None
                if source is None:
                    last_reason = "伴随分解奇异"
                    continue
                with np.errstate(all="ignore"):
                    norm = storage.max_row_sum()
                ratio = 0.0 if norm == 0.0 else float(source.min_pivot) / norm
                if ratio < self.pivot_floor:
                    last_reason = f"伴随主元比率 {ratio:.3e} 低于门限 {self.pivot_floor:.3e}"
                    continue
            with np.errstate(all="ignore"):
                try:
                    raw = _factor_solve_transpose(
                        source, cotangent.astype(_DTYPES[precision])
                    )
                except (ValueError, FloatingPointError):
                    raw = None
            if raw is None or not np.all(np.isfinite(raw)):
                last_reason = "伴随解非有限"
                continue
            y = raw.astype(np.float64)
            with np.errstate(all="ignore"):
                certificate = _transpose_certificate(storage, y, cotangent)
            if not math.isfinite(certificate):
                last_reason = "转置证书非有限"
                continue
            if certificate > self.tol:
                last_reason = f"转置证书 {certificate:.3e} 大于门限 {self.tol:.3e}"
                if position + 1 < len(order):
                    self._promotions.append(
                        {
                            "n": n,
                            "from": precision,
                            "to": order[position + 1],
                            "reason": last_reason,
                        }
                    )
                continue
            return (y, precision, certificate, self._peak), None, tuple(attempts)
        return None, last_reason, tuple(attempts)

    # -- 查询 -------------------------------------------------------------

    def results(self, label: str | None = None) -> list[dict[str, Any]]:
        """训练端查询票据（默认最近一次）每路证书与绑定信息。"""
        if label is None:
            if not self._tickets:
                return []
            label = next(reversed(self._tickets))
        if label not in self._tickets:
            raise CaptureError(f"未知票据 {label!r}")
        ticket = self._tickets[label]
        return [
            {
                "label": label,
                "profile": ticket.profile,
                "key": key,
                "boundary_version": ticket.version,
                "boundary": ticket.boundary,
                "accepted": record.accepted,
                "precision": record.precision,
                "certificate": record.certificate,
                "reason": record.reason,
                "ranges": [
                    list(segment.segment.range) for segment in record.segments if segment.accepted
                ],
                "trainable_slots": [
                    slot for slot in record.slots_used if ticket.slots[slot][1]
                ],
                "fixed_slots": [
                    slot for slot in record.slots_used if not ticket.slots[slot][1]
                ],
                "segments": [
                    {
                        "range": list(segment.segment.range),
                        "precision": segment.precision,
                        "certificate": segment.certificate,
                        "pivot_ratio": segment.pivot_ratio,
                        "accepted": segment.accepted,
                        "reason": segment.reason,
                    }
                    for segment in record.segments
                ],
            }
            for key, record in ticket.channels.items()
        ]


# ---------------------------------------------------------------------------
# 辅助
# ---------------------------------------------------------------------------


def _channel_result(key: str, record: _ChannelRecord, ticket: _Ticket) -> ChannelResult:
    return ChannelResult(
        key=key,
        status=VALID if record.accepted else REJECTED,
        accepted=record.accepted,
        reason=record.reason,
        precision=record.precision,
        certificate=record.certificate,
        retried=record.retried,
        segments=tuple(_segment_view(segment) for segment in record.segments),
        boundary_version=ticket.version,
        precisions=record.precisions,
        trainable_slots=tuple(slot for slot in record.slots_used if ticket.slots[slot][1]),
        fixed_slots=tuple(slot for slot in record.slots_used if not ticket.slots[slot][1]),
        ranges=tuple(segment.segment.range for segment in record.segments if segment.accepted),
    )


def _segment_view(record: _SegmentRecord) -> dict[str, Any]:
    return {
        "range": list(record.segment.range),
        "precision": record.precision,
        "certificate": record.certificate,
        "residual": record.residual,
        "pivot_ratio": record.pivot_ratio,
        "accepted": record.accepted,
        "reason": record.reason,
        "retried": list(record.retried),
    }


def _finite_seq(values: Any, what: str) -> np.ndarray:
    if not isinstance(values, (list, tuple, np.ndarray)):
        raise CaptureError(f"{what} 必须是数值序列")
    try:
        array = np.asarray(values, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise CaptureError(f"{what} 必须是数值序列") from exc
    if array.ndim != 1 or not np.all(np.isfinite(array)):
        raise CaptureError(f"{what} 必须是一维有限数值")
    return array


def _split_cotangent(
    values: np.ndarray,
    record: _ChannelRecord,
    point_count: int,
    active_count: int,
    active: np.ndarray,
    label: str,
    key: str,
) -> list[np.ndarray]:
    if values.shape[0] == point_count:
        full = np.zeros(point_count, dtype=np.float64)
        full[active] = values[active]
    elif values.shape[0] == active_count:
        full = np.zeros(point_count, dtype=np.float64)
        full[active] = values
    else:
        raise CaptureError(
            f"票据 {label!r} 通道 {key!r} 的 cotangent 长度应为"
            f" {point_count} 或 {active_count}，实际 {values.shape[0]}"
        )
    return [full[list(segment.segment.points)] for segment in record.segments]


# ---------------------------------------------------------------------------
# 采集回放
# ---------------------------------------------------------------------------


def replay(capture: Mapping[str, Any], *, budget_bytes: int | None = None) -> list[dict[str, Any]]:
    """按接收顺序执行采集事件，返回每一步的可序列化结果。"""
    solver = BatchSolver(capture, budget_bytes=budget_bytes)
    events = capture.get("events")
    if not isinstance(events, list):
        raise CaptureError("采集资料缺少事件列表")
    log: list[dict[str, Any]] = []
    for index, event in enumerate(events, start=1):
        if not isinstance(event, (list, tuple)) or not event or not isinstance(event[0], str):
            raise CaptureError(f"第 {index} 个事件无效")
        kind = event[0]
        try:
            if kind == "solve":
                _, label, profile, keys = event
                report = solver.solve(label, profile, keys)
                log.append({"event": index, "kind": kind, "result": report.to_dict()})
            elif kind == "publish":
                _, version, definitions = event
                solver.publish(version, definitions)
                log.append(
                    {
                        "event": index,
                        "kind": kind,
                        "version": version,
                        "definitions": definitions,
                        "budget": solver.budget_report(),
                    }
                )
            elif kind == "back":
                _, label, key, cotangent = event
                report = solver.back(label, key, cotangent)
                log.append({"event": index, "kind": kind, "result": report.to_dict()})
            else:
                raise CaptureError(f"第 {index} 个事件类型 {kind!r} 未知")
        except (ValueError, TypeError, KeyError) as exc:
            raise CaptureError(f"第 {index} 个事件（{kind}）无法执行：{exc}") from exc
    return log


if __name__ == "__main__":
    import argparse
    import json
    from pathlib import Path

    from read_capture import read_capture

    parser = argparse.ArgumentParser(description="回放交接采集并输出求解与反传结果")
    parser.add_argument("capture", type=Path)
    parser.add_argument("--budget", type=int, default=None, help="覆盖采集中的 scratch_bytes")
    args = parser.parse_args()
    payload = replay(read_capture(args.capture), budget_bytes=args.budget)
    print(json.dumps(payload, ensure_ascii=False, indent=2, default=float))
