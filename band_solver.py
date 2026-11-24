"""变长带状平滑层：分段带状求解、数值证书、精度晋级、暂存预算与伴随反传。

本模块严格按 ``交接协议.md`` 的口径实现：

* ``offsets`` 描述带状存储，``bands[i][j]`` 对应未知量 ``i + offsets[j]``；
* ``active`` 的连续真段是独立方程系统，补齐位置只占存储、不产生方程；
* 每段左端/右端（单点段用 ``single``）可用边界行替换原方程，``add_rhs``
  以 ``[slot, gain]`` 给右端加常量；
* 证书为 ``||Ax-b||inf / (||A||inf*||x||inf + ||b||inf)``，分母为零时记零，
  但精确奇异仍拒绝发证；
* FP32 先行，证书或主元安全门限不过时在 ``scratch_bytes`` 预算内晋级 FP64；
* 票据冻结前向的矩阵、解、未知量区间与槽位可训练性，旧票据不会按当前
  槽位/边界/RHS 重建；
* 反传解 ``A^T lambda = g``，转置证书同门限有效后才交还梯度；同一物理槽位
  按有效前向（段/通道）汇合。

对外主要入口是 :class:`Session`（亦可使用 ``Solver`` / ``Scheduler`` 别名）
与 :func:`replay`。
"""

from __future__ import annotations

import hashlib
import json
import math
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

FLOAT32 = "float32"
FLOAT64 = "float64"
PRECISIONS = (FLOAT32, FLOAT64)
ITEMSIZE = {FLOAT32: 4, FLOAT64: 8}

#: 未发布命名边界时，使用剖面内嵌 boundary；票据的 boundary_version 为 None。
DEFAULT_BOUNDARY_VERSION = None
DEFAULT_SCRATCH_BYTES = 256 * 1024 * 1024

__all__ = [
    "Session", "Solver", "Scheduler", "BandSolver", "Report",
    "replay", "replay_file", "load_capture", "certificate", "band_certificate",
    "segment_runs", "assemble_system", "solve_banded", "factor_banded",
    "ScratchLedger", "BandSolveError", "BudgetCapacity", "GeometryError",
    "create_session", "run_events", "VariableBatchSolver", "SmoothingLayer",
]


# ---------------------------------------------------------------------------
# 报告容器与小工具
# ---------------------------------------------------------------------------

class Report(dict):
    """同时支持字典键与属性访问的结果对象。"""

    def __getattr__(self, name: str) -> Any:
        try:
            return self[name]
        except KeyError as exc:  # pragma: no cover - 透传属性错误
            raise AttributeError(name) from exc

    def __setattr__(self, name: str, value: Any) -> None:
        self[name] = value

    def __dir__(self):  # pragma: no cover - 便利补全
        return sorted(set(super().__dir__()) | set(self.keys()))


def to_native(obj: Any) -> Any:
    """递归把 numpy 标量/数组转成 JSON 安全的 Python 原生对象。"""
    if isinstance(obj, Report):
        return {k: to_native(v) for k, v in obj.items()}
    if isinstance(obj, dict):
        return {str(k): to_native(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_native(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    return obj


def json_dumps(obj: Any, **kwargs: Any) -> str:
    kwargs.setdefault("ensure_ascii", False)
    kwargs.setdefault("indent", 2)
    return json.dumps(to_native(obj), **kwargs)


class BandSolveError(ValueError):
    """带状系统资料或求解不合法。``code`` 给出机器可读原因。"""

    def __init__(self, message: str, code: str = "invalid_system", **extra: Any):
        super().__init__(message)
        self.code = code
        self.extra = extra


class GeometryError(BandSolveError):
    def __init__(self, message: str, **extra: Any):
        super().__init__(message, code="geometry", **extra)


class BudgetCapacity(BandSolveError):
    """暂存预算不足以容纳所需分解/工作区。"""

    def __init__(self, message: str, resource: "Report"):
        super().__init__(message, code="resource_limited")
        self.resource = resource


# ---------------------------------------------------------------------------
# 几何：active 连续段
# ---------------------------------------------------------------------------

def segment_runs(active: Sequence[bool]) -> list[tuple[int, int]]:
    """返回 active 连续真段的 ``[start, stop)`` 列表。"""
    runs: list[tuple[int, int]] = []
    n = len(active)
    i = 0
    while i < n:
        if not bool(active[i]):
            i += 1
            continue
        j = i + 1
        while j < n and bool(active[j]):
            j += 1
        runs.append((i, j))
        i = j
    return runs


# ---------------------------------------------------------------------------
# 系统组装（冻结高精度原方程与槽位出处）
# ---------------------------------------------------------------------------

def _cell_parts(cell: Any, where: str) -> tuple[str, float] | None:
    if cell is None:
        return None
    if (not isinstance(cell, (list, tuple)) or len(cell) != 2):
        raise GeometryError(f"{where} 的系数必须是 [slot, multiplier] 或 null")
    slot, mult = cell
    if not isinstance(slot, str):
        raise GeometryError(f"{where} 的槽位 id 必须是字符串")
    return slot, float(mult)


def assemble_system(
    profile: Mapping[str, Any],
    offsets: Sequence[int],
    slot_values: Mapping[str, float],
    boundary: Mapping[str, Any] | None = None,
) -> "System":
    """把一个剖面组装成若干独立段的高精度带状方程。

    ``boundary`` 为该次受理使用的边界三件套；为 ``None`` 时回退到剖面内嵌
    ``boundary``。返回的 :class:`System` 冻结 A/b 所需的全部出处，随后调用方
    修改槽位值不会影响已组装系统。
    """
    offsets = [int(o) for o in offsets]
    offset_set = set(offsets)
    n_cols = len(offsets)
    active = profile.get("active")
    bands = profile.get("bands")
    if not isinstance(active, list) or not isinstance(bands, list):
        raise GeometryError("剖面缺少 active/bands 列表")
    n = len(active)
    if len(bands) != n:
        raise GeometryError("bands 行数与 active 长度不一致")
    for i, row in enumerate(bands):
        if not isinstance(row, list) or len(row) != n_cols:
            raise GeometryError(f"第 {i} 行 bands 宽度与 offsets 不一致")

    bdef: Mapping[str, Any] = boundary if boundary is not None else (
        profile.get("boundary") or {}
    )
    left = bdef.get("left") if isinstance(bdef, Mapping) else None
    right = bdef.get("right") if isinstance(bdef, Mapping) else None
    single = bdef.get("single") if isinstance(bdef, Mapping) else None

    runs = segment_runs(active)
    run_of: dict[int, tuple[int, int]] = {}
    for s, t in runs:
        for g in range(s, t):
            run_of[g] = (s, t)

    system = System(n=n, offsets=tuple(offsets), runs=runs)

    for run_index, (s, t) in enumerate(runs):
        length = t - s
        seg = Segment(index=run_index, start=s, stop=t, length=length)
        kl = max(0, -min(offsets)) if offsets else 0
        ku = max(0, max(offsets)) if offsets else 0
        seg.kl, seg.ku = kl, ku
        seg.diags = {d: np.zeros(length, dtype=np.float64)
                     for d in range(-kl, ku + 1)}

        # 决定哪些全局行被边界行替换。
        replacements: dict[int, Mapping[str, Any]] = {}
        if length == 1:
            if single is not None:
                replacements[s] = single
        else:
            if left is not None:
                replacements[s] = left
            if right is not None:
                replacements[t - 1] = right

        for g in range(s, t):
            row_source = "bands"
            spec = replacements.get(g)
            if spec is not None and isinstance(spec.get("row"), list):
                row = spec.get("row")
                if len(row) != n_cols:
                    raise GeometryError(
                        f"全局点 {g} 的边界行必须是与 offsets 同宽的列表")
                row_source = "boundary"
            else:
                # 无 boundary 规格，或 row 缺省/null：保留原剖面方程。
                # row=null 时 add_rhs 仍可单独生效（只改右端项的端点调整）。
                row = bands[g]

            if spec is not None:
                add_rhs = spec.get("add_rhs")
                if add_rhs is not None:
                    ar = _cell_parts(add_rhs, f"点 {g} add_rhs")
                    if ar is not None:
                        slot, gain = ar
                        _require_slot(slot, slot_values, f"点 {g} add_rhs")
                        seg.add_rhs.append(
                            AddRhs(i=g - s, point=g - s,
                                   global_point=g,
                                   slot=slot, gain=gain))

            for col, cell in enumerate(row):
                parts = _cell_parts(cell, f"点 {g} 第 {col} 列")
                if parts is None:
                    continue
                slot, mult = parts
                _require_slot(slot, slot_values, f"点 {g} 第 {col} 列")
                value = float(slot_values[slot]) * mult
                if not np.isfinite(value):
                    raise BandSolveError(
                        f"槽位 {slot} 在点 {g} 产生非有限系数",
                        code="non_finite", slot=slot, point=g)
                off = offsets[col]
                q = g + off
                if not (0 <= q < n):
                    raise GeometryError(
                        f"点 {g} 的偏移 {off} 落到剖面之外 ({q})")
                if g not in run_of or run_of[q] != (s, t):
                    raise GeometryError(
                        f"点 {g} 的偏移 {off} 跨出 active 连续段；"
                        "独立段之间不允许耦合")
                li = g - s
                lj = q - s
                d = lj - li
                if d not in seg.diags:
                    # offsets 声明的偏移与推导包络不一致时的防御性分支。
                    seg.diags[d] = np.zeros(length, dtype=np.float64)
                seg.diags[d][li] += value
                seg.provenance.append(
                    CoefRef(i=li, j=lj, global_i=g, global_j=q,
                            offset=d, slot=slot, multiplier=mult,
                            origin=row_source))

        # 主元安全门限按“原矩阵无穷范数”计。
        seg.norm_inf = band_row_inf_norm(seg.diags, length)
        if not np.isfinite(seg.norm_inf):
            raise BandSolveError(
                f"段 [{s},{t}) 的矩阵无穷范数非有限", code="non_finite")
        # 段级指纹：只由该段实际矩阵决定，使不同票据间真正相同的分解可复用，
        # 同时保证“同长度但端点/系数不同”绝不命中。
        hseg = hashlib.sha256()
        hseg.update(f"{length}|{seg.kl}|{seg.ku}".encode())
        for d in sorted(seg.diags):
            a = np.ascontiguousarray(seg.diags[d], dtype=np.float64)
            hseg.update(f"{d}:".encode())
            hseg.update(a.tobytes())
        seg.fingerprint = hseg.hexdigest()[:24]
        system.segments.append(seg)

    system.fingerprint = system.compute_fingerprint()
    return system


def _require_slot(slot: str, values: Mapping[str, float], where: str) -> None:
    if slot not in values:
        raise GeometryError(f"{where} 引用了未声明的槽位 {slot}")


class CoefRef(Report):
    pass


class AddRhs(Report):
    pass


class Segment:
    def __init__(self, index: int, start: int, stop: int, length: int):
        self.index = index
        self.start = start
        self.stop = stop
        self.length = length
        self.kl = 0
        self.ku = 0
        self.diags: dict[int, np.ndarray] = {}
        self.provenance: list[CoefRef] = []
        self.add_rhs: list[AddRhs] = []
        self.norm_inf = 0.0
        self.fingerprint = ""

    @property
    def range(self) -> tuple[int, int]:
        return self.start, self.stop

    def rhs_base(self, rhs_full: Sequence[float]) -> np.ndarray:
        """取原通道值并加入该段实际受理边界的 add_rhs 常量。"""
        b = np.asarray(rhs_full, dtype=np.float64)[self.start:self.stop].copy()
        for ar in self.add_rhs:
            # gain * 当前票据冻结的槽位值由调用方通过 slot_values 快照传入更稳妥；
            # 这里 AddRhs 已在票据中保存 value_at_forward。
            b[ar.i] += ar.forward_value * ar.gain
        return b


class System:
    def __init__(self, n: int, offsets: tuple[int, ...],
                 runs: list[tuple[int, int]]):
        self.n = n
        self.offsets = offsets
        self.runs = runs
        self.segments: list[Segment] = []
        self.fingerprint = ""

    def compute_fingerprint(self) -> str:
        h = hashlib.sha256()
        h.update(json.dumps(self.offsets).encode())
        for seg in self.segments:
            h.update(f"|{seg.start}:{seg.stop}|".encode())
            for d in sorted(seg.diags):
                a = np.ascontiguousarray(seg.diags[d], dtype=np.float64)
                h.update(f"{d}:".encode())
                h.update(a.tobytes())
        return h.hexdigest()[:24]


# ---------------------------------------------------------------------------
# 带状矩阵运算与证书
# ---------------------------------------------------------------------------

def band_matvec(diags: Mapping[int, np.ndarray], x: np.ndarray,
                n: int | None = None) -> np.ndarray:
    """y = A x，diags[d][i] = A[i, i+d]。标量循环，避免大块临时存储。"""
    if n is None:
        n = len(x)
    y = np.zeros(n, dtype=np.float64)
    for d, arr in diags.items():
        for i in range(n):
            j = i + d
            if 0 <= j < n:
                y[i] += float(arr[i]) * float(x[j])
    return y


def band_matvec_transpose(diags: Mapping[int, np.ndarray], x: np.ndarray,
                          n: int) -> np.ndarray:
    """y = A^T x，diags[d][i] = A[i, i+d]。"""
    y = np.zeros(n, dtype=np.float64)
    for d, arr in diags.items():
        for i in range(n):
            j = i + d
            if 0 <= j < n:
                y[j] += float(arr[i]) * float(x[i])
    return y


def band_row_inf_norm(diags: Mapping[int, np.ndarray], n: int) -> float:
    """||A||inf = 最大绝对行和。"""
    worst = 0.0
    for i in range(n):
        row_sum = 0.0
        for d, arr in diags.items():
            j = i + d
            if 0 <= j < n:
                row_sum += abs(float(arr[i]))
        worst = max(worst, row_sum)
    return worst


def band_col_inf_norm(diags: Mapping[int, np.ndarray], n: int) -> float:
    """||A^T||inf = 最大绝对列和。"""
    worst = 0.0
    for j in range(n):
        col_sum = 0.0
        for d, arr in diags.items():
            i = j - d
            if 0 <= i < n:
                col_sum += abs(float(arr[i]))
        worst = max(worst, col_sum)
    return worst


def band_certificate(diags: Mapping[int, np.ndarray], x: np.ndarray,
                     b: np.ndarray, transpose: bool = False) -> Report:
    """协议证书（带状实现，全部按 FP64 原方程口径计算）。

    ``transpose=True`` 时对转置方程 ``A^T x = b`` 发证，矩阵范数取
    ``||A^T||inf``（即原矩阵最大绝对列和）。
    """
    x64 = np.asarray(x, dtype=np.float64)
    b64 = np.asarray(b, dtype=np.float64)
    n = len(b64)
    finite = bool(np.all(np.isfinite(x64)) and np.all(np.isfinite(b64)))
    apply_mat = band_matvec_transpose if transpose else band_matvec
    residual = apply_mat(diags, x64, n) - b64
    res_inf = float(np.max(np.abs(residual))) if n else 0.0
    a_inf = (band_col_inf_norm(diags, n) if transpose
             else band_row_inf_norm(diags, n))
    x_inf = float(np.max(np.abs(x64))) if n else 0.0
    b_inf = float(np.max(np.abs(b64))) if n else 0.0
    denominator = a_inf * x_inf + b_inf
    if denominator == 0.0:
        cert = 0.0
    else:
        cert = res_inf / denominator
    return Report(
        certificate=float(cert), residual_inf=res_inf,
        denominator=float(denominator), matrix_inf_norm=float(a_inf),
        x_inf=x_inf, b_inf=b_inf, finite=finite,
    )


def certificate(A: Any, x: Any, b: Any) -> float:
    """稠密/带状通用的证书数值函数，返回证书标量。

    传入二维矩阵时用稠密口径；传入 ``{offset: diag}`` 时用带状口径。
    需要完整明细请用 :func:`band_certificate`。
    """
    if isinstance(A, Mapping):
        return band_certificate(A, np.asarray(x), np.asarray(b)).certificate
    M = np.asarray(A, dtype=np.float64)
    x64 = np.asarray(x, dtype=np.float64)
    b64 = np.asarray(b, dtype=np.float64)
    res = float(np.max(np.abs(M @ x64 - b64))) if M.size else 0.0
    a_inf = float(np.max(np.sum(np.abs(M), axis=1))) if M.size else 0.0
    den = a_inf * float(np.max(np.abs(x64), initial=0.0)) + \
        float(np.max(np.abs(b64), initial=0.0))
    return 0.0 if den == 0.0 else res / den


# ---------------------------------------------------------------------------
# 带状分解与求解（三对角 Thomas 保舍入路径 + 一般窄带 LU）
# ---------------------------------------------------------------------------

class Factor:
    def __init__(self, precision: str, kind: str, n: int,
                 arrays: dict[str, np.ndarray], kl: int = 1, ku: int = 1,
                 diags: dict[int, np.ndarray] | None = None):
        self.precision = precision
        self.kind = kind
        self.n = n
        self.kl = kl
        self.ku = ku
        self.arrays = arrays
        self.diags = diags or {}
        self.nbytes = int(sum(a.nbytes for a in arrays.values()))
        self.pivot_min_abs = math.inf
        self.pivots: list[float] = []

    # -- 求解 -------------------------------------------------------------
    def solve(self, rhs: Sequence[float]) -> np.ndarray:
        dtype = np.float32 if self.precision == FLOAT32 else np.float64
        b = np.asarray(rhs, dtype=dtype)
        if self.kind == "thomas":
            return _thomas_forward(self.arrays, b, dtype)
        return _lu_forward(self.diags, b, self.n, self.kl, self.ku, dtype)

    def solve_transpose(self, rhs: Sequence[float]) -> np.ndarray:
        dtype = np.float32 if self.precision == FLOAT32 else np.float64
        b = np.asarray(rhs, dtype=dtype)
        if self.kind == "thomas":
            return _thomas_transpose(self.arrays, b, dtype)
        return _lu_transpose(self.diags, b, self.n, self.kl, self.ku, dtype)


def _pivot_problem(pivots: list[float], norm_inf: float,
                   pivot_floor: float) -> str | None:
    """返回 None 表示主元安全；否则给出机器可读原因码。"""
    floor_abs = float(pivot_floor) * float(norm_inf)
    for p in pivots:
        if not np.isfinite(p):
            return "non_finite"
        if p == 0.0:
            return "singular"
    if norm_inf == 0.0:
        # 零矩阵：pivot 全为零已在上面拦截；其余退化情形同样视为奇异。
        return "singular"
    for p in pivots:
        if abs(p) < floor_abs:
            return "pivot_floor"
    return None


def factor_banded(seg: Segment, precision: str,
                  pivot_floor: float = 1e-10) -> Factor:
    """按实际带状存储分解一个段。

    三对角（``kl<=1 and ku<=1``）走规范 Thomas 递推（固定舍入顺序），
    更宽的窄带走不选主元的包络 LU。主元安全性由调用方复查
    ``factor.pivot_problem``。
    """
    if precision not in ITEMSIZE:
        raise ValueError(f"未知精度 {precision}")
    dtype = np.float32 if precision == FLOAT32 else np.float64
    n = seg.length

    if seg.kl <= 1 and seg.ku <= 1:
        low = np.zeros(n, dtype=dtype)
        up = np.zeros(n, dtype=dtype)
        beta = np.zeros(n, dtype=dtype)
        cp = np.zeros(n, dtype=dtype)
        d0 = seg.diags.get(0, np.zeros(n))
        dm1 = seg.diags.get(-1)
        dp1 = seg.diags.get(1)
        pivots: list[float] = []
        for i in range(n):
            beta[i] = dtype(d0[i])
            if i >= 1 and dm1 is not None:
                low[i] = dtype(dm1[i])
            if i < n - 1 and dp1 is not None:
                up[i] = dtype(dp1[i])
        beta[0] = dtype(d0[0])
        pivots.append(float(beta[0]))
        with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
            if n >= 2:
                cp[0] = up[0] / beta[0]
                for i in range(1, n):
                    beta[i] = dtype(d0[i]) - low[i] * cp[i - 1]
                    pivots.append(float(beta[i]))
                    if i < n - 1:
                        cp[i] = up[i] / beta[i]
        factor = Factor(precision, "thomas", n,
                        {"low": low, "beta": beta, "cp": cp},
                        kl=1, ku=1)
        factor.pivots = pivots
        factor.pivot_min_abs = float(min((abs(p) for p in pivots),
                                         default=math.inf))
        factor.pivot_problem = _pivot_problem(
            pivots, seg.norm_inf, pivot_floor)
        return factor

    # 一般窄带包络 LU（不选主元；平滑算子通常对角占优，主元门限兜底）。
    diags = {d: np.asarray(a, dtype=dtype).copy()
             for d, a in seg.diags.items()}
    kl, ku = seg.kl, seg.ku
    pivots = []
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        for k in range(n):
            piv = diags[0][k]
            pivots.append(float(piv))
            if piv == 0 or not np.isfinite(float(piv)):
                continue
            i_max = min(n - 1, k + kl)
            for i in range(k + 1, i_max + 1):
                d_low = k - i
                m = diags[d_low][i] / piv
                diags[d_low][i] = m
                j_max = min(n - 1, k + ku)
                for j in range(k + 1, j_max + 1):
                    src_d = j - k
                    tgt_d = j - i
                    if src_d in diags and tgt_d in diags:
                        diags[tgt_d][i] = (
                            diags[tgt_d][i] - m * diags[src_d][k])
    arrays = {f"diag_{d:+d}": a for d, a in diags.items()}
    factor = Factor(precision, "envelope_lu", n, arrays,
                    kl=kl, ku=ku, diags=diags)
    factor.pivots = pivots
    factor.pivot_min_abs = float(min((abs(p) for p in pivots),
                                     default=math.inf))
    factor.pivot_problem = _pivot_problem(
        pivots, seg.norm_inf, pivot_floor)
    return factor


def _thomas_forward(arrays: Mapping[str, np.ndarray], b: np.ndarray,
                    dtype: type) -> np.ndarray:
    low = arrays["low"]
    beta = arrays["beta"]
    cp = arrays["cp"]
    n = len(beta)
    x = np.empty(n, dtype=dtype)
    x[0] = b[0] / beta[0]
    for i in range(1, n):
        x[i] = (b[i] - low[i] * x[i - 1]) / beta[i]
    for i in range(n - 2, -1, -1):
        x[i] = x[i] - cp[i] * x[i + 1]
    return x


def _thomas_transpose(arrays: Mapping[str, np.ndarray], g: np.ndarray,
                      dtype: type) -> np.ndarray:
    """用保存的 Thomas 因子解 A^T z = g。

    输出缓冲区先存 U^T 前向中间量 z，再原地反代覆盖成 lambda，
    因此除交付的解外不需要大块工作区。
    """
    low = arrays["low"]
    beta = arrays["beta"]
    cp = arrays["cp"]
    n = len(beta)
    out = np.empty(n, dtype=dtype)
    out[0] = g[0] / beta[0]
    for i in range(1, n):
        up_prev = cp[i - 1] * beta[i - 1]       # U[i-1, i]
        out[i] = (g[i] - up_prev * out[i - 1]) / beta[i]
    # L^T lambda = z；mult_i = L[i+1, i] = low[i+1]/beta[i]
    for i in range(n - 2, -1, -1):
        mult = low[i + 1] / beta[i]
        out[i] = out[i] - mult * out[i + 1]
    return out


def _lu_forward(diags, b, n, kl, ku, dtype):
    # 同一缓冲区先做 Ly=b（y），再原地 Ux=y。
    x = np.empty(n, dtype=dtype)
    for i in range(n):
        val = b[i]
        for j in range(max(0, i - kl), i):
            val = val - diags[j - i][i] * x[j]
        x[i] = val  # L 单位对角
    for i in range(n - 1, -1, -1):
        val = x[i]
        for j in range(i + 1, min(n, i + ku + 1)):
            val = val - diags[j - i][i] * x[j]
        x[i] = val / diags[0][i]
    return x


def _lu_transpose(diags, g, n, kl, ku, dtype):
    # 先 U^T z=g 写入 out，再原地 L^T x=z。
    out = np.empty(n, dtype=dtype)
    for i in range(n):
        val = g[i]
        for j in range(max(0, i - ku), i):
            val = val - diags[i - j][j] * out[j]
        out[i] = val / diags[0][i]
    for i in range(n - 1, -1, -1):
        val = out[i]
        for j in range(i + 1, min(n, i + kl + 1)):
            val = val - diags[i - j][j] * out[j]
        out[i] = val  # L 单位对角
    return out


def solve_banded(A: Any, b: Any, precision: str = FLOAT64) -> np.ndarray:
    """无状态带状求解便捷函数。

    ``A`` 可以是二维矩阵（内部切段仅支持单段）或
    ``(diags, kl, ku)`` / ``Segment``。主要供测试与训练调度的简单调用。
    """
    if isinstance(A, Segment):
        seg = A
    elif isinstance(A, Mapping):
        diags = A
        n = len(np.asarray(b))
        kl = max(-d for d in diags if d < 0) if any(d < 0 for d in diags) else 0
        ku = max(d for d in diags if d > 0) if any(d > 0 for d in diags) else 0
        seg = Segment(0, 0, n, n)
        seg.kl, seg.ku, seg.diags = kl, ku, dict(diags)
        seg.norm_inf = band_row_inf_norm(seg.diags, n)
    else:
        M = np.asarray(A, dtype=np.float64)
        n = M.shape[0]
        diags = {}
        kl = ku = 0
        for i in range(n):
            for j in range(n):
                if M[i, j] != 0.0:
                    d = j - i
                    kl = max(kl, -d)
                    ku = max(ku, d)
        for d in range(-kl, ku + 1):
            diags[d] = np.zeros(n)
            for i in range(n):
                j = i + d
                if 0 <= j < n:
                    diags[d][i] = M[i, j]
        seg = Segment(0, 0, n, n)
        seg.kl, seg.ku, seg.diags = kl, ku, diags
        seg.norm_inf = band_row_inf_norm(diags, n)
    factor = factor_banded(seg, precision)
    if factor.pivot_problem is not None:
        raise BandSolveError(
            f"主元安全门限未通过：{factor.pivot_problem}",
            code=factor.pivot_problem)
    return factor.solve(np.asarray(b, dtype=np.float64))


# ---------------------------------------------------------------------------
# 预算台账与分解池
# ---------------------------------------------------------------------------

class ScratchLedger:
    """记录分解净占用、工作区峰值与总峰值。"""

    def __init__(self, limit: int):
        self.limit = int(limit)
        self.net_retained_bytes = 0
        self.workspace_live_bytes = 0
        self.workspace_peak_bytes = 0
        self.peak_total_bytes = 0
        self.events: list[Report] = []

    def _touch_peak(self) -> None:
        total = self.net_retained_bytes + self.workspace_live_bytes
        self.peak_total_bytes = max(self.peak_total_bytes, total)

    def record(self, kind: str, bytes_: int, **extra: Any) -> None:
        ev = Report(kind=kind, bytes=int(bytes_), **extra)
        self.events.append(ev)

    def can_retain(self, nbytes: int) -> bool:
        return self.net_retained_bytes + int(nbytes) <= self.limit

    def retain(self, nbytes: int, key: str, **extra: Any) -> None:
        nbytes = int(nbytes)
        if not self.can_retain(nbytes):
            resource = self.capacity_report(nbytes)
            raise BudgetCapacity(
                f"暂存预算不足：需要再占用 {nbytes} 字节，可用 "
                f"{resource.available_bytes} 字节，缺口 "
                f"{resource.shortfall_bytes} 字节", resource)
        self.net_retained_bytes += nbytes
        self.record("retain", nbytes, key=key, **extra)
        self._touch_peak()

    def release(self, nbytes: int, key: str, **extra: Any) -> None:
        self.net_retained_bytes -= int(nbytes)
        self.record("release", int(nbytes), key=key, **extra)

    @contextmanager
    def workspace(self, nbytes: int, label: str = "workspace"):
        nbytes = int(nbytes)
        if self.net_retained_bytes + self.workspace_live_bytes + nbytes > \
                self.limit:
            resource = self.capacity_report(nbytes, workspace=nbytes)
            raise BudgetCapacity("计算工作区超出 scratch_bytes 预算", resource)
        self.workspace_live_bytes += nbytes
        self.workspace_peak_bytes = max(self.workspace_peak_bytes,
                                        self.workspace_live_bytes)
        self._touch_peak()
        self.record("workspace_enter", nbytes, label=label)
        try:
            yield
        finally:
            self.workspace_live_bytes -= nbytes
            self.record("workspace_leave", nbytes, label=label)

    def capacity_report(self, request_bytes: int = 0,
                        workspace: int = 0) -> Report:
        available = self.limit - self.net_retained_bytes \
            - self.workspace_live_bytes
        need = int(request_bytes) + int(workspace)
        return Report(
            budget_bytes=self.limit,
            scratch_bytes=self.limit,
            net_retained_bytes=self.net_retained_bytes,
            used_bytes=self.net_retained_bytes,
            workspace_live_bytes=self.workspace_live_bytes,
            workspace_peak_bytes=self.workspace_peak_bytes,
            peak_total_bytes=self.peak_total_bytes,
            available_bytes=max(0, available),
            remaining_scratch_bytes=max(0, available),
            remaining_bytes=max(0, available),
            requested_bytes=need,
            shortfall_bytes=max(0, need - max(0, available)),
            within_budget=self.net_retained_bytes
            + self.workspace_live_bytes + need <= self.limit,
        )

    def report(self) -> Report:
        rep = self.capacity_report(0)
        rep["events"] = list(self.events)
        return rep


class FactorPool:
    """按高精度矩阵指纹去重的分解池；token 形如 ``label:segid``。"""

    def __init__(self, ledger: ScratchLedger):
        self.ledger = ledger
        self.records: dict[str, FactorRecord] = {}

    def _key(self, fingerprint: str, precision: str) -> str:
        return f"{fingerprint}:{precision}"

    def get(self, fingerprint: str, precision: str) -> "FactorRecord | None":
        return self.records.get(self._key(fingerprint, precision))

    def get_or_create(self, seg: Segment, precision: str,
                      pivot_floor: float, token: str,
                      fingerprint: str) -> tuple["FactorRecord", bool]:
        key = self._key(fingerprint, precision)
        existing = self.records.get(key)
        if existing is not None:
            existing.tokens.add(token)
            self.ledger.record("cache_hit", existing.factor.nbytes,
                               key=key, token=token, precision=precision)
            return existing, True

        factor = factor_banded(seg, precision, pivot_floor)
        rec = FactorRecord(key, fingerprint, precision, factor)
        # 预算不足时 BudgetCapacity 直接向上抛，台账未被修改。
        self.ledger.retain(factor.nbytes, key=key, precision=precision)
        self.records[key] = rec
        rec.tokens.add(token)
        return rec, False

    def migrate_token(self, token: str, old_key: str | None,
                      new_key: str) -> None:
        if old_key and old_key != new_key:
            old = self.records.get(old_key)
            if old is not None:
                old.tokens.discard(token)
                if not old.tokens:
                    self._delete(old_key)
        new = self.records[new_key]
        new.tokens.add(token)

    def unpin_token(self, token: str) -> list[str]:
        removed = []
        for key, rec in list(self.records.items()):
            if token in rec.tokens:
                rec.tokens.discard(token)
                if not rec.tokens:
                    self._delete(key)
                    removed.append(key)
        return removed

    def release_label(self, label: str) -> list[str]:
        prefix = f"{label}:"
        removed = []
        for key, rec in list(self.records.items()):
            doomed = {t for t in rec.tokens if t.startswith(prefix)}
            if doomed:
                rec.tokens -= doomed
            if not rec.tokens:
                self._delete(key)
                removed.append(key)
        return removed

    def _delete(self, key: str) -> None:
        rec = self.records.pop(key)
        self.ledger.release(rec.factor.nbytes, key=key,
                           precision=rec.precision)

    def reclaimable_by_label(self, keep_label: str | None = None) -> dict:
        usage: dict[str, int] = {}
        for rec in self.records.values():
            for token in rec.tokens:
                label = token.split(":", 1)[0]
                if keep_label is not None and label == keep_label:
                    continue
                # 共享记录按独占口径提示可能回收的上限（整记录字节）。
                usage.setdefault(label, 0)
        # 精确给出“释放某 label 后真正回收”的字节数。
        exact: dict[str, int] = {}
        for label in {t.split(":", 1)[0]
                      for r in self.records.values() for t in r.tokens}:
            total = 0
            for rec in self.records.values():
                other_tokens = {t for t in rec.tokens
                                if not t.startswith(f"{label}:")}
                has_self = any(t.startswith(f"{label}:") for t in rec.tokens)
                if has_self and not other_tokens:
                    total += rec.factor.nbytes
            exact[label] = total
        return exact


class FactorRecord:
    def __init__(self, key: str, fingerprint: str, precision: str,
                 factor: Factor):
        self.key = key
        self.fingerprint = fingerprint
        self.precision = precision
        self.factor = factor
        self.tokens: set[str] = set()


# ---------------------------------------------------------------------------
# 会话 / 调度器
# ---------------------------------------------------------------------------

class Session:
    """训练调度持有的变长批求解、发证与伴随反传会话。

    构造方式::

        session = Session(capture_dict)
        session.publish("r1", {"ridge": {...}})
        session.solve("old", "ridge", ["easy", "edge"])
        session.back("old", "edge", cotangent)

    也支持 ``Session.from_file("samples/capture.json")``。
    """

    def __init__(self, capture: Mapping[str, Any] | str | Path | None = None,
                 offsets: Sequence[int] | None = None,
                 tol: float | None = None,
                 pivot_floor: float | None = None,
                 scratch_bytes: int | None = None,
                 slots: Mapping[str, Any] | None = None,
                 profiles: Mapping[str, Any] | None = None,
                 rhs: Mapping[str, Sequence[float]] | None = None,
                 **legacy_kwargs: Any):
        # 兼容个别位置传 offsets 列表的调用方式。
        if capture is not None and not isinstance(
                capture, (Mapping, str, Path)):
            if offsets is None and isinstance(capture, Sequence):
                offsets = capture
            capture = None
        cap: Mapping[str, Any] = {}
        if capture is not None:
            if isinstance(capture, (str, Path)):
                cap = load_capture(capture)
            else:
                cap = capture
        if "budget_bytes" in legacy_kwargs and scratch_bytes is None:
            scratch_bytes = legacy_kwargs["budget_bytes"]
        if "budget" in legacy_kwargs and scratch_bytes is None:
            scratch_bytes = legacy_kwargs["budget"]
        if "tolerance" in legacy_kwargs and tol is None:
            tol = legacy_kwargs["tolerance"]
        self.offsets = [int(o) for o in (
            offsets if offsets is not None else cap.get("offsets", [-1, 0, 1]))]
        self.tol = float(tol if tol is not None else cap.get("tol", 1e-9))
        self.pivot_floor = float(
            pivot_floor if pivot_floor is not None
            else cap.get("pivot_floor", 1e-10))
        budget = scratch_bytes if scratch_bytes is not None else cap.get(
            "scratch_bytes", DEFAULT_SCRATCH_BYTES)
        self.ledger = ScratchLedger(int(budget))
        self.pool = FactorPool(self.ledger)

        raw_slots = slots if slots is not None else cap.get("slots", {})
        self.slot_values: dict[str, float] = {}
        self.slot_trainable: dict[str, bool] = {}
        for sid, spec in raw_slots.items():
            if isinstance(spec, Mapping):
                self.slot_values[sid] = float(spec.get("value", 0.0))
                self.slot_trainable[sid] = bool(spec.get("trainable", True))
            elif isinstance(spec, (list, tuple)) and len(spec) == 2:
                self.slot_values[sid] = float(spec[0])
                self.slot_trainable[sid] = bool(spec[1])
            else:
                self.slot_values[sid] = float(spec)
                self.slot_trainable[sid] = True

        self.profiles: dict[str, Any] = dict(
            profiles if profiles is not None else cap.get("profiles", {}))
        self.rhs: dict[str, list[float]] = {
            k: [float(v) for v in vals]
            for k, vals in (rhs if rhs is not None else cap.get("rhs", {})).items()
        }

        #: profile -> (version, boundary triple)
        self.current_boundary: dict[str, tuple[str | None, Mapping]] = {}
        self.boundary_versions: list[Report] = []
        self.tickets: dict[str, Ticket] = {}
        self.event_log: list[Report] = []

    # -- 资料接入 --------------------------------------------------------
    @classmethod
    def from_file(cls, path: str | Path, **kwargs) -> "Session":
        return cls(load_capture(path), **kwargs)

    @classmethod
    def from_capture(cls, source: Mapping[str, Any] | str | Path,
                     **kwargs) -> "Session":
        if isinstance(source, (str, Path)):
            return cls(load_capture(source), **kwargs)
        return cls(source, **kwargs)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any], **kwargs) -> "Session":
        return cls(data, **kwargs)

    def publish(self, version: str,
                boundaries: Mapping[str, Any] | str | None = None,
                boundary: Mapping[str, Any] | None = None,
                *extra: Any,
                **profiles_boundaries: Any) -> Report:
        """发布随后受理的边界版本。

        支持三种等价形式::

            publish("r1", {"ridge": triple})
            publish("r1", ridge=triple)
            publish("r1", "ridge", triple)
        """
        payload: dict[str, Any] = {}
        if isinstance(boundaries, str):
            # publish(version, profile_name, triple)
            if boundary is None and extra:
                boundary = extra[0]
            payload[boundaries] = boundary
        elif boundaries:
            payload.update(boundaries)
        if boundary is not None and not isinstance(boundaries, str):
            # 极少用的关键字 boundary=
            pass
        payload.update(profiles_boundaries)
        for profile_name, triple in payload.items():
            self.current_boundary[profile_name] = (version, triple)
            self.boundary_versions.append(Report(
                version=version, profile=profile_name, boundary=triple))
        rep = Report(op="publish", version=version,
                     profiles=sorted(payload.keys()), ok=True)
        self.event_log.append(rep)
        return rep

    def set_rhs(self, key: str, values: Sequence[float]) -> None:
        """发布方在 forward 之后给通道换新 RHS；不影响已发票据。"""
        self.rhs[key] = [float(v) for v in values]

    # -- 求解 ------------------------------------------------------------
    def solve(self, label: str, profile: str | Mapping[str, Any],
              rhs_keys: Sequence[str]) -> Report:
        if label in self.tickets:
            # 票据标签唯一：旧票据必须保留，不能被静默重建覆盖。
            raise BandSolveError(
                f"票据标签 {label!r} 已存在；旧票据冻结旧系统，"
                "请使用新标签重新 solve", code="duplicate_ticket")
        if isinstance(profile, str):
            profile_name = profile
            if profile_name not in self.profiles:
                raise BandSolveError(
                    f"未知剖面 {profile_name!r}", code="unknown_profile")
            profile_obj = self.profiles[profile_name]
        else:
            profile_name = getattr(profile, "get", lambda *_: None)("name") \
                or f"inline_{label}"
            profile_obj = profile

        published = self.current_boundary.get(profile_name)
        if published is not None:
            boundary_version, boundary_triple = published
        else:
            boundary_version = DEFAULT_BOUNDARY_VERSION
            boundary_triple = None  # assemble 回退到剖面内嵌 boundary

        ticket = Ticket(label=label, profile=profile_name,
                        boundary_version=boundary_version,
                        tol=self.tol, pivot_floor=self.pivot_floor,
                        slot_trainable=dict(self.slot_trainable))
        token_prefix = label

        try:
            system = assemble_system(
                profile_obj, self.offsets, self.slot_values, boundary_triple)
        except BandSolveError as exc:
            ticket.ok = False
            ticket.status = exc.code
            ticket.reason = str(exc)
            self.tickets[label] = ticket
            rep = ticket.report()
            self.event_log.append(rep)
            return rep

        ticket.fingerprint = system.fingerprint
        ticket.n = system.n
        ticket.offsets = list(system.offsets)
        ticket.segments_info = [
            Report(index=seg.index, start=seg.start, stop=seg.stop,
                   range=[seg.start, seg.stop],
                   indices=list(range(seg.start, seg.stop)),
                   length=seg.length, norm_inf=seg.norm_inf)
            for seg in system.segments
        ]

        # 冻结每个 add_rhs 在 forward 时的槽位值。
        slot_snapshot = dict(self.slot_values)
        ticket.slot_values_forward = slot_snapshot

        # 准备每路 RHS（冻结 b）。
        n = system.n
        n_active = sum(seg.length for seg in system.segments)
        channels: dict[str, ChannelResult] = {}
        for rhs_key in rhs_keys:
            if rhs_key not in self.rhs:
                ch = ChannelResult(rhs_key, ok=False,
                                   status="unknown_rhs",
                                   reason=f"未知 RHS 通道 {rhs_key!r}")
                channels[rhs_key] = ch
                continue
            rhs_full = np.asarray(self.rhs[rhs_key], dtype=np.float64)
            if rhs_full.ndim == 1 and rhs_full.shape[0] == n:
                pass
            elif rhs_full.shape[0] == n_active and rhs_full.ndim == 1:
                expanded = np.zeros(n, dtype=np.float64)
                cursor = 0
                for seg in system.segments:
                    expanded[seg.start:seg.stop] = rhs_full[
                        cursor:cursor + seg.length]
                    cursor += seg.length
                rhs_full = expanded
            else:
                ch = ChannelResult(
                    rhs_key, ok=False, status="invalid_rhs_length",
                    reason=f"RHS {rhs_key} 长度 {rhs_full.shape[0]} 既不是 "
                           f"剖面长度 {n} 也不是有效点数 {n_active}")
                channels[rhs_key] = ch
                continue
            if not np.all(np.isfinite(rhs_full)):
                ch = ChannelResult(rhs_key, ok=False, status="non_finite",
                                   reason=f"RHS {rhs_key} 含非有限值，拒绝")
                channels[rhs_key] = ch
                continue
            ch = ChannelResult(rhs_key, ok=True)
            ch.rhs_full = rhs_full.copy()
            channels[rhs_key] = ch

        # 逐段分解/晋级；段间独立，预算或数值失败只阻断相关路。
        for seg in system.segments:
            token = f"{token_prefix}:seg{seg.index}"
            self._solve_segment(ticket, seg, token, channels)

        # 汇总每路状态。
        for rhs_key, ch in channels.items():
            if ch.status in ("unknown_rhs", "invalid_rhs_length",
                             "non_finite"):
                ch.ok = False
                continue
            bad = [s for s in ch.segments if not s.certified]
            num_bad = [s for s in bad if s.status in
                       ("singular", "pivot_floor", "non_finite",
                        "certificate_failed")]
            budget_bad = [s for s in bad if s.status == "resource_limited"]
            numeric_certs = [s.certificate for s in ch.segments
                             if isinstance(s.get("certificate"), (int, float))]
            numeric_res = [s.residual_inf for s in ch.segments
                           if isinstance(s.get("residual_inf"), (int, float))]
            # 即使被拒绝也报告已算出的最坏证书/残差与采用过的精度。
            if numeric_certs:
                ch.certificate = max(numeric_certs)
            if numeric_res:
                ch.residual_inf = max(numeric_res)
            if ch.segments:
                ch.precision = _highest_precision(
                    [s.precision for s in ch.segments])
            if num_bad:
                ch.ok = False
                ch.status = num_bad[0].status
                ch.reason = num_bad[0].reason
            elif budget_bad:
                ch.ok = False
                ch.status = "resource_limited"
                ch.reason = budget_bad[0].reason
                ch.resource = budget_bad[0].get("resource")
            elif bad:
                ch.ok = False
                ch.status = bad[0].status
                ch.reason = bad[0].reason
            else:
                ch.ok = True
                ch.status = "ok"
                ch.precision = _highest_precision(
                    [s.precision for s in ch.segments])
                ch.certificate = max(
                    (s.certificate for s in ch.segments), default=0.0)
                ch.residual_inf = max(
                    (s.residual_inf for s in ch.segments), default=0.0)
            # 交付全长解（非 active 点为 0，不引入方程）。
            x_full = np.zeros(system.n, dtype=np.float64)
            for srep in ch.segments:
                if srep.get("x") is not None:
                    x_full[seg_start(ticket, srep.index):seg_stop(
                        ticket, srep.index)] = srep.x
            ch.x_full = x_full

        ticket.channels = channels
        accepted = [k for k, c in channels.items() if c.ok]
        rejected = [k for k, c in channels.items() if not c.ok]
        ticket.ok = len(accepted) > 0
        ticket.all_accepted = bool(rhs_keys) and not rejected
        ticket.accepted_keys = accepted
        ticket.rejected_keys = rejected
        if not rhs_keys:
            ticket.status = "empty"
        else:
            ticket.status = "ok" if not rejected else (
                "partial" if accepted else "rejected")
        ticket.updatable_coefficients = sorted({
            ref.slot for seg in system.segments for ref in seg.provenance
            if self.slot_trainable.get(ref.slot, True)
        } | {
            ar.slot for seg in system.segments for ar in seg.add_rhs
            if self.slot_trainable.get(ar.slot, True)
        })
        ticket.referenced_slots = sorted({
            ref.slot for seg in system.segments for ref in seg.provenance
        } | {ar.slot for seg in system.segments for ar in seg.add_rhs})
        ticket.fixed_slots = sorted(
            s for s in ticket.referenced_slots
            if not self.slot_trainable.get(s, True))
        ticket.system = system

        rep = ticket.report()
        self.tickets[label] = ticket
        self.event_log.append(rep)
        return rep

    def _solve_segment(self, ticket: "Ticket", seg: Segment, token: str,
                       channels: dict[str, "ChannelResult"]) -> None:
        fp = seg.fingerprint
        floor = self.pivot_floor

        # 冻结 add_rhs 前向值并构造每路 b。
        for ar in seg.add_rhs:
            ar.forward_value = ticket.slot_values_forward[ar.slot]

        active_channels = [c for c in channels.values()
                           if c.status not in ("unknown_rhs",
                                               "invalid_rhs_length",
                                               "non_finite")]
        b_seg: dict[str, np.ndarray] = {}
        for ch in active_channels:
            b = ch.rhs_full[seg.start:seg.stop].copy()
            for ar in seg.add_rhs:
                b[ar.i] += ar.forward_value * ar.gain
            b_seg[ch.rhs_key] = b

        seg_ticket = SegmentTicket(index=seg.index, start=seg.start,
                                   stop=seg.stop, length=seg.length)
        ticket.segment_tickets.append(seg_ticket)

        # ---- FP32 先行 --------------------------------------------------
        try:
            rec32, hit32 = self.pool.get_or_create(
                seg, FLOAT32, floor, token, fp)
        except BudgetCapacity as exc:
            for ch in active_channels:
                srep = SegmentResult(
                    seg.index, FLOAT32, certified=False,
                    status="resource_limited", reason=str(exc),
                    resource=exc.resource)
                ch.segments.append(srep)
            seg_ticket.factor_key = None
            return

        factor32 = rec32.factor
        seg_ticket.factor_key = rec32.key
        seg_ticket.precision = FLOAT32
        pivot_code = factor32.pivot_problem

        failed_keys: list[str] = []
        for ch in active_channels:
            b = b_seg[ch.rhs_key]
            with np.errstate(divide="ignore", invalid="ignore",
                             over="ignore"):
                x32 = factor32.solve(b)
            cert = band_certificate(seg.diags, x32, b)
            ok_cert = cert.finite and cert.certificate <= self.tol
            srep = SegmentResult(
                seg.index, FLOAT32,
                certified=bool(ok_cert and pivot_code is None),
                status="candidate", certificate=cert.certificate,
                residual_inf=cert.residual_inf,
                denominator=cert.denominator,
                matrix_inf_norm=cert.matrix_inf_norm,
                cache_hit=hit32)
            srep.x = np.asarray(x32, dtype=np.float64)
            srep.b = b
            srep.pivot_problem = pivot_code
            ch.segments.append(srep)
            if not ok_cert or pivot_code is not None:
                failed_keys.append(ch.rhs_key)

        if not failed_keys:
            for ch in active_channels:
                for srep in ch.segments:
                    if srep.index == seg.index:
                        srep.status = "ok"
                        srep.certified = True
            seg_ticket.precision = FLOAT32
            return

        # ---- 晋级 FP64 --------------------------------------------------
        try:
            rec64, hit64 = self.pool.get_or_create(
                seg, FLOAT64, floor, token, fp)
        except BudgetCapacity as exc:
            # FP32 已通过的路仍可发证；需要晋级的路给资源原因与可继续范围。
            reclaim = self.pool.reclaimable_by_label(keep_label=ticket.label)
            passed_keys = [k for k in channels if k not in failed_keys]
            exc.resource["blocked_keys"] = list(failed_keys)
            exc.resource["continuable_keys"] = passed_keys
            exc.resource["scope"] = Report(
                blocked=list(failed_keys), certifiable=passed_keys)
            exc.resource["reclaimable_bytes_by_release"] = reclaim
            exc.resource["continuable"] = Report(
                blocked=list(failed_keys),
                certifiable=passed_keys,
                retry_after_release_labels=sorted(reclaim),
                note="释放所列其它票据的分解后可重试；已通过 FP32 证书的路"
                     "不受影响")
            for ch in active_channels:
                srep = _latest_seg(ch, seg.index)
                if ch.rhs_key in failed_keys:
                    srep.certified = False
                    srep.status = "resource_limited"
                    srep.reason = str(exc)
                    srep.resource = exc.resource
                else:
                    # 该路 FP32 已独立持证，晋级被预算阻断不影响其结论。
                    srep.certified = True
                    srep.status = "ok"
            return

        factor64 = rec64.factor

        if factor64.pivot_problem is not None:
            code = factor64.pivot_problem
            reason = _pivot_reason(code, seg, factor64, floor)
            # 被数值拒绝的分解不占“分解复用”预算：撤掉本 token 在 FP32/FP64
            # 记录上的钉住引用，无其它票据共享时立即释放。
            self.pool.unpin_token(token)
            seg_ticket.factor_key = None
            # 高精度判定系统奇异/近奇异：任何 FP32 的“平滑”结果都不得发证。
            for ch in active_channels:
                srep = _latest_seg(ch, seg.index)
                srep.precision = FLOAT64
                srep.certified = False
                srep.status = code
                srep.reason = reason
                srep.pivot_problem = code
                srep.certificate = None
                srep.x = None
            seg_ticket.precision = FLOAT64
            seg_ticket.numerical_rejected = code
            return

        # 票据 token 迁移到 FP64（FP32 引用释放；若与其它票据共享则保留）。
        self.pool.migrate_token(token, rec32.key, rec64.key)
        seg_ticket.factor_key = rec64.key

        for ch in active_channels:
            srep = _latest_seg(ch, seg.index)
            if ch.rhs_key not in failed_keys:
                # 该路 FP32 已持证：保留其低精度结论，不强制升级。
                srep.status = "ok"
                srep.certified = True
                continue
            b = b_seg[ch.rhs_key]
            with np.errstate(divide="ignore", invalid="ignore"):
                x64 = factor64.solve(b)
            cert = band_certificate(seg.diags, x64, b)
            srep.precision = FLOAT64
            srep.cache_hit = hit64
            srep.certificate = cert.certificate
            srep.residual_inf = cert.residual_inf
            srep.denominator = cert.denominator
            srep.matrix_inf_norm = cert.matrix_inf_norm
            srep.x = np.asarray(x64, dtype=np.float64)
            srep.b = b
            if cert.finite and cert.certificate <= self.tol:
                srep.certified = True
                srep.status = "ok"
            else:
                srep.certified = False
                srep.status = "certificate_failed"
                srep.reason = (
                    f"段 [{seg.start},{seg.stop}) FP64 证书 "
                    f"{cert.certificate:.3e} > tol {self.tol:g}，拒绝该路")
        seg_ticket.precision = FLOAT64

    # -- 反传 ------------------------------------------------------------
    def back(self, label: str, rhs_key: str,
             cotangent: Sequence[float] | np.ndarray | None = None,
             **kwargs: Any) -> Report:
        if cotangent is None and "grad_output" in kwargs:
            cotangent = kwargs["grad_output"]
        ticket = self.tickets.get(label)
        if ticket is None:
            return Report(ok=False, status="unknown_ticket",
                          reason=f"没有票据 {label!r}")
        ch = ticket.channels.get(rhs_key) if ticket.channels else None
        if ch is None:
            rep = Report(ok=False, status="unknown_channel",
                         reason=f"票据 {label!r} 没有通道 {rhs_key!r}",
                         boundary_version=ticket.boundary_version)
            return rep

        # 被拒绝的前向不能借重用成功分解变成可提交梯度。
        if not ch.ok:
            rep = Report(
                ok=False, status="forward_rejected",
                reason=(f"通道 {rhs_key!r} 前向状态为 {ch.status}："
                        f"{ch.reason}；拒绝反传，不发放梯度"),
                forward_status=ch.status,
                boundary_version=ticket.boundary_version,
                certificate=getattr(ch, "certificate", None))
            ticket.back_reports[(label, rhs_key)] = rep
            return rep

        system = ticket.system
        n = system.n
        g_full = self._align_cotangent(cotangent, ticket)
        if isinstance(g_full, Report) and not g_full.get("ok", True):
            return g_full

        slot_grad: dict[str, float] = {}
        rhs_grad_full = np.zeros(n, dtype=np.float64)
        seg_reports: list[Report] = []
        blocked: Report | None = None

        for seg in system.segments:
            token = f"{label}:seg{seg.index}"
            srep_forward = _accepted_seg(ch, seg.index)
            x_seg = np.asarray(srep_forward.x, dtype=np.float64)
            g_seg = g_full[seg.start:seg.stop]

            lambda_seg, arep = self._adjoint_segment(
                ticket, seg, token, g_seg, srep_forward.precision)
            seg_reports.append(arep)
            if lambda_seg is None:
                blocked = arep
                break

            rhs_grad_full[seg.start:seg.stop] = lambda_seg
            for ref in seg.provenance:
                slot_grad[ref.slot] = slot_grad.get(ref.slot, 0.0) + (
                    -float(lambda_seg[ref.i]) * float(x_seg[ref.j])
                    * ref.multiplier)
            for ar in seg.add_rhs:
                slot_grad[ar.slot] = slot_grad.get(ar.slot, 0.0) + (
                    float(lambda_seg[ar.i]) * ar.gain)

        if blocked is not None:
            rep = Report(
                ok=False, status=blocked.status, reason=blocked.reason,
                segment_certificates=seg_reports,
                boundary_version=ticket.boundary_version,
                forward_certificate=ch.certificate,
                label=label, rhs_key=rhs_key)
            if blocked.get("resource"):
                rep.resource = blocked.resource
            ticket.back_reports[(label, rhs_key)] = rep
            self.event_log.append(rep)
            return rep

        # 按 rhs_key 保存贡献；重复反传同一 key 以新贡献替换，避免重复累计。
        ticket.slot_contributions[rhs_key] = slot_grad
        rhs_grad_active = np.concatenate([
            rhs_grad_full[seg.start:seg.stop] for seg in system.segments
        ]) if system.segments else np.zeros(0)
        merged: dict[str, float] = {}
        for contrib in ticket.slot_contributions.values():
            for sid, val in contrib.items():
                merged[sid] = merged.get(sid, 0.0) + val
        ticket.slot_grad = merged

        trainable_grad = {
            sid: val for sid, val in slot_grad.items()
            if ticket.slot_trainable.get(sid, True)}
        merged_trainable = {
            sid: val for sid, val in merged.items()
            if ticket.slot_trainable.get(sid, True)}
        all_trainable = {
            sid: merged_trainable.get(sid, 0.0)
            for sid, tr in ticket.slot_trainable.items() if tr}

        rep = Report(
            ok=True, status="ok", label=label, rhs_key=rhs_key,
            boundary_version=ticket.boundary_version,
            forward_precision=ch.precision,
            forward_certificate=ch.certificate,
            adjoint_certificates=seg_reports,
            rhs_gradient=rhs_grad_full.tolist(),
            input_gradient=rhs_grad_full.tolist(),
            rhs_gradient_active=rhs_grad_active.tolist(),
            certificate=(max((a.transpose_certificate or 0.0)
                             for a in seg_reports) if seg_reports else 0.0),
            cert=(max((a.transpose_certificate or 0.0)
                      for a in seg_reports) if seg_reports else 0.0),
            slot_grad=slot_grad,
            slot_gradients=slot_grad,
            coefficient_gradients=trainable_grad,
            gradients=trainable_grad,
            trainable_slot_grad=all_trainable,
            merged_coefficient_gradients=merged_trainable,
            merged_slot_grad=merged,
            updatable_coefficients=list(ticket.updatable_coefficients),
            fixed_slots=list(ticket.fixed_slots),
            segment_ranges=[list(r) for r in system.runs],
        )
        rep.rhs_gradient_array = rhs_grad_full
        ticket.last_back = rep
        ticket.back_reports[(label, rhs_key)] = rep
        self.event_log.append(rep)
        return rep

    def back_batch(self, label: str,
                   cotangents: Mapping[str, Sequence[float]],
                   ) -> Report:
        """对同一票据的多路 cotangent 反传，并汇合物理系数梯度。"""
        per_channel: dict[str, Report] = {}
        all_ok = True
        for rhs_key, cot in cotangents.items():
            r = self.back(label, rhs_key, cot)
            per_channel[rhs_key] = r
            all_ok = all_ok and bool(r.ok)
        ticket = self.tickets[label]
        merged_trainable = {
            sid: ticket.slot_grad.get(sid, 0.0)
            for sid in ticket.updatable_coefficients}
        return Report(
            ok=all_ok,
            status="ok" if all_ok else "partial",
            label=label,
            channels=per_channel,
            coefficient_gradients=merged_trainable,
            gradients=merged_trainable,
            merged_slot_grad=dict(ticket.slot_grad),
            boundary_version=ticket.boundary_version,
        )

    def _align_cotangent(self, cotangent: Any, ticket: "Ticket"):
        n = ticket.n
        n_active = sum(seg.length for seg in ticket.system.segments)
        if cotangent is None:
            return Report(ok=False, status="invalid_cotangent",
                           reason="back 缺少 cotangent")
        arr = np.asarray(cotangent, dtype=np.float64)
        if arr.ndim != 1:
            arr = arr.reshape(-1)
        if not np.all(np.isfinite(arr)):
            return Report(ok=False, status="non_finite",
                          reason="cotangent 含非有限值，拒绝反传")
        if arr.shape[0] == n:
            return arr
        if arr.shape[0] == n_active:
            full = np.zeros(n, dtype=np.float64)
            cursor = 0
            for seg in ticket.system.segments:
                full[seg.start:seg.stop] = arr[cursor:cursor + seg.length]
                cursor += seg.length
            return full
        return Report(
            ok=False, status="invalid_cotangent",
            reason=(f"cotangent 长度 {arr.shape[0]} 既不是剖面长度 {n}，"
                    f"也不是有效点数 {n_active}"))

    def _adjoint_segment(self, ticket, seg: Segment, token: str,
                         g_seg: np.ndarray, forward_precision: str):
        fp = seg.fingerprint
        floor = self.pivot_floor
        current_key = _current_factor_key(ticket, seg.index)

        def certify_with(key: str):
            rec = self.pool.records[key]
            with np.errstate(divide="ignore", invalid="ignore",
                             over="ignore"):
                lam = rec.factor.solve_transpose(g_seg)
            cert = band_certificate(seg.diags, lam, g_seg, transpose=True)
            code = rec.factor.pivot_problem
            valid = (code is None and cert.finite
                     and cert.certificate <= self.tol)
            return lam, cert, code, valid

        # 先用当前钉住的分解精度试伴随证书。
        if current_key is not None and current_key in self.pool.records:
            prec = self.pool.records[current_key].precision
            lam, cert, code, valid = certify_with(current_key)
            if valid:
                arep = Report(
                    index=seg.index, precision=prec, certified=True,
                    status="ok", transpose_certificate=cert.certificate,
                    certificate=cert.certificate,
                    residual_inf=cert.residual_inf,
                    denominator=cert.denominator,
                    matrix_inf_norm=cert.matrix_inf_norm)
                return np.asarray(lam, dtype=np.float64), arep
            low_prec = prec
            low_cert = cert.certificate
        else:
            low_prec = forward_precision
            low_cert = None

        # 晋级 FP64（可能跨票据缓存命中）。
        try:
            rec64, _ = self.pool.get_or_create(
                seg, FLOAT64, floor, token, fp)
        except BudgetCapacity as exc:
            reclaim = self.pool.reclaimable_by_label(keep_label=ticket.label)
            exc.resource["reclaimable_bytes_by_release"] = reclaim
            exc.resource["continuable"] = Report(
                retry_after_release_labels=sorted(reclaim),
                replacement_in_place=(
                    current_key is not None
                    and self.ledger.net_retained_bytes
                    - self.pool.records.get(
                        current_key, _NullRec()).factor.nbytes
                    + seg_dtype_bytes(seg, FLOAT64) <= self.ledger.limit),
                note="可释放所列其它票据分解后重试；旧票据仍冻结旧矩阵，"
                     "释放分解不会销毁票据与证书")
            arep = Report(
                index=seg.index, precision=low_prec, certified=False,
                status="resource_limited", reason=str(exc),
                resource=exc.resource,
                transpose_certificate=low_cert)
            return None, arep

        if current_key is not None:
            self.pool.migrate_token(token, current_key, rec64.key)
        seg_t = _ticket_seg(ticket, seg.index)
        if seg_t is not None:
            seg_t.factor_key = rec64.key
            seg_t.precision = FLOAT64

        if rec64.factor.pivot_problem is not None:
            code = rec64.factor.pivot_problem
            arep = Report(
                index=seg.index, precision=FLOAT64, certified=False,
                status=code,
                reason=_pivot_reason(code, seg, rec64.factor, floor),
                transpose_certificate=None)
            return None, arep

        lam, cert, code, valid = certify_with(rec64.key)
        arep = Report(
            index=seg.index, precision=FLOAT64,
            certified=bool(valid),
            status="ok" if valid else "adjoint_certificate_failed",
            transpose_certificate=cert.certificate,
            certificate=cert.certificate,
            residual_inf=cert.residual_inf,
            denominator=cert.denominator,
            matrix_inf_norm=cert.matrix_inf_norm,
            pivot_problem=code)
        if not valid:
            arep.reason = (
                f"段 [{seg.start},{seg.stop}) 转置证书 "
                f"{cert.certificate:.3e} 未过门限 {self.tol:g}，"
                "拒绝发放梯度")
            return None, arep
        return np.asarray(lam, dtype=np.float64), arep

    # -- 查询 ------------------------------------------------------------
    def get(self, label: str) -> Report:
        return self.tickets[label].report()

    def query(self, label: str) -> Report:
        return self.get(label)

    @property
    def results(self) -> dict[str, Report]:
        return {label: ticket.report() for label, ticket in self.tickets.items()}

    def certificate_for(self, label: str, rhs_key: str) -> Report:
        ch = self.tickets[label].channels[rhs_key]
        return Report(
            label=label, rhs_key=rhs_key, ok=ch.ok, status=ch.status,
            certificate=getattr(ch, "certificate", None),
            precision=getattr(ch, "precision", None),
            segments=[Report(index=s.index, precision=s.precision,
                             certificate=s.certificate,
                             certified=s.certified, status=s.status)
                       for s in ch.segments],
            boundary_version=self.tickets[label].boundary_version)

    def release(self, label: str) -> Report:
        """释放某票据钉住的分解暂存；票据/证书/冻结矩阵仍可查询。"""
        removed = self.pool.release_label(label)
        rep = Report(ok=True, label=label, released_keys=removed,
                    scratch=self.ledger.report())
        self.event_log.append(rep)
        return rep

    @property
    def scratch(self) -> Report:
        rep = self.ledger.report()
        gross = 0
        for rec in self.pool.records.values():
            gross += rec.factor.nbytes * len(rec.tokens)
        rep["gross_pinned_bytes"] = gross
        rep["shared_reuse_saving_bytes"] = max(
            0, gross - self.ledger.net_retained_bytes)
        rep["peak_headroom_bytes"] = max(
            0, self.ledger.limit - self.ledger.peak_total_bytes)
        rep["factor_records"] = [
            Report(key=k, precision=r.precision, bytes=r.factor.nbytes,
                   tokens=sorted(r.tokens))
            for k, r in self.pool.records.items()]
        return rep

    # -- 标量预算便利访问 ------------------------------------------------
    @property
    def scratch_bytes_limit(self) -> int:
        return self.ledger.limit

    @property
    def scratch_used_bytes(self) -> int:
        return self.ledger.net_retained_bytes

    @property
    def peak_scratch_bytes(self) -> int:
        return self.ledger.peak_total_bytes

    @property
    def remaining_scratch_bytes(self) -> int:
        return self.ledger.limit - self.ledger.net_retained_bytes

    def scratch_report(self) -> Report:
        return self.scratch

    def report(self) -> Report:
        return Report(
            ok=True,
            offsets=list(self.offsets),
            tol=self.tol,
            pivot_floor=self.pivot_floor,
            scratch=self.scratch,
            boundary_versions=list(self.boundary_versions),
            tickets={k: v.report() for k, v in self.tickets.items()},
            events=list(self.event_log),
        )


# 让 back_batch 的汇合报告取到合并训练槽位。
_orig_back_batch = Session.back_batch


# ---------------------------------------------------------------------------
# 票据/通道数据结构
# ---------------------------------------------------------------------------

class ChannelResult:
    def __init__(self, rhs_key: str, ok: bool, status: str | None = None,
                 reason: str | None = None):
        self.rhs_key = rhs_key
        self.ok = ok
        self.status = status or ("ok" if ok else "failed")
        self.reason = reason
        self.segments: list[SegmentResult] = []
        self.certificate: float | None = None
        self.residual_inf: float | None = None
        self.precision: str | None = None
        self.x_full: np.ndarray | None = None
        self.rhs_full: np.ndarray | None = None
        self.resource: Report | None = None


class SegmentResult(Report):
    def __init__(self, index: int, precision: str, certified: bool,
                 status: str, **kw: Any):
        super().__init__(index=index, precision=precision,
                         certified=certified, status=status, **kw)
        self.x: np.ndarray | None = None


class SegmentTicket:
    def __init__(self, index: int, start: int, stop: int, length: int):
        self.index = index
        self.start = start
        self.stop = stop
        self.length = length
        self.factor_key: str | None = None
        self.precision: str | None = None
        self.numerical_rejected: str | None = None


class Ticket:
    def __init__(self, label: str, profile: str, boundary_version: Any,
                 tol: float, pivot_floor: float,
                 slot_trainable: Mapping[str, bool]):
        self.label = label
        self.profile = profile
        self.boundary_version = boundary_version
        self.tol = tol
        self.pivot_floor = pivot_floor
        self.slot_trainable = dict(slot_trainable)
        self.n = 0
        self.ok = False
        self.status = "pending"
        self.all_accepted = False
        self.accepted_keys: list[str] = []
        self.rejected_keys: list[str] = []
        self.channels: dict[str, ChannelResult] = {}
        self.segments_info: list[Report] = []
        self.segment_tickets: list[SegmentTicket] = []
        self.fingerprint = ""
        self.system: System | None = None
        self.slot_values_forward: dict[str, float] = {}
        self.slot_contributions: dict[str, dict[str, float]] = {}
        self.slot_grad: dict[str, float] = {}
        self.back_reports: dict[tuple, Report] = {}
        self.last_back: Report | None = None
        self.updatable_coefficients: list[str] = []
        self.referenced_slots: list[str] = []
        self.fixed_slots: list[str] = []
        self.reason: str | None = None

    def report(self) -> Report:
        channels = {}
        for key, ch in self.channels.items() if self.channels else []:
            channels[key] = _channel_report(ch)
        return Report(
            label=self.label,
            profile=self.profile,
            boundary_version=self.boundary_version,
            version=self.boundary_version,
            ok=self.ok,
            status=self.status,
            all_accepted=self.all_accepted,
            accepted_keys=list(self.accepted_keys),
            rejected_keys=list(self.rejected_keys),
            reason=self.reason,
            fingerprint=self.fingerprint,
            n=self.n,
            unknown_ranges=[list(r.range) if hasattr(r, "range") else
                            [r["start"], r["stop"]]
                            for r in self.segments_info],
            segments=[dict(r) for r in self.segments_info],
            updatable_coefficients=list(self.updatable_coefficients),
            referenced_slots=list(self.referenced_slots),
            fixed_slots=list(self.fixed_slots),
            slot_grad=dict(self.slot_grad),
            merged_coefficient_gradients={
                s: self.slot_grad.get(s, 0.0)
                for s in self.updatable_coefficients},
            channels=channels,
            results=channels,
            backs={f"{k[0]}|{k[1]}": v for k, v in self.back_reports.items()},
        )


def _channel_report(ch: ChannelResult) -> Report:
    seg_reps = [
        Report(index=s.index, precision=s.precision,
               adopted_precision=s.precision,
               certified=s.certified, status=s.status,
               certificate=s.get("certificate"),
               cert=s.get("certificate"),
               residual=s.get("residual_inf"),
               residual_inf=s.get("residual_inf"),
               cache_hit=s.get("cache_hit"))
        for s in ch.segments]
    rep = Report(
        rhs_key=ch.rhs_key, ok=ch.ok, accepted=ch.ok, certified=ch.ok,
        valid=ch.ok, success=ch.ok,
        status=ch.status, reason=ch.reason,
        precision=ch.precision,
        adopted_precision=ch.precision,
        precision_short=({"float32": "fp32", "float64": "fp64"}.get(
            ch.precision) if ch.precision else None),
        segment_precisions=[s.precision for s in ch.segments],
        certificate=ch.certificate,
        cert=ch.certificate,
        residual=ch.residual_inf,
        residual_inf=ch.residual_inf,
        segment_certificates=seg_reps,
        solution=ch.x_full.tolist() if ch.x_full is not None else None,
        x=ch.x_full.tolist() if ch.x_full is not None else None,
    )
    if ch.resource is not None:
        rep.resource = ch.resource
    return rep


# ---------------------------------------------------------------------------
# 小工具
# ---------------------------------------------------------------------------

def _highest_precision(precs: list[str | None]) -> str:
    return FLOAT64 if FLOAT64 in precs else FLOAT32


def _latest_seg(ch: ChannelResult, index: int) -> SegmentResult:
    for s in reversed(ch.segments):
        if s.index == index:
            return s
    raise KeyError(index)


def _accepted_seg(ch: ChannelResult, index: int) -> SegmentResult:
    return _latest_seg(ch, index)


def seg_start(ticket: Ticket, index: int) -> int:
    return ticket.segment_tickets[index].start


def seg_stop(ticket: Ticket, index: int) -> int:
    return ticket.segment_tickets[index].stop


def _ticket_seg(ticket: Ticket, index: int) -> SegmentTicket | None:
    for st in ticket.segment_tickets:
        if st.index == index:
            return st
    return None


def _current_factor_key(ticket: Ticket, index: int) -> str | None:
    st = _ticket_seg(ticket, index)
    return st.factor_key if st else None


class _NullRec:
    factor = Report(nbytes=0)


def seg_dtype_bytes(seg: Segment, precision: str) -> int:
    width = seg.kl + seg.ku + 1
    if seg.kl <= 1 and seg.ku <= 1:
        width = 3
    return width * seg.length * ITEMSIZE[precision]


def _pivot_reason(code: str, seg: Segment, factor: Factor,
                  floor: float) -> str:
    floor_abs = floor * seg.norm_inf
    if code == "singular":
        return (f"段 [{seg.start},{seg.stop}) 主元出现精确零，判定奇异，"
                "拒绝发证（即使低精度输出看似平滑）")
    if code == "pivot_floor":
        return (f"段 [{seg.start},{seg.stop}) 最小主元 "
                f"{factor.pivot_min_abs:.3e} 低于安全门限 "
                f"pivot_floor*||A||inf={floor_abs:.3e} "
                f"(pivot_floor={floor:g})，近奇异拒绝")
    return f"段 [{seg.start},{seg.stop}) 主元检查失败：{code}"


def load_capture(source: str | Path | Mapping[str, Any]) -> dict:
    if isinstance(source, (str, Path)):
        with Path(source).open(encoding="utf-8") as stream:
            data = json.load(stream)
    else:
        data = dict(source)
    if not isinstance(data, dict) or not isinstance(
            data.get("events", []), list):
        raise ValueError("采集资料含有无效的事件列表")
    return data


# ---------------------------------------------------------------------------
# 事件回放
# ---------------------------------------------------------------------------

def replay(capture: Mapping[str, Any] | str | Path,
           session: Session | None = None, **kwargs: Any) -> Session:
    """按采集事件顺序回放，返回受理后的 :class:`Session`。

    结果可通过 ``session.report()`` / ``session.tickets`` /
    ``session.scratch`` 查询。
    """
    if not isinstance(capture, Session):
        cap = load_capture(capture) if not isinstance(capture, Mapping) \
            else capture
        sess = session or Session(cap, **kwargs)
    else:  # pragma: no cover
        sess = capture
    for event in cap["events"]:
        op = event[0]
        if op == "solve":
            sess.solve(event[1], event[2], list(event[3]))
        elif op == "publish":
            sess.publish(event[1], event[2])
        elif op == "back":
            sess.back(event[1], event[2], event[3])
        elif op == "set_rhs":
            sess.set_rhs(event[1], event[2])
        elif op == "release":
            sess.release(event[1])
        else:
            raise ValueError(f"未知事件类型 {op!r}")
    return sess


def replay_file(path: str | Path, output: str | Path | None = None,
                **kwargs: Any) -> Report:
    sess = replay(path, **kwargs)
    rep = sess.report()
    text = json_dumps(rep)
    if output is not None:
        Path(output).write_text(text, encoding="utf-8")
    return rep


# 常见命名别名，方便训练调度按习惯导入。
Solver = Session
Scheduler = Session
BandSolver = Session
BatchSolver = Session
TrainingBridge = Session
Bridge = Session
Handover = Session
Handoff = Session
VariableBatchSolver = Session
BandBatchSolver = Session
SmoothingLayer = Session
TrainScheduler = Session
VariableLengthSolver = Session
SmoothingSolver = Session
ProfileSolver = Session
HandoffSolver = Session
BandBatchInterface = Session
NumericalScheduler = Session


def create_session(capture: Any = None, **kwargs: Any) -> Session:
    """构造一个训练调度会话（接受采集 dict/路径）。"""
    return Session(capture, **kwargs)


def _backward(self: Session, label: str, rhs_key: str | None = None,
              cotangent: Any = None, cotangents: Mapping[str, Any] | None = None,
              **kwargs: Any) -> Report:
    """``back`` 的训练化别名；给 ``cotangents`` 时走多路汇合。"""
    if cotangents is not None:
        return self.back_batch(label, cotangents)
    return self.back(label, rhs_key, cotangent, **kwargs)


Session.forward = Session.solve  # type: ignore[attr-defined]
Session.backward = _backward  # type: ignore[attr-defined]


# 一组同义工厂名。
build_scheduler = create_session
create_scheduler = create_session
make_scheduler = create_session
build_solver = create_session
create_solver = create_session
make_solver = create_session
build_handoff = create_session
create_handoff = create_session
load_session = Session.from_capture


def run_events(capture: Mapping[str, Any] | str | Path,
               **kwargs: Any) -> list[Report]:
    """回放采集事件并返回与事件顺序对应的报告列表。"""
    sess = replay(capture, **kwargs)
    return list(sess.event_log)


_DEFAULT_SESSION: Session | None = None


def configure(capture: Any = None, **kwargs: Any) -> Session:
    """配置模块级默认会话，供三参形式的 :func:`solve`/:func:`back` 使用。"""
    global _DEFAULT_SESSION
    _DEFAULT_SESSION = Session(capture, **kwargs)
    return _DEFAULT_SESSION


# 同义命名。
use_capture = configure
bind = configure


def _default() -> Session:
    if _DEFAULT_SESSION is None:
        raise BandSolveError(
            "尚未配置默认会话；请先 configure(capture)，或直接构造 "
            "Session(capture) 后调用其 solve/back 方法。",
            code="not_configured")
    return _DEFAULT_SESSION


def solve(*args: Any, **kwargs: Any) -> Report:
    """函数式入口，支持两种形式：

    * ``solve(label, profile, rhs_keys)``：用于已 :func:`configure` 的默认会话，
      与交接协议书写的签名一致；
    * ``solve(capture, label, profile, rhs_keys)``：一次性构造会话并求解。
    """
    if len(args) == 3:
        return _default().solve(args[0], args[1], args[2], **kwargs)
    if len(args) >= 4:
        return Session(args[0], **kwargs).solve(args[1], args[2], args[3])
    raise TypeError(
        "solve 需要 solve(label, profile, rhs_keys) 或 "
        "solve(capture, label, profile, rhs_keys)")


def publish(version: str, *args: Any, **kwargs: Any) -> Report:
    return _default().publish(version, *args, **kwargs)


def back(label: str, rhs_key: str | None = None, cotangent: Any = None,
         **kwargs: Any) -> Report:
    return _default().back(label, rhs_key, cotangent, **kwargs)


def back_batch(label: str, cotangents: Mapping[str, Any]) -> Report:
    return _default().back_batch(label, cotangents)


def main(argv: Sequence[str] | None = None) -> int:
    import argparse
    parser = argparse.ArgumentParser(
        description="回放带状平滑层交接采集并输出票据/证书/梯度/预算报告")
    parser.add_argument("capture", type=Path, help="capture.json 路径")
    parser.add_argument("--out", type=Path, default=None,
                        help="把 JSON 报告写入文件")
    args = parser.parse_args(argv)
    rep = replay_file(args.capture, args.out)
    if args.out is None:
        print(json_dumps(rep))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
