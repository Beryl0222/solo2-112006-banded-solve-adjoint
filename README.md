# 变长带状求解器的边界更新和伴随交接

`samples/capture.json` 是离线合成的交接采集，包含原始数据和接收顺序。[交接协议](交接协议.md) 描述各字段及上下游共同使用的数值口径。当前入口可读取采集文件并列出操作顺序。

环境使用 Python 3.11 和锁定的 NumPy。首次在本地准备虚拟环境后运行以下命令。

```bash
python -m pip install -r requirements.txt
python -m read_capture samples/capture.json
python -m unittest discover -s tests -v
```

## 训练调度接口（变长批求解 + 反传）

权威实现是单文件模块 [`band_solver.py`](band_solver.py)，只依赖 NumPy。
`solver.py / scheduler.py / smoother.py / handoff.py / solution.py / core.py`
等只是把全部名字再导出的兼容别名。

```python
from band_solver import Session

s = Session("samples/capture.json")     # 也接受 capture dict
s.solve("old", "ridge", ["easy", "edge"])
s.solve("fault", "fault", ["bad"])      # 精确奇异：保留原因，不影响 ridge
s.publish("r1", {"ridge": {...}})       # 随后受理的边界版本
s.solve("new", "ridge", ["drift"])

g = [1.0, 0.0, -2.0, 0.0, 2.0]          # 与剖面全长对齐（无效点置 0）
old = s.back("old", "edge", g)
old["coefficient_gradients"]            # 仅 trainable=true 的物理槽位
old["rhs_gradient"]                     # dL/db = A^{-T} g，全长散布回剖面
old["slot_grad"]                        # 含固定槽（如 p）的原始汇合梯度
```

也可一键回放整个采集：

```bash
python -m band_solver samples/capture.json --out report.json
python -m read_capture samples/capture.json --replay
```

或代码内 `session = replay("samples/capture.json")`。

### 票据冻结语义

`solve(label, profile, rhs_keys)` 返回报告并在 `session.tickets[label]`
保存票据。票据冻结：

* 组装后的高精度矩阵 `A`、每路含 `add_rhs` 的右端 `b`、被接受的解 `x`；
* 每个 active 连续段的未知量区间 `unknown_ranges`（如 `[[0,3],[4,5]]`）；
* 受理时的边界版本 `boundary_version`（剖面内嵌边界为 `None`，publish 后为 `"r1"`）；
* 槽位的 `trainable` 快照与系数出处（物理槽 id + multiplier）。

之后修改槽位值、`publish` 新边界或 `set_rhs` 更换通道 RHS，都不会重建旧票据；
旧 `back` 仍对旧 `A` 解转置方程。同一物理槽被多段/多路引用时按有效前向求和；
`back_batch(label, {key: cot})` 会汇合多路系数梯度，同一 `rhs_key` 重复反传
以新贡献替换，不会双倍累计。被拒绝（奇异/主元门限/证书不过/预算不足）的通道
调用 `back` 只返回 `status="forward_rejected"`，绝不发放梯度。

### 数值证书与精度晋级

* 段由 `active` 的连续真段切出，单点段使用 `boundary.single`；补齐位置只占
  带状数组宽度，不产生方程。
* 三对角走固定舍入顺序的 Thomas 递推，更宽窄带走包络 LU + 转置三角求解，
  均为 `O(n·bw²)` 的实际带状存储，不做稠密展开。
* 证书严格按协议：
  `||Ax-b||∞ / (||A||∞·||x||∞ + ||b||∞)`，FP64 对原高精度 `A,b` 计算；
  分母为零时记 `0`，但零主元/精确奇异仍拒绝发证。
* 每段先试 FP32；残差证书或主元安全比率不过时，在预算内晋级 FP64。
  高精度分解若判定 `singular / pivot_floor / non_finite`，该段任何 FP32
  “看似平滑”的结果都会被撤销。每路报告 `certificate / residual_inf /
  precision` 及逐段明细。
* 反传解 `Aᵀλ=g`，并对转置方程用同一 `tol` 发证（矩阵范数取 `||Aᵀ||∞`）；
  前向、反向证书都有效才交还梯度。梯度为
  `dL/dθ = Σ multiplier·(-λ_i x_j) + Σ gain·λ_boundary`，
  `dL/drhs = λ`。

### 暂存预算 `scratch_bytes`

台账 `session.scratch` 用真实 `ndarray.nbytes` 计量：

* `net_retained_bytes`：分解池净占用；相同段矩阵（按 FP64 矩阵内容取指纹，
  与剖面长度无关）跨票据/通道复用时只计一次；
* 晋级时新高精度分解在旧分解仍钉住期间分配，二者同时计入峰值；
* `peak_total_bytes / remaining_scratch_bytes / peak_headroom_bytes`；
* 预算不足时相关路返回 `status="resource_limited"` 与 `resource`
  （缺口、可用量、`reclaimable_bytes_by_release`、`continuable` 范围）。
  独立的段/路照常发证；调用 `session.release(label)` 释放其它票据的分解后
  可重试，票据本身与冻结矩阵/证书仍在，反传会按需重新分解。

样例采集（预算 220 B）的确定性台账峰值为 204 B：旧 `edge` 的三对角段 FP32
证书约 `6.46e-10` 被接受，长度一段晋级 FP64；新 `r1` 系统晋级 FP64；旧反传
的伴随晋级 FP64 时与新分解共存，峰值 204 B，两路反传均成功。

改变变长 RHS 批的拆法（一次多路 vs 多次单路）得到的解、证书与梯度逐位一致，
见 `tests/test_band_solver.py`。
