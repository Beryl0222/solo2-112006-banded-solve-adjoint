# 变长带状求解器的边界更新和伴随交接

`samples/capture.json` 是离线合成的交接采集，包含原始数据和接收顺序。[交接协议](交接协议.md) 描述各字段及上下游共同使用的数值口径。

`banded_solver.py` 给训练调度提供 Python 版的变长批求解与反传接口。每个通道、
每一段都按交接资料里声明的实际带状存储装配方程，不按剖面长度补齐，也不为补齐
位置引入方程。

## 运行

环境使用 Python 3.11 和锁定的 NumPy。首次在本地准备虚拟环境后运行以下命令。

```bash
python -m pip install -r requirements.txt
python -m read_capture samples/capture.json
python -m banded_solver samples/capture.json        # 回放并输出求解、证书与反传
python -m banded_solver samples/capture.json --budget 4096
python -m unittest discover -s tests -v
```

## 接口

```python
from read_capture import read_capture
from banded_solver import BatchSolver

solver = BatchSolver(read_capture("samples/capture.json"))
# 可选显式预算；默认使用采集里的 scratch_bytes。
solver = BatchSolver(capture, budget_bytes=1 << 20)

report = solver.solve("old", "ridge", ["easy", "edge"])
report["easy"].accepted          # 前向证书是否有效
report["easy"].precision         # 该路采用的最高精度（逐段精度见 segments）
report["easy"].certificate       # 按协议口径的残差证书
report["easy"].segments          # 每段的范围、精度、残差、主元比率与被拒原因
report["easy"].boundary_version  # 票据绑定的边界版本
report.continuation()            # 能继续训练的范围与剩余资源原因

solver.publish("r1", {"ridge": {...}})      # 只影响随后受理的前向
report2 = solver.solve("new", "ridge", ["drift"])

grad = solver.back("old", "edge", cotangent)
grad.status                      # "ok" / "partial" / "rejected"
grad.rhs_grad                    # 整条剖面长度；非有效点为 0
grad.coefficient_grads           # 可训练槽位；固定边界槽位不出现
grad.coefficient_grads_by_key    # 多路时按通道拆分
solver.results("old")            # 训练端查询每路证书与绑定信息
```

### 语义要点

* **边界版本**：`publish` 之后新受理的前向用新版本；旧票据保留旧矩阵、旧解、
  旧右端项与旧未知量范围，不能按当前条件重建。`update_rhs` 只换右端项，矩阵
  指纹不变时缓存仍可命中。缓存键是实际方程指纹（布局与系数），不是剖面长度。
* **段**：内点掩码中的断开把剖面切成独立段；长度为一的段使用 `single` 边界行
  （未给出时保留该点原方程），有独立边界方程。补齐位置不生成列也不生成方程。
* **证书**：每段 `||Ax-b||∞ / (||A||∞·||x||∞+||b||∞)`，矩阵范数为最大绝对行和，
  分母为零时证书为零；按原高精度输入计算并拒绝任何非有限。近奇异的伴随也须
  通过同一门限的转置方程证书。
* **精度与晋级**：FP32 优先；证书或 `pivot_floor` 不达标时在预算内晋级到 FP64。
  晋级失败、奇异或预算不足都保留原因，可认证的段与通道照常继续。
* **预算**：`scratch_bytes` 计分解复用所占空间与晋级暂存，报告占用、峰值与
  剩余额度；被票据引用的分解不会被淘汰，预算冲突时返回剩余资源原因。
  被拒绝的前向不保留分解，也不会把成功的分解借给后续乘用。
* **反传**：`cotangent` 对应通道与有效点，可给整条剖面长度或有效点个数；
  多路用 `{key: cotangent}`。多段或多路共享同一物理系数时按有效前向汇合梯度，
  作为固定边界常数使用的槽位（`trainable=false`）不训练。
* **批拆分无关**：改变变长批的拆分方式（单路反传后相加，或一次多路反传，
  或改变通道顺序）不改变数值结果。
