# Q1逐预测距离收缩校准

## 方法

对冻结GRU的预测位移使用逐距离标量收缩：

\[
\hat y_h^{\mathrm{shrink}}=\alpha_h\hat y_h,
\qquad
\alpha_h=\arg\min_\alpha\sum_i\|y_{i,h}-\alpha\hat y_{i,h}\|_2^2.
\]

0.2秒原始预测已经最优，因此固定 \(\alpha_{0.2}=1\)，仅校准0.5和1.0秒。验证采用leave-one-environment-out：每个环境的系数仅由另外三个环境拟合，再应用到被留出的环境。

## 结果

| 模型 | 0.2s/m | 0.5s/m | 1.0s/m | Overall ADE/m |
|---|---:|---:|---:|---:|
| MSE GRU | 0.02981 | 0.07767 | 0.15359 | 0.08702 |
| MSE GRU + Shrink | 0.02981 | 0.07576 | 0.14253 | 0.08270 |
| Gaussian GRU | 0.02681 | 0.07841 | 0.15701 | 0.08741 |
| Gaussian GRU + Shrink | **0.02681** | **0.07511** | 0.14287 | **0.08160** |

Gaussian收缩模型的ADE相对原模型下降约6.6%，1秒误差下降约9.0%，并优于Last Position的0.15097m。四个验证环境的ADE均改善，环境Bootstrap 95%差值区间为[-0.00791,-0.00406]m。

## 冻结系数

用于后续一次性test推理的全validation系数为：

| 模型 | 0.2s | 0.5s | 1.0s |
|---|---:|---:|---:|
| MSE GRU | 1.0000 | 0.7786 | 0.6016 |
| Gaussian GRU | 1.0000 | 0.7353 | 0.5491 |

收缩后Gaussian NLL在0.5和1.0秒均改善，联合覆盖率基本保持。由于只修正均值，不缩放预测方差，仍可直接为Q2提供不确定性。

## 最终候选

第一问主模型调整为 `Gaussian GRU + horizon-wise shrinkage`。MSE GRU + Shrink作为确定性消融，Last Position、Constant Velocity和Kalman作为物理基线。
