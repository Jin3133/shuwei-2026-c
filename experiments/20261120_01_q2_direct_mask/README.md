# Q2直接遮挡预测实验

## 目的

比较Last Position、Constant Velocity和冻结的Gaussian GRU在模型原生支持的0.2、0.5和1.0秒人工mask终点上的表现。仅使用validation，test保持未查看。

## 信息边界

模型只接收mask开始前的15帧历史、速度、质量、hand和最后可见位置。`target_position`仅在所有预测生成后用于评分。

## 结果摘要

| Mask | 方法 | 平均误差 | RMSE | P90 |
|---:|---|---:|---:|---:|
| 0.2s | Last Position | 0.04079 | 0.05595 | 0.09234 |
| 0.2s | Constant Velocity | 0.03030 | 0.04088 | 0.06105 |
| 0.2s | Gaussian GRU | **0.02680** | **0.03690** | **0.05377** |
| 0.5s | Last Position | 0.08767 | 0.12035 | 0.20492 |
| 0.5s | Constant Velocity | 0.09876 | 0.12757 | 0.20565 |
| 0.5s | Gaussian GRU | **0.07605** | **0.09807** | **0.15504** |
| 1.0s | Last Position | **0.14685** | 0.19724 | 0.31436 |
| 1.0s | Constant Velocity | 0.23272 | 0.29229 | 0.45433 |
| 1.0s | Gaussian GRU | 0.15074 | **0.19195** | **0.30174** |

Gaussian GRU的90%校准覆盖率依次为0.8964、0.9097和0.8960，校准迁移到Q2 mask样本后仍接近目标。

## 结论与下一步

0.2和0.5秒直接使用Gaussian GRU；1.0秒GRU减少大误差但平均误差略差于保持位置。下一轮只调整一个因素：在validation上拟合逐horizon位移收缩系数，使长时预测可向Last Position收缩。暂不外推1.5和2.0秒。

