# Q3前缀时长分位数实验

## 目的

检验0.2、0.6、1.0秒数值前缀能否在Family Quantile基础上改善时长预测。仅使用train拟合、validation评价，test未参与。

## 方法

- Family Quantile：按P/A/L在train计算Q50/Q80/Q90；
- Gradient Boosting Quantile：使用当前获准前缀的聚合数值特征、family和prefix duration；
- ID、dataset、split和真实时长均不进入输入；
- 缺失数值用train中位数填充；
- 独立分位数预测后使用累积最大值修复交叉。

## 主要结果

| 方法 | Prefix | Q50 MAE/s | Q50 RMSE/s | Q80覆盖率 | Q90覆盖率 | Q90 Pinball | Q90平均Idle/s |
|---|---:|---:|---:|---:|---:|---:|---:|
| Family | 任意 | 3.477 | 4.785 | 0.800 | 0.914 | 1.111 | 9.365 |
| Quantile GBDT | 0.2s | 3.425 | 4.596 | 0.657 | 0.857 | 1.184 | 9.627 |
| Quantile GBDT | 0.6s | 3.369 | 4.582 | 0.629 | 0.857 | 1.167 | 7.757 |
| Quantile GBDT | 1.0s | **3.334** | **4.557** | 0.543 | 0.800 | **1.004** | **6.521** |

## 解释

- 前缀从0.2扩展到1.0秒时，Q50点预测和Q90 pinball逐步改善；
- 1秒GBDT相对Family的Q50 MAE改善约4.1%；
- 1秒GBDT在更低idle下取得更好的Q90 pinball，表明前缀确实提供了时长信息；
- 但Q80/Q90覆盖率明显低于名义值，不能未经校准直接进入风险调度；
- 原始分位数交叉率15.24%，小样本下分别拟合多个quantile不够稳定。

## 决策

保留两条方法：

```text
Family Quantile：强概率基线
Quantile GBDT + calibration：主时长模型候选
```

不增加深度时长模型。下一步只增加校准，并检查校准后Q80/Q90的覆盖、overrun和idle；若test仍不稳定，调度主方案回退到Family Quantile。

