# GitHub调研与复用决策

> 原则：先调研、后开发；优先直接依赖，其次做薄适配，最后才自行实现题目专用逻辑。

## 1. 选型表

| 问题 | 项目 | 许可证/状态 | 决策 | 原因 |
|---|---|---|---|---|
| Q1缺失感知 | [GRU-D参考实现](https://github.com/PeterChe1990/GRU-D) | MIT；实现较旧 | 参考设计，不直接依赖 | “数值+mask+距上次观测时间”适合本题，但旧框架不适合当前PyTorch |
| Q1/Q2概率预测 | [Microsoft ProbTS](https://github.com/microsoft/ProbTS) | 开源；模型完整但依赖较重 | 参考统一概率预测接口 | 本题样本和输出结构较小，不引入Lightning与整套benchmark |
| Q1/Q2校准 | [MAPIE](https://github.com/scikit-learn-contrib/MAPIE) | BSD-3-Clause；维护活跃 | 后续作为可选依赖 | split conformal与validation校准符合题目环境级划分 |
| Q1 TCN | [locuslab/TCN](https://github.com/locuslab/TCN) | MIT；经典实现 | 已适配核心结构 | 只保留因果、膨胀卷积和残差块 |
| Q2主动观测 | [cpomdp](https://github.com/inferogenesis/cpomdp) | JAX连续主动推断 | 仅参考滚动信息价值思想 | 题目动作和信息返回协议特殊，不适合直接依赖 |
| Q3概率时长 | [NeuralForecast](https://github.com/Nixtla/neuralforecast) | Apache-2.0；维护活跃 | 参考quantile/pinball接口 | 数据是短前缀轨迹而非标准长时间序列表 |
| Q3非交叉分位数 | [Timestral](https://github.com/mistralai/timestral) | 代码Apache-2.0 | 参考非交叉quantile head | 模型权重和规模明显超过本题需要，不直接引入 |
| Q3调度 | [Google OR-Tools](https://github.com/google/or-tools) | Apache-2.0；成熟 | 作为可选依赖 | 用于离线Oracle和滚动确定性子问题 |
| Q3调度建模 | [PyJobShop](https://github.com/PyJobShop/PyJobShop) | 开源；基于CP-SAT | 参考约束表达 | 题目在线review和预留占用仍需自建状态机 |

## 2. 直接采用的设计

### 缺失感知编码

输入显式区分：

```text
observed value
observation mask
time since last valid observation
raw quality evidence
```

不把插值后的数值当成真实观测。衰减和质量门控在模型内部完成。

### 概率输出与校准

神经模型先输出Gaussian参数或多个条件分位数；随后只用validation环境残差校准区间。测试环境只做一次冻结评估。

当前轨迹校准采用MAPIE式split-conformal流程，但针对多步3D输出实现为逐horizon的标准化残差半径校准；校准参数必须随冻结模型一并保存。

### 非交叉时长分位数

模型不独立输出互相无约束的Q50/Q80/Q90，而是输出：

```text
Q50 = positive(base)
Q80 = Q50 + positive(gap_1)
Q90 = Q80 + positive(gap_2)
```

从结构上保证分位数顺序。

### 滚动调度

求解器只读取当前已公开的信息和冻结的时长预测。每次review返回或执行完成后重新规划，不能把真实未来时长传给策略。

## 3. 不直接引入的项目

- 不直接使用Trajectron++：面向多智能体交通图和多模态交互，改造成本超过收益；
- 不直接使用Timestral权重：任务规模小，且输入不是标准连续时间序列；
- 不直接使用完整ProbTS/NeuralForecast训练框架：会将数据协议和大量依赖带入项目；
- 不使用强化学习调度仓库：四天内难以完成可信训练、约束保证和公平对比；
- 不用通用POMDP库替代题目回放器：无法天然保证“延迟结束前不可查看”和“只能返回原记录已有信息”。

## 4. 复用审计要求

每次新增第三方方案必须记录：仓库链接、许可证、采用的模块或思想、没有直接采用的原因、版本或提交。复制源码时保留原许可证和版权头；能通过包依赖完成时不复制源码。

