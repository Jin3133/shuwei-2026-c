# 2026 数维杯 C 题：不确定性感知状态预测与调度

本项目面向2026年第十二届数维杯全国大学生数学建模挑战赛C题，构建从手部状态估计、遮挡期间主动重新观测到风险感知任务调度的统一实验框架。

## 技术主线

```text
3D手部关键点
→ 运动状态与观测可靠性
→ 概率轨迹预测
→ 遮挡回放与重新观测
→ 前缀时长分位数预测
→ 风险感知滚动调度
```

## 当前模块

- Q1：Last Observation、Constant Velocity、Kalman、GRU、TCN、质量感知概率GRU；
- Q2：严格因果人工mask回放、固定间隔策略、不确定性触发策略；
- Q3：非交叉时长分位数模型、风险优先级策略、官方规则模拟器；
- 数据：官方索引审计、交付验收、合并工具和模型窗口适配器；
- 训练：配置驱动、早停、学习率调度、统一指标和权重保存。

## 目录

```text
config/        数据交付接口配置
configs/       模型与实验配置
docs/          数据、方法和实验说明
scripts/       训练、转换与测试入口
shuwei_c/      核心Python包
src/           数据审计与交付工具
```

原始数据、Parquet、NPZ、模型权重和本地实验输出不会提交到仓库。

## 环境

```bash
python -m pip install -e .
```

可选依赖：

```bash
python -m pip install -e '.[data,kalman,q3,calibration,dev]'
```

## 快速检查

```bash
python scripts/smoke_test_models.py
python scripts/smoke_test_quality_model.py
python scripts/smoke_test_replay_and_scheduler.py
```

## 训练示例

```bash
python scripts/train_q1.py --config configs/q1_delivery_gru_mse.yaml
```

完整方法、数据接口和实验边界见 `docs/`。

