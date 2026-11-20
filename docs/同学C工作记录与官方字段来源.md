# 同学C工作记录与官方字段来源

> 开始日期：2026-11-20  
> 当前阶段：官方索引整理、字段冻结、合并验收准备

## 1. 已读取的官方文件

| 文件 | 行数 | 主键或核心键 | 作用 |
|---|---:|---|---|
| `record_split.csv` | 69 | `episode_uuid` | 记录身份和官方数据划分 |
| `source_segment_index.csv` | 349 | `unit_id` | 操作区间及其记录映射 |
| `tasks.csv` | 36 | `job_id` | S1/S2/S3调度任务 |
| `common_conditions.json` | 1份 | `case_id` | 第三问公共规则和随机种子 |
| `job_families.csv` | 3 | `family_id` | P/A/L任务族解释 |
| `service_zones.csv` | 5 | `zone_id` | Z0—Z4二维服务区坐标 |

## 2. `records`字段分别从哪里来

| 统一字段 | 来源 | 性质 | 处理规则 |
|---|---|---|---|
| `episode_uuid` | `record_split.csv` | 官方字段 | 原样保留，作为记录主键 |
| `dataset` | `record_split.csv` | 官方字段 | 原样保留 |
| `environment_id` | `record_split.csv` | 官方字段 | 原样保留 |
| `scene_id` | `record_split.csv` | 官方字段 | 原样保留 |
| `operator_id` | `record_split.csv` | 官方字段 | 原样保留 |
| `split` | `record_split.csv` | 官方字段 | 原样保留，不自行重新划分 |
| `sample` | `source_segment_index.csv`，并与原始目录核对 | 映射字段 | 按`episode_uuid`连接后取得 |
| `source_path` | `dataset + sample` | 派生字段 | 保存相对数据根目录的路径 |
| `n_frames` | 统一帧表 | 派生统计 | 每条记录的帧表行数 |
| `duration_s` | 统一帧表 | 派生统计 | 最大时间减最小时间 |
| `actual_fps` | 统一帧表 | 派生统计 | 时间差中位数的倒数 |
| `n_segments` | `source_segment_index.csv`或读取结果 | 派生统计 | 按`episode_uuid`计数 |
| `left_bad_ratio` | 质量标记与帧表对齐结果 | 派生统计 | 不从dataset名称推断 |
| `right_bad_ratio` | 质量标记与帧表对齐结果 | 派生统计 | 不从dataset名称推断 |
| 其他时间质量统计 | 统一帧表 | 派生统计 | 由同一套验收函数计算 |

结论：`records`不是原始数据里现成的一张表，而是以官方`record_split.csv`为骨架、补充`sample`映射和后续统计形成的统一索引。

## 3. `segments`字段分别从哪里来

`source_segment_index.csv`已直接提供：

```text
unit_id
dataset
sample
episode_uuid
environment_id
split
family_id
task
source_segment_index
start_s
end_s
duration_s
description
eligible
exclusion_reason
```

统一表只做以下规范化：

- `unit_id`重命名为`segment_id`；
- `eligible`由0/1规范化为true/false；
- 保留所有官方区间，不因`eligible=false`删除；
- 后续增加`n_frames、good_ratio、bad_ratio、mapping_status`；
- 区间使用半开区间`[start_s, end_s)`映射帧。

## 4. 第三问官方条件

### 场景规模

| 场景 | 任务数 | 时间上限/s | 成本预算 | 付费查看上限 |
|---|---:|---:|---:|---:|
| S1 | 8 | 100 | 32 | 4 |
| S2 | 12 | 100 | 34 | 6 |
| S3 | 16 | 125 | 40 | 8 |

### 观察规则

```text
可见前缀：0.2s、0.6s、1.0s
返回延迟：1.0s
单次查看成本：1.0
单个任务最多查看：2次
```

付费观察是“释放原记录中更多前缀”，不是产生新的摄像机视角。

### 服务、移动和费用

```text
服务时间 = alpha × source_annotation_duration_s + beta_s
默认 alpha = 1.0
默认 beta_s = 0.5
移动速度 = 1.0
切换时间 = 0.5s
固定服务费 = 2.0
预留时间费 = 0.1 / s
移动费 = 0.2 / 距离单位
```

服务区坐标是合成调度坐标，不能解释为商品或手部空间位置。

## 5. 已完成核对

```text
总记录：69
EgoDemo / EgoPro / EgoStandard：31 / 14 / 24
train / validation / test：55 / 7 / 7
环境数：16 / 4 / 4
官方区间：349
eligible区间：343
S1 / S2 / S3任务：8 / 12 / 16
原始目录映射失败：0
错误：0
警告：0
```

## 6. 当前冻结的基准文件

```text
work/official_index_audit/official_records.csv
work/official_index_audit/official_segments.csv
work/official_index_audit/field_dictionary.csv
work/official_index_audit/summary.json
config/delivery_schema.json
```

另外两位同学后续读取轨迹时，以这些文件为准，不能自行重建数据划分。

## 7. 接下来同学C要做的实际工作

1. 从三套数据各抽一条记录，检查读取结果能否与官方`episode_uuid/sample`对应；
2. 接收另外两人的单条样例交付；
3. 运行`validate_delivery.py`，反馈具体到记录和字段的问题；
4. 三套数据全部通过后运行`merge_deliveries.py`；
5. 对合并后的69条记录再次执行总量、外键、划分和区间边界检查。

