# 同学C：官方索引、统一规则与合并验收操作

> 日期：2026-11-20  
> 状态：官方索引首轮审计已通过  
> 适用目录：`Restocking_Problem_and_Data`

## 1. 这部分工作的边界

同学C暂时不读取和展开大规模手部轨迹，先把三个人共同依赖的“标准答案”固定下来。

本部分只做四件事：

1. 核对官方索引与原始记录目录；
2. 生成三个人共同使用的记录表和区间表；
3. 冻结统一字段名、类型、单位和枚举；
4. 在另外两人交付后执行合并前验收。

这一步不做插值、平滑、速度计算、人工mask和模型窗口生成。

## 2. 官方文件及用途

| 文件 | 用途 | 是否允许人工修改 |
|---|---|---:|
| `record_split.csv` | 记录主键、数据源、环境和训练/验证/测试划分 | 否 |
| `source_segment_index.csv` | 官方操作区间、任务族、起止时间和可用性 | 否 |
| `tasks.csv` | 第三问S1/S2/S3任务定义 | 否 |
| `common_conditions.json` | 第三问公共参数和验证随机种子 | 否 |
| `job_families.csv` | P/A/L任务族定义 | 否 |
| `service_zones.csv` | Z0—Z4服务区定义 | 否 |

原则：官方文件只读。任何标准化和补充字段写入派生文件，不覆盖原件。

## 3. 已完成的首轮审计结果

| 检查项 | 结果 |
|---|---:|
| 总记录数 | 69 |
| EgoDemo / EgoPro / EgoStandard | 31 / 14 / 24 |
| train / validation / test记录数 | 55 / 7 / 7 |
| train / validation / test环境数 | 16 / 4 / 4 |
| 环境总数 | 24 |
| 官方区间数 | 349 |
| eligible区间数 | 343 |
| S1 / S2 / S3任务数 | 8 / 12 / 16 |
| 原始目录与官方记录映射 | 全部匹配 |
| 错误 / 警告 | 0 / 0 |

这说明三个人可以直接以本次生成的 `official_records.csv` 和 `official_segments.csv` 为共同基准，不需要各自重新猜测目录和划分规则。

## 4. 实际操作顺序

### 第一步：运行官方索引审计

在项目目录执行：

```bash
python3 src/audit_official_indices.py \
  --data-root "$DATA_ROOT/Restocking_Problem_and_Data" \
  --output work/official_index_audit
```

终端最后显示 `"status": "pass"` 才能向下进行。若为 `fail`，先查看 `audit.csv`，不要绕过错误继续合并。

### 第二步：把共同基准发给另外两人

必须发出的文件：

```text
work/official_index_audit/official_records.csv
work/official_index_audit/official_segments.csv
work/official_index_audit/field_dictionary.csv
docs/预处理接口与交付规范.md
```

另外两人不能自己生成或修改 `episode_uuid`、`split`、`environment_id` 和官方 `segment_id`。

每个人只在 `official_records.csv` 中筛选自己负责的数据源或样本，然后逐条读取。

### 第三步：要求另外两人先交一条样例

在批量运行前，每人先交付一条完整记录，至少包含：

```text
records.csv
frames.parquet
segments.csv
audit.csv
run_info.json
```

同学C检查：

- 主键、列名和类型是否一致；
- 时间是否统一为记录内相对秒；
- 坐标单位是否为米；
- 缺失值是否使用null，而不是0或-1；
- `dataset`、`split`、`environment_id`是否与官方表相同；
- `segment_id`和区间边界是否可回溯到官方表；
- 是否保留原始定位字段，出现问题时能回到原文件。

样例通过后再批量处理，避免三套数据全部跑完才发现接口不一致。

### 第四步：接收批量交付并验收

每份交付先单独验收，不要直接拼接。验收顺序为：

1. 文件是否齐全；
2. 字段是否齐全且无多义列名；
3. 主键是否唯一；
4. 官方字段是否一致；
5. 时间是否单调，重复和大间隔是否已记录；
6. 区间是否满足 `start_s < end_s`；
7. 比例是否处于0到1；
8. `fit_ok`、`ref_ok`和质量三值是否符合定义；
9. `audit.csv`中的错误是否有明确处理结果。

发现问题时只退回对应记录，不要求整套重新运行。问题记录写入 `audit.csv`，修复后重新提交该记录。

对每位同学的交付执行：

```bash
python3 src/validate_delivery.py \
  --delivery work/deliveries/同学A \
  --schema config/delivery_schema.json \
  --official-records work/official_index_audit/official_records.csv \
  --output work/validation/同学A.json
```

同学B和同学C的目录同理替换。输出必须为 `"status": "pass"`。开发阶段允许用 `frames.csv`联调，但会产生warning；最终交付必须换成 `frames.parquet`。

### 第五步：合并

验收通过后按下列键合并：

| 表 | 唯一键 |
|---|---|
| `records` | `episode_uuid` |
| `frames` | `frame_id`，并复查 `(episode_uuid, timestamp_s)` |
| `segments` | `segment_id` |

合并采用纵向追加，不通过列位置拼接；列顺序不同不影响，但字段名和含义必须一致。

合并后再次检查：

- `records`必须正好69行；
- 三个数据源必须为31/14/24；
- 划分必须为55/7/7；
- 环境必须为16/4/4；
- 每个帧和区间的 `episode_uuid`必须能在 `records`中找到；
- 同一环境不得跨越不同split；
- 官方区间映射不得出现无说明的丢失或重复。

三份交付全部通过后执行：

```bash
python3 src/merge_deliveries.py \
  --deliveries work/deliveries/同学A work/deliveries/同学B work/deliveries/同学C \
  --output work/merged
```

合并程序会拒绝字段集合不一致、主键重复或三份交付使用不同帧文件格式的情况，并生成 `merge_manifest.json`记录输入目录和合并行数。

## 5. 统一字段冻结规则

开工后必须冻结的核心字段：

```text
episode_uuid
dataset
sample
environment_id
scene_id
operator_id
split
timestamp_s
left_palm_x
left_palm_y
left_palm_z
right_palm_x
right_palm_y
right_palm_z
left_raw_quality_flag
right_raw_quality_flag
left_bad_reasons
right_bad_reasons
left_quality
right_quality
left_fit_ok
right_fit_ok
left_ref_ok
right_ref_ok
segment_id
continuity_block_id
```

注意：原先口头写法 `left_palm_raw`、`right_palm_raw` 不作为最终帧表列名。最终表拆为x/y/z三列，便于类型校验和模型读取；如读取器内部暂时使用三元数组，只能作为中间结构。

原先单数写法 `left_bad_reason`、`right_bad_reason` 统一改为复数：

```text
left_bad_reasons
right_bad_reasons
```

因为同一时刻可能同时存在 `occlusion` 和 `near_edge`，字段类型为列表而不是单一字符串。

## 6. 同学C的交付物

同学C第一阶段应交付：

```text
src/audit_official_indices.py
src/validate_delivery.py
src/merge_deliveries.py
config/delivery_schema.json
work/official_index_audit/official_records.csv
work/official_index_audit/official_segments.csv
work/official_index_audit/field_dictionary.csv
work/official_index_audit/audit.csv
work/official_index_audit/summary.json
docs/同学C官方索引与合并验收操作.md
```

其中：

- `official_records.csv`是记录级共同入口；
- `official_segments.csv`是区间映射共同入口；
- `field_dictionary.csv`是机器可读字段字典；
- `summary.json`保存计数、文件哈希和审计结论；
- `audit.csv`保存审计问题，本次为空表表示没有发现问题。

## 7. 时间安排

| 工作 | 建议耗时 |
|---|---:|
| 官方索引审计与基准表生成 | 已完成 |
| 两位同学各一条样例联调 | 约30—60分钟 |
| 批量交付逐份验收 | 约1—2小时，视问题数量而定 |
| 三套数据合并及总量复核 | 约30分钟 |

同学C完成官方基准后不需要等待：可以马上开始做一条跨数据源的读取抽查，或者准备最终合并目录。真正耗时的是另外两人的轨迹读取，同学C的任务重点是尽早拦截接口错误。

## 8. 止损规则

- 一条样例未通过时，不允许直接批量运行；
- 官方字段冲突时，以官方索引为准，不凭目录名猜测；
- 单条记录有问题时先隔离该记录，其余记录继续；
- 不为追求“零缺失”进行无依据插值；
- 开赛后接口一旦冻结，任何字段变更必须同时更新字段字典、验收规则和版本号。

