# 2026 数维杯 C 题：数据集与预处理规范

> 文档版本：v0.2（竞赛高质量精简版）  
> 更新日期：2026-11-20  
> 适用范围：问题1状态与观测质量、问题2遮挡期间预测与重新观测、问题3时长预测与任务调度  
> 实施目标：在不牺牲数据可信度、官方协议合规性和实验可复现性的前提下，将预处理压缩为竞赛期间可完成的最小高质量闭环。

---

## 1. 总体原则

本方案不追求将所有可能的视频、几何和异常检测方法一次做完，而是优先保证以下四点：

1. **不破坏原始数据**：原始数值永远保留，不就地修改、插值或覆盖；
2. **不造成信息泄漏**：严格使用官方环境切分，回放时不使用未来或未披露信息；
3. **不误用低质量数据**：遮挡、近边缘、缺失和对齐失败均显式标记，不作为精确真值；
4. **能支撑三问闭环**：同一掌心状态表示延续到问题2，语义区间的前缀特征进入问题3。

预处理分为三级优先级：

| 级别 | 含义 | 执行策略 |
|---|---|---|
| **MUST** | 影响题面合规、数据正确性或三问实验 | 必须完成 |
| **CHECK** | 只在数据检查发现问题时才扩展 | 先检查，异常时触发 |
| **DEFER** | 成本高、对当前主线收益不确定 | 暂不实施，有余力再加 |

---

## 2. 数据位置与规模

### 2.1 数据根目录

```text
$DATA_ROOT/
└── Restocking_Problem_and_Data/
    ├── Raw_Data/
    │   ├── EgoDemo/
    │   ├── EgoPro/
    │   └── EgoStandard/
    └── Restocking_Tasks_and_Constraints/
        ├── Problem_3_Requirements.pdf
        ├── common_conditions.json
        ├── job_families.csv
        ├── record_split.csv
        ├── service_zones.csv
        ├── source_segment_index.csv
        └── tasks.csv
```

### 2.2 已核验规模

| 数据源 | 记录数 | 题面轨迹时间戳数 | 主格式 |
|---|---:|---:|---|
| EgoDemo | 31 | 16,414 | Parquet + JSON + MP4 |
| EgoPro | 14 | 17,642 | CSV + MP4 |
| EgoStandard | 24 | 32,014 | CSV + MP4 |
| **合计** | **69** | **66,070** | - |

66,070按题面解释为轨迹时间戳口径。当前展开数据中的质量行数与该数量一致，但实施时仍分别统计左手轨迹、右手轨迹、质量行和视频帧，不在对齐验证前将它们当作同一计数。

官方环境切分：

| 划分 | 记录数 | 环境数 | 用途 |
|---|---:|---:|---|
| train | 55 | 16 | 模型、标准化参数、质量统计量拟合 |
| validation | 7 | 4 | 模型选择、阈值和超参数确定 |
| test | 7 | 4 | 冻结方案后最终评价 |

语义区间：

| 项目 | 数量 |
|---|---:|
| 原始语义区间 | 349 |
| 时长至少1.2秒、进入公共采样池 | 343 |
| EgoDemo / EgoPro / EgoStandard区间 | 118 / 71 / 160 |
| train / validation / test区间 | 252 / 35 / 62 |

注意：`eligible=1` 只说明该区间时长符合问题3公共采样条件，不说明手部观测一定高质量。

### 2.3 官方文件的使用边界

| 文件 | 用途 | 是否可改 |
|---|---|---|
| `record_split.csv` | 环境级训练/验证/测试切分 | 否 |
| `source_segment_index.csv` | 349个源语义区间索引 | 否 |
| `tasks.csv` | S1/S2/S3的36个调度作业 | 否 |
| `common_conditions.json` | 预算、时域、查看条件和验证种子 | 否 |
| `Problem_3_Requirements.pdf` | 问题3回放和评价协议 | 否 |

---

## 3. 数据源字段差异

### 3.1 EgoDemo

主轨迹位于：

```text
EgoDemo/<sample>/data/chunk-000/file-000.parquet
```

已核验字段：

| 字段 | 含义 | 形状 |
|---|---|---|
| `timestamp` | 从记录开始计的秒数 | float |
| `frame_index` | 帧索引 | int |
| `observation.state.hand_left_world` | 左手21个3D关键点 | `[21,3]` |
| `observation.state.hand_right_world` | 右手21个3D关键点 | `[21,3]` |
| `observation.state.hand_left_world_rotation` | 左手21组四元数 | `[21,4]` |
| `observation.state.hand_right_world_rotation` | 右手21组四元数 | `[21,4]` |
| `observation.state.head_world` | 头部世界坐标 | `[1,3]` |
| `observation.state.head_world_rotation` | 头部旋转 | `[1,4]` |

元数据：

- `meta/info.json`：帧率、字段形状和关键点名称；
- `meta/episodes/...parquet`：`episode_uuid`、`environment_id`、`scene_id`、`operator_id`；
- `annotation.json`：语义区间的开始、结束和描述；
- `bad_frame_ratio.json`：左右手的质量原因。

### 3.2 EgoPro与EgoStandard

两套数据使用展开CSV：

| 文件 | 主要内容 |
|---|---|
| `session_metadata.csv` | episode、环境、场景、操作者和设备帧率 |
| `tables/pose_left_hand.csv` | 左手21点位置和四元数 |
| `tables/pose_right_hand.csv` | 右手21点位置和四元数 |
| `tables/pose_head.csv` | 头部位姿 |
| `tables/annotation_bad_frame_pose_hand.csv` | `is_bad`、手别和质量原因 |
| `tables/annotation_semantic_segments.csv` | 语义区间开始、结束和描述 |

手部字段展开为：

```text
transforms[i].pos.x/y/z
transforms[i].quat.w/x/y/z
i = 0,...,20
```

主时间字段：

```text
_offset_from_first_log_s
_log_time_ns
_publish_time_ns
header.ts
```

主建模时间使用 `_offset_from_first_log_s`，保留纳秒原始时间作为审计依据。

当前手头数据包已经是CSV/MP4展开版，因此主流程直接读取展开文件，不增加MCAP解码工作。若后续发现原MCAP或官方字段字典，只对每个数据源抽查一条记录，核对行数、时间跨度、21点数值和质量标记，不重复构建第二套完整读取链。

### 3.3 `sample` 不等于 `episode_uuid`

对EgoPro/EgoStandard，目录名可能与记录内部 `task_info.episode_uuid` 不同。统一规则：

- `sample`：保存原始目录名，用于定位文件；
- `episode_uuid`：作为去重、表关联和官方切分主键。

---

## 4. 精简后的数据分层

只保留四个主数据产品，不再为每个中间步骤建立大表。

### 4.1 `records.csv`：记录级审计表（MUST）

一条采集记录一行：

| 字段 | 来源 | 含义 |
|---|---|---|
| `episode_uuid` | 原始+官方 | 记录唯一主键 |
| `dataset` | 官方 | EgoDemo/EgoPro/EgoStandard |
| `sample` | 原始目录 | 文件定位标识 |
| `environment_id` | 原始+官方 | 环境标识 |
| `scene_id` | 原始+官方 | 场景标识 |
| `operator_id` | 原始+官方 | 操作者标识，主要用于说明数据构成 |
| `split` | 官方 | train/validation/test |
| `n_frames` | 派生 | 主手部时间轴数 |
| `duration_s` | 派生 | 最后时间戳减首时间戳 |
| `actual_fps` | 派生 | `1/median(diff(timestamp))` |
| `n_segments` | 派生 | 语义区间数 |
| `left_bad_ratio` | 派生 | 左手至少有一个官方质量原因的时间戳比例 |
| `right_bad_ratio` | 派生 | 右手至少有一个官方质量原因的时间戳比例 |
| `duplicate_timestamp_count` | 派生 | 重复时间戳数 |
| `nonmonotonic_count` | 派生 | 非单调时间戳数 |
| `large_gap_count` | 派生 | 显著大于常规采样间隔的断点数 |
| `duplicate_status` | 派生 | 根据 `episode_uuid` 判定 |
| `usable` | 派生 | 是否至少可用于一项建模任务 |
| `exclusion_reason` | 派生 | 排除或限制使用原因 |

说明：

- `records.csv` 是审计表，不是时序模型训练表；
- `episode_uuid` 重复检查必须完成；
- 内容指纹去重属于CHECK，只在UUID或元数据异常时触发。

### 4.2 `frames.parquet`：统一帧级状态表（MUST）

一个统一时间戳一行。原始21点不必在主表中展开，但原始文件保留不变。

```text
episode_uuid
dataset
split
timestamp_s
timestamp_raw
frame_index
source_locator

left_palm_raw[3]
right_palm_raw[3]
left_palm_relative[3]
right_palm_relative[3]
left_velocity[3]
right_velocity[3]

left_quality          # good / bad / missing
right_quality         # good / bad / missing
left_raw_quality_flag
right_raw_quality_flag
left_bad_reason       # occlusion / near_edge / other / empty
right_bad_reason
left_fit_ok
right_fit_ok
left_ref_ok
right_ref_ok

segment_id
continuity_block_id
```

可选附加字段，只在不显著增加工作量时保留：

```text
head_position_raw[3]
head_quaternion_raw[4]
alignment_error_ms
```

暂不进入主表：21组四元数、手指张开度、角速度、加速度、视频特征。

### 4.3 `segments.csv`：语义区间表（MUST）

每个语义区间一行：

| 字段 | 来源 | 用途 |
|---|---|---|
| `unit_id` | 官方 | 区间标识 |
| `episode_uuid` | 官方 | 关联原始记录 |
| `dataset` | 官方 | 数据源 |
| `sample` | 官方/目录 | 文件定位 |
| `split` | 官方 | 环境级切分 |
| `family_id` | 官方 | P/A/L采样家族 |
| `start_s` | 官方 | 区间开始 |
| `end_s` | 官方 | 区间结束 |
| `duration_s` | 官方/校验 | `end_s-start_s` |
| `eligible` | 官方 | 是否进入公共采样池 |
| `description` | 官方 | 离线分析，不进入问题3在线输入 |
| `n_frames` | 派生 | 映射到的时间戳数 |
| `good_ratio` | 派生 | 双手至少可用状态比例 |
| `bad_ratio` | 派生 | 质量问题比例 |

语义区间统一使用半开区间：

\[
[\mathrm{start\_s},\mathrm{end\_s}).
\]

### 4.4 三问样本索引（MUST）

不重复保存大量轨迹，只保存样本索引和回放条件：

```text
q1_windows.parquet
q2_masks.parquet
q3_prefix_index.parquet
```

所有样本可通过 `episode_uuid + timestamp/segment_id` 追溯回 `frames.parquet` 和 `segments.csv`。

---

## 5. 核心预处理流程

## 5.1 步骤A：三套数据统一读取（MUST）

分别实现：

```text
load_egodemo()
load_egopro()
load_egostandard()
```

三个读取器统一输出：

```text
episode metadata
left/right 21x3 positions
left/right 21x4 quaternions (readable but not necessarily used)
head pose (readable but optional for main model)
timestamps
quality flags
semantic intervals
```

读取时只转换字段名和数组形状，不对数值做插值或平滑。

## 5.2 步骤B：记录级审计（MUST）

对每条记录完成：

1. `episode_uuid` 与官方 `record_split.csv` 关联；
2. 检查UUID缺失、重复和元数据冲突；
3. 检查左右手所有帧是否为21点；
4. 统计NaN/Inf和缺失值；
5. 检查时间戳单调性、重复和大间隔；
6. 统计左右手官方质量标记；
7. 检查语义区间是否落在记录时间范围内；
8. 输出 `records.csv`。

不得因少量坏帧删除整条记录。优先保留记录，再按时间区间划分用途。

## 5.3 步骤C：时间对齐（MUST）

### EgoDemo

左手、右手和头部位姿已在同一Parquet行中，直接使用 `timestamp`。

### EgoPro/EgoStandard

1. 以手部较完整的通道作为主时间轴；
2. 另一只手和质量标记使用最近时间戳对齐；
3. 初始对齐容差为0.02秒；
4. 超过容差记为 `missing`，不强行匹配；
5. 语义区间按 \(start_s\le t<end_s\) 分配。

### 容差质量检查

0.02秒不是盲目固定值。程序必须输出对齐误差的中位数、95%分位数和失败率：

- 若大多数对齐误差远小于0.02秒，保留该值；
- 若大量正常数据对齐失败，先检查时间基准和漂移，不直接放大容差；
- 对齐容差在验证后冻结，测试集不调参。

## 5.4 步骤D：掌心点和基础状态（MUST）

21点索引：

| 索引 | 含义 |
|---:|---|
| 0 | Wrist |
| 5 | Index MCP |
| 9 | Middle MCP |
| 13 | Ring MCP |
| 17 | Little MCP |

掌心代表点定义为：

\[
p_t^{\mathrm{palm}}=
\frac{p_{0,t}+p_{5,t}+p_{9,t}+p_{13,t}+p_{17,t}}{5}.
\]

依据：这些点属于掌部稳定骨架，比21点整体平均更少受指尖弯曲影响。

为了避免不同记录世界坐标原点不同，主模型使用窗口内相对位移：

\[
\Delta p_t=p_t-p_{t_0},
\]

其中 \(t_0\) 是当前模型输入窗口的第一个时刻。

基础状态：

\[
x_t=[\Delta p_t,v_t].
\]

速度在同一连续区间中用实际时间间隔的因果后向差分计算：

\[
v_t=\frac{p_t-p_{t-1}}{t_t-t_{t-1}}.
\]

使用后向差分是为了保证问题2和问题3回放时只依赖已经发生的观测。中心差分可仅作为离线诊断视图，不进入在线模型输入。每个连续区间的首个速度留空。

暂不将加速度和掌心方向列为MUST，原因是它们对噪声和四元数约定更敏感。

## 5.5 步骤E：基础几何质量检查（MUST，精简版）

为避免官方未标记的数值崩溃进入模型，保留两项低成本检查。

### 1. 掌部关键点完整性

计算掌心所需的5个关键点是否：

- 数量正确；
- 数值有限；
- 未全部填充为相同值。

任一失败，该手该时刻记为 `missing`。

### 2. 掌部尺度突变

使用掌部边长，例如：

\[
l_{05,t}=\|p_{0,t}-p_{5,t}\|,
\quad
l_{09,t}=\|p_{0,t}-p_{9,t}\|,
\quad
l_{0,17,t}=\|p_{0,t}-p_{17,t}\|.
\]

在每条记录、每只手内用中位数和MAD标记明显异常。这里只标记，不自动修复。

不实施完整21点骨架边的复杂质量评分，除非上述精简检查发现大量异常。

## 5.6 步骤F：质量分类（MUST）

采用三值质量标签：

| 类别 | 判定 | 用途 |
|---|---|---|
| `good` | 所需掌部点完整，对齐成功，官方未标记坏帧，无明显掌部尺度异常 | 模型拟合、人工mask和误差评价 |
| `bad` | 数值存在，但官方标记遮挡/近边缘，或精简几何检查异常 | 观测质量建模、自然遮挡分析；不作精确真值 |
| `missing` | 所需点缺失、非有限或时间对齐失败 | 缺失统计，不进入位置误差评价 |

同一时刻可分别为左手和右手记录质量，不要因一只手遮挡就必然删除另一只手。

为避免“数值存在”、“可以拟合”和“可以作为误差参考”三种概念混淆，同时保留：

```text
raw_quality_flag     # 原始数据集的质量标记
quality              # 统一后的 good / bad / missing
fit_ok               # 是否可进入指定模型拟合
ref_ok               # 是否可作为人工mask的误差参考
```

默认 `ref_ok` 比 `fit_ok` 严格：一些可用于质量分类器拟合的 `bad` 样本，不能作为精确位置参考。

## 5.7 步骤G：连续区间（MUST）

在以下位置开启新的 `continuity_block_id`：

- episode边界；
- 时间戳大间隔；
- 所研究的手变为 `missing`；
- 人工mask边界。

速度、滑动窗口和预测样本均不得跨连续区间。

语义边界不改变物理时钟的连续性，因此不强制切断基础轨迹；但如某项实验需要阶段内样本，在生成Q1/Q3样本索引时限制窗口不跨 `segment_id`。在线模式不使用尚未合法披露的语义终点判定连续性。

## 5.8 步骤H：插值和平滑（默认不做）

主流程不对缺失或坏帧插值，原因：

- 数据量足够，可优先使用连续 `good` 区间；
- 插值会弱化问题2对缺失和遮挡的真实分析；
- 复杂插值需额外验证且容易引入未来信息。

只有在连续 `good` 区间的基线速度仍存在大量高频抖动时，才触发CHECK：

- 比较不平滑和轻量平滑后的基线预测误差；
- 回放时只能使用因果平滑，不使用未来帧；
- 平滑参数在验证集确定后冻结。

---

## 6. 三个问题的数据生成

## 6.1 问题1：状态与质量

从连续区间中生成历史窗口。模型预测目标与输入的相对坐标口径保持一致，不从相对坐标突然切换到世界绝对坐标。推荐直接预测从当前时刻开始的未来位移增量：

\[
X_{t-L+1:t}\rightarrow
\left\{p_{t+h}-p_t\right\}_{h=1}^{H}.
\]

恢复到当前记录坐标系时：

\[
\hat p_{t+h}=p_t+\widehat{(p_{t+h}-p_t)}.
\]

单步预测是 \(H=1\) 的特例。该定义与问题2的缺失期间多步轨迹预测直接衔接。

初始参数：

- \(L=15\)，约0.5秒历史；
- 单步预测作为基础；
- 增加5帧或更长预测作为多步实验。

输入：

```text
relative palm position
velocity
hand side
historical quality
```

`dataset` 默认不进入主模型。EgoDemo/EgoPro/EgoStandard的标识可能同时编码设备、采集协议、场景、操作者和质量差异，直接输入可能使模型记住数据源而不是学习运动规律。`dataset` 只用于：

- 分层统计和各数据源独立误差报告；
- domain shift和鲁棒性分析；
- 额外消融实验 `without dataset ID` 对比 `with dataset ID`。

如加入dataset ID仅提升总体指标、却降低跨环境泛化，主模型仍使用不含dataset ID的版本。

样本索引 `q1_windows.parquet` 至少包含：

```text
episode_uuid
hand
split
history_start_s
history_end_s
target_end_s
continuity_block_id
segment_id
```

基线至少包括Last Position和Constant Velocity。

## 6.2 问题2：人工mask与自然遮挡

人工mask必须从原本连续 `good` 的区间中生成，以便被遮挡部分仍有合格参考状态。

初始mask时长分层：

```text
0.2 s
0.5 s
1.0 s
2.0 s
```

1.5秒作为CHECK层级：先运行上述四个时长。如1.0秒仍满足容差、而2.0秒已超出容差，再增加1.5秒以缩小“最大可容忍缺失时间”的定位区间。如结论已在1.0秒之前或2.0秒之后明确，不增加该实验。

`q2_masks.parquet` 包含：

```text
mask_id
episode_uuid
hand
split
segment_id
mask_start_s
mask_end_s
mask_duration_s
observation_budget
request_cost
return_delay_s
random_seed
```

必须满足：

1. mask后的位置不进入预测器输入；
2. 从被mask位置派生的速度等特征同时隐藏；
3. 重新观测在返回延迟结束前不可使用；
4. 所有方法共用相同mask、预算、费用和延迟；
5. 自然 `bad` 区间用于遮挡发生、持续和恢复分析，不在无独立真值时计算真实位置误差。

## 6.3 问题3：前缀时长预测

每个eligible语义区间生成三个前缀层级：

\[
X_j^{0.2},\quad X_j^{0.6},\quad X_j^{1.0}.
\]

第一版使用低成本、可解释的前缀特征：

```text
prefix_level_s
left_observed_ratio
right_observed_ratio
both_observed_ratio
left_displacement
right_displacement
left_mean_speed
right_mean_speed
left_max_speed
right_max_speed
left_speed_std
right_speed_std
left_terminal_speed
right_terminal_speed
left_speed_trend
right_speed_trend
mean_two_hand_distance
bad_ratio
family_id
dataset
```

其中终端速度取前缀末端最近一个合法因果速度；趋势特征可用：

\[
\Delta v=v_{\mathrm{last}}-v_{\mathrm{first}}.
\]

如前缀内合法速度少于2个，趋势留空并增加缺失指示。该特征只使用已披露前缀，不需要计算高噪声的逐帧加速度。

目标是区间总时长 \(d_j\)，最终输出时长分位数或分布，而不只是均值。

### 在线回放信息隔离（MUST）

在线决策程序不直接读取 `segments.csv` 或全量原始轨迹，只通过受控函数读取：

```text
get_visible(job_id, trial_state)
```

可见前缀层级由回放器根据作业释放状态、已完成付费查看次数和返回时间决定，不由策略传入 `visible_prefix_s`，防止策略请求越权前缀。

允许返回：

- 当前已合法披露的数值前缀；
- 时间戳；
- 左右手位置和原始方向；
- 头部位姿；
- 原始手部质量标记。

禁止返回：

- `unit_id`、`episode_uuid`、`sample` 等可反查原始区间的标识；
- 语义描述；
- `end_s`、`duration_s`；
- 前缀之后的任何状态或汇总特征。

问题3数据分为公开与私有两个命名空间：

```text
q3/public/      # 策略可读：公开作业条件和当前已返回信息
q3/private/     # 仅回放器/评分器可读：源ID、区间边界、真实时长和样本分配
```

训练阶段可使用train区间的真实时长作为监督标签；在线调度策略只能通过 `get_visible` 获得当前合法信息。

---

## 7. 信息泄漏检查清单（MUST）

### 7.1 划分泄漏

- [ ] 所有样本先按官方 `split` 分开，再构造窗口；
- [ ] 同一 `environment_id` 未出现在多个split；
- [ ] 标准化参数只用train拟合；
- [ ] 掌心定义、对齐容差和异常阈值未使用test调整。

### 7.2 时序泄漏

- [ ] 回放时未使用当前时刻之后的帧；
- [ ] mask时同时隐藏相关派生特征；
- [ ] 返回延迟结束前无法访问新观测；
- [ ] 输入特征不包含真实区间进度或终点。

### 7.3 问题3源记录反查

- [ ] 策略输入不包含源ID和语义描述；
- [ ] 策略不能直接读取 `source_segment_index.csv`；
- [ ] 策略不能通过前缀匹配全量原始数据；
- [ ] 训练器和在线回放器使用不同的数据访问接口。

---

## 8. 被删减或延后的处理

### 8.1 DEFER：暂不实施

| 项目 | 暂不实施原因 | 何时重新考虑 |
|---|---|---|
| 全视频解码和视觉特征 | 数据量大，题目已提供3D状态，问题3又不使用视频 | 只用于最终失败案例展示 |
| 头部相对坐标主流程 | 需先验证三套数据的旋转方向和坐标轴 | 相对位移仍有明显跨记录偏差时 |
| 掌心方向、角速度 | 需增加四元数和几何稳定性处理 | 位置+速度模型效果不足时 |
| 加速度 | 二阶差分放大噪声 | 验证集证明其显著提升时 |
| 复杂Q0/Q1/Q2连续评分 | 规则和阈值过多 | 三值 `good/bad/missing` 无法支撑不确定性建模时 |
| 全骨架所有骨长异常 | 实现和解释成本高 | 精简掌部尺度检查发现大量异常时 |
| 四元数全流程修复 | 主模型暂不使用旋转 | 将方向加入主模型时 |
| 缺失插值 | 可能破坏问题2缺失结构 | 连续good样本严重不足时 |

### 8.2 CHECK：发现问题才触发

- 对齐失败率过高：检查时间基准和时钟漂移；
- 窗口内速度抖动严重：检查轻量因果平滑；
- 掌部尺度异常过多：扩展到完整骨架检查；
- 相对位移跨记录差异过大：验证头部相对坐标；
- 基础时长特征效果差：再考虑小型GRU/TCN编码前缀。

---

## 9. 输出目录

```text
data_processed/
├── records.csv
├── frames.parquet
├── segments.csv
├── field_dictionary.csv
├── q1_windows.parquet
├── q2_masks.parquet
├── q3/
│   ├── public/
│   ├── train/
│   └── private/
├── audit/
│   ├── timestamp_summary.csv
│   ├── alignment_summary.csv
│   ├── quality_summary.csv
│   └── segment_mapping_summary.csv
└── preprocessing_run.json
```

`field_dictionary.csv` 至少包含：

```text
field_name
table_name
source_type        # raw / official / derived / experiment
source_fields
definition
unit
version
```

---

## 10. 可复现记录

`preprocessing_run.json` 至少记录：

```json
{
  "run_id": "...",
  "created_at": "...",
  "source_root": "...",
  "code_version": "...",
  "config_hash": "...",
  "record_split_hash": "...",
  "source_segment_index_hash": "...",
  "input_record_count": 69,
  "input_timestamp_count": 66070,
  "output_frame_count": null,
  "good_left_count": null,
  "good_right_count": null,
  "bad_left_count": null,
  "bad_right_count": null,
  "missing_left_count": null,
  "missing_right_count": null,
  "segment_count": 349,
  "eligible_segment_count": 343,
  "warnings": []
}
```

论文中的数据保留率、质量比例和样本数必须能从以上输出重新计算。

---

## 11. 实施顺序与时间预算

| 阶段 | 产出 | 预计时间 |
|---|---|---:|
| 1. 三套读取器 | 统一记录对象 | 1.5-2小时 |
| 2. 记录审计 | `records.csv` | 0.5-1小时 |
| 3. 时间对齐与掌心状态 | `frames.parquet` | 2-3小时 |
| 4. 质量标签与区间映射 | `segments.csv` | 1-2小时 |
| 5. 三问索引 | Q1/Q2/Q3样本索引 | 1.5-2小时 |
| 6. 核对与报告 | 审计表和运行记录 | 1-2小时 |
| **合计** | 可建模的高质量预处理闭环 | **8-12小时** |

该时间不包括视频全量处理和DEFER项。如CHECK项发现坐标或时间异常，再单独增加处理时间。不以5.5-8.5小时作为团队进度承诺，避免低估首次读取异构数据的调试成本。

### 进度止损节点

#### 开始后8小时

至少必须完成：

- `records.csv`；
- `frames.parquet`；
- `segments.csv`；
- EgoDemo、EgoPro、EgoStandard各至少成功打印并绘制一条掌心轨迹；
- 记录数、时间戳数和语义区间数的首轮核对。

如尚未完成，停止所有CHECK和DEFER特征，只修复阻断主表产出的问题。

#### 开始后12小时

必须完成：

- Q1连续窗口生成；
- Last Position基线；
- Constant Velocity基线；
- 至少一张按预测时域分层的验证误差表。

如尚未完成，不增加头部坐标、方向、加速度、视频或复杂质量特征，直到基线闭环跑通。

---

## 12. 完成标准

预处理在满足以下条件时视为完成：

1. 69条记录都出现在 `records.csv` 中，或有明确排除原因；
2. 记录数与官方切分的55/7/7一致；
3. 同一环境未跨split；
4. 每条记录的时间单调性、重复时间戳和大间隔都已统计；
5. 左右手掌心可追溯回5个指定关键点；
6. 每个帧的 `good/bad/missing` 来源明确；
7. 349个语义区间已映射到原始时间轴，343个eligible区间可生成前缀；
8. Q1窗口和Q2 mask不跨episode、split或连续区间；
9. Q2 mask后的真值及派生特征无法进入模型输入；
10. Q3在线策略无法访问真实区间ID、语义描述、结束时间和总时长；
11. 所有阈值只在train/validation上决定，test只用于冻结后评价；
12. 论文中的数据数量和保留率可从输出表和 `preprocessing_run.json` 复现。

### 回放边界测试

在验收时必须额外通过两类小型对抗测试：

1. **问题2延迟测试**：人为修改一个已请求但尚未返回的观测值。返回时刻之前，策略的所有可见输入必须完全不变；只有延迟结束后才允许出现差异。
2. **问题3前缀测试**：人为修改源区间第0.85秒的数值。0.2秒和0.6秒可见层的输出必须保持不变，1.0秒层只能在第二次付费查看返回后反映该修改。普通等待不得自动扩展前缀。

---

## 13. 当前冻结决策

| 决策 | 当前选择 | 状态 |
|---|---|---|
| 记录主键 | `episode_uuid` | 冻结 |
| 数据切分 | 官方环境级split | 冻结 |
| 掌心点 | Wrist + 4个MCP的几何中心 | 待用简单稳定性统计确认 |
| 主状态 | 窗口内相对位移 + 速度 | 冻结为第一版 |
| Q1预测目标 | 从当前时刻到未来时刻的位移增量 | 冻结 |
| dataset ID | 不进入默认主模型，只用于分层报告和消融 | 冻结 |
| 时间对齐 | 最近时间戳，初始容差0.02秒 | 待对齐统计确认 |
| 质量标签 | `good/bad/missing` | 冻结 |
| 插值 | 主流程不插值 | 冻结 |
| 视频特征 | 不进入主流程 | 冻结 |
| 头部相对坐标 | CHECK项 | 延后 |
| 方向、角速度、加速度 | 不进入第一版 | 延后 |
| Q2的1.5秒mask | 当1.0秒合格而2.0秒不合格时触发 | CHECK项 |
| Q3趋势特征 | 终端速度 + 首末速度差 | 冻结为第一版 |

本文档后续只在两种情况下扩展预处理：

1. 已实施的质量检查明确发现数据问题；
2. 基线和主模型实验证明某个延后特征具有清晰收益。

不因“可能有用”而在建模前无限增加预处理步骤。
