# MomAD + 潜世界模型（nuScenes 开环 6s）

## 需求对应关系

- **Long Horizon Planning**：保持 nuScenes 的 0.5s 采样间隔，固定预测 12 步，即 6s。
- **潜世界模型**：把 ego、检测目标、地图特征与 MomAD 的多模态 agent 预测编码为未来感知 token，再用 GRU 在 latent space 中 rollout 12 步。
- **感知与规划联合预测**：输入包含每个 agent 的 6 模态未来轨迹；输出包含 3 个驾驶指令 × 6 个规划模态，共 18 条 ego 轨迹。
- **可学习监督**：新增 world ego trajectory、future scene-flow、mode diversity 三类 loss。latent 不做 `detach`，能够真正反向传播。
- **保护 6s baseline**：默认从 `work_dirs/MomAD_small_stage2_roboAD_6s/iter_11720.pth` 微调；world-query 融合系数初始化为 0，最终 world trajectory 仅以 0.1 倍小残差修正 baseline。
- **多卡/随机采样安全**：删除旧实验中 head 内部跨 batch 的全局状态 cache，只使用 MomAD 已有、带场景 mask 的 `InstanceQueue`。

## 老师建议的两处是否正确

方向正确，但只改这两处无法实现世界模型：

1. `nuscenes_3d_dataset_roboAD_6s.py` 原 405 行附近负责构造规划碰撞评测所需的未来 box。新文件补充了数组越界保护，并接入独立的 6s planning evaluator。
2. `nuscenes_converter_6s.py` 中真正生成 agent/ego 未来轨迹的是原 344--409 行附近，而原 449 行附近主要是 CAN bus ego status。新 converter 保留 12 步标签，并额外生成 `(x, y, vx, vy, speed, heading)` 的未来世界状态。
3. 真正的潜世界模型还必须改 motion/planning head、loss、最终 decoder 选择、配置注册和评测，所以这些都创建了独立文件。

## 新增文件（原文件均未修改）

1. `projects/configs/MomAD_small_stage2_MomAD_World_model_6s.py`
2. `projects/mmdet3d_plugin/models/motion/latent_world_model_MomAD_World_model_6s.py`
3. `projects/mmdet3d_plugin/models/motion/motion_planning_head_MomAD_World_model_6s.py`
4. `projects/mmdet3d_plugin/datasets/nuscenes_3d_dataset_MomAD_World_model_6s.py`
5. `projects/mmdet3d_plugin/datasets/evaluation/planning/planning_eval_MomAD_World_model_6s.py`
6. `tools/data_converter/nuscenes_converter_MomAD_World_model_6s.py`
7. `scripts/train_MomAD_World_model_6s.sh`
8. `projects/mmdet3d_plugin/datasets/pipelines/world_model_pipeline_MomAD_World_model_6s.py`
9. `MomAD_World_model_6s_CHANGELOG.md`

新 dataset/head 通过配置中的 `custom_imports` 注册，因此没有改动原有 `datasets/__init__.py` 或 `models/motion/__init__.py`。

## 原文件备份校验值（SHA-256）

```text
f84c895b4c762bb3e089c1d86c1e9b81479ae0b6e7b66040f973ecc4bf949a19  projects/mmdet3d_plugin/datasets/nuscenes_3d_dataset_roboAD_6s.py
646908e3c59af5e4a07dc7b1d07997de0f27fb3d459fd1269b0233e47d78d409  tools/data_converter/nuscenes_converter_6s.py
f1b25ef841d9cca9cfe3400ec1a70f1d0a4100d79b93ff08e7324d807c80b8bf  projects/mmdet3d_plugin/datasets/evaluation/planning/planning_eval_roboAD_6s.py
cb8bc538531b6a015a2f63e7ed8dc7e9b3bcbdb75844f49f2c1236443dbd72e7  projects/mmdet3d_plugin/models/motion/motion_planning_head_roboAD_6s.py
8b489d518b31fad1a1197c8fb0290559bbba558583769309dcec981be7bee055  projects/configs/MomAD_small_stage2_roboAD_6s.py
a8073fbabff227aeae1146ee1f435f1046940f7819bf943dbe395be61149d166  scripts/train_6s.sh
```

## 数据生成

在 `MomAD/open_loop` 下执行（若 CAN bus 路径不同，只修改 `--canbus`）：

```bash
python tools/data_converter/nuscenes_converter_MomAD_World_model_6s.py \
  nuscenes \
  --root-path /data/share/nuscenes \
  --canbus /data/share/nuscenes \
  --version v1.0 \
  --out-dir data/infos \
  --extra-tag nuscenes
```

会生成：

```text
data/infos/nuscenes_infos_train_MomAD_World_model_6s.pkl
data/infos/nuscenes_infos_val_MomAD_World_model_6s.pkl
data/infos/nuscenes_infos_test_MomAD_World_model_6s.pkl
```

## 训练与测试

```bash
bash scripts/train_MomAD_World_model_6s.sh
```

```bash
bash tools/dist_test.sh \
  projects/configs/MomAD_small_stage2_MomAD_World_model_6s.py \
  work_dirs/MomAD_small_stage2_MomAD_World_model_6s/latest.pth \
  1 \
  --deterministic \
  --eval bbox
```

训练日志应出现以下新增项，否则说明 world model 没有进入 loss：

```text
world_model_loss_cls
world_model_loss_reg
world_model_loss_scene_flow
world_model_loss_diversity
world_model_loss_latent
world_model_loss_state_reconstruction
```

评测保留原 6s baseline 的 `L2`（1s/2s/3s 均值）以便公平比较，同时新增 `L2_ADE_4s_5s_6s` 与作为真实 6s 单点误差的 `L2_at_6s`，用于验证 Long Horizon Planning 是否真正改善。

## 2026-06-20 第二版关键修正

第一版完成静态审查后发现的关键问题与处理如下。

### 1. converter 世界状态未进入训练

问题：第一版虽然生成了未来 `(x, y, vx, vy, speed, heading)`，但 pipeline 没有收集，loss 也没有使用。

修正：新增独立 world-state pipeline，对 agent 世界状态执行与检测框一致的类别过滤、距离过滤和旋转增强，再转换为 DataContainer。训练配置已经收集四个新字段：

```text
gt_world_model_agent_states
gt_world_model_agent_masks
gt_world_model_ego_states
gt_world_model_ego_masks
```

### 2. agent 六模态被平均为一条轨迹

问题：左右两个高概率未来直接坐标平均，可能得到并不存在的中间轨迹。

修正：每个 agent 的六条轨迹先分别编码成 latent token，保留 `agent × motion mode × future step` 结构；每个 ego planning query 再通过带 motion probability prior 的 attention 查询这些 token。

### 3. rollout 没有 ego action 条件

问题：18 个规划模态共享相同的 future-scene 输入，无法表达不同 ego 行为对应的条件未来。

修正：每个 rollout step 都输入对应候选规划模态的 `delta action + cumulative position`，形成 action-conditioned latent transition。

### 4. 缺少真实未来 latent 监督

问题：第一版 latent 仅通过 trajectory/scene-flow 间接训练，容易被质疑只是 GRU planner。

修正：真实未来 agent/ego 世界状态经过 teacher state encoder 得到 future latent target。GT 指令下与真实轨迹最接近的预测模态执行 stop-gradient latent consistency，同时 teacher/predicted latent 共同解码重建未来 ego 和 agent 状态，防止 target encoder collapse。

### 5. 训练分支和最终 decoder 不一致

问题：refined trajectory 参与 loss，但测试使用另一条 world trajectory。

修正：world trajectory 与 action-conditioned refined trajectory 通过可学习 sigmoid gate 融合为唯一 `final_prediction`。该结果同时接受 final planning loss，并由 planning decoder 直接使用。

### 6. 6s 指标语义和 Consistency 不准确

问题：旧 `_at_6s` 实际是 0.5--6s 的前缀平均，不是 6s 单点误差；上一帧规划也没有变换到当前 ego 坐标系。

修正：

- `L2_at_6s` 现在是第 12 步真实 FDE；
- `L2_ADE_4s_5s_6s` 报告长时域累计 ADE；
- Consistency 先执行上一帧 lidar → global → 当前 lidar 变换，再向前平移一个 0.5s step；
- 场景首帧不计入 Consistency。

### 7. 训练可观测性

checkpoint/evaluation 周期从只在第 20 epoch 执行一次，改为每 2 epoch 执行，便于及时发现 world loss 发散或 baseline 性能退化。

### 老师要求的实现链路

当前第二版对应关系为：

```text
MomAD 图像感知/检测/地图特征
        +
周围 agent 六模态 6s motion trajectory
        ↓
mode-aware future perception latent tokens
        +
18 个候选 ego planning actions
        ↓
action-conditioned 潜世界模型 rollout（12 × 0.5s）
        ↓
真实未来 world-state latent 对齐与状态重建
        ↓
3 commands × 6 modes 的最终 ego planning trajectory
```

因此已经实现“MomAD + 潜世界模型，通过感知预测和规划预测未来多模态轨迹来完成 nuScenes 6s Long Horizon Planning”的代码链路。最终是否提升仍必须通过与原 6s checkpoint 相同数据、batch size 和评测协议的训练实验确认。
