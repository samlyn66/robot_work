# 肌腱吻合分层模仿 VAM 项目（参照 SRT-H）

参照论文 *SRT-H: A Hierarchical Framework for Autonomous Surgery via
Language-Conditioned Imitation Learning*（Kim et al., arXiv:2505.10251）与
《面向肌腱吻合的分层模仿视频-动作模型》技术方案，实现面向肌腱吻合的
分层模仿学习系统。

## 代码现状说明（重要）

- **官方 SRT-H 没有公开训练代码与预训练权重**，官方只放出论文图表分析数据
  （Zenodo: 15637074）。
- 可参考的开源实现：`lucidrains/SRT-H`（架构级社区实现，含 flow matching 组件，
  WIP、无权重）。本项目自包含实现，不依赖该仓库，但模块设计与其对齐。
- **本项目使用的预训练权重**（均为公开权重）：
  - `torchvision` 的 `EfficientNet-B3`（LL 策略视觉编码，SRT-H 同款）
  - `torchvision` 的 `Swin-T`（HL 规划器视觉编码，SRT-H 同款）
  - 可选：`facebook/vit-base` / VideoMAE / V-JEPA 视觉自监督权重（视频骨干，
    对应方案中的"手术视频模型先验"，见 `video_encoder.py` 中钩子）

## 与论文/方案的模块对应

| 本项目模块 | 对应 SRT-H / 技术方案内容 |
|---|---|
| `tendon_vam/hl_policy.py` | 高层策略 π_HL：Swin-T + Transformer Decoder，输出子任务指令、纠错标志、纠正指令；CE 损失 × L1 类距加权；中心裁剪 + 4 帧历史 |
| `tendon_vam/ll_policy.py` | 低层策略 π_LL：EfficientNet-B3 + FiLM 语言调制 + Transformer Decoder + 动作分块；hybrid-relative 20 维动作 |
| `tendon_vam/video_encoder.py` | 共享视觉编码器 + **局部去噪（Partial Denoising）潜在视觉计划** |
| `tendon_vam/flow_decoder.py` | **Flow Matching 动作解码器（IDM）** + 显式时空对齐算子（动作块时序 ↔ 视频潜在时序绑定） |
| `tendon_vam/dataset.py` | 18 组双腕部相机视频（left_endo / right_endo）→ 配对帧数据集；运动学 CSV 可选接入 |

子任务集合（技术方案定义的 6 阶段）：
`idle / 持针(grasp_needle) / 进针(insert) / 出针(exit) / 拉线(pull) / 打结(knot) / 剪线(cut)`

## 数据目录约定

```
data/
  recording_20260826_162504/
    left_endo/*.mp4          # 左腕部相机视频（原始）
    right_endo/*.mp4         # 右腕部相机视频
    kinematics.csv           # 可选：末端位姿/关节角/夹持（训练 LL 时必需）
    subtasks.json            # 子任务标注（label 工具生成，训练 HL 时必需）
```

标注文件格式（`subtasks.json`）：

```json
{
  "fps_original": 30,
  "events": [
    {"t_start": 12.4, "t_end": 30.1, "phase": "grasp_needle"},
    {"t_start": 30.1, "t_end": 46.0, "phase": "insert"}
  ]
}
```

## 快速开始

```bash
pip install -r requirements.txt

# 1. 抽帧（对 18 组视频，双相机按统一目标帧率抽帧并对齐）
python scripts/extract_frames.py --data_root data --fps 10

# 2. 子任务标注（半自动：分段打点，生成 subtasks.json）
python scripts/make_subtask_labels.py --recording data/recording_20260826_162504

# 3. 训练高层子任务规划器（纯视频即可训练）
python scripts/train_hl.py --config configs/default.yaml

# 4. 训练低层策略（LL: SRT-H 风格 ACT）—— 需要运动学数据
python scripts/train_ll.py --config configs/default.yaml

# 5. 训练完整 VAM（视频骨干 + 潜在视觉计划 + Flow Matching 动作解码器）
python scripts/train_vam.py --config configs/default.yaml

# 6. 评估 / 可视化（在留出视频上滚动预测子任务与动作块）
python scripts/evaluate.py --config configs/default.yaml --ckpt runs/hl/best.pt
```

## 数据量提示

18 条视频对模仿学习而言是小数据：HL 规划器（分类任务）可以先训起来并做
留出视频上的阶段识别评估；LL / VAM 动作解码建议在补充运动学数据、按
《专家演示数据采集规范》放量后启动。项目按"运动学数据缺失时仅训练 HL +
视频表征"自动降级。

## 项目结构

```
tendon_vam_project/
├── configs/default.yaml          # 全部超参
├── tendon_vam/
│   ├── config.py                 # 配置加载
│   ├── dataset.py                # 双相机配对数据集 + hybrid-relative 动作生成
│   ├── video_encoder.py          # 共享视觉编码器 + 局部去噪潜在视觉计划
│   ├── hl_policy.py              # 高层子任务规划器（SRT-H 式）
│   ├── ll_policy.py              # 低层 ACT 式策略（FiLM + 分块）
│   ├── flow_decoder.py           # Flow Matching 动作解码器 + 时空对齐
│   └── utils.py                  # 6D 旋转、插值、损失等
├── scripts/
│   ├── extract_frames.py         # 视频抽帧与双相机对齐
│   ├── make_subtask_labels.py    # 子任务标注工具（键盘打点）
│   ├── train_hl.py               # 训练高层规划器
│   ├── train_ll.py               # 训练低层策略
│   ├── train_vam.py              # 训练完整 VAM（flow matching）
│   └── evaluate.py               # 留出视频评估与动作可视化
└── requirements.txt
```

## 引用

```
@misc{kim2025srthhierarchicalframeworkautonomous,
  title  = {SRT-H: A Hierarchical Framework for Autonomous Surgery via
            Language-Conditioned Imitation Learning},
  author = {Ji Woong Kim and others},
  year   = {2025},
  eprint = {2505.10251},
  archivePrefix = {arXiv}
}
```
