# DEPLOY.md — Windows 服务器部署指南（VS Code + PowerShell）

> 本指南对应 **Windows 服务器** 环境。示例使用你的实际路径与环境：
> - 项目目录：`D:\ancddx\vm\tendor_t3\tendor_project`
> - conda 环境：`tendor311`（Python 3.11）
> - 服务器用户目录：`C:\Users\_exception`（以你的截图为准）
> - 所有命令在 **PowerShell** 中执行（VS Code 终端默认就是）

---

## 第 0 步：本地 VS Code 连接 Windows 服务器

1. 本地 VS Code 安装扩展 **Remote - SSH**
2. `Ctrl+Shift+P` → **Remote-SSH: Connect to Host** → 输入
   `ssh 用户名@服务器IP -p 端口`（若服务器 SSH 是默认 22 端口可省略 `-p`）
3. 首次连接选择平台时选 **Windows**，输密码登录
4. **Open Folder** → 打开 `D:\ancddx\vm\tendor_t3\tendor_project`
5. `` Ctrl+` `` 打开集成终端，确认提示符是 `PS D:\...>`（PowerShell）

> 你的截图显示已经连上了（conda 环境 `tendor311` 已激活），说明第 0 步已完成，可直接从第 1 步开始。

---

## 第 1 步：拉取代码

在服务器 PowerShell 中：

```powershell
cd D:\ancddx\vm\tendor_t3
git clone https://github.com/samlyn66/robot_work.git tendor_project
# 如果 clone 超时（国内服务器常见），改用镜像：
git clone https://ghproxy.net/https://github.com/samlyn66/robot_work.git tendor_project

cd tendor_project
git log --oneline -1   # 应能看到最新 commit
```

> 若之后手动更新代码：`git pull` 即可。

---

## 第 2 步：从本地电脑传数据

数据（`dataset` 文件夹）在**本地电脑**桌面，通过 scp 上传到服务器。
在**本地** PowerShell / Git Bash 中执行（不是在服务器上）：

```powershell
scp -P 端口 -r "C:\Users\25965\Desktop\dataset" 用户名@服务器IP:"D:/ancddx/vm/tendor_t3/tendor_project/data/"
```

- Windows 自带 OpenSSH 客户端，scp 直接可用
- 注意目标路径用**正斜杠 `/`** 并加引号（含空格路径必须加引号）
- 传完后在服务器上确认目录结构（recording 文件夹必须直接在 `data\` 下）：

```powershell
dir D:\ancddx\vm\tendor_t3\tendor_project\data
# 应看到 data\recording_20260826_172240\{ads_data.csv, left_endo_*.mp4, right_endo_*.mp4}
```

> 数据量大、scp 慢时，也可以用 WinSCP / FileZilla 图形化拖拽，效果相同。

---

## 第 3 步：配置 Python 环境（conda）

```powershell
conda activate tendor311          # 你已有的环境；若要新建见下方注释
# 新建环境（可选）：conda create -n tendor311 python=3.11 -y

# 确认 GPU 驱动与 CUDA 上限
nvidia-smi

# 安装 GPU 版 PyTorch（按 nvidia-smi 显示的 CUDA 版本选 index-url）
# CUDA >= 12.1:
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
# CUDA 11.8:
# pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
# 没有独立 GPU（仅 CPU）:
# pip install torch torchvision

# 安装其余依赖
cd D:\ancddx\vm\tendor_t3\tendor_project
pip install -r requirements.txt

# 验证 GPU 可用（必须输出 True）
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

> - 国内服务器 pip 慢可加清华镜像：`pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple`
> - **不要**用 `pip install opencv-python-headless` 替代——Windows 服务器有桌面环境，标准 `opencv-python` 没有 Linux 上的 libGL 问题

---

## 第 4 步：跑通数据流水线（CPU 即可，几分钟内完成）

```powershell
cd D:\ancddx\vm\tendor_t3\tendor_project

# 4.1 数据体检：自动分析无表头 CSV 的列结构，核对列映射
python scripts/inspect_ads.py --rec data\recording_20260826_172240

# 4.2 运动学转换：ads_data.csv -> kinematics.csv（含时间对齐、四元数修复）
python scripts/convert_ads.py --config configs\default.yaml --data_root data

# 4.3 双腕相机抽帧（10fps，跑完整数据请去掉 --max_frames）
python scripts/extract_frames.py --data_root data --fps 10
#   小规模试跑：--fps 10 --max_frames 200
#   中断后重跑会自动跳过已抽帧（断点续抽）
```

完成后每个 recording 下应出现 `left_endo_frames\`、`right_endo_frames\`、`kinematics.csv`。

---

## 第 5 步：标注子任务（在服务器上就能做，Windows 有 GUI）

```powershell
python scripts/make_subtask_labels.py --recording data\recording_20260826_172240
```

会弹出窗口播放视频，按键在 6 个阶段边界打点（持针/进针/出针/拉线/打结/剪线），生成 `subtasks.json`。逐个 recording 标注。

---

## 第 6 步：训练

**Windows 没有 tmux**，防断线做法二选一：

- **方案 A（推荐）**：训练时保持 VS Code 窗口开着；万一远程连接断了，Windows 上的 python 进程通常仍在跑（不是 SSH 子进程时），重连后看日志文件即可
- **方案 B**：用独立窗口 + 日志重定向，即使 VS Code 关掉也继续运行：

```powershell
Start-Process powershell -ArgumentList '-NoExit','-Command',
  "conda activate tendor311; cd D:\ancddx\vm\tendor_t3\tendor_project; python scripts\train_hl.py --config configs\default.yaml *>&1 | Tee-Object runs\train_hl.log"
```

训练命令：

```powershell
# 6.1 先训高层阶段规划器（只需视频帧 + subtasks.json）
python scripts\train_hl.py --config configs\default.yaml

# 6.2 低层策略（LL: SRT-H 风格 ACT）—— 需要 kinematics.csv
python scripts\train_ll.py --config configs\default.yaml

# 6.3 完整 VAM（视频骨干 + 潜在视觉计划 + Flow Matching 动作解码器）
python scripts\train_vam.py --config configs\default.yaml
```

产物保存在 `runs\`（模型权重 `*.pt`、日志、曲线图）。

**显存不足（CUDA out of memory）时**，编辑 `configs\default.yaml`：
`train.batch_size: 8` → 调到 `4` 或 `2`。

---

## 第 7 步：评估

```powershell
python scripts\evaluate.py --config configs\default.yaml --ckpt runs\hl\best.pt
```

在留出 recording 上滚动预测子任务序列，输出阶段识别准确率与时间轴可视化图（PNG）。

---

## 常见问题（Windows 版）

| 问题 | 解决 |
|---|---|
| `conda activate` 报 "无法加载 profiles" | 在 PowerShell 执行一次 `conda init powershell`，重开终端 |
| clone / pip 超时 | clone 用 `ghproxy.net` 镜像前缀；pip 加 `-i https://pypi.tuna.tsinghua.edu.cn/simple` |
| `python` 指向错误环境 | 终端右上角切换 Python 解释器到 `tendor311`，或重新 `conda activate tendor311` |
| GUI 标注窗口弹不出（远程会话） | 用 RDP/向日葵等图形会话登录服务器再跑，不要在纯 SSH 会话里跑 |
| 训练中断后想继续 | 脚本支持 `--resume runs\<model>\last.pt`（如脚本提供该参数） |
| 路径含空格报错 | 全部加引号；本项目路径无空格，保持数据放 `data\` 下即可 |

---

## 目录速查

```
D:\ancddx\vm\tendor_t3\tendor_project\
├── configs\default.yaml       # 所有超参与数据路径配置
├── data\                      # ← scp 上传的 recording 数据放这里
│   └── recording_20260826_172240\
│       ├── ads_data.csv               # 原始 367 列无表头运动学
│       ├── left_endo_*.mp4            # 原始视频
│       ├── left_endo_frames\          # 抽帧产出
│       ├── right_endo_frames\
│       ├── kinematics.csv             # 转换产出
│       └── subtasks.json              # 标注产出
├── runs\                      # 训练权重与日志
├── scripts\                   # 流水线脚本
└── tendon_vam\                # 模型代码
```
