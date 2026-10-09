# 服务器部署与运行指南（VS Code Remote-SSH）

适用：GPU Linux 服务器（Ubuntu 20.04/22.04），本地 Windows + VS Code。
全部命令在**服务器终端**执行（除非特别说明）。

---

## 第 0 步：VS Code 连上服务器

1. 本地 VS Code 安装扩展 **Remote - SSH**（ms-vscode-remote.remote-ssh）。
2. `Ctrl+Shift+P` → **Remote-SSH: Add New SSH Host** → 输入：
   ```bash
   ssh <用户名>@<服务器IP> -p <端口，默认22可省>
   ```
3. `Ctrl+Shift+P` → **Remote-SSH: Connect to Host** → 选刚添加的主机 → 输密码（或提前配好 ssh-key 免密）。
4. 连接后 `文件 → 打开文件夹`，选一个工作目录，如 `~/tendon/`，之后所有命令都在 VS Code 里打开的**服务器终端**（`` Ctrl+` ``）里执行。

## 第 1 步：拉代码

```bash
cd ~ && mkdir -p tendon && cd tendon
git clone https://github.com/samlyn66/robot_work.git tendon_vam_project
cd tendon_vam_project
```

> 服务器访问 GitHub 慢/失败时：换镜像 `git clone https://ghproxy.net/https://github.com/samlyn66/robot_work.git`，或先在本地打包再 `scp` 上去（见第 2 步的 scp 用法）。

## 第 2 步：上传数据（视频 + ads_data.csv，不走 git）

数据是大文件，**不要**放进 git 仓库，直接传到服务器：

本地 Windows PowerShell（`scp` 用法，注意路径）：

```powershell
# 整个 dataset 文件夹（18 个 recording）传到服务器
scp -P <端口> -r "C:\Users\25965\Desktop\dataset" <用户名>@<服务器IP>:~/tendon/data/
```

数据量大时建议压缩后再传（或用 rsync 断点续传，服务器若装了 rsync）：

```powershell
# 压缩
cd C:\Users\25965\Desktop
tar -a -c -f dataset.zip dataset
scp -P <端口> dataset.zip <用户名>@<服务器IP>:~/tendon/
```

服务器上解压并放入项目约定位置：

```bash
cd ~/tendon
unzip dataset.zip            # 没有 unzip 就: python -m zipfile -e dataset.zip .
mkdir -p tendon_vam_project/data
mv dataset/* tendon_vam_project/data/
ls tendon_vam_project/data/  # 应看到 recording_20260826_172240 等
```

## 第 3 步：装环境（conda）

```bash
# 3.1 Miniconda（已装可跳过）
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh   # 一路回车/yes，完成后重开终端
conda init bash && exec bash

# 3.2 建环境
conda create -n tendon python=3.10 -y
conda activate tendon

# 3.3 PyTorch（按服务器 CUDA 版本选；先 nvidia-smi 看驱动支持的 CUDA 上限）
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
# 纯 CPU 服务器就: pip install torch torchvision

# 3.4 其余依赖
cd ~/tendon/tendon_vam_project
pip install -r requirements.txt
# requirements 里若没有 opencv: pip install opencv-python pyyaml tqdm matplotlib einops
```

验证 GPU：

```bash
python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"
```

> 国内服务器 pip 慢：`pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple`

## 第 4 步：数据体检 + 转换 + 抽帧（CPU 即可）

```bash
cd ~/tendon/tendon_vam_project
conda activate tendon

# 4.1 新数据先体检：自动分析 ads_data.csv 列结构/采样率/夹爪候选
python scripts/inspect_ads.py --rec data/recording_20260826_172240

# 4.2 运动学转换：ads_data.csv → kinematics.csv（30Hz，对齐视频时间轴）
python scripts/convert_ads.py --config configs/default.yaml --data_root data

# 4.3 抽帧：60fps 视频 → 10fps jpg（18 个 recording 全跑，耗时取决于 IO）
python scripts/extract_frames.py --data_root data --fps 10
```

跑完检查：每个 recording 下应有 `kinematics.csv`、`left_endo_frames/`（约 3400 张）、`right_endo_frames/`。

## 第 5 步：子任务标注（HL 训练的前置）

```bash
# 需要能弹 GUI 的环境；服务器无显示器时用 X11 转发，或在自己电脑上标注后传 subtasks.json
python scripts/make_subtask_labels.py --recording data/recording_20260826_172240
```

> 建议在**本地**标（双击视频看更方便），把生成的 `subtasks.json` scp 到对应 recording 目录。6 个阶段：持针/进针/出针/拉线/打结/剪线。

## 第 6 步：训练

```bash
# 6.1 高层子任务规划器（小数据即可起步，先拿阶段识别准确率）
python scripts/train_hl.py --config configs/default.yaml

# 6.2 低层策略（需要 kinematics.csv，GPU 建议 ≥8GB 显存）
python scripts/train_ll.py --config configs/default.yaml

# 6.3 完整 VAM（flow matching 动作解码器）
python scripts/train_vam.py --config configs/default.yaml
```

长时间训练放后台（断开 SSH 也不停）：

```bash
# 方式一：tmux（推荐）
tmux new -s train
conda activate tendon && python scripts/train_hl.py --config configs/default.yaml
# Ctrl+B 再按 D 脱离；下次 tmux attach -t train 回来

# 方式二：nohup
nohup python scripts/train_hl.py --config configs/default.yaml > train_hl.log 2>&1 &
tail -f train_hl.log
```

## 第 7 步：评估

```bash
python scripts/evaluate.py --config configs/default.yaml --ckpt runs/hl/best.pt
```

---

## 常见问题

| 现象 | 处理 |
|---|---|
| `git clone` 超时 | 用镜像前缀（ghproxy.net 等），或本地 `git bundle` 后 scp |
| `pip install` 卡住 | 换清华源（见 3.4 注） |
| `torch.cuda.is_available()` 为 False | 驱动/PyTorch CUDA 版本不匹配，`nvidia-smi` 看上限后重装对应版本 |
| cv2 报 `libGL.so.1` | `conda install -y libgl` 或 `pip install opencv-python-headless` |
| 中文路径报错 | 服务器上全用英文路径（本指南的 `~/tendon/` 已避免） |
| 显存不足 | 调小 `configs/default.yaml` 的 `train.batch_size`（8→2/4） |
| 训练中断 | `tmux attach` 回去看进度；抽帧支持断点续跑（已抽的自动跳过） |

## 推荐目录结构（服务器上）

```
~/tendon/
├── dataset.zip                  # 上传的压缩包（可删）
└── tendon_vam_project/          # git clone 的代码
    ├── data/                    # 18 个 recording（视频+csv+抽帧+kinematics）
    ├── runs/                    # 训练输出（checkpoint、日志）
    └── ...
```
