# DEPLOY.md — Windows 服务器部署与运行指南（robot_work）

> 本指南对应 **Windows 服务器**，与当前实际状态一致：
>
> - 项目目录：`C:\Users\服务器1\robot_work`（代码在仓库根目录，即 robot_work 就是项目根）
> - conda 环境：`tendon311`（Python 3.11，位于 `D:\anaconda\envs\tendon311`）
> - PyTorch 2.5.1 已装好，`torch.cuda.is_available() == True` ✅
> - requirements.txt 依赖已安装 ✅
> - 所有命令在 **PowerShell** 中执行（VS Code 终端默认就是）

---

## 第 0 步：VS Code 连接服务器（已完成 ✅）

1. 本地 VS Code 装扩展 **Remote - SSH**
2. `Ctrl+Shift+P` → **Remote-SSH: Connect to Host** → `ssh 用户名@服务器IP -p 端口`
3. 首次连接平台选 **Windows**，输密码
4. **Open Folder** → `C:\Users\服务器1\robot_work`
5. `` Ctrl+` `` 打开终端，确认提示符为 `PS C:\Users\服务器1\robot_work>` 且前面有 `(tendon311)`

> 从你的截图看第 0~3 步（连服务器、clone、装 torch、装依赖）都已完成，**直接从第 4 步开始**。

---

## 第 1 步：环境自检（已完成 ✅，换新环境时再跑）

```powershell
conda activate tendon311
cd C:\Users\服务器1\robot_work
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
# 预期输出: 2.5.1 True
```

- 若输出 `False`：`nvidia-smi` 看驱动，重装对应 CUDA 版 torch（`pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121`）
- 缺依赖时：`pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple`

---

## 第 2 步：上传数据

数据（`dataset` 文件夹，含 recording_xxx）在**本地电脑**桌面，在**本地**终端执行 scp（不是在服务器上）：

```powershell
scp -P 端口 -r "C:\Users\25965\Desktop\dataset" 用户名@服务器IP:"C:/Users/服务器1/robot_work/data/"
```

要点：

- 目标路径用**正斜杠 `/`** 并加引号
- recording 文件夹必须直接位于 `data\` 下
- 传完在服务器上确认：

```powershell
dir C:\Users\服务器1\robot_work\data
# 应看到 data\recording_20260826_172240\{ads_data.csv, left_endo_*.mp4, right_endo_*.mp4}
```

> 数据大、scp 慢时可用 WinSCP / FileZilla 图形化拖拽，效果相同。

---

## 第 3 步：跑通数据流水线（CPU 即可，几分钟）

```powershell
cd C:\Users\服务器1\robot_work

# 3.1 数据体检：自动分析无表头 CSV 的列结构，核对列映射

# 3.2 运动学转换：ads_data.csv -> kinematics.csv（含时间对齐、四元数修复）
python scripts\convert_ads.py --config configs\default.yaml --data_root data

# 3.3 双腕相机抽帧（10fps）
#    先小规模试跑（每路只抽 200 帧，几分钟）：
python scripts\extract_frames.py --data_root data --fps 10 --max_frames 200
#    确认无误后跑完整数据（去掉 --max_frames）：
python scripts\extract_frames.py --data_root data --fps 10
#    中断后重跑会自动跳过已抽帧（断点续抽）
```

完成后每个 recording 下应出现 `left_endo_frames\`、`right_endo_frames\`、`kinematics.csv`。

---

## 第 4 步：标注子任务（服务器有桌面，直接弹窗标注）

```powershell
python scripts\make_subtask_labels.py --recording data\recording_20260826_172240
```

弹出窗口播放视频，在 6 个阶段边界打点（持针/进针/出针/拉线/打结/剪线），生成 `subtasks.json`。逐个 recording 标注。

> 必须在图形会话（RDP / 向日葵 / 服务器本机）里跑，纯 SSH 会话弹不出窗口。

---

## 第 5 步：训练

**Windows 没有 tmux**，防断线二选一：

- **方案 A（简单）**：训练时保持 VS Code 窗口开着，直接前台跑
- **方案 B（断线不停）**：独立 PowerShell 窗口 + 日志重定向：

```powershell
Start-Process powershell -ArgumentList '-NoExit','-Command',
  "conda activate tendon311; cd C:\Users\服务器1\robot_work; python scripts\train_hl.py --config configs\default.yaml *>&1 | Tee-Object runs\train_hl.log"
```

训练命令（按顺序）：

```powershell
# 5.1 高层阶段规划器（只需视频帧 + subtasks.json，先跑这个拿第一版结果）
python scripts\train_hl.py --config configs\default.yaml

# 5.2 低层策略（LL: SRT-H 风格 ACT）—— 需要 kinematics.csv
python scripts\train_ll.py --config configs\default.yaml

# 5.3 完整 VAM（视频骨干 + 潜在视觉计划 + Flow Matching 动作解码器）
python scripts\train_vam.py --config configs\default.yaml
```

产物保存在 `runs\`（权重 `*.pt`、日志、曲线）。

**显存不足（CUDA out of memory）**：编辑 `configs\default.yaml`，`train.batch_size: 8` → `4` 或 `2`。

---

## 第 6 步：评估

```powershell
python scripts\evaluate.py --config configs\default.yaml --ckpt runs\hl\best.pt
```

在留出 recording 上滚动预测子任务序列，输出阶段识别准确率与时间轴可视化 PNG。

---

## 常见问题（Windows 版）

| 问题                                 | 解决                                                                                                               |
| ---------------------------------- | ---------------------------------------------------------------------------------------------------------------- |
| `conda activate` 报 "无法加载 profiles" | 执行一次 `conda init powershell`，重开终端                                                                                |
| clone / pip 超时                     | clone 加镜像前缀 `https://ghproxy.net/https://github.com/...`；pip 加清华镜像 `-i https://pypi.tuna.tsinghua.edu.cn/simple` |
| `python` 指向错误环境                    | 终端右上角切换解释器到 `tendon311`，或重新 `conda activate tendon311`                                                           |
| 标注窗口弹不出                            | 用 RDP/向日葵等图形会话登录服务器再跑                                                                                            |
| 训练被远程断连打断                          | 用方案 B 的 `Start-Process` 方式重跑；日志在 `runs\*.log`                                                                    |
| 路径含空格报错                            | 全部加引号；保持数据放 `data\` 下即可                                                                                          |

---

## 目录速查（当前实际布局）

```
C:\Users\服务器1\robot_work\        ← 仓库根 = 项目根
├── configs\default.yaml       # 所有超参与数据路径配置
├── data\                      # ← scp 上传的 recording 数据放这里
│   └── recording_20260826_172240\
│       ├── ads_data.csv               # 原始 367 列无表头运动学
│       ├── left_endo_*.mp4            # 原始视频
│       ├── left_endo_frames\          # 抽帧产出
│       ├── right_endo_frames\
│       ├── kinematics.csv             # 转换产出
│       └── subtasks.json              # 标注产出
├── runs\                      # 训练权重与日志（训练后生成）
├── scripts\                   # 流水线脚本
└── tendon_vam\                # 模型代码
```

## 更新代码

本地或服务器上改动推送后，服务器同步：

```powershell
cd C:\Users\服务器1\robot_work
git pull
# 国内服务器 pull 超时：
git config http.proxy http://127.0.0.1:10808   # 若服务器有代理
# 或改用镜像一次性拉取：
git pull https://ghproxy.net/https://github.com/samlyn66/robot_work.git main
```
