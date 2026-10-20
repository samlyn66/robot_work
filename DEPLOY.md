# DEPLOY.md — Windows 服务器部署与运行指南（robot_work）

> 本指南对应 **Windows 服务器**，与当前实际状态一致：
>
> - 项目目录：`C:\Users\服务器1\robot_work`（代码在仓库根目录，即 robot_work 就是项目根）
> - conda 环境：`tendon311`（Python 3.11，位于 `D:\anaconda\envs\tendon311`）
> - PyTorch 2.5.1 已装好，`torch.cuda.is_available() == True` ✅
> - requirements.txt 依赖已安装 ✅
> - **18 条原始数据已就位：`C:\data\gczjj`（⚠️ 只读，绝不写入/修改）**
> - 所有命令在 **PowerShell** 中执行（VS Code 终端默认就是）

---

## ⚠️ 数据隔离原则（重要）

**原始目录 `C:\data\gczjj` 只进不出** —— 里面只有 `ads_data.csv` + 两个 `*_endo*.mp4`，
任何脚本都不会往里写东西。

所有生成数据（kinematics.csv、抽帧 `*_frames`、subtasks.json、报告 JSON）一律写到
**独立目录 `C:\data\gczjj_processed`**（流水线自动创建）。标注与训练时的
`--data_root` / `--recording` 都指向 `_processed` 目录。

```
C:\data\gczjj             ← 原始数据（只读，永不改动）
C:\data\gczjj_processed   ← 全部生成数据（流水线产出 + 人工标注）
```

---

## 第 0 步：VS Code 连接服务器（已完成 ✅）

1. 本地 VS Code 装扩展 **Remote - SSH**
2. `Ctrl+Shift+P` → **Remote-SSH: Connect to Host** → `ssh 用户名@服务器IP -p 端口`
3. 首次连接平台选 **Windows**，输密码
4. **Open Folder** → `C:\Users\服务器1\robot_work`
5. `` Ctrl+` `` 打开终端，确认提示符为 `PS C:\Users\服务器1\robot_work>` 且前面有 `(tendon311)`

---

## 第 1 步：拉取最新代码 + 环境自检

```powershell
conda activate tendon311
cd C:\Users\服务器1\robot_work
git pull
git log --oneline -3        # 最新应包含批量标注/数据隔离相关提交

python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
# 预期输出: 2.5.1 True
```

- `git pull` 超时（国内服务器）：见文末「更新代码」一节的镜像/代理办法
- 若输出 `False`：`nvidia-smi` 看驱动，重装对应 CUDA 版 torch
- 缺依赖时：`pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple`

---

## 第 2 步：一键流水线（体检 → 转换 → 抽帧，处理全部 18 条）

```powershell
cd C:\Users\服务器1\robot_work
python scripts\run_pipeline.py --data_root C:\data\gczjj --fps 10
```

- 自动遍历 `gczjj` 下**全部** recording（不是只处理某一条）：
  1. **体检**：文件齐全性 / 双机一致 / 行帧比 / 列映射（有 fail 项会中止并列出）
  2. **转换**：`ads_data.csv → kinematics.csv`（时间对齐、四元数修复、夹爪归一化）
  3. **抽帧**：双腕相机按 10 fps 抽帧（耗时较长；中断后重跑同一条命令会**断点续抽**）
- **生成物全部落在 `C:\data\gczjj_processed`**，原始 `gczjj` 零改动
- 想自定义生成目录位置：加 `--output_root D:\somewhere\gczjj_gen`

完成后检查（都在 `_processed` 里）：

```powershell
dir C:\data\gczjj_processed
# 应看到 18 个 recording_xxx + validate_report.json + ads_convert_report.json
dir C:\data\gczjj_processed\recording_20260826_172240
# 应看到 left_endo_frames\ 、right_endo_frames\ 、kinematics.csv
dir C:\data\gczjj
# 应仍然只有 ads_data.csv 和 mp4，没有任何新文件 ✅
```

---

## 第 3 步：标注子任务（⚠️ 服务器本机 PowerShell，不要在 VS Code Remote-SSH 终端跑）

SSH 会话里启动的 OpenCV 窗口收不到键盘消息（窗口显示"未响应"），必须用
**RDP / 向日葵 / 服务器本机桌面** 打开 PowerShell：

```powershell
conda activate tendon311
cd C:\Users\服务器1\robot_work

# 批量模式：遍历全部 18 条，逐条弹窗，已标过的自动跳过
python scripts\make_subtask_labels.py --root C:\data\gczjj_processed --all --fps 10

# 或单条标注：
python scripts\make_subtask_labels.py --recording C:\data\gczjj_processed\recording_20260826_172240 --fps 10
```

按键（先点视频窗口获得焦点）：

- `空格` 播放/暂停；`,` / `.` 上一帧 / 下一帧
- `1`~`6` 打阶段开始标记：1=持针 grasp_needle、2=进针 insert、3=出针 exit、4=拉线 pull、5=打结 knot、6=剪线 cut
- `9` 打恢复片段标记（recover_idle）；`u` 撤销上一个标记
- `s` 保存；`q` / `Esc` / 点窗口 ✕ 退出（退出时自动保存 `subtasks.json`）

**标 3~5 条即可先训 HL**；批量模式下随时关终端退出，已标的都已保存，
下次再跑同一条命令会自动从没标过的继续。

---

## 第 4 步：训练（`--data_root` 指向生成目录！）

**Windows 没有 tmux**，防断线二选一：

- **方案 A（简单）**：训练时保持 VS Code 窗口开着，直接前台跑
- **方案 B（断线不停）**：独立 PowerShell 窗口 + 日志重定向：

```powershell
Start-Process powershell -ArgumentList '-NoExit','-Command',
  "conda activate tendon311; cd C:\Users\服务器1\robot_work; python scripts\train_hl.py --config configs\default.yaml --data_root C:\data\gczjj_processed *>&1 | Tee-Object runs\train_hl.log"
```

训练命令（按顺序，注意全部指向 `gczjj_processed`）：

```powershell
# 4.1 高层阶段规划器（只需视频帧 + subtasks.json，先跑这个拿第一版结果）
python scripts\train_hl.py --config configs\default.yaml --data_root C:\data\gczjj_processed

# 4.2 低层策略（LL: SRT-H 风格 ACT）—— 需要 kinematics.csv
python scripts\train_ll.py --config configs\default.yaml --data_root C:\data\gczjj_processed

# 4.3 完整 VAM（视频骨干 + 潜在视觉计划 + Flow Matching 动作解码器）
python scripts\train_vam.py --config configs\default.yaml --data_root C:\data\gczjj_processed
```

- 训练轮数在 `configs\default.yaml`：`train.epochs_hl / epochs_ll / epochs_vam`
  （默认 60/300/300；想快些验证可先把 `epochs_hl` 改成 20）
- `Ctrl+C` 随时中断
- 产物保存在 `runs\`（权重 `*.pt`、日志、曲线）

**显存不足（CUDA out of memory）**：编辑 `configs\default.yaml`，`train.batch_size: 8` → `4` 或 `2`。

---

## 第 5 步：评估

```powershell
python scripts\evaluate.py --config configs\default.yaml --ckpt runs\hl\best.pt --data_root C:\data\gczjj_processed
```

在留出 recording 上滚动预测子任务序列，输出阶段识别准确率与时间轴可视化 PNG。

---

## 常见问题（Windows 版）

| 问题                                 | 解决                                                                                                               |
| ---------------------------------- | --------------------------------------------------------------------------------------------------------------- |
| `conda activate` 报 "无法加载 profiles" | 执行一次 `conda init powershell`，重开终端                                                                                |
| clone / pip 超时                     | clone 加镜像前缀 `https://ghproxy.net/https://github.com/...`；pip 加清华镜像 `-i https://pypi.tuna.tsinghua.edu.cn/simple` |
| `python` 指向错误环境                    | 终端右上角切换解释器到 `tendon311`，或重新 `conda activate tendon311`                                                           |
| 标注窗口"未响应"/收不到键盘                  | 不要在 Remote-SSH 终端跑标注；用 RDP/向日葵/服务器本机 PowerShell，且先点视频窗口获得焦点                                                      |
| 标注窗口弹不出                            | 用 RDP/向日葵等图形会话登录服务器再跑                                                                                            |
| 训练被远程断连打断                          | 用方案 B 的 `Start-Process` 方式重跑；日志在 `runs\*.log`                                                                    |
| 路径含空格报错                            | 全部加引号                                                                                                            |
| 之前用旧版脚本跑过、gczjj 里残留了生成文件            | 手动把 `kinematics.csv`、`*_frames`、`*_report.json` 从 `gczjj` 移到别处即可；新流水线不会再往 gczjj 写任何东西                             |

---

## 目录速查（当前实际布局）

```
C:\data\gczjj\                       ← 原始数据（只读，只有下面这些文件）
└── recording_20260826_101644\       # 共 18 条，结构相同
    ├── ads_data.csv                 # 原始 367 列无表头运动学
    ├── left_endo_*.mp4              # 原始左腕视频
    └── right_endo_*.mp4             # 原始右腕视频

C:\data\gczjj_processed\             ← 全部生成数据（流水线 + 标注产出）
├── validate_report.json             # 体检报告
├── ads_convert_report.json          # 转换报告
└── recording_20260826_101644\
    ├── left_endo_frames\            # 抽帧产出
    ├── right_endo_frames\           # 抽帧产出
    ├── kinematics.csv               # 转换产出
    └── subtasks.json                # 人工标注产出

C:\Users\服务器1\robot_work\          ← 仓库根 = 项目根
├── configs\default.yaml             # 所有超参配置
├── runs\                            # 训练权重与日志（训练后生成）
├── scripts\                         # 流水线脚本
└── tendon_vam\                      # 模型代码
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
