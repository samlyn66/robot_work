"""tendon_vam: 面向肌腱吻合的分层模仿视频-动作模型（参照 SRT-H）."""

__version__ = "0.1.0"

PHASES = ["idle", "grasp_needle", "insert", "exit", "pull", "knot", "cut"]
PHASE_ZH = {
    "idle": "空闲",
    "grasp_needle": "持针",
    "insert": "进针",
    "exit": "出针",
    "pull": "拉线",
    "knot": "打结",
    "cut": "剪线",
}
