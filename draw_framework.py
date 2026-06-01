"""
Research framework diagram with proper LaTeX formulas.
Run: python draw_framework.py
Output: research_framework.pdf  (vector, print-quality)
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import matplotlib.patheffects as pe

# ── colour palette ────────────────────────────────────────────────────────────
C_TITLE_BG  = "#1565C0"   # dark blue   – top/bottom banner
C_TOPIC_BG  = "#1E88E5"   # medium blue – research-content box
C_CARD_BG   = "#1976D2"   # card header
C_CARD_BODY = "#E3F2FD"   # card body (light blue)
C_ARROW     = "#0D47A1"
C_WHITE     = "white"
C_BLACK     = "#1A1A1A"
C_GREY      = "#90A4AE"
C_GREEN     = "#43A047"

import matplotlib.font_manager as fm
_wqy = "/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc"
fm.fontManager.addfont(_wqy)
_fp = fm.FontProperties(fname=_wqy)
_cn_font = _fp.get_name()   # 'WenQuanYi Zen Hei'

plt.rcParams.update({
    "text.usetex": False,
    "mathtext.fontset": "dejavusans",
    "font.family": _cn_font,
    "font.size": 8,
})

fig, ax = plt.subplots(figsize=(18, 12))
ax.set_xlim(0, 18)
ax.set_ylim(0, 12)
ax.axis("off")

# ─────────────────────────────────────────────────────────────────────────────
# helpers
# ─────────────────────────────────────────────────────────────────────────────
def rect(ax, x, y, w, h, fc, ec="none", lw=1, radius=0.15, zorder=2):
    p = FancyBboxPatch((x, y), w, h,
                       boxstyle=f"round,pad=0,rounding_size={radius}",
                       fc=fc, ec=ec, lw=lw, zorder=zorder)
    ax.add_patch(p)
    return p

def text(ax, x, y, s, **kw):
    return ax.text(x, y, s, **kw)

def up_arrow(ax, x, y, length=0.55, color=C_ARROW):
    ax.annotate("", xy=(x, y + length), xytext=(x, y),
                arrowprops=dict(arrowstyle="->,head_width=0.18,head_length=0.18",
                                color=color, lw=2), zorder=5)

def down_arrow(ax, x, y, length=0.45, color=C_WHITE):
    ax.annotate("", xy=(x, y - length), xytext=(x, y),
                arrowprops=dict(arrowstyle="->,head_width=0.16,head_length=0.16",
                                color=color, lw=1.8), zorder=5)

# ─────────────────────────────────────────────────────────────────────────────
# 1. TOP BANNER  – research goal
# ─────────────────────────────────────────────────────────────────────────────
rect(ax, 0.2, 10.8, 17.6, 0.95, fc=C_TITLE_BG, radius=0.2, zorder=3)
text(ax, 9.0, 11.27,
     "研究目标：建模真实网络流量的非平稳特性、提升预测准确度",
     ha="center", va="center", fontsize=13, fontweight="bold",
     color=C_WHITE, zorder=4)

# ─────────────────────────────────────────────────────────────────────────────
# 2. RESEARCH CONTENT 1  – non-stationarity analysis box
# ─────────────────────────────────────────────────────────────────────────────
rect(ax, 0.5, 8.5, 17.0, 2.1, fc=C_TOPIC_BG, ec=C_WHITE, lw=1.2,
     radius=0.15, zorder=2)

# label
rect(ax, 0.6, 9.9, 4.5, 0.55, fc=C_CARD_BG, radius=0.12, zorder=3)
text(ax, 2.85, 10.17,
     "研究内容1：网络流量非平稳性分析",
     ha="center", va="center", fontsize=8.5, fontweight="bold",
     color=C_WHITE, zorder=4)

# main formula  P(x_{t1+τ},...,x_{tk+τ}) ≠ P(x_{t1},...,x_{tk})  ∀τ
text(ax, 9.0, 9.55,
     r"$P(x_{t_1+\tau},\ldots,x_{t_k+\tau}) \neq P(x_{t_1},\ldots,x_{t_k}),\quad \forall\,\tau$",
     ha="center", va="center", fontsize=10.5, color=C_WHITE, zorder=4)

# four sub-formula columns
col_xs = [1.6, 5.5, 9.4, 13.7]
sub_rows = [
    # col 0 – mobile (moment shift)
    [r"$P(x_{t_1},\ldots,x_{t_m}) \neq P(x_{t_{m+1}},\ldots,x_{t_{m+n}})$",
     r"$P(x_t,\ldots) = P(x_{t+\tau},\ldots)$"],
    # col 1 – video (long-range dep)
    [r"$P(x_{t_1},y_{t_2},\ldots) \neq P(x_{t_1+\tau},y_{t_2+\tau},\ldots)$",
     r"$P(x_t,\ldots) = P(x_{t+\tau},\ldots)$",
     r"$P(y_t,\ldots) = P(y_{t+\tau},\ldots)$"],
    # col 2 – academic (multi-scale)
    [r"$P(X_t, Y_t,\ldots) \neq P(X_{t+\tau}, Y_{t+\tau},\ldots)$",
     r"$P(X_t,\ldots) = P(X_{t+\tau},\ldots)$",
     r"$P(Y_t,\ldots) = P(Y_{t+\tau},\ldots)$"],
    # col 3 – satellite (event-driven)
    [r"$P(x_{t_1}',\ldots,x_{t_m}') \neq P(x_{t_{m+1}}',\ldots,x_{t_{m+n}}')$",
     r"$P(x_{t_1}',\ldots,x_{t_m}'|E_1) = P(x_{t_{m+1}}',\ldots,x_{t_{m+n}}'|E_1)$",
     r"$P(x_{t_1}',\ldots,x_{t_m}'|E_2) = P(x_{t_{m+1}}',\ldots,x_{t_{m+n}}'|E_2)$"],
]

for ci, (cx, rows) in enumerate(zip(col_xs, sub_rows)):
    for ri, row in enumerate(rows):
        text(ax, cx, 9.15 - ri * 0.26, row,
             ha="center", va="center", fontsize=6.8, color=C_WHITE, zorder=4)

# ─────────────────────────────────────────────────────────────────────────────
# 3. FOUR TOPIC CARDS  (header row)
# ─────────────────────────────────────────────────────────────────────────────
card_titles = [
    "移动网络流量的\n高阶矩分布偏移",
    "视频流量的\n长程依赖",
    "学术网络流量的\n多尺度依赖",
    "卫星网络流量的\n事件驱动分布偏移",
]
card_x = [0.55, 5.0, 9.45, 13.5]
card_w = 3.9

for cx, title in zip(card_x, card_titles):
    rect(ax, cx, 7.3, card_w, 0.95, fc=C_CARD_BG, radius=0.15, zorder=3)
    text(ax, cx + card_w / 2, 7.77, title,
         ha="center", va="center", fontsize=9, fontweight="bold",
         color=C_WHITE, zorder=4, linespacing=1.4)

# ─────────────────────────────────────────────────────────────────────────────
# 4. ARROWS  down from topic cards  +  labels on arrows
# ─────────────────────────────────────────────────────────────────────────────
arrow_cx = [2.5, 6.95, 11.4, 15.45]
arrow_labels = ["非平稳变换", None, "非平稳分解", "非平稳归因"]

for cx, lbl in zip(arrow_cx, arrow_labels):
    down_arrow(ax, cx, 7.3, length=0.55)
    if lbl:
        text(ax, cx, 6.88, lbl, ha="center", va="center",
             fontsize=7.5, color=C_WHITE, zorder=5)

# horizontal "非平稳分解" label spanning cols 1-2
text(ax, 8.7, 6.88, "←  非平稳分解  →",
     ha="center", va="center", fontsize=7.5, color=C_WHITE, zorder=5)

# ─────────────────────────────────────────────────────────────────────────────
# 5. CONTENT LABELS  (研究内容2/3/4/5)
# ─────────────────────────────────────────────────────────────────────────────
content_labels = ["研究内容4", "研究内容2", "研究内容3", "研究内容5"]
label_cx = [2.5, 6.95, 11.4, 15.45]
for cx, lbl in zip(label_cx, content_labels):
    rect(ax, cx - 0.75, 6.38, 1.5, 0.38, fc=C_CARD_BG, ec=C_WHITE,
         lw=0.8, radius=0.1, zorder=4)
    text(ax, cx, 6.57, lbl, ha="center", va="center",
         fontsize=7.5, color=C_WHITE, fontweight="bold", zorder=5)

# ─────────────────────────────────────────────────────────────────────────────
# 6. MODEL CARDS  (body)
# ─────────────────────────────────────────────────────────────────────────────
model_headers = [
    "高阶矩交互预测模型",
    "自注意力序列预测模型",
    "自适应时钟神经网络模型",
    "事件感知非平稳预测模型",
]
model_bodies = [
    (
        "- 高阶矩归一化运算：扩展均值、\n"
        "  方差标准化，提出高阶矩标准化\n"
        "- 库普曼算子：统计矩动态非线性\n"
        "  演化映射到线性空间\n"
        "- 统计矩交互注意力机制：多头\n"
        "  注意力建模高阶矩相关性"
    ),
    (
        "- 编码器＋解码器架构、多头\n"
        "  自注意力机制：建模多时间\n"
        "  序列间长程依赖\n"
        "- 视频帧位置编码：视频类别\n"
        "  信息显示编码到视频流量序列\n"
        "- \"教师强制\"训练策略：训练\n"
        "  阶段提示预测，缓解误差累积"
    ),
    (
        "- 时钟 LSTM：将 LSTM 隐藏状态\n"
        "  拆为 $k$ 个不同频率子单元，\n"
        "  捕捉不同尺度的流量特征\n"
        "- 自适应采样间隔：对输入流量\n"
        "  序列进行滑动自相关分析，\n"
        "  选取 $k$ 个最高自相关系数\n"
        "  对应的时延作为采样间隔"
    ),
    (
        "- 自回归扩散模型：概率生成方\n"
        "  式建模网络流量不确定性\n"
        "- 条件注意力机制：时序分类器\n"
        "  自由机制建模事件对网络\n"
        "  流量的影响\n"
        "- 事件感知时间步：不同事件下\n"
        "  时间序列建模难度不同"
    ),
]
metrics = [
    "使用后将骨干模型的预测\n误差平均降低 20%",
    "相比标准 Transformer 平均\n误差损失降低 9.7%",
    "相比 LSTM 的精度平均提升\n23% 且推理时间较 Transformer\n减少 65%",
    "概率预测精度平均提升 41.3%\n确定性预测精度平均提升 27.5%",
]

body_y_top = 6.15
body_h     = 4.1
hdr_h      = 0.42

for i, (cx, hdr, body, met) in enumerate(
        zip(card_x, model_headers, model_bodies, metrics)):
    # card background
    rect(ax, cx, body_y_top - body_h, card_w, body_h,
         fc=C_CARD_BODY, ec="#90CAF9", lw=0.8, radius=0.12, zorder=3)
    # header stripe
    rect(ax, cx, body_y_top - hdr_h, card_w, hdr_h,
         fc=C_CARD_BG, radius=0.12, zorder=4)
    text(ax, cx + card_w / 2, body_y_top - hdr_h / 2, hdr,
         ha="center", va="center", fontsize=8.2, fontweight="bold",
         color=C_WHITE, zorder=5)
    # body text
    text(ax, cx + 0.12, body_y_top - hdr_h - 0.18, body,
         ha="left", va="top", fontsize=7.0, color=C_BLACK,
         zorder=5, linespacing=1.45)
    # metric footer (dashed separator)
    sep_y = body_y_top - body_h + 0.75
    ax.plot([cx + 0.15, cx + card_w - 0.15], [sep_y, sep_y],
            ls="--", lw=0.8, color=C_GREY, zorder=5)
    text(ax, cx + card_w / 2, sep_y - 0.08, met,
         ha="center", va="top", fontsize=6.8, color="#546E7A",
         style="italic", zorder=5, linespacing=1.35)

# ─────────────────────────────────────────────────────────────────────────────
# 7. BOTTOM UP-ARROWS  + labels
# ─────────────────────────────────────────────────────────────────────────────
support_xs = [2.5, 6.95, 11.4, 15.45]
support_y  = body_y_top - body_h   # top of the gap below cards

for cx in support_xs:
    up_arrow(ax, cx, support_y - 0.65, length=0.5, color=C_ARROW)
    text(ax, cx, support_y - 0.72, "↑  支撑",
         ha="center", va="top", fontsize=8.5, color=C_ARROW,
         fontweight="bold", zorder=5)

# ─────────────────────────────────────────────────────────────────────────────
# 8. BOTTOM BANNER  – final task
# ─────────────────────────────────────────────────────────────────────────────
rect(ax, 0.2, 0.15, 17.6, 0.85, fc=C_TITLE_BG, radius=0.2, zorder=3)
text(ax, 9.0, 0.57,
     "最终任务：为网络规划、流量调度等智能网络管理与自智网络应用提供理论支撑与技术基础",
     ha="center", va="center", fontsize=10.5, fontweight="bold",
     color=C_WHITE, zorder=4)

# ─────────────────────────────────────────────────────────────────────────────
# 9. OUTER BORDER
# ─────────────────────────────────────────────────────────────────────────────
rect(ax, 0.1, 0.05, 17.8, 11.85, fc="none", ec=C_TITLE_BG,
     lw=2.0, radius=0.25, zorder=1)

plt.tight_layout(pad=0)
plt.savefig("/home/user/EventTSF/research_framework.pdf",
            dpi=300, bbox_inches="tight", facecolor="white")
plt.savefig("/home/user/EventTSF/research_framework.png",
            dpi=200, bbox_inches="tight", facecolor="white")
print("Saved: research_framework.pdf  &  research_framework.png")
