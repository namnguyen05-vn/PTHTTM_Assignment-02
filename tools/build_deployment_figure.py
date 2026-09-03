from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "figures" / "deployment_architecture.png"


def box(ax, x, y, width, height, title, detail, color):
    patch = FancyBboxPatch(
        (x, y), width, height,
        boxstyle="round,pad=0.02,rounding_size=0.025",
        linewidth=1.4, edgecolor=color, facecolor="white",
    )
    ax.add_patch(patch)
    ax.text(x + width / 2, y + height * 0.64, title, ha="center", va="center", fontsize=11, weight="bold", color=color)
    ax.text(x + width / 2, y + height * 0.34, detail, ha="center", va="center", fontsize=8.5, color="#333333")


fig, ax = plt.subplots(figsize=(11, 4.2))
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
ax.axis("off")

nodes = [
    (0.03, 0.34, 0.18, 0.34, "Responsive Web", "Desktop / tablet / phone\nraw user inputs", "#8E5A00"),
    (0.28, 0.34, 0.18, 0.34, "FastAPI", "Pydantic validation\n3 prediction endpoints", "#0B7285"),
    (0.53, 0.34, 0.19, 0.34, "Persisted Pipeline", "Impute / encode / TF-IDF\ntrained estimator", "#5F3DC4"),
    (0.79, 0.34, 0.18, 0.34, "Prediction", "class / price / interest\nprobability or range", "#2B8A3E"),
]
for node in nodes:
    box(ax, *node)

for start, end in [((0.21, 0.51), (0.28, 0.51)), ((0.46, 0.51), (0.53, 0.51)), ((0.72, 0.51), (0.79, 0.51))]:
    ax.add_patch(FancyArrowPatch(start, end, arrowstyle="-|>", mutation_scale=15, linewidth=1.5, color="#495057"))

ax.text(0.5, 0.88, "Kiến trúc triển khai thống nhất cho ba ứng dụng", ha="center", fontsize=14, weight="bold", color="#1F4E79")
ax.text(0.5, 0.13, "Frontend gửi JSON về cùng origin; backend nạp pipeline đã lưu và không fit lại preprocessing khi inference.", ha="center", fontsize=9.5, color="#495057")
fig.tight_layout()
OUT.parent.mkdir(exist_ok=True)
fig.savefig(OUT, dpi=180, bbox_inches="tight", facecolor="white")
print(OUT)
