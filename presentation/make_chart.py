"""Draw the "what we tried" chart for the slides: validation and test accuracy of every model.

Run it from the repository folder: uv run python presentation/make_chart.py
"""

import matplotlib.pyplot as plt
import pandas as pd

# validation and test accuracy from the notebooks (60 validation photos, 30 test photos)
results = pd.DataFrame(
    [
        ("Chance (guessing)", 1 / 3, 1 / 3),
        ("CNN from scratch", 0.633, 0.633),
        ("Eva: dense head (224)", 0.800, 0.967),
        ("Eva: + augmentation (224)", 0.800, 0.967),
        ("Eva: strong augmentation (224)", 0.817, 0.933),
        ("Eva: Keras Tuner (224, Eva's best)", 0.817, 0.933),
        ("Amir: + augmentation (160)", 0.850, 0.867),
        ("Amir: features + dense head (160, chosen)", 0.867, 0.900),
    ],
    columns=["model", "validation", "test"],
).set_index("model")

ax = results.plot.barh(figsize=(11, 5.5), color=["#9ecae1", "#3182bd"], width=0.8)
for container in ax.containers:
    ax.bar_label(container, fmt="%.2f", padding=3)
ax.set_xlim(0, 1.08)
ax.set_xlabel("accuracy")
ax.set_ylabel("")
ax.set_title("What we tried: accuracy on photos the models never trained on")
ax.legend(loc="lower right")
for side in ["top", "right"]:
    ax.spines[side].set_visible(False)
plt.tight_layout()
plt.savefig("presentation/images/what_we_tried.png", dpi=150)
print("saved presentation/images/what_we_tried.png")
