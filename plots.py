import pandas as pd
import matplotlib.pyplot as plt
import os

csv_path = "model/logs/training_metrics_1000.csv"
plot_dir = "plots"
plot_path = os.path.join(plot_dir, "training_loss_1000.png")

os.makedirs(plot_dir, exist_ok=True)

df = pd.read_csv(csv_path)

early = df[df["epoch"] <= 10]
late = df[df["epoch"] > 9]

plt.figure()

# Train loss — blue
plt.plot(
    early["epoch"],
    early["train_loss"],
    label="Train Loss",
    color="tab:blue",
    alpha=1.0,
    linewidth=2.5
)
plt.plot(
    late["epoch"],
    late["train_loss"],
    color="tab:blue",
    alpha=0.5,
    linewidth=2.5
)

# Validation loss — orange
plt.plot(
    early["epoch"],
    early["val_loss"],
    label="Validation Loss",
    color="tab:orange",
    alpha=1.0,
    linewidth=2.5
)
plt.plot(
    late["epoch"],
    late["val_loss"],
    color="tab:orange",
    alpha=0.5,
    linewidth=2.5
)

plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training vs Validation Loss")
plt.legend()
plt.grid(False)

plt.axvline(
    x=17,
    linestyle="--",
    linewidth=1.5,
    color="tab:red",
    alpha=0.7,
    label="Selected model (epoch 17)"
)

plt.savefig(plot_path, bbox_inches="tight")
plt.close()

plot_path = os.path.join(plot_dir, "training_validation_1000.png")

plt.figure()

plt.plot(
    early["epoch"],
    early["val_top1"],
    label="Top-1 accuracy",
    color="tab:blue",
    alpha=1.0,
    linewidth=2.5
)
plt.plot(
    late["epoch"],
    late["val_top1"],
    color="tab:blue",
    alpha=0.5,
    linewidth=2.5
)

plt.plot(
    early["epoch"],
    early["val_top3"],
    label="Top-3 accuracy",
    color="tab:orange",
    alpha=1.0,
    linewidth=2.5
)
plt.plot(
    late["epoch"],
    late["val_top3"],
    color="tab:orange",
    alpha=0.5,
    linewidth=2.5
)

plt.plot(
    early["epoch"],
    early["val_top5"],
    label="Top-5 accuracy",
    color="tab:green",
    alpha=1.0,
    linewidth=2.5
)
plt.plot(
    late["epoch"],
    late["val_top5"],
    color="tab:green",
    alpha=0.5,
    linewidth=2.5
)

plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training Validation Accuracy")
plt.legend()
plt.grid(False)

plt.axvline(
    x=17,
    linestyle="--",
    linewidth=1.5,
    color="tab:red",
    alpha=0.7,
    label="Selected model (epoch 17)"
)

plt.savefig(plot_path, bbox_inches="tight")
plt.close()
