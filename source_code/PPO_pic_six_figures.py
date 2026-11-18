import csv
from pathlib import Path

import matplotlib.pyplot as plt


# ============================================================
# Path settings
# ============================================================

BASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = BASE_DIR / "ppo_figures_selected"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

PROGRESS_CSV = BASE_DIR / "progress.csv"
STAGE_EVAL_CSV = BASE_DIR / "stage_eval_rewards.csv"


# ============================================================
# Helpers
# ============================================================

def to_float(x):
    if x is None or x == "":
        return None
    try:
        return float(x)
    except (ValueError, TypeError):
        return None


def moving_average(values, window=10):
    result = []
    for i in range(len(values)):
        start = max(0, i - window + 1)
        valid = [v for v in values[start:i + 1] if v is not None]
        result.append(sum(valid) / len(valid) if valid else None)
    return result


def filter_none(x, y):
    pairs = [(a, b) for a, b in zip(x, y) if a is not None and b is not None]
    return [p[0] for p in pairs], [p[1] for p in pairs]


def read_csv_dicts(path):
    if not path.exists():
        raise FileNotFoundError(f"Required file not found: {path}")
    with open(path, "r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def save_current_figure(filename):
    out_path = OUTPUT_DIR / filename
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"Saved: {out_path}")


def sort_rows_by_numeric_key(rows, key):
    def safe_key(row):
        value = to_float(row.get(key))
        return value if value is not None else float("inf")
    return sorted(rows, key=safe_key)


def get_latest_value_before_or_near(source_x, source_y, target_x):
    """
    For each target_x (evaluation timestep), get the last valid y from training log
    with step <= target_x. If none exists, use the nearest valid point.
    """
    valid_pairs = [(x, y) for x, y in zip(source_x, source_y) if x is not None and y is not None]
    if not valid_pairs:
        return None

    valid_pairs.sort(key=lambda p: p[0])

    candidates = [(x, y) for x, y in valid_pairs if x <= target_x]
    if candidates:
        return candidates[-1][1]

    # fallback: nearest point if all x > target_x
    nearest = min(valid_pairs, key=lambda p: abs(p[0] - target_x))
    return nearest[1]


# ============================================================
# Read training log
# ============================================================

progress_rows = read_csv_dicts(PROGRESS_CSV)

train_steps = []
train_rewards = []
train_value_loss = []
train_explained_variance = []

for row in progress_rows:
    step = to_float(row.get("time/total_timesteps"))
    if step is None:
        continue

    train_steps.append(step)
    train_rewards.append(to_float(row.get("rollout/ep_rew_mean")))
    train_value_loss.append(to_float(row.get("train/value_loss")))
    train_explained_variance.append(to_float(row.get("train/explained_variance")))


# ============================================================
# Read evaluation log
# ============================================================

eval_rows = sort_rows_by_numeric_key(read_csv_dicts(STAGE_EVAL_CSV), "timesteps")

eval_timesteps = [to_float(r.get("timesteps")) for r in eval_rows]
eval_stages = [to_float(r.get("stage")) for r in eval_rows]
eval_avg_reward = [to_float(r.get("avg_eval_reward")) for r in eval_rows]
eval_std_reward = [to_float(r.get("std_eval_reward")) for r in eval_rows]

# Since stage_eval_rewards.csv does not contain evaluation value loss or
# evaluation explained variance directly, we sample the nearest stage-aligned
# value from progress.csv at each evaluation timestep.
eval_stage_value_loss = [
    get_latest_value_before_or_near(train_steps, train_value_loss, t)
    if t is not None else None
    for t in eval_timesteps
]

eval_stage_explained_variance = [
    get_latest_value_before_or_near(train_steps, train_explained_variance, t)
    if t is not None else None
    for t in eval_timesteps
]


# ============================================================
# 1-3. Training curves
# ============================================================

def plot_training_curve(steps, values, ylabel, title, filename, label, window=10):
    values_ma = moving_average(values, window=window)

    x_raw, y_raw = filter_none(steps, values)
    x_ma, y_ma = filter_none(steps, values_ma)

    plt.figure(figsize=(8, 5))
    plt.plot(x_raw, y_raw, linewidth=1, alpha=0.35, label=label)
    plt.plot(x_ma, y_ma, linewidth=2, label=f"Moving average (window={window})")
    plt.xlabel("Training timesteps")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    plt.grid(True, alpha=0.3)
    save_current_figure(filename)


plot_training_curve(
    train_steps,
    train_rewards,
    "Mean episode reward",
    "PPO Training Reward Curve",
    "training_reward_curve.png",
    "Episode reward mean",
)

plot_training_curve(
    train_steps,
    train_value_loss,
    "Value loss",
    "PPO Training Value Loss Curve",
    "training_value_loss_curve.png",
    "Value loss",
)

plot_training_curve(
    train_steps,
    train_explained_variance,
    "Explained variance",
    "PPO Training Explained Variance Curve",
    "training_explained_variance_curve.png",
    "Explained variance",
)


# ============================================================
# 4-6. Evaluation curves
# ============================================================

# 4. Evaluation reward curve with std band
valid = [
    (x, y, s)
    for x, y, s in zip(eval_timesteps, eval_avg_reward, eval_std_reward)
    if x is not None and y is not None
]

if valid:
    x = [v[0] for v in valid]
    y = [v[1] for v in valid]
    std = [0.0 if v[2] is None else v[2] for v in valid]
    lower = [a - b for a, b in zip(y, std)]
    upper = [a + b for a, b in zip(y, std)]

    plt.figure(figsize=(8, 5))
    plt.plot(x, y, marker="o", linewidth=2, label="Mean evaluation reward")
    plt.fill_between(x, lower, upper, alpha=0.2, label="Mean ± 1 std")
    plt.xlabel("Training timesteps")
    plt.ylabel("Average evaluation episode reward")
    plt.title("PPO Evaluation Reward Curve")
    plt.legend()
    plt.grid(True, alpha=0.3)
    save_current_figure("eval_reward_curve_with_std.png")


# 5. Stage-aligned value loss curve
x_eval_v, y_eval_v = filter_none(eval_timesteps, eval_stage_value_loss)
if x_eval_v:
    plt.figure(figsize=(8, 5))
    plt.plot(x_eval_v, y_eval_v, marker="o", linewidth=2, label="Stage-aligned value loss")
    plt.xlabel("Training timesteps")
    plt.ylabel("Value loss")
    plt.title("Evaluation-Stage Value Loss Curve")
    plt.legend()
    plt.grid(True, alpha=0.3)
    save_current_figure("eval_value_loss_curve.png")


# 6. Stage-aligned explained variance curve
x_eval_e, y_eval_e = filter_none(eval_timesteps, eval_stage_explained_variance)
if x_eval_e:
    plt.figure(figsize=(8, 5))
    plt.plot(x_eval_e, y_eval_e, marker="o", linewidth=2, label="Stage-aligned explained variance")
    plt.xlabel("Training timesteps")
    plt.ylabel("Explained variance")
    plt.title("Evaluation-Stage Explained Variance Curve")
    plt.legend()
    plt.grid(True, alpha=0.3)
    save_current_figure("eval_explained_variance_curve.png")


print("\nDone. Six figures are saved in:")
print(OUTPUT_DIR)
print("\nNote:")
print("The evaluation reward curve is drawn directly from stage_eval_rewards.csv.")
print("The evaluation-stage value loss and explained variance curves are sampled")
print("from progress.csv at the corresponding evaluation timesteps, because")
print("stage_eval_rewards.csv does not contain these two metrics directly.")
