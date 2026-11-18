import csv
import matplotlib.pyplot as plt

csv_path = "progress.csv"

steps = []
rewards = []
value_loss = []
explained_variance = []
approx_kl = []
clip_fraction = []
entropy_loss = []

def to_float(x):
    if x is None or x == "":
        return None
    try:
        return float(x)
    except ValueError:
        return None

with open(csv_path, "r", encoding="utf-8-sig", newline="") as f:
    reader = csv.DictReader(f)
    for row in reader:
        step = to_float(row.get("time/total_timesteps"))
        if step is None:
            continue

        steps.append(step)
        rewards.append(to_float(row.get("rollout/ep_rew_mean")))
        value_loss.append(to_float(row.get("train/value_loss")))
        explained_variance.append(to_float(row.get("train/explained_variance")))
        approx_kl.append(to_float(row.get("train/approx_kl")))
        clip_fraction.append(to_float(row.get("train/clip_fraction")))
        entropy_loss.append(to_float(row.get("train/entropy_loss")))

def moving_average(values, window=10):
    result = []
    for i in range(len(values)):
        start = max(0, i - window + 1)
        valid = [v for v in values[start:i+1] if v is not None]
        result.append(sum(valid) / len(valid) if valid else None)
    return result

def filter_none(x, y):
    pairs = [(a, b) for a, b in zip(x, y) if b is not None]
    return [p[0] for p in pairs], [p[1] for p in pairs]

# 1. Reward curve
reward_ma = moving_average(rewards, window=10)
x1, y1 = filter_none(steps, rewards)
x1_ma, y1_ma = filter_none(steps, reward_ma)

plt.figure(figsize=(8, 5))
plt.plot(x1, y1, linewidth=1, alpha=0.35, label="Episode reward mean")
plt.plot(x1_ma, y1_ma, linewidth=2, label="Moving average")
plt.xlabel("Training timesteps")
plt.ylabel("Mean episode reward")
plt.title("PPO Training Reward Curve")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("ppo_reward_curve.png", dpi=300)
plt.close()

# 2. Value loss curve
x2, y2 = filter_none(steps, value_loss)
value_loss_ma = moving_average(value_loss, window=10)
x2_ma, y2_ma = filter_none(steps, value_loss_ma)

plt.figure(figsize=(8, 5))
plt.plot(x2, y2, linewidth=1, alpha=0.35, label="Value loss")
plt.plot(x2_ma, y2_ma, linewidth=2, label="Moving average")
plt.xlabel("Training timesteps")
plt.ylabel("Value loss")
plt.title("PPO Value Loss Curve")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("ppo_value_loss_curve.png", dpi=300)
plt.close()

# 3. Explained variance curve
x3, y3 = filter_none(steps, explained_variance)
explained_ma = moving_average(explained_variance, window=10)
x3_ma, y3_ma = filter_none(steps, explained_ma)

plt.figure(figsize=(8, 5))
plt.plot(x3, y3, linewidth=1, alpha=0.35, label="Explained variance")
plt.plot(x3_ma, y3_ma, linewidth=2, label="Moving average")
plt.xlabel("Training timesteps")
plt.ylabel("Explained variance")
plt.title("PPO Explained Variance Curve")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("ppo_explained_variance_curve.png", dpi=300)
plt.close()

print("Figures saved:")
print("ppo_reward_curve.png")
print("ppo_value_loss_curve.png")
print("ppo_explained_variance_curve.png")
