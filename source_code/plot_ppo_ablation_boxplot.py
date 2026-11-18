import csv
from pathlib import Path

import matplotlib.pyplot as plt


BASE_DIR = Path(__file__).resolve().parent
RUNS_DIR = BASE_DIR / "ppo_runs"

OUTPUT_DIR = BASE_DIR / "ppo_ablation_figures"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


CONFIG_LABELS = {
    "full_6action": "PPO-6action",
    "ablate_action_0": "PPO-w/o A0",
    "ablate_action_1": "PPO-w/o A1",
    "ablate_action_2": "PPO-w/o A2",
    "ablate_action_3": "PPO-w/o A3",
    "ablate_action_4": "PPO-w/o A4",
    "ablate_action_5": "PPO-w/o A5",
}


def to_float(x):
    if x is None or x == "":
        return None
    try:
        return float(x)
    except ValueError:
        return None


def read_csv_dicts(path):
    with open(path, "r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def load_ppo_tardiness_by_config():
    data = {}
    missing = []

    for config_name, label in CONFIG_LABELS.items():
        csv_path = RUNS_DIR / config_name / "logs" / "final_eval_raw_results.csv"

        if not csv_path.exists():
            missing.append(str(csv_path))
            continue

        rows = read_csv_dicts(csv_path)
        values = []

        for row in rows:
            method = row.get("method", "")

            # 每个 final_eval_raw_results.csv 里可能同时有 PPO 和 Fixed-action。
            # 这里只取 PPO 这一组。
            if not method.startswith("PPO"):
                continue

            tardiness = to_float(row.get("tardiness"))
            if tardiness is not None:
                values.append(tardiness)

        if values:
            data[label] = values

    if missing:
        print("Missing files:")
        for path in missing:
            print(path)

    return data


def plot_boxplot(data):
    labels = list(data.keys())
    values = [data[label] for label in labels]

    plt.figure(figsize=(11, 5.8))

    plt.boxplot(
        values,
        tick_labels=labels,
        showmeans=True,
        patch_artist=True,
        meanline=False,
    )

    plt.xlabel("Method")
    plt.ylabel("Weighted total tardiness")
    plt.title("Weighted Total Tardiness Distribution of PPO Ablation Models")
    plt.xticks(rotation=30, ha="right")
    plt.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()

    output_path = OUTPUT_DIR / "ppo_ablation_tardiness_boxplot.png"
    plt.savefig(output_path, dpi=300)
    plt.close()

    print(f"Saved: {output_path}")


if __name__ == "__main__":
    data = load_ppo_tardiness_by_config()

    print("Loaded data:")
    for label, values in data.items():
        print(f"{label}: {len(values)} samples")

    if len(data) < 2:
        raise RuntimeError("At least two PPO configs are required for boxplot.")

    plot_boxplot(data)
