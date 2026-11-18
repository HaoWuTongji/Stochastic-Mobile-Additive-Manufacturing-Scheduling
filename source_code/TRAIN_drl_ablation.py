import os
import csv
import contextlib
import io
from datetime import datetime

import numpy as np

from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.logger import configure

from MAM_DRL_ENV_ablation import MobileAMEnv, ENV_VERSION

try:
    from MAM_DRL_ENV_ablation import ACTION_RULES
except ImportError:
    ACTION_RULES = {
        0: {
            "name": "nearest + EDD",
            "route_rule": "nearest",
            "print_rule": "EDD",
        },
        1: {
            "name": "urgent + EDD",
            "route_rule": "urgent",
            "print_rule": "EDD",
        },
        2: {
            "name": "nearest + SPT",
            "route_rule": "nearest",
            "print_rule": "SPT",
        },
        3: {
            "name": "risk + SPT",
            "route_rule": "risk",
            "print_rule": "SPT",
        },
        4: {
            "name": "batch_gain + CR",
            "route_rule": "batch_gain",
            "print_rule": "CR",
        },
        5: {
            "name": "nearest + same_node_cluster",
            "route_rule": "nearest",
            "print_rule": "same_node_cluster",
        },
    }


# ============================================================
# Basic experiment settings
# ============================================================

INSTANCE_PATH = "instance/problem_instance_10cust_20parts.json"

STAGE_TIMESTEPS = 100000
N_STAGES = 5
MAX_EPISODE_STEPS = 1000

TRAIN_SEED = 43

# Seeds used for selecting the best checkpoint during training.
STAGE_EVAL_SEEDS = list(range(200, 210))

# Seeds used for final evaluation. 100-149 means 50 independent tests.
EVAL_SEEDS = list(range(100, 150))

# Select one config here when you only want to run one experiment.
CURRENT_CONFIG_NAME = "full_6action"

# If True, train all configs in ACTION_CONFIGS one by one.
# For thesis experiments this is useful, but it can take a long time.
RUN_ALL_CONFIGS = False

ACTION_CONFIGS = {
    "full_6action": [0, 1, 2, 3, 4, 5],

    "ablate_action_0": [1, 2, 3, 4, 5],
    "ablate_action_1": [0, 2, 3, 4, 5],
    "ablate_action_2": [0, 1, 3, 4, 5],
    "ablate_action_3": [0, 1, 2, 4, 5],
    "ablate_action_4": [0, 1, 2, 3, 5],
    "ablate_action_5": [0, 1, 2, 3, 4],
}

RUN_TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")


# ============================================================
# Utility functions
# ============================================================

def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def get_action_name(global_action):
    if global_action in ACTION_RULES:
        return ACTION_RULES[global_action].get("name", f"Action {global_action}")
    return f"Action {global_action}"


def get_paper_action(global_action):
    """
    Program action ids are 0-5. In the thesis, you may call them Action 1-6.
    """
    return int(global_action) + 1


def build_paths(config_name):
    """
    Keep every ablation experiment in an independent folder to avoid overwriting.
    """
    base_dir = os.path.join("ppo_runs", config_name)
    model_dir = os.path.join(base_dir, "models")
    log_dir = os.path.join(base_dir, "logs")
    tensorboard_dir = os.path.join(base_dir, "tensorboard")
    train_log_dir = os.path.join(log_dir, "ppo_train_logs")
    eval_log_dir = os.path.join(log_dir, "ppo_eval_logs")

    for d in [base_dir, model_dir, log_dir, tensorboard_dir, train_log_dir, eval_log_dir]:
        ensure_dir(d)

    return {
        "base_dir": base_dir,
        "model_dir": model_dir,
        "log_dir": log_dir,
        "tensorboard_dir": tensorboard_dir,
        "train_log_dir": train_log_dir,
        "eval_log_dir": eval_log_dir,
        "final_model": os.path.join(model_dir, f"mam_ppo_{config_name}_final"),
        "best_model": os.path.join(model_dir, f"mam_ppo_{config_name}_best"),
        "result_log": os.path.join(log_dir, f"eval_summary_{RUN_TIMESTAMP}.txt"),
        "stage_eval_csv": os.path.join(eval_log_dir, "stage_eval_rewards.csv"),
        "stage_eval_per_seed_csv": os.path.join(eval_log_dir, "stage_eval_rewards_per_seed.csv"),
        "decision_log_csv": os.path.join(eval_log_dir, "ppo_decision_logs.csv"),
    }

def get_ppo_method_name(config_name):
    if config_name == "full_6action":
        return "PPO-6action"

    if config_name.startswith("ablate_action_"):
        action_id = config_name.replace("ablate_action_", "")
        return f"PPO-w/o-A{action_id}"

    return f"PPO-{config_name}"


def create_env(active_actions, seed=None, monitor=True):
    """
    Create one environment.

    This script assumes MobileAMEnv supports:
        MobileAMEnv(instance_path=..., active_actions=...)

    If your MAM_DRL_ENV.py has not been updated yet, add active_actions support
    in MobileAMEnv first; otherwise ablation experiments cannot be performed
    correctly.
    """
    env = MobileAMEnv(
        instance_path=INSTANCE_PATH,
        active_actions=active_actions,
    )

    if seed is not None:
        env.reset(seed=seed)
        env.action_space.seed(seed)

    if monitor:
        env = Monitor(env)

    return env


def make_train_env(active_actions, seed=TRAIN_SEED):
    """
    Used by DummyVecEnv.
    """
    def _init():
        return create_env(
            active_actions=active_actions,
            seed=seed,
            monitor=True,
        )
    return _init


def write_experiment_header(result_log_path, config_name, active_actions):
    lines = []
    lines.append(f"Environment version: {ENV_VERSION}")
    lines.append(f"Instance path: {INSTANCE_PATH}")
    lines.append(f"Config name: {config_name}")
    lines.append(f"Active global actions: {active_actions}")
    lines.append("Active action meanings:")
    for global_action in active_actions:
        lines.append(
            f"  local={active_actions.index(global_action)}, "
            f"global={global_action}, "
            f"paper_action={get_paper_action(global_action)}, "
            f"name={get_action_name(global_action)}"
        )
    lines.append(f"Train seed: {TRAIN_SEED}")
    lines.append(f"Stage eval seeds: {STAGE_EVAL_SEEDS}")
    lines.append(f"Final eval seeds: {EVAL_SEEDS}")
    lines.append(f"Stage timesteps: {STAGE_TIMESTEPS}")
    lines.append(f"Number of stages: {N_STAGES}")
    lines.append("")

    with open(result_log_path, "a", encoding="utf-8") as f:
        f.write("\n".join(lines))
        f.write("\n")


def append_stage_eval_summary(csv_path, row):
    file_exists = os.path.exists(csv_path)

    fieldnames = [
        "config_name",
        "stage",
        "timesteps",
        "avg_eval_reward",
        "std_eval_reward",
        "avg_eval_tardiness",
        "std_eval_tardiness",
        "avg_eval_steps",
        "completion_rate",
    ]

    with open(csv_path, "a", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)


def append_stage_eval_per_seed(csv_path, config_name, stage, timesteps, stage_results):
    file_exists = os.path.exists(csv_path)

    fieldnames = [
        "config_name",
        "stage",
        "timesteps",
        "seed",
        "episode_reward",
        "tardiness",
        "steps",
        "done",
    ]

    with open(csv_path, "a", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()

        for r in stage_results:
            writer.writerow({
                "config_name": config_name,
                "stage": stage,
                "timesteps": timesteps,
                "seed": r["seed"],
                "episode_reward": r["total_reward"],
                "tardiness": r["tardiness"],
                "steps": r["steps"],
                "done": r["done"],
            })


# ============================================================
# Core evaluation functions
# ============================================================

def evaluate_model(
    model,
    seeds,
    active_actions,
    deterministic=True,
    print_each_seed=True,
):
    """
    Evaluate PPO model on multiple seeds.

    Important:
    - local_action is the PPO output: 0 to len(active_actions)-1.
    - global_action is the original action id in the full action space: 0 to 5.
    - action_count is counted by global_action, so ablation logs are not confusing.
    """
    results = []

    for seed in seeds:
        env = create_env(active_actions=active_actions, seed=seed, monitor=False)
        obs, info = env.reset(seed=seed)

        done = False
        total_reward = 0.0
        step_count = 0
        info = {}

        action_count = {global_action: 0 for global_action in active_actions}

        while not done and step_count < MAX_EPISODE_STEPS:
            local_action, _ = model.predict(obs, deterministic=deterministic)
            local_action = int(local_action)

            global_action = active_actions[local_action]
            action_count[global_action] += 1

            with contextlib.redirect_stdout(io.StringIO()):
                obs, reward, terminated, truncated, info = env.step(local_action)

            done = bool(terminated or truncated)
            total_reward += float(reward)
            step_count += 1

        final_tardiness = info.get("tardiness", env._calculate_total_tardiness())

        result = {
            "seed": seed,
            "tardiness": final_tardiness,
            "total_reward": total_reward,
            "steps": step_count,
            "done": done,
            "action_count": action_count,
        }
        results.append(result)

        if print_each_seed:
            print(
                f"[PPO] seed={seed}, "
                f"tardiness={final_tardiness:.2f}, "
                f"reward={total_reward:.2f}, "
                f"steps={step_count}, "
                f"done={done}, "
                f"global_actions={action_count}"
            )

    return results


def evaluate_fixed_global_action(global_action, seeds):
    """
    Test one fixed original/global action.

    Here the environment is constructed with only this action active.
    Therefore local action is always 0, and it maps to the required global action.
    """
    active_actions = [global_action]
    results = []

    for seed in seeds:
        env = create_env(active_actions=active_actions, seed=seed, monitor=False)
        obs, info = env.reset(seed=seed)

        done = False
        total_reward = 0.0
        step_count = 0
        info = {}

        while not done and step_count < MAX_EPISODE_STEPS:
            local_action = 0

            with contextlib.redirect_stdout(io.StringIO()):
                obs, reward, terminated, truncated, info = env.step(local_action)

            done = bool(terminated or truncated)
            total_reward += float(reward)
            step_count += 1

        final_tardiness = info.get("tardiness", env._calculate_total_tardiness())

        results.append({
            "seed": seed,
            "global_action": global_action,
            "paper_action": get_paper_action(global_action),
            "action_name": get_action_name(global_action),
            "tardiness": final_tardiness,
            "total_reward": total_reward,
            "steps": step_count,
            "done": done,
        })

    return results


def benchmark_fixed_actions(seeds, result_log_path):
    """
    Test six original fixed actions.
    These are not affected by the current ablation config.
    """
    all_benchmark_results = {}

    for global_action in sorted(ACTION_RULES.keys()):
        results = evaluate_fixed_global_action(global_action, seeds)
        all_benchmark_results[global_action] = results
        summarize_results(
            name=(
                f"Fixed Global Action {global_action} "
                f"(Paper Action {get_paper_action(global_action)}: {get_action_name(global_action)})"
            ),
            results=results,
            result_log_path=result_log_path,
            active_actions=[global_action],
        )

    return all_benchmark_results


def summarize_results(name, results, result_log_path, active_actions=None):
    """
    Print and save summary.
    """
    tardiness_values = np.array([r["tardiness"] for r in results], dtype=float)
    reward_values = np.array([r["total_reward"] for r in results], dtype=float)
    step_values = np.array([r["steps"] for r in results], dtype=float)
    done_values = np.array([1 if r["done"] else 0 for r in results], dtype=float)

    lines = []
    lines.append(f"\n--- {name} Summary ---")
    lines.append(f"Test count: {len(results)}")
    lines.append(f"Average total tardiness: {np.mean(tardiness_values):.2f}")
    lines.append(f"Tardiness std: {np.std(tardiness_values):.2f}")
    lines.append(f"Minimum tardiness: {np.min(tardiness_values):.2f}")
    lines.append(f"Maximum tardiness: {np.max(tardiness_values):.2f}")
    lines.append(f"Average total reward: {np.mean(reward_values):.2f}")
    lines.append(f"Average steps: {np.mean(step_values):.2f}")
    lines.append(f"Completion rate: {np.mean(done_values) * 100:.1f}%")

    if "action_count" in results[0]:
        if active_actions is None:
            active_actions = sorted(results[0]["action_count"].keys())

        total_action_count = {global_action: 0 for global_action in active_actions}
        for r in results:
            for global_action, count in r["action_count"].items():
                total_action_count[global_action] = total_action_count.get(global_action, 0) + count

        lines.append("Action usage count by global action id:")
        for global_action in sorted(total_action_count.keys()):
            lines.append(
                f"  global={global_action}, "
                f"paper_action={get_paper_action(global_action)}, "
                f"name={get_action_name(global_action)}, "
                f"count={total_action_count[global_action]}"
            )

    summary_text = "\n".join(lines)
    print(summary_text)

    with open(result_log_path, "a", encoding="utf-8") as f:
        f.write(summary_text)
        f.write("\n")


def compare_ppo_with_baseline(
    ppo_results,
    baseline_results,
    result_log_path,
    baseline_name="Fixed Global Action 0",
):
    baseline_map = {r["seed"]: r for r in baseline_results}

    better_count = 0
    worse_count = 0
    equal_count = 0
    diffs = []

    lines = []
    lines.append(f"\n--- PPO vs {baseline_name} by seed ---")

    for r in ppo_results:
        seed = r["seed"]
        if seed not in baseline_map:
            continue

        ppo_t = r["tardiness"]
        base_t = baseline_map[seed]["tardiness"]
        diff = ppo_t - base_t
        diffs.append(diff)

        if diff < 0:
            better_count += 1
            flag = "PPO better"
        elif diff > 0:
            worse_count += 1
            flag = "PPO worse"
        else:
            equal_count += 1
            flag = "Equal"

        lines.append(
            f"seed={seed}, "
            f"PPO={ppo_t:.2f}, "
            f"{baseline_name}={base_t:.2f}, "
            f"diff={diff:.2f}, "
            f"{flag}, "
            f"actions={r.get('action_count', {})}"
        )

    diffs = np.array(diffs, dtype=float)

    lines.append("\n--- Comparison summary ---")
    lines.append(f"PPO better count: {better_count}")
    lines.append(f"PPO worse count: {worse_count}")
    lines.append(f"Equal count: {equal_count}")
    if len(diffs) > 0:
        lines.append(f"Average diff PPO - baseline: {np.mean(diffs):.2f}")

    text = "\n".join(lines)
    print(text)

    with open(result_log_path, "a", encoding="utf-8") as f:
        f.write(text)
        f.write("\n")


def save_final_raw_results(output_csv, method_name, results, append=True):
    file_exists = os.path.exists(output_csv)

    mode = "a" if append else "w"

    with open(output_csv, mode, newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "method",
                "seed",
                "tardiness",
                "total_reward",
                "steps",
                "done",
            ]
        )

        if (not file_exists) or (not append):
            writer.writeheader()

        for r in results:
            writer.writerow({
                "method": method_name,
                "seed": r["seed"],
                "tardiness": r["tardiness"],
                "total_reward": r["total_reward"],
                "steps": r["steps"],
                "done": r["done"],
            })

# ============================================================
# Decision log and observation inspection
# ============================================================

def get_decision_context(env):
    """
    Extract interpretable state information at a decision point.
    """
    unassigned = [o for o in env.orders.values() if o.status == "unassigned"]
    printing = [o for o in env.orders.values() if o.status == "printing"]
    printed = [o for o in env.orders.values() if o.status == "printed"]
    delivered = [o for o in env.orders.values() if o.status == "delivered"]

    current_batch = list(env.printer.current_batch) if env.printer.is_printing else []

    if printed:
        route_orders = printed
        route_type = "printed"
    elif current_batch:
        route_orders = current_batch
        route_type = "current_batch"
    else:
        route_orders = []
        route_type = "none"

    if route_orders:
        min_due = min(o.due_date for o in route_orders)
        min_slack = min(o.due_date - env.time for o in route_orders)
        node_count = len(set(o.node_id for o in route_orders))
        order_count = len(route_orders)
    else:
        min_due = None
        min_slack = None
        node_count = 0
        order_count = 0

    return {
        "time": env.time,
        "truck_state": env.truck.state,
        "truck_node": env.truck.to_node,
        "planned_target": getattr(env.truck, "planned_target", None),
        "printer_busy": env.printer.is_printing,
        "route_type": route_type,
        "route_order_count": order_count,
        "route_node_count": node_count,
        "min_due": min_due,
        "min_slack": min_slack,
        "num_unassigned": len(unassigned),
        "num_printing": len(printing),
        "num_printed": len(printed),
        "num_delivered": len(delivered),
    }


def analyze_policy_decisions(
    model,
    seeds,
    active_actions,
    output_csv,
):
    """
    Save PPO decision logs.
    The output includes both local and global action ids.
    """
    rows = []

    for seed in seeds:
        env = create_env(active_actions=active_actions, seed=seed, monitor=False)
        obs, info = env.reset(seed=seed)

        done = False
        step_count = 0
        info = {}

        while not done and step_count < MAX_EPISODE_STEPS:
            context_before = get_decision_context(env)

            local_action, _ = model.predict(obs, deterministic=True)
            local_action = int(local_action)
            global_action = active_actions[local_action]

            with contextlib.redirect_stdout(io.StringIO()):
                obs, reward, terminated, truncated, info = env.step(local_action)

            done = bool(terminated or truncated)
            context_after = get_decision_context(env)

            row = {
                "seed": seed,
                "step": step_count,
                "local_action": local_action,
                "global_action": global_action,
                "paper_action": get_paper_action(global_action),
                "action_name": get_action_name(global_action),
                "reward": float(reward),
                "tardiness_after_step": info.get("tardiness", None),

                "pre_time": context_before["time"],
                "pre_truck_state": context_before["truck_state"],
                "pre_truck_node": context_before["truck_node"],
                "pre_planned_target": context_before["planned_target"],
                "pre_printer_busy": context_before["printer_busy"],
                "pre_route_type": context_before["route_type"],
                "pre_route_order_count": context_before["route_order_count"],
                "pre_route_node_count": context_before["route_node_count"],
                "pre_min_due": context_before["min_due"],
                "pre_min_slack": context_before["min_slack"],
                "pre_num_unassigned": context_before["num_unassigned"],
                "pre_num_printing": context_before["num_printing"],
                "pre_num_printed": context_before["num_printed"],
                "pre_num_delivered": context_before["num_delivered"],

                "post_time": context_after["time"],
                "post_truck_state": context_after["truck_state"],
                "post_truck_node": context_after["truck_node"],
                "post_planned_target": context_after["planned_target"],
                "post_printer_busy": context_after["printer_busy"],
                "post_route_type": context_after["route_type"],
                "post_route_order_count": context_after["route_order_count"],
                "post_route_node_count": context_after["route_node_count"],
                "post_min_due": context_after["min_due"],
                "post_min_slack": context_after["min_slack"],
                "post_num_unassigned": context_after["num_unassigned"],
                "post_num_printing": context_after["num_printing"],
                "post_num_printed": context_after["num_printed"],
                "post_num_delivered": context_after["num_delivered"],

                "done": done,
            }

            rows.append(row)
            step_count += 1

        final_tardiness = info.get("tardiness", env._calculate_total_tardiness())

        for row in rows:
            if row["seed"] == seed:
                row["final_tardiness"] = final_tardiness

    if not rows:
        print("No decision data recorded.")
        return

    fieldnames = list(rows[0].keys())
    with open(output_csv, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Decision log saved to: {output_csv}")


def inspect_observations(seeds, active_actions, actions=None, max_steps=1000):
    """
    Inspect observation vectors.
    """
    if actions is None:
        actions = list(range(len(active_actions))) + ["random"]

    all_obs = []
    all_raw_obs = []
    all_obs_low = []
    all_obs_high = []

    for mode in actions:
        for seed in seeds:
            env = create_env(active_actions=active_actions, seed=seed, monitor=False)

            with contextlib.redirect_stdout(io.StringIO()):
                obs, info = env.reset(seed=seed)

            done = False
            step_count = 0

            while not done and step_count < max_steps:
                all_obs.append(np.array(obs, dtype=np.float32))

                if hasattr(env, "last_raw_obs"):
                    all_raw_obs.append(np.array(env.last_raw_obs, dtype=np.float32))
                    all_obs_low.append(np.array(env.last_obs_low, dtype=bool))
                    all_obs_high.append(np.array(env.last_obs_high, dtype=bool))

                if mode == "random":
                    local_action = env.action_space.sample()
                else:
                    local_action = int(mode)

                with contextlib.redirect_stdout(io.StringIO()):
                    obs, reward, terminated, truncated, info = env.step(local_action)

                done = bool(terminated or truncated)
                step_count += 1

    all_obs = np.array(all_obs, dtype=np.float32)

    print("\n========== Observation Inspection: Obs ==========")
    print(f"sample count: {len(all_obs)}")
    print(f"obs shape: {all_obs.shape}")
    print(f"nan count: {np.isnan(all_obs).sum()}")
    print(f"inf count: {np.isinf(all_obs).sum()}")

    for i in range(all_obs.shape[1]):
        col = all_obs[:, i]
        print(
            f"obs[{i:02d}] "
            f"min={np.min(col): .4f}, "
            f"max={np.max(col): .4f}, "
            f"mean={np.mean(col): .4f}, "
            f"std={np.std(col): .4f}, "
            f"zero_ratio={np.mean(col == 0.0): .2%}, "
            f"one_ratio={np.mean(col == 1.0): .2%}, "
            f"neg_one_ratio={np.mean(col == -1.0): .2%}"
        )

    if all_raw_obs:
        all_raw_obs = np.array(all_raw_obs, dtype=np.float32)
        all_obs_low = np.array(all_obs_low, dtype=bool)
        all_obs_high = np.array(all_obs_high, dtype=bool)

        print("\n========== Observation Inspection: Raw Obs Before Clip ==========")
        print(f"raw sample count: {len(all_raw_obs)}")
        print(f"raw nan count: {np.isnan(all_raw_obs).sum()}")
        print(f"raw inf count: {np.isinf(all_raw_obs).sum()}")

        for i in range(all_raw_obs.shape[1]):
            col = all_raw_obs[:, i]
            print(
                f"raw[{i:02d}] "
                f"min={np.min(col): .4f}, "
                f"max={np.max(col): .4f}, "
                f"mean={np.mean(col): .4f}, "
                f"std={np.std(col): .4f}, "
                f"below_low={np.mean(all_obs_low[:, i]): .2%}, "
                f"above_high={np.mean(all_obs_high[:, i]): .2%}"
            )


# ============================================================
# Training and testing
# ============================================================

def train_one_config(config_name):
    if config_name not in ACTION_CONFIGS:
        raise ValueError(f"Unknown config_name: {config_name}")

    active_actions = ACTION_CONFIGS[config_name]
    paths = build_paths(config_name)

    print(f"\n================ Training config: {config_name} ================")
    print(f"Using environment version: {ENV_VERSION}")
    print(f"Active global actions: {active_actions}")
    for local_id, global_id in enumerate(active_actions):
        print(
            f"  local={local_id}, "
            f"global={global_id}, "
            f"paper_action={get_paper_action(global_id)}, "
            f"name={get_action_name(global_id)}"
        )

    write_experiment_header(paths["result_log"], config_name, active_actions)

    # Check environment only once for this config.
    check_env_raw = create_env(active_actions=active_actions, seed=TRAIN_SEED, monitor=False)
    print("Checking environment...")
    check_env(check_env_raw)

    train_env = DummyVecEnv([
        make_train_env(active_actions=active_actions, seed=TRAIN_SEED)
    ])

    model = PPO(
        "MlpPolicy",
        train_env,
        verbose=1,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=64,
        ent_coef=0.01,
        seed=TRAIN_SEED,
        device="auto",
        tensorboard_log=paths["tensorboard_dir"],
    )

    new_logger = configure(
        paths["train_log_dir"],
        ["stdout", "csv", "tensorboard"],
    )
    model.set_logger(new_logger)

    best_tardiness = float("inf")

    print("Start staged PPO training...")

    for stage in range(N_STAGES):
        stage_id = stage + 1
        print(f"\n========== Training stage {stage_id}/{N_STAGES} ==========")

        model.learn(
            total_timesteps=STAGE_TIMESTEPS,
            reset_num_timesteps=(stage == 0),
        )

        checkpoint_path = os.path.join(paths["model_dir"], f"checkpoint_stage_{stage_id}")
        model.save(checkpoint_path)
        print(f"Stage {stage_id} checkpoint saved: {checkpoint_path}")

        stage_results = evaluate_model(
            model=model,
            seeds=STAGE_EVAL_SEEDS,
            active_actions=active_actions,
            deterministic=True,
            print_each_seed=False,
        )

        tardiness_values = np.array([r["tardiness"] for r in stage_results], dtype=float)
        reward_values = np.array([r["total_reward"] for r in stage_results], dtype=float)
        step_values = np.array([r["steps"] for r in stage_results], dtype=float)
        done_values = np.array([1 if r["done"] else 0 for r in stage_results], dtype=float)

        avg_tardiness = float(np.mean(tardiness_values))
        avg_reward = float(np.mean(reward_values))

        current_timesteps = stage_id * STAGE_TIMESTEPS

        append_stage_eval_summary(
            paths["stage_eval_csv"],
            {
                "config_name": config_name,
                "stage": stage_id,
                "timesteps": current_timesteps,
                "avg_eval_reward": avg_reward,
                "std_eval_reward": float(np.std(reward_values)),
                "avg_eval_tardiness": avg_tardiness,
                "std_eval_tardiness": float(np.std(tardiness_values)),
                "avg_eval_steps": float(np.mean(step_values)),
                "completion_rate": float(np.mean(done_values)),
            },
        )

        append_stage_eval_per_seed(
            paths["stage_eval_per_seed_csv"],
            config_name=config_name,
            stage=stage_id,
            timesteps=current_timesteps,
            stage_results=stage_results,
        )

        print(
            f"[Stage {stage_id}] "
            f"avg_tardiness={avg_tardiness:.2f}, "
            f"avg_reward={avg_reward:.2f}"
        )

        if avg_tardiness < best_tardiness:
            best_tardiness = avg_tardiness
            model.save(paths["best_model"])
            print(
                f"New best model saved: {paths['best_model']}, "
                f"best_avg_tardiness={best_tardiness:.2f}"
            )

    model.save(paths["final_model"])
    print(f"\nFinal model saved: {paths['final_model']}")
    print(f"Best model saved: {paths['best_model']}")

    print("\n================ Best PPO final multi-seed test ================")
    best_model = PPO.load(paths["best_model"])

    ppo_results = evaluate_model(
        model=best_model,
        seeds=EVAL_SEEDS,
        active_actions=active_actions,
        deterministic=True,
        print_each_seed=True,
    )

    summarize_results(
        name=f"Best PPO - {config_name}",
        results=ppo_results,
        result_log_path=paths["result_log"],
        active_actions=active_actions,
    )

    raw_csv = os.path.join(paths["log_dir"], "final_eval_raw_results.csv")

    save_final_raw_results(
        output_csv=raw_csv,
        method_name=get_ppo_method_name(config_name),
        results=ppo_results,
        append=False
    )

    analyze_policy_decisions(
        model=best_model,
        seeds=EVAL_SEEDS,
        active_actions=active_actions,
        output_csv=paths["decision_log_csv"],
    )

    print("\n================ Fixed-action benchmark ================")
    benchmark_results = benchmark_fixed_actions(
        seeds=EVAL_SEEDS,
        result_log_path=paths["result_log"],
    )

    for action_id, results in benchmark_results.items():
        save_final_raw_results(
            output_csv=raw_csv,
            method_name=f"Fixed-action{action_id}",
            results=results,
            append=True
        )

    if 0 in benchmark_results:
        compare_ppo_with_baseline(
            ppo_results=ppo_results,
            baseline_results=benchmark_results[0],
            result_log_path=paths["result_log"],
            baseline_name="Fixed Global Action 0",
        )

    return {
        "config_name": config_name,
        "active_actions": active_actions,
        "paths": paths,
        "ppo_results": ppo_results,
        "benchmark_results": benchmark_results,
    }


def train_all_configs():
    all_results = []

    for config_name in ACTION_CONFIGS.keys():
        result = train_one_config(config_name)
        all_results.append(result)

    return all_results


def load_and_test(config_name=CURRENT_CONFIG_NAME):
    """
    Load an already trained best model and test it.
    """
    if config_name not in ACTION_CONFIGS:
        raise ValueError(f"Unknown config_name: {config_name}")

    active_actions = ACTION_CONFIGS[config_name]
    paths = build_paths(config_name)

    print(f"Using environment version: {ENV_VERSION}")
    print(f"Loading config: {config_name}")
    print(f"Best model path: {paths['best_model']}")

    model = PPO.load(paths["best_model"])

    analyze_policy_decisions(
        model=model,
        seeds=EVAL_SEEDS,
        active_actions=active_actions,
        output_csv=paths["decision_log_csv"],
    )

    print("\n================ Loaded model multi-seed test ================")
    ppo_results = evaluate_model(
        model=model,
        seeds=EVAL_SEEDS,
        active_actions=active_actions,
        deterministic=True,
        print_each_seed=True,
    )

    summarize_results(
        name=f"Loaded Best PPO - {config_name}",
        results=ppo_results,
        result_log_path=paths["result_log"],
        active_actions=active_actions,
    )

    raw_csv = os.path.join(paths["log_dir"], "final_eval_raw_results.csv")

    save_final_raw_results(
        raw_csv,
        "PPO-6action",
        ppo_results,
        append=False
    )


    print("\n================ Fixed-action benchmark ================")
    benchmark_results = benchmark_fixed_actions(
        seeds=EVAL_SEEDS,
        result_log_path=paths["result_log"],
    )

    for action_id, results in benchmark_results.items():
        save_final_raw_results(
            output_csv=raw_csv,
            method_name=f"Fixed-action{action_id}",
            results=results,
            append=True
        )

    if 0 in benchmark_results:
        compare_ppo_with_baseline(
            ppo_results=ppo_results,
            baseline_results=benchmark_results[0],
            result_log_path=paths["result_log"],
            baseline_name="Fixed Global Action 0",
        )


def load_and_test_all_ablation_configs():
    for config_name in ACTION_CONFIGS.keys():
        load_and_test(config_name)


if __name__ == "__main__":
    # To inspect observations:
    # inspect_observations(
    #     seeds=list(range(90, 100)),
    #     active_actions=ACTION_CONFIGS[CURRENT_CONFIG_NAME],
    #     actions=None,
    #     max_steps=1000,
    # )

    # if RUN_ALL_CONFIGS:
    #     train_all_configs()
    # else:
    #     train_one_config(CURRENT_CONFIG_NAME)

    # If you only want to test an existing model, comment the training lines above
    # and uncomment the following line:
    ##TODO:和固定规则对比版本
    # load_and_test(CURRENT_CONFIG_NAME)
    ##TODO:消融版本
    load_and_test_all_ablation_configs()