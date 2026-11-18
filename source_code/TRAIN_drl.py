import os
import numpy as np
import contextlib
import io
import csv

from datetime import datetime

from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.logger import configure


from MAM_DRL_ENV import MobileAMEnv,ENV_VERSION


INSTANCE_PATH = "instance/problem_instance_10cust_20parts.json"

MODEL_PATH = "mam_ppo_reward_5action_43_final_v00"
BEST_MODEL_PATH = "mam_ppo_reward_5action_43_best_v00"

STAGE_TIMESTEPS = 100000
N_STAGES = 5

MAX_EPISODE_STEPS = 1000

TRAIN_SEED = 43
# 训练过程中用于挑选最佳模型的 seed，可以少一点，避免太慢
STAGE_EVAL_SEEDS = list(range(200, 210))

# 最终正式测试用 20 个 seed
EVAL_SEEDS = list(range(100, 150))


RESULT_LOG_PATH = f"eval_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"







def make_train_env(seed=TRAIN_SEED):
    """
    创建训练环境。
    注意：训练环境和测试环境要分开创建。
    """

    def _init():
        env = MobileAMEnv(instance_path=INSTANCE_PATH)
        env.reset(seed=seed)
        env = Monitor(env)
        return env

    return _init




def train():
    """
    分阶段训练 PPO。
    每训练 STAGE_TIMESTEPS 步，就用 STAGE_EVAL_SEEDS 测一次平均总延误。
    保存平均总延误最低的模型，而不是只保存最后一步模型。
    """

    print(f"Using environment version: {ENV_VERSION}")

    with open(RESULT_LOG_PATH, "a", encoding="utf-8") as f:
        f.write(f"Environment version: {ENV_VERSION}\n")
        f.write(f"Instance path: {INSTANCE_PATH}\n")
        f.write(f"Train seed: {TRAIN_SEED}\n")
        f.write(f"Stage eval seeds: {STAGE_EVAL_SEEDS}\n")
        f.write(f"Final eval seeds: {EVAL_SEEDS}\n\n")

    # 1. 检查环境规范
    check_env_raw = MobileAMEnv(instance_path=INSTANCE_PATH)
    print("正在检查环境规范...")
    check_env(check_env_raw)

    # 2. 创建训练环境
    train_env = DummyVecEnv([make_train_env(TRAIN_SEED)])

    # 3. 定义 PPO 模型
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
        tensorboard_log="./ppo_tensorboard/",
    )

    new_logger = configure(
        "./ppo_train_logs/",
        ["stdout", "csv", "tensorboard"]
    )

    model.set_logger(new_logger)

    best_tardiness = float("inf")

    print("开始分阶段训练智能体...")

    for stage in range(N_STAGES):
        print(f"\n========== Training stage {stage + 1}/{N_STAGES} ==========")

        # 第一阶段可以重置计数；后续阶段不要重置，否则日志和学习进度会混乱
        model.learn(
            total_timesteps=STAGE_TIMESTEPS,
            reset_num_timesteps=(stage == 0)
        )

        model.save(f"checkpoint_stage_{stage + 1}")
        print(f"阶段 {stage + 1} 临时模型已保存。")

        # 用少量 seed 做阶段性评估，目的是挑选模型，不是最终报告
        stage_results = evaluate_model(
            model,
            STAGE_EVAL_SEEDS,
            deterministic=True,
            print_each_seed=False
        )

        avg_tardiness = np.mean([r["tardiness"] for r in stage_results])
        avg_reward = np.mean([r["total_reward"] for r in stage_results])

        print(
            f"[Stage {stage + 1}] "
            f"avg_tardiness={avg_tardiness:.2f}, "
            f"avg_reward={avg_reward:.2f}"
        )

        # 关键：按平均延误保存最佳模型
        if avg_tardiness < best_tardiness:
            best_tardiness = avg_tardiness
            model.save(BEST_MODEL_PATH)
            print(
                f"保存新的最佳模型: {BEST_MODEL_PATH}, "
                f"best_avg_tardiness={best_tardiness:.2f}"
            )

    # 保存最后一步模型，方便对比
    model.save(MODEL_PATH)
    print(f"\n最后一步模型已保存为: {MODEL_PATH}")
    print(f"最佳延误模型已保存为: {BEST_MODEL_PATH}")

    # 4. 加载最佳模型做最终 20 seed 测试
    print("\n================ Best PPO 最终多 seed 测试 ================")
    best_model = PPO.load(BEST_MODEL_PATH)

    ppo_results = evaluate_model(
        best_model,
        EVAL_SEEDS,
        deterministic=True,
        print_each_seed=True
    )
    summarize_results("Best PPO", ppo_results)

    # 5. 固定动作基准对比
    print("\n================ 固定动作规则基准测试 ================")
    benchmark_results = benchmark_fixed_actions(EVAL_SEEDS)

    # 6. 和 Fixed Action 0 做逐 seed 对比
    if 0 in benchmark_results:
        compare_ppo_with_baseline(
            ppo_results,
            benchmark_results[0],
            "Fixed Action 0"
        )



def evaluate_model(model, seeds, deterministic=True, print_each_seed=True):

    """
    用多个随机种子测试 PPO 模型。
    这里不要只看 reward，要重点记录 tardiness。
    """

    results = []
    decision_logs = []

    for seed in seeds:
        env = MobileAMEnv(instance_path=INSTANCE_PATH)
        obs, info = env.reset(seed=seed)

        done = False
        total_reward = 0.0
        step_count = 0

        action_count = {i: 0 for i in range(env.action_space.n)}

        while not done and step_count < MAX_EPISODE_STEPS:
            context = get_decision_context(env)

            action, _ = model.predict(obs, deterministic=deterministic)
            action = int(action)
            action_count[action] += 1

            decision_logs.append({
                "seed": seed,
                "step": step_count,
                "action": action,
                **context
            })

        #TODO:回退版
            # raw_action, _ = model.predict(obs, deterministic=deterministic)
            # raw_action = int(raw_action)
            #
            # action = raw_action
            # if action in [2]:
            #     action = 0
            #
            # assert action not in [2], f"回退失败: raw_action={raw_action}, action={action}"

            # action_count[action] += 1
            # decision_logs.append({
            #     "seed": seed,
            #     "step": step_count,
            #     "raw_action": raw_action,
            #     "exec_action": action,
            #     **context
            # })

            with contextlib.redirect_stdout(io.StringIO()):
                obs, reward, terminated, truncated, info = env.step(action)

            done = bool(terminated or truncated)
            total_reward += float(reward)
            step_count += 1

        # info 里正常有 tardiness；如果没有，就从环境函数里兜底取
        final_tardiness = info.get("tardiness", env._calculate_total_tardiness())

        results.append({
            "seed": seed,
            "tardiness": final_tardiness,
            "total_reward": total_reward,
            "steps": step_count,
            "done": done,
            "action_count": action_count,
        })

    #TODO:回退版本字典
        # results.append({
        #     "seed": seed,
        #     "tardiness": final_tardiness,
        #     "total_reward": total_reward,
        #     "steps": step_count,
        #     "done": done,
        #     "action_count": action_count,
        # })

        if print_each_seed:
            print(
                f"[PPO] seed={seed}, "
                f"tardiness={final_tardiness:.2f}, "
                f"reward={total_reward:.2f}, "
                f"steps={step_count}, "
                f"done={done}, "
                f"actions={action_count}"
            )

    return results


def evaluate_fixed_action(action_id, seeds):
    """
    测试固定动作规则。
    例如一直 action=0、一直 action=1 等。
    """

    results = []

    for seed in seeds:
        env = MobileAMEnv(instance_path=INSTANCE_PATH)
        obs, info = env.reset(seed=seed)

        done = False
        total_reward = 0.0
        step_count = 0

        while not done and step_count < MAX_EPISODE_STEPS:
            with contextlib.redirect_stdout(io.StringIO()):
                obs, reward, terminated, truncated, info = env.step(action_id)

            done = bool(terminated or truncated)
            total_reward += float(reward)
            step_count += 1

        final_tardiness = info.get("tardiness", env._calculate_total_tardiness())

        results.append({
            "seed": seed,
            "action": action_id,
            "tardiness": final_tardiness,
            "total_reward": total_reward,
            "steps": step_count,
            "done": done,
        })

    return results


def benchmark_fixed_actions(seeds):
    """
    把 PPO 和 6 个固定规则动作进行对比。
    如果 PPO 还不如最好的固定动作，通常说明 reward / observation / action 设计还有问题。
    """

    all_benchmark_results = {}

    # for action_id in range(6):
    #     results = evaluate_fixed_action(action_id, seeds)
    #     all_benchmark_results[action_id] = results
    #     summarize_results(f"Fixed Action {action_id}", results)

#TODO：五动作版本
    for action_id in range(5):
        results = evaluate_fixed_action(action_id, seeds)
        all_benchmark_results[action_id] = results
        summarize_results(f"Fixed Action {action_id}", results)

    return all_benchmark_results


def summarize_results(name, results):
    """
    汇总平均延误、标准差、最大最小值、完成率。
    同时打印到控制台，并保存到 txt 文件。
    """

    tardiness_values = np.array([r["tardiness"] for r in results], dtype=float)
    reward_values = np.array([r["total_reward"] for r in results], dtype=float)
    step_values = np.array([r["steps"] for r in results], dtype=float)
    done_values = np.array([1 if r["done"] else 0 for r in results], dtype=float)

    lines = []
    lines.append(f"\n--- {name} 汇总结果 ---")
    lines.append(f"测试次数: {len(results)}")
    lines.append(f"平均总延误: {np.mean(tardiness_values):.2f}")
    lines.append(f"延误标准差: {np.std(tardiness_values):.2f}")
    lines.append(f"最小总延误: {np.min(tardiness_values):.2f}")
    lines.append(f"最大总延误: {np.max(tardiness_values):.2f}")
    lines.append(f"平均总奖励: {np.mean(reward_values):.2f}")
    lines.append(f"平均步数: {np.mean(step_values):.2f}")
    lines.append(f"完成率: {np.mean(done_values) * 100:.1f}%")

    # if "action_count" in results[0]:
    #     total_action_count = {i: 0 for i in range(6)}
    #     for r in results:
    #         for action_id, count in r["action_count"].items():
    #             total_action_count[action_id] += count
    #     lines.append(f"动作使用次数: {total_action_count}")

   #TODO:五动作版本
    if "action_count" in results[0]:
        total_action_count = {i: 0 for i in range(5)}
        for r in results:
            for action_id, count in r["action_count"].items():
                total_action_count[action_id] += count
        lines.append(f"动作使用次数: {total_action_count}")

    summary_text = "\n".join(lines)

    print(summary_text)

    with open(RESULT_LOG_PATH, "a", encoding="utf-8") as f:
        f.write(summary_text)
        f.write("\n")


def load_and_test():
    """
    如果你已经训练过模型，只想重新测试，可以运行这个函数。
    """
    print(f"Using environment version: {ENV_VERSION}")

    model = PPO.load(BEST_MODEL_PATH)

    analyze_policy_decisions(
        model,
        EVAL_SEEDS,
        output_csv="ppo_decision_logs.csv"
    )

    print("\n================ 已训练模型多 seed 测试 ================")
    ppo_results = evaluate_model(model, EVAL_SEEDS , deterministic=True ,print_each_seed=True)
    fixed0_results = evaluate_fixed_action(0, EVAL_SEEDS)

    summarize_results("Loaded Best PPO", ppo_results)
    summarize_results("Fixed Action 0", fixed0_results)

    compare_ppo_with_baseline(ppo_results, fixed0_results, "Fixed Action 0")

    print("\n================ 固定动作规则基准测试 ================")
    benchmark_results = benchmark_fixed_actions(EVAL_SEEDS)

    if 0 in benchmark_results:
        compare_ppo_with_baseline(
            ppo_results,
            benchmark_results[0],
            "Fixed Action 0"
        )

def compare_ppo_with_baseline(ppo_results, baseline_results, baseline_name="Fixed Action 0"):
    baseline_map = {
        r["seed"]: r for r in baseline_results
    }

    better_count = 0
    worse_count = 0
    equal_count = 0

    diffs = []

    print(f"\n--- PPO vs {baseline_name} 逐 seed 对比 ---")

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
            flag = "PPO更好"
        elif diff > 0:
            worse_count += 1
            flag = "PPO更差"
        else:
            equal_count += 1
            flag = "相同"

        print(
            f"seed={seed}, "
            f"PPO={ppo_t:.2f}, "
            f"{baseline_name}={base_t:.2f}, "
            f"diff={diff:.2f}, "
            f"{flag}"
            f"actions={r.get('action_count', {})}"
        )

    diffs = np.array(diffs, dtype=float)

    print("\n--- 对比汇总 ---")
    print(f"PPO 更好的次数: {better_count}")
    print(f"PPO 更差的次数: {worse_count}")
    print(f"相同次数: {equal_count}")
    print(f"平均差值 PPO - baseline: {np.mean(diffs):.2f}")


def inspect_observations(seeds, actions=None, max_steps=1000):
    """
    检查当前环境状态向量是否合理：
    1. 是否有 nan / inf
    2. 是否有恒为 0 的维度
    3. 是否经常贴近边界 0 / 1 / -1
    4. raw_obs 是否经常超出 observation_space
    """

    if actions is None:
        actions = list(range(6)) + ["random"]

    all_obs = []
    all_raw_obs = []
    all_obs_low = []
    all_obs_high = []

    for mode in actions:
        for seed in seeds:
            env = MobileAMEnv(instance_path=INSTANCE_PATH)

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
                    action = env.action_space.sample()
                else:
                    action = int(mode)

                with contextlib.redirect_stdout(io.StringIO()):
                    obs, reward, terminated, truncated, info = env.step(action)

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





def get_decision_context(env):
    """
    提取当前决策点的可解释状态，用于分析 PPO 为什么选择某个动作。
    """

    unassigned = [o for o in env.orders.values() if o.status == "unassigned"]
    printing = [o for o in env.orders.values() if o.status == "printing"]
    printed = [o for o in env.orders.values() if o.status == "printed"]
    delivered = [o for o in env.orders.values() if o.status == "delivered"]

    current_batch = list(env.printer.current_batch) if env.printer.is_printing else []

    # 当前真正影响卡车决策的候选订单
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


def analyze_policy_decisions(model, seeds, output_csv="ppo_decision_logs.csv"):
    """
    记录 PPO 在每个 seed、每个决策点的动作和状态。
    输出 CSV，方便后面用 Excel 看规律。
    """

    rows = []

    for seed in seeds:
        env = MobileAMEnv(instance_path=INSTANCE_PATH)
        obs, info = env.reset(seed=seed)

        done = False
        step_count = 0
        total_reward = 0.0

        while not done and step_count < MAX_EPISODE_STEPS:
            context_before = get_decision_context(env)

            action, _ = model.predict(obs, deterministic=True)
            action = int(action)

            obs, reward, terminated, truncated, info = env.step(action)
            done = bool(terminated or truncated)

            context_after = get_decision_context(env)

            row = {
                "seed": seed,
                "step": step_count,
                "action": action,
                "reward": float(reward),
                "tardiness_after_step": info.get("tardiness", None),

                "pre_num_unassigned": context_before["num_unassigned"],
                "pre_num_printing": context_before["num_printing"],
                "pre_num_printed": context_before["num_printed"],
                "pre_num_delivered": context_before["num_delivered"],

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
        print("没有记录到决策数据。")
        return

    fieldnames = list(rows[0].keys())

    with open(output_csv, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"决策日志已保存到: {output_csv}")



if __name__ == "__main__":
    #检查各维度向量情况
    # inspect_observations(
    #     seeds=list(range(90, 100)),
    #     actions=list(range(6)) + ["random"],
    #     max_steps=1000
    # )

    train()
    # 如果只想测试已有模型，就注释 train()，改成：
    # load_and_test()
