

# -*- coding: utf-8 -*-
"""
MAM Evaluate Deterministic Solutions
Runs MILP models (BASE, MVP, PDE) to get solutions,
then evaluates their quality using the convergent simulation environment.
"""
import gurobipy as gp
from gurobipy import GRB
import matplotlib.pyplot as plt
import numpy as np
import os
import re
from scipy.stats import norm
import scipy.stats as stats
import random
import sys

# 设置 Matplotlib 字体以支持中文
try:
    plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False
except Exception:
    pass  # If no Chinese font available, skip.


# =============================================================================
# A. UTILITIES (Data Loading, Log-Normal, and Simulation Functions)
#
# Note: These functions are copied from mam_simulation_environment.py to make
# this file fully self-contained and runnable.
# =============================================================================

from mam_simulation_environment import calculate_lognormal_quantile, load_problem_data


def convert_to_simulation_format(deterministic_solution):
    """Converts MILP solution dictionary to the standard simulation format."""
    simulation_solution = {}

    # 路径处理
    simulation_solution["route"] = deterministic_solution.get("route", [])

    # 批次组成和序列处理 (已由 MILP 求解器格式化)
    simulation_solution["batch_composition"] = deterministic_solution.get("batch_composition", {})
    simulation_solution["batch_sequence"] = deterministic_solution.get("batch_sequence", [])

    return simulation_solution


from mam_simulation_environment import simulate_mam_vprp_convergent, deterministic_mam_vprp_evaluation


# =============================================================================
# C. MILP SOLVER (Based on mam_vprp_p_milp.py)
# =============================================================================

from MAM_Deterministic_Equivalent_MILP import create_and_solve_model


# =============================================================================
# D. COMPARISON LOGIC
# =============================================================================


def compare_deterministic_models(problem_data):
    """运行MILP求解BASE、MVP、PDE模型，并用仿真评估它们的质量"""
    # 调整仿真收敛参数，避免过长的运行时间
    SIM_PARAMS = {'alpha': 0.05, 'epsilon': 0.01, 'N_min': 100, 'N_max': 5000}

    models_to_solve = [
        {"name": "BASE-MODE (p=0)", "percentile": 0.0, "use_mean": False},
        {"name": "MVP-MEDIAN (p=0.50)", "percentile": 0.50, "use_mean": False},
        {"name": "PDE-P95 (p=0.95)", "percentile": 0.95, "use_mean": False},
        {"name": "MVP-MEAN", "percentile": None, "use_mean": True}
    ]

    results = {}

    for model_info in models_to_solve:
        name = model_info["name"]
        p = model_info["percentile"]
        use_mean = model_info["use_mean"]

        print(f"\n{'=' * 50}")
        print(f"1. Solving {name} MILP...")

        # 1. 求解MILP
        solution = create_and_solve_model(
            problem_data,
            percentile=p,
            model_name=name,
            use_mean=use_mean
        )

        if solution["status"] == GRB.OPTIMAL or solution["status"] == GRB.TIME_LIMIT:
            simulation_solution = convert_to_simulation_format(solution)

            print(f"2. Evaluating {name} via Convergent Simulation...")
            tardiness_results, mean_tardiness, convergence_info = simulate_mam_vprp_convergent(
                simulation_solution, problem_data, **SIM_PARAMS
            )

            # 2. 直接使用MILP的objective作为确定性值
            milp_deterministic = solution.get('objective', 0)

            # 3. （可选）用评估函数验证 - 需要先实现get_correct_model_type函数
            # 这里简化：根据模型信息选择正确的评估类型
            if use_mean:
                model_type_for_eval = "mean"
            elif p == 0.0:
                model_type_for_eval = "mode"
            elif p == 0.50:
                model_type_for_eval = "median"
            elif p == 0.95:
                model_type_for_eval = "p95"
            else:
                model_type_for_eval = "mode"  # 默认

            eval_deterministic = deterministic_mam_vprp_evaluation(
                simulation_solution, problem_data,
                model_type=model_type_for_eval
            )

            # 记录差异（用于调试）
            diff = abs(milp_deterministic - eval_deterministic)
            if diff > 0.1:  # 如果差异显著
                print(
                    f"  差异警告：MILP值({milp_deterministic:.2f}) vs 评估值({eval_deterministic:.2f})，差异：{diff:.4f}")

            # 4. 使用MILP值（更准确，因为与优化一致）
            results[name] = {
                'deterministic_obj': solution.get('objective', 0),
                'simulation_mean': mean_tardiness,
                'simulation_std': np.std(tardiness_results),
                'convergence_info': convergence_info,
                'solution': solution,
                'tardiness_results': tardiness_results,
                'det_eval_value': eval_deterministic,  # 评估函数计算的值
                'difference': diff  # 差异值
            }

            print(f"3. {name} Results Summary:")
            if model_info["use_mean"]:
                print(f"   - MILP Target (均值模型): {solution.get('objective'):.2f}")
            else:
                print(f"   - MILP Target (p={p * 100:.0f}%): {solution.get('objective'):.2f}")
            print(f"   - Simulation Mean: {mean_tardiness:.2f}")
            print(f"   - 评估函数值: {eval_deterministic:.2f}")
            print(f"   - 差异: {diff:.4f}\n")

            plot_distribution(tardiness_results, eval_deterministic, mean_tardiness, name)

        else:
            print(f"MILP for {name} failed with status {solution['status']}. Skipping simulation.")

    if results:
        plot_comparison_summary(results)

    return results


# def plot_distribution(tardiness_results, deterministic_mode_value, simulation_mean, label_name):
#     """绘制总延迟分布图"""
#     plt.figure(figsize=(10, 6))
#     plt.hist(tardiness_results, bins=30, density=True, alpha=0.7, color='skyblue', edgecolor='black')
#
#     plt.axvline(simulation_mean, color='yellow', linestyle='dashed', linewidth=2,
#                 label=f'Simulation Mean: {simulation_mean:.2f}')
#     plt.axvline(deterministic_mode_value, color='red', linestyle='dashed', linewidth=2,
#                 label=f'Deterministic (Mode) Value: {deterministic_mode_value:.2f}')

def plot_distribution(tardiness_results, eval_deterministic, simulation_mean, label_name):
    """绘制总延迟分布图"""
    plt.figure(figsize=(10, 6))
    plt.hist(tardiness_results, bins=30, density=True, alpha=0.7, color='skyblue', edgecolor='black')

    plt.axvline(simulation_mean, color='yellow', linestyle='dashed', linewidth=2,
                label=f'Simulation Mean: {simulation_mean:.2f}')
    plt.axvline(eval_deterministic, color='red', linestyle='dashed', linewidth=2,
            label=f'Deterministic Value: {eval_deterministic:.2f}')

    plt.title(f"总延迟分布 - {label_name} 方案")
    plt.xlabel("总延迟 (Total Tardiness)")
    plt.ylabel("概率密度 (Probability Density)")
    plt.legend()
    plt.grid(axis='y', alpha=0.75)
    plt.tight_layout()
    plt.show()


def plot_comparison_summary(results):
    """绘制模型比较图"""
    valid_results = {k: v for k, v in results.items() if v.get('simulation_mean') is not None}

    if not valid_results:
        print("没有有效的仿真结果可绘制")
        return

    models = list(valid_results.keys())
    milp_targets = [valid_results[m]['deterministic_obj'] for m in models]
    simulation_means = [valid_results[m]['simulation_mean'] for m in models]
    simulation_stds = [valid_results[m]['simulation_std'] for m in models]

    x = np.arange(len(models))
    width = 0.35

    plt.figure(figsize=(14, 7))

    # 左图：确定性目标 vs 仿真均值
    plt.subplot(1, 2, 1)
    rects1 = plt.bar(x - width / 2, milp_targets, width, label='MILP Target (p%)', color='darkblue', alpha=0.7)
    rects2 = plt.bar(x + width / 2, simulation_means, width, label='Simulation Mean ($\mathbb{E}$)',
                     yerr=simulation_stds, capsize=5, color='orange', alpha=0.8)

    plt.title('确定性模型目标与仿真评估均值对比')
    plt.xticks(x, models)
    plt.ylabel('总延迟')
    plt.legend()

    # def autolabel(rects, is_mean=False):
    #     for rect in rects:
    #         height = rect.get_height()
    #         plt.text(rect.get_x() + rect.get_width() / 2., height + 5,
    #                  f'{height:.1f}',
    #                  ha='center', va='bottom', fontsize=9)
    #         if is_mean:
    #             N = valid_results[models[np.where(rects == rect)[0][0]]]['convergence_info']['final_N']
    #             plt.text(rect.get_x() + rect.get_width() / 2., height / 2,
    #                      f'N={N}', ha='center', va='center', color='black', fontsize=8, fontweight='bold')
    #
    # autolabel(rects1)
    # autolabel(rects2, is_mean=True)

    # 右图：性能分析
    vss_values = [simulation_means[0] - mean for mean in simulation_means]

    plt.subplot(1, 2, 2)
    # VSS with respect to MVP (Simulation Mean)
    vss_baseline = simulation_means[1]  # Use MVP's simulation mean as VSS benchmark

    vss_data = [vss_baseline - mean for mean in simulation_means]

    bars = plt.bar(x, vss_data, color=['gray', 'green', 'red'], alpha=0.7)

    plt.axhline(0, color='black', linestyle='--', linewidth=1)

    plt.title('方案质量 ($\mathbb{E}[Tard]$ 差距) - 以MVP为基准')
    plt.xticks(x, models)
    plt.ylabel('差距值 ($\mathbb{E}[Tard]_{MVP} - \mathbb{E}[Tard]_{Model}$)')

    # 标签显示
    for i, v in enumerate(vss_data):
        label = f'{v:.1f}' if i != 1 else f'VDS ({vss_baseline:.1f})'
        plt.text(i, v + np.sign(v) * 5, label, ha='center', va='center', fontsize=10)

    plt.tight_layout()
    plt.show()


# =============================================================================
# E. MAIN EXECUTION
# =============================================================================
if __name__ == "__main__":

    # 1. 加载数据 (使用 TXT 文件)
    PROBLEM_FILE = "instance/problem_instance_5cust_10parts.json"

    problem_data = load_problem_data(PROBLEM_FILE)

    if problem_data is None:
        sys.exit("Execution halted: Failed to load problem data.")

    print("\n=========================================================")
    print(" 确定性模型解的鲁棒性评估 ")
    print("=========================================================\n")

    # 2. 运行评估
    final_results = compare_deterministic_models(problem_data)

    print("\n---------------------------------------------------------")
    print("Final Comparative Summary of Simulation Means ($\mathbb{E}[Tard]$)")
    print("---------------------------------------------------------")
    if final_results:
        for name, res in final_results.items():
            print(
                f"[{name}] MILP Target: {res['deterministic_obj']:.2f} | Simulation Mean: {res['simulation_mean']:.2f}")# (p={res['solution']["percentile"] * 100:.0f}%)