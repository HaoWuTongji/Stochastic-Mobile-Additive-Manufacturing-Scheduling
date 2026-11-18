# -*- coding: utf-8 -*-
import math
import random
import copy
import numpy as np
import time
import os
import json
import argparse
import logging  # 用于ID处理的日志
import textwrap

# ========== 导入统一的仿真环境 ==========
from mam_simulation_environment import (
    simulate_mam_vprp_convergent,
    deterministic_mam_vprp_evaluation,
    load_problem_data,
    calculate_lognormal_quantile
)


# ========== 通用ID处理工具函数 ==========
def safe_id_convert(id_val, target_type=int):
    """安全的ID类型转换"""
    try:
        return target_type(id_val)
    except (ValueError, TypeError):
        logging.warning(f"ID {id_val} 转换为{target_type}失败")
        return None


def validate_and_filter_ids(problem_data, solution):
    """显式ID过滤：清理无效客户/零件/批次ID，保证数据合法性"""
    # 1. 过滤客户ID（保留存在的数字型客户ID，0是仓库节点）
    valid_customer_ids = set(problem_data["customer_data"].keys())
    solution["route"] = [node for node in solution["route"] if node == 0 or node in valid_customer_ids]

    # 2. 过滤零件ID
    valid_part_ids = set(problem_data["part_data"].keys())
    empty_batches = []
    for batch_id, part_list in solution["batch_composition"].items():
        filtered_parts = [p for p in part_list if p in valid_part_ids]
        solution["batch_composition"][batch_id] = filtered_parts
        if not filtered_parts:
            empty_batches.append(batch_id)

    # 清理空批次
    for b_id in empty_batches:
        del solution["batch_composition"][b_id]
    solution["batch_sequence"] = [b for b in solution["batch_sequence"] if b in solution["batch_composition"]]

    return solution, problem_data


# ==============================================================================
# SECTION 1: SIMULATION EVALUATOR (统一仿真调用)
# ==============================================================================
def simulate_mam_vprp(solution, problem_data, num_replications=300):
    """
    复用mam_simulation_environment中的收敛式仿真，保证环境完全一致
    """
    solution, problem_data = validate_and_filter_ids(problem_data, solution)

    # 调用统一的收敛仿真函数（固定次数，关闭自适应收敛提前退出逻辑）
    tardiness_results, mean_tardiness, convergence_info = simulate_mam_vprp_convergent(
        solution=solution,
        problem_data=problem_data,
        alpha=0.05,
        epsilon=0.05,
        N_min=num_replications,
        N_max=num_replications
    )

    return mean_tardiness


def deterministic_mam_vprp(solution, problem_data):
    """
    复用mam_simulation_environment中的确定性评估
    """
    solution, problem_data = validate_and_filter_ids(problem_data, solution)

    # 包含对未服务客户的惩罚
    total_tardiness = deterministic_mam_vprp_evaluation(
        solution=solution,
        problem_data=problem_data,
        model_type="mode"
    )

    return total_tardiness


def generate_initial_solution(problem_data):
    """Generates a simple but valid initial solution."""
    # Route generation (Nearest Neighbor)
    customers_to_visit = list(problem_data["customer_data"].keys())
    route = [0]
    current_node = 0

    def get_travel_time(c):
        # 使用统一环境的 tuple 元组作为键
        key = (current_node, c)
        if key in problem_data["travel_time_params"]:
            return problem_data["travel_time_params"][key]['value']
        return float('inf')

    while customers_to_visit:
        next_node = min(customers_to_visit, key=lambda c: get_travel_time(c))
        route.append(next_node)
        customers_to_visit.remove(next_node)
        current_node = next_node

    route.append(0)

    # Batching and Sequencing (Greedy)
    parts_data = problem_data["part_data"]
    all_parts = list(parts_data.keys())
    random.shuffle(all_parts)

    batch_composition = {}
    batch_counter = 1

    while all_parts:
        batch_id = f"Batch{batch_counter}"
        batch_composition[batch_id] = []
        current_batch_area = 0
        parts_to_remove_from_list = []

        for part_id in all_parts:
            part_area = parts_data[part_id]['area']
            if current_batch_area + part_area <= problem_data["printer_params"]['A_max']:
                batch_composition[batch_id].append(part_id)
                current_batch_area += part_area
                parts_to_remove_from_list.append(part_id)

        all_parts = [p for p in all_parts if p not in parts_to_remove_from_list]
        batch_counter += 1

    batch_sequence = list(batch_composition.keys())

    return {
        "route": route,
        "batch_composition": batch_composition,
        "batch_sequence": batch_sequence
    }


# ==============================================================================
# HELPER: SOLUTION POOL MANAGEMENT
# ==============================================================================
def update_solution_pool(pool, solution, fitness, pool_size=10):
    """
    更新随机解池：
    1. 检查重复（基于路线是否相同）。
    2. 如果池未满，直接加入。
    3. 如果池已满，且新解优于池中最差解，则替换最差解。
    """
    # 1. 查重：如果路线完全一样，视为同一个解，保留fitness更准确的那个（可选），这里简单跳过
    current_route = solution['route']
    for item in pool:
        if item['solution']['route'] == current_route:
            # 如果新测算的fitness更小（更优），可以更新它，或者直接跳过
            if fitness < item['fitness']:
                item['fitness'] = fitness
                item['solution'] = copy.deepcopy(solution)
            return pool

    # 包装新条目
    new_entry = {'solution': copy.deepcopy(solution), 'fitness': fitness}

    # 2. 池未满，直接加入
    if len(pool) < pool_size:
        pool.append(new_entry)
        # 加入后重新按 fitness 排序 (从小到大，因为是最小化问题)
        pool.sort(key=lambda x: x['fitness'])
        return pool

    # 3. 池已满，尝试替换最差解 (列表最后一名是最大的/最差的)
    worst_entry = pool[-1]
    if fitness < worst_entry['fitness']:
        pool.pop() # 移除最差
        pool.append(new_entry)
        pool.sort(key=lambda x: x['fitness']) # 重新排序

    return pool


# ==============================================================================
# SECTION 2: ALNS OPERATORS (DESTROY & REPAIR)
# ==============================================================================
#region破坏算子
def random_customer_removal(solution, problem_data, q):
    sol_partial = copy.deepcopy(solution)
    removed_customers = []

    available_customers = [c for c in sol_partial["route"] if c != 0]
    if not available_customers:
        return sol_partial, []

    num_to_remove = min(q, len(available_customers))
    removed_customers = random.sample(available_customers, num_to_remove)

    sol_partial["route"] = [c for c in sol_partial["route"] if c not in removed_customers]

    parts_to_remove = set()
    for cust_id in removed_customers:
        for part_id in problem_data["customer_data"][cust_id]['parts']:
            parts_to_remove.add(part_id)

    empty_batches = []
    for batch_id, part_list in list(sol_partial["batch_composition"].items()):
        sol_partial["batch_composition"][batch_id] = [p for p in part_list if p not in parts_to_remove]
        if not sol_partial["batch_composition"][batch_id]:
            empty_batches.append(batch_id)

    for b_id in empty_batches:
        del sol_partial["batch_composition"][b_id]
    sol_partial["batch_sequence"] = [b for b in sol_partial["batch_sequence"] if b not in empty_batches]

    return sol_partial, removed_customers

def batch_blocking_part_removal(solution, problem_data, q):
    sol_partial = copy.deepcopy(solution)
    parts_data = problem_data["part_data"]
    blocking_score = {}

    for parts in sol_partial["batch_composition"].values():
        for p in parts:
            blocking_score[p] = 0.0

    for batch_id, parts in sol_partial["batch_composition"].items():
        if len(parts) <= 1: continue

        heights = [(p, parts_data[p]["height"]) for p in parts]
        heights.sort(key=lambda x: x[1], reverse=True)

        # 阻塞的严重程度 = 最高零件高度 - 第二高零件高度
        max_part, max_h = heights[0]
        second_h = heights[1][1]
        blocking_score[max_part] += (max_h - second_h)

    sorted_parts = sorted(blocking_score.items(), key=lambda x: x[1], reverse=True)#阻塞分数降序排列
    num_remove = min(q, len(sorted_parts))
    removed_parts = []

    # 3. 偏向性随机选择 (Biased Random Selection)
    p_bias = 3 # 偏置参数，越大越容易选中排在前面的（分数高的）
    while len(removed_parts) < num_remove and sorted_parts:
        r = random.random()
        idx = int((r ** p_bias) * len(sorted_parts))
        part_id = sorted_parts[idx][0]
        if part_id not in removed_parts:
            removed_parts.append(part_id)

    empty_batches = []
    for batch_id, parts in sol_partial["batch_composition"].items():
        new_parts = [p for p in parts if p not in removed_parts]
        sol_partial["batch_composition"][batch_id] = new_parts
        if not new_parts: empty_batches.append(batch_id)

    for b in empty_batches: del sol_partial["batch_composition"][b]
    sol_partial["batch_sequence"] = [b for b in sol_partial["batch_sequence"] if b not in empty_batches]

    return sol_partial, removed_parts

def worst_tardiness_removal(solution, problem_data, q):
    sol_partial = copy.deepcopy(solution)

    # 临时计算方案的罚分，以找出影响最大的节点
    customer_tardiness = {}

    # 为此算子引入确定性的逐节点评估逻辑
    part_completion_times = {}
    production_time_accumulator = 0.0
    for batch_id in sol_partial["batch_sequence"]:
        parts = sol_partial["batch_composition"].get(batch_id, [])
        if not parts: continue
        vol = sum(problem_data["part_data"][p]['volume'] for p in parts)
        h = max(problem_data["part_data"][p]['height'] for p in parts)
        t = problem_data["printer_params"]['alpha'] * vol + problem_data["printer_params"]['beta'] * h
        production_time_accumulator += t
        for p in parts: part_completion_times[p] = production_time_accumulator

    truck_time = 0.0
    for i in range(len(sol_partial["route"]) - 1):
        s_node, n_node = sol_partial["route"][i], sol_partial["route"][i + 1]
        travel_time = problem_data["travel_time_params"].get((s_node, n_node), {'value': 0})['value']
        truck_time += travel_time
        if n_node == 0: break

        c_parts = problem_data["customer_data"][n_node]['parts']
        last_part_time = max([part_completion_times.get(p, 0) for p in c_parts] + [0])
        truck_time = max(truck_time, last_part_time)
        tardiness = max(0, truck_time - problem_data["customer_data"][n_node]['due_date'])
        customer_tardiness[n_node] = tardiness

    if not customer_tardiness:
        return random_customer_removal(solution, problem_data, q)

    sorted_customers = sorted(customer_tardiness.items(), key=lambda item: item[1], reverse=True)
    num_to_remove = min(q, len(sorted_customers))
    removed_customers = [c[0] for c in sorted_customers[:num_to_remove]]

    sol_partial["route"] = [c for c in sol_partial["route"] if c not in removed_customers]
    parts_to_remove = {p for cust_id in removed_customers for p in problem_data["customer_data"][cust_id]['parts']}

    empty_batches = []
    for batch_id, part_list in list(sol_partial["batch_composition"].items()):
        sol_partial["batch_composition"][batch_id] = [p for p in part_list if p not in parts_to_remove]
        if not sol_partial["batch_composition"][batch_id]:
            empty_batches.append(batch_id)

    for b_id in empty_batches:
        del sol_partial["batch_composition"][b_id]
    sol_partial["batch_sequence"] = [b for b in sol_partial["batch_sequence"] if b not in empty_batches]

    return sol_partial, removed_customers


def sim_high_variance_exposure_removal(solution, problem_data, q):
    """
    破坏算子：基于High-Variance-Exposure-Removal思想，移除风险分数最高的q个客户。
    已同步蒙特卡洛对数正态抽样逻辑与 mam_simulation_environment.py 一致。
    """
    sol_partial = copy.deepcopy(solution)
    available_customers = [c for c in sol_partial["route"] if c != 0]
    if not available_customers:
        return sol_partial, []

    parts_data = problem_data["part_data"]
    customer_data = problem_data["customer_data"]
    printer_params = problem_data["printer_params"]
    travel_time_params = problem_data["travel_time_params"]
    proc_uncertainty = problem_data["proc_uncertainty"]

    α = printer_params['alpha']
    β = printer_params['beta']

    batch_base_times = {}
    for batch_id, part_list in sol_partial["batch_composition"].items():
        if not part_list:
            batch_base_times[batch_id] = 0.0
            continue
        total_volume = sum(parts_data[p]['volume'] for p in part_list)
        max_height = max(parts_data[p]['height'] for p in part_list)
        batch_base_times[batch_id] = α * total_volume + β * max_height

    customer_parts_map = {cust_id: customer_data[cust_id]['parts'] for cust_id in available_customers}

    RiskScore = {}
    N_sim = 50
    λ = 1.0

    # 🌟【核心修复：CRN 机制】在评估所有客户前，提前生成并固定这 50 种场景
    scenarios_proc = []
    scenarios_travel = []

    proc_std = proc_uncertainty['std']
    mu_proc = proc_std ** 2

    for _ in range(N_sim):
        # 场景1：生成本轮固定的加工时间乘数
        proc_multipliers = {}
        for b_id in sol_partial["batch_composition"]:
            val = np.random.lognormal(mean=mu_proc, sigma=proc_std)
            proc_multipliers[b_id] = min(max(val, 0.5), 2.0)
        scenarios_proc.append(proc_multipliers)

        # 场景2：生成本轮固定的旅行时间乘数
        travel_times = {}
        for (s_node, n_node), params in travel_time_params.items():
            travel_std = params['std']
            mu_travel = travel_std ** 2
            multiplier = np.random.lognormal(mean=mu_travel, sigma=travel_std)
            travel_times[(s_node, n_node)] = min(max(multiplier, 0.5), 2.0) * params['value']
        scenarios_travel.append(travel_times)

    # 然后再遍历客户，使用刚才生成的固定场景进行公平对比
    for n in available_customers:
        departure_times = []
        n_parts = customer_parts_map.get(n, [])
        route = sol_partial["route"]
        if not n_parts or n not in route:
            RiskScore[n] = 0
            continue

        for sim_idx in range(N_sim):
            # 🌟【使用统一场景】：直接提取，不再重新 np.random
            proc_time_multipliers = scenarios_proc[sim_idx]
            sampled_travel_times = scenarios_travel[sim_idx]

            part_completion_times = {}
            production_time_accumulator = 0.0

            for batch_id in sol_partial["batch_sequence"]:
                base_time = batch_base_times.get(batch_id, 0.0)
                if base_time <= 0: continue
                proc_multiplier = proc_time_multipliers.get(batch_id, 1.0)
                sampled_proc_time = base_time * proc_multiplier
                production_time_accumulator += min(max(sampled_proc_time, 1e-3), 1e5)

                for part_id in sol_partial["batch_composition"].get(batch_id, []):
                    part_completion_times[part_id] = production_time_accumulator

            truck_time = 0.0
            departure_time = 0.0
            found_customer = False

            for i in range(len(route) - 1):
                s_node, n_node = route[i], route[i + 1]
                travel_key = (s_node, n_node)
                travel_time = sampled_travel_times.get(travel_key, 0.0)
                arrival_time = truck_time + travel_time

                if n_node == n:
                    found_customer = True
                    last_part_completion_time = max([part_completion_times.get(p, 0.0) for p in n_parts] + [0.0])
                    departure_time = max(arrival_time, last_part_completion_time)
                    break

                if n_node != 0:
                    other_parts = customer_parts_map.get(n_node, [])
                    last_part_completion_time = max([part_completion_times.get(p, 0.0) for p in other_parts] + [0.0])
                    truck_time = max(arrival_time, last_part_completion_time)

            if found_customer:
                departure_times.append(departure_time)

        if departure_times:
            μn = np.mean(departure_times)
            σn = np.std(departure_times) if len(departure_times) > 1 else 0.0
            RiskScore[n] = μn + λ * σn##风险分数计算
        else:
            RiskScore[n] = 0

    if not RiskScore:
        return random_customer_removal(solution, problem_data, q)

    sorted_customers = sorted(RiskScore.items(), key=lambda item: item[1], reverse=True)
    num_to_remove = min(q, len(sorted_customers))
    removed_customers = [c[0] for c in sorted_customers[:num_to_remove]]

    sol_partial["route"] = [c for c in sol_partial["route"] if c not in removed_customers]

    parts_to_remove = set()
    for cust_id in removed_customers:
        parts_to_remove.update(customer_parts_map.get(cust_id, []))

    empty_batches = []
    for batch_id, part_list in list(sol_partial["batch_composition"].items()):
        sol_partial["batch_composition"][batch_id] = [p for p in part_list if p not in parts_to_remove]
        if not sol_partial["batch_composition"][batch_id]:
            empty_batches.append(batch_id)

    for b_id in empty_batches:
        del sol_partial["batch_composition"][b_id]
    sol_partial["batch_sequence"] = [b for b in sol_partial["batch_sequence"] if b not in empty_batches]

    return sol_partial, removed_customers


def sim_synchronization_critical_removal(solution, problem_data, q):
    sol_partial = copy.deepcopy(solution)
    available_customers = [c for c in sol_partial["route"] if c != 0]
    if not available_customers: return sol_partial, []

    AvgWait = {}
    N_sim = 50

    proc_uncertainty = problem_data["proc_uncertainty"]
    travel_time_params = problem_data["travel_time_params"]
    parts_data = problem_data["part_data"]
    printer_params = problem_data["printer_params"]

    # 🌟【核心修复：CRN 机制】在评估所有客户前，提前生成并固定这 50 种平行宇宙（场景）
    scenarios_proc = []
    scenarios_travel = []

    proc_std = proc_uncertainty['std']
    mu_proc = proc_std ** 2

    for _ in range(N_sim):
        # 场景1：生成本轮固定的加工时间乘数
        proc_multipliers = {}
        for b_id in sol_partial["batch_composition"]:
            val = np.random.lognormal(mean=mu_proc, sigma=proc_std)
            proc_multipliers[b_id] = min(max(val, 0.5), 2.0)
        scenarios_proc.append(proc_multipliers)

        # 场景2：生成本轮固定的旅行时间乘数
        travel_times = {}
        for (s_node, n_node), params in travel_time_params.items():
            travel_std = params['std']
            mu_travel = travel_std ** 2
            multiplier = np.random.lognormal(mean=mu_travel, sigma=travel_std)
            travel_times[(s_node, n_node)] = min(max(multiplier, 0.5), 2.0) * params['value']
        scenarios_travel.append(travel_times)

    # 然后再遍历客户，使用刚才生成的固定场景进行公平对比
    for n in available_customers:
        total_wait = 0

        for sim_idx in range(N_sim):
            # 🌟【使用统一场景】：直接提取，不再重新 np.random
            proc_time_multipliers = scenarios_proc[sim_idx]
            sampled_travel_times = scenarios_travel[sim_idx]

            part_completion_times = {}
            production_time_accumulator = 0.0

            for batch_id in sol_partial["batch_sequence"]:
                part_list = sol_partial["batch_composition"].get(batch_id, [])
                if not part_list: continue

                total_volume = sum(parts_data[j]['volume'] for j in part_list)
                max_height = max(parts_data[j]['height'] for j in part_list)
                base_proc_time = printer_params['alpha'] * total_volume + printer_params['beta'] * max_height

                sampled_proc_time = base_proc_time * proc_time_multipliers.get(batch_id, 1.0)
                production_time_accumulator += min(max(sampled_proc_time, 1e-3), 1e5)

                for part_id in part_list:
                    part_completion_times[part_id] = production_time_accumulator

            route = sol_partial["route"]
            truck_time = 0.0
            arrival_time = 0.0
            departure_time = 0.0
            found_customer = False

            for i in range(len(route) - 1):
                s_node, n_node = route[i], route[i + 1]
                travel_time = sampled_travel_times.get((s_node, n_node), 0)
                arrival_time = truck_time + travel_time

                if n_node == n:
                    found_customer = True
                    customer_parts = problem_data["customer_data"][n]['parts']
                    last_part_completion_time = max([part_completion_times.get(p, 0.0) for p in customer_parts] + [0.0])
                    departure_time = max(arrival_time, last_part_completion_time)
                    break

                if n_node != 0:
                    customer_parts = problem_data["customer_data"][n_node]['parts']
                    last_part_completion_time = max([part_completion_times.get(p, 0.0) for p in customer_parts] + [0.0])
                    truck_time = max(arrival_time, last_part_completion_time)

            if found_customer:
                wait_time = max(0, departure_time - arrival_time)
                total_wait += wait_time

        AvgWait[n] = total_wait / N_sim if N_sim > 0 else 0

    if not AvgWait:
        return random_customer_removal(solution, problem_data, q)

    sorted_customers = sorted(AvgWait.items(), key=lambda item: item[1], reverse=True)
    num_to_remove = min(q, len(sorted_customers))
    removed_customers = [c[0] for c in sorted_customers[:num_to_remove]]

    sol_partial["route"] = [c for c in sol_partial["route"] if c not in removed_customers]

    parts_to_remove = set()
    for cust_id in removed_customers:
        parts_to_remove.update(problem_data["customer_data"][cust_id]['parts'])

    empty_batches = []
    for batch_id, part_list in list(sol_partial["batch_composition"].items()):
        sol_partial["batch_composition"][batch_id] = [p for p in part_list if p not in parts_to_remove]
        if not sol_partial["batch_composition"][batch_id]: empty_batches.append(batch_id)

    for b_id in empty_batches: del sol_partial["batch_composition"][b_id]
    sol_partial["batch_sequence"] = [b for b in sol_partial["batch_sequence"] if b not in empty_batches]

    return sol_partial, removed_customers

# ==============================================================================
# EXP版本（确定性）的破坏算子
# ==============================================================================

def exp_synchronization_critical_removal(solution, problem_data, q):
    """
    同步关键节点移除（确定性等价版）
    不使用仿真，直接用确定性完工时间和到达时间计算等待时间。
    """
    sol_partial = copy.deepcopy(solution)
    available_customers = [c for c in sol_partial["route"] if c != 0]
    if not available_customers: return sol_partial, []

    # 1. 确定性地计算所有零件的完工时间
    part_completion_times = {}
    time_acc = 0.0
    for batch_id in sol_partial["batch_sequence"]:
        parts = sol_partial["batch_composition"].get(batch_id, [])
        if not parts: continue
        vol = sum(problem_data["part_data"][p]['volume'] for p in parts)
        h = max(problem_data["part_data"][p]['height'] for p in parts)
        # 用众数/均值直接计算
        base_t = problem_data["printer_params"]['alpha'] * vol + problem_data["printer_params"]['beta'] * h
        time_acc += base_t * problem_data["proc_uncertainty"].get("base_value", 1.0)
        for p in parts:
            part_completion_times[p] = time_acc

    # 2. 确定性地推演车辆路线，计算等待时间
    WaitScore = {}
    truck_time = 0.0
    for i in range(len(sol_partial["route"]) - 1):
        s_node, n_node = sol_partial["route"][i], sol_partial["route"][i + 1]
        travel_time = problem_data["travel_time_params"].get((s_node, n_node), {'value': 0})['value']
        arrival_time = truck_time + travel_time

        if n_node == 0: break

        c_parts = problem_data["customer_data"][n_node]['parts']
        last_part_time = max([part_completion_times.get(p, 0) for p in c_parts] + [0])

        # 确定性等待时间 = max(0, 零件完工时间 - 卡车到达时间)
        wait_time = max(0, last_part_time - arrival_time)
        WaitScore[n_node] = wait_time

        truck_time = max(arrival_time, last_part_time)

    if sum(WaitScore.values()) == 0:
        return random_customer_removal(solution, problem_data, q)

    # 3. 按等待时间排序并移除
    sorted_customers = sorted(WaitScore.items(), key=lambda x: x[1], reverse=True)
    removed_customers = [c[0] for c in sorted_customers[:min(q, len(sorted_customers))]]

    sol_partial["route"] = [c for c in sol_partial["route"] if c not in removed_customers]
    parts_to_remove = set()
    for cust_id in removed_customers:
        parts_to_remove.update(problem_data["customer_data"][cust_id]['parts'])

    empty_batches = []
    for batch_id, part_list in list(sol_partial["batch_composition"].items()):
        sol_partial["batch_composition"][batch_id] = [p for p in part_list if p not in parts_to_remove]
        if not sol_partial["batch_composition"][batch_id]: empty_batches.append(batch_id)
    for b_id in empty_batches: del sol_partial["batch_composition"][b_id]
    sol_partial["batch_sequence"] = [b for b in sol_partial["batch_sequence"] if b not in empty_batches]

    return sol_partial, removed_customers


def exp_high_variance_exposure_removal(solution, problem_data, q):
    """
    高方差暴露移除（解析近似版）
    不使用仿真，而是通过累加路径和打印的参数标准差，作为风险代理指标。
    """
    sol_partial = copy.deepcopy(solution)
    available_customers = [c for c in sol_partial["route"] if c != 0]
    if not available_customers: return sol_partial, []

    # 1. 计算均值和方差代理（打印机侧）
    part_completion_times = {}
    part_variance = {}
    time_acc = 0.0
    var_acc = 0.0

    for batch_id in sol_partial["batch_sequence"]:
        parts = sol_partial["batch_composition"].get(batch_id, [])
        if not parts: continue
        vol = sum(problem_data["part_data"][p]['volume'] for p in parts)
        h = max(problem_data["part_data"][p]['height'] for p in parts)
        base_t = problem_data["printer_params"]['alpha'] * vol + problem_data["printer_params"]['beta'] * h

        mean_t = base_t * problem_data["proc_uncertainty"].get("base_value", 1.0)
        # 用基础时间 * std作为方差累加的代理值
        batch_var = (base_t * problem_data["proc_uncertainty"].get("std", 0.1)) ** 2

        time_acc += mean_t
        var_acc += batch_var
        for p in parts:
            part_completion_times[p] = time_acc
            part_variance[p] = var_acc

    # 2. 计算均值和方差代理（车辆侧）
    RiskScore = {}
    truck_time = 0.0
    truck_var = 0.0

    for i in range(len(sol_partial["route"]) - 1):
        s_node, n_node = sol_partial["route"][i], sol_partial["route"][i + 1]
        travel_params = problem_data["travel_time_params"].get((s_node, n_node), {'value': 0, 'std': 0})

        truck_time += travel_params['value']
        truck_var += (travel_params.get('std', 0) * travel_params['value']) ** 2

        if n_node == 0: break

        c_parts = problem_data["customer_data"][n_node]['parts']
        max_part_time = max([part_completion_times.get(p, 0) for p in c_parts] + [0])
        max_part_var = max([part_variance.get(p, 0) for p in c_parts] + [0])

        departure_time = max(truck_time, max_part_time)

        # 风险 = 离开时间均值 + 综合标准差
        combined_std = math.sqrt(truck_var) + math.sqrt(max_part_var)
        RiskScore[n_node] = departure_time + 1.0 * combined_std

        truck_time = departure_time

    # 3. 排序并移除
    sorted_customers = sorted(RiskScore.items(), key=lambda x: x[1], reverse=True)
    removed_customers = [c[0] for c in sorted_customers[:min(q, len(sorted_customers))]]

    sol_partial["route"] = [c for c in sol_partial["route"] if c not in removed_customers]
    parts_to_remove = set()
    for cust_id in removed_customers:
        parts_to_remove.update(problem_data["customer_data"][cust_id]['parts'])

    empty_batches = []
    for batch_id, part_list in list(sol_partial["batch_composition"].items()):
        sol_partial["batch_composition"][batch_id] = [p for p in part_list if p not in parts_to_remove]
        if not sol_partial["batch_composition"][batch_id]: empty_batches.append(batch_id)
    for b_id in empty_batches: del sol_partial["batch_composition"][b_id]
    sol_partial["batch_sequence"] = [b for b in sol_partial["batch_sequence"] if b not in empty_batches]

    return sol_partial, removed_customers

def related_removal(solution, problem_data, q):
    sol_partial = copy.deepcopy(solution)
    available_customers = [c for c in sol_partial["route"] if c != 0]
    if len(available_customers) <= q:
        return random_customer_removal(solution, problem_data, q)

    seed_customer = random.choice(available_customers)
    relatedness = []
    w1_time = 0.6
    w2_due_date = 0.4

    for cust_id in available_customers:
        if cust_id == seed_customer: continue

        key = (seed_customer, cust_id)
        if key in problem_data["travel_time_params"]:
            travel_time_metric = problem_data["travel_time_params"][key]['value']
        else:
            travel_time_metric = float('inf')

        due_date_metric = abs(problem_data["customer_data"][seed_customer]['due_date'] -
                              problem_data["customer_data"][cust_id]['due_date'])

        score = w1_time * travel_time_metric + w2_due_date * due_date_metric
        relatedness.append((cust_id, score))

    relatedness.sort(key=lambda item: item[1])
    num_to_remove = min(q - 1, len(relatedness))
    removed_customers = [seed_customer] + [item[0] for item in relatedness[:num_to_remove]]

    sol_partial["route"] = [c for c in sol_partial["route"] if c not in removed_customers]
    parts_to_remove = {p for cust_id in removed_customers for p in problem_data["customer_data"][cust_id]['parts']}

    empty_batches = []
    for batch_id, part_list in list(sol_partial["batch_composition"].items()):
        sol_partial["batch_composition"][batch_id] = [p for p in part_list if p not in parts_to_remove]
        if not sol_partial["batch_composition"][batch_id]: empty_batches.append(batch_id)
    for b_id in empty_batches: del sol_partial["batch_composition"][b_id]
    sol_partial["batch_sequence"] = [b for b in sol_partial["batch_sequence"] if b not in empty_batches]

    return sol_partial, removed_customers

#endregion


#region修复算子
def random_repair(sol_partial, removed_customers, problem_data):
    """
    随机修复算子 (Random Repair Operator)
    完全不考虑成本与时间，随机选择合法位置插入客户和零件。
    目的：最大化解空间的多样性（Exploration），协助算法跳出局部最优陷阱。
    """
    new_sol = copy.deepcopy(sol_partial)

    # ==========================================
    # 1. 随机修复车辆路线 (Random Route Repair)
    # ==========================================
    # 打乱被移除客户的顺序，增加随机性
    customers_to_insert = copy.deepcopy(removed_customers)
    random.shuffle(customers_to_insert)

    for cust_id in customers_to_insert:
        # 路线的头尾是仓库 0，因此可插入的有效索引范围是 1 到 len(route) - 1
        # 例如 route = [0, 2, 5, 0]，可插入的索引为 1, 2, 3
        insert_pos = random.randint(1, len(new_sol["route"]) - 1)
        new_sol["route"].insert(insert_pos, cust_id)

    # ==========================================
    # 2. 随机修复打印批次 (Random Batch Repair)
    # ==========================================
    # 收集所有需要重新安排的零件，并打乱顺序
    parts_to_insert = []
    for cust_id in removed_customers:
        # 兼容 key 是 int 或 str 的情况
        cust_key = str(cust_id) if str(cust_id) in problem_data["customer_data"] else cust_id
        parts_to_insert.extend(problem_data["customer_data"][cust_key]['parts'])

    random.shuffle(parts_to_insert)

    for part_id in parts_to_insert:
        part_area = problem_data["part_data"][part_id]['area']

        # 找出当前所有“面积足够装下该零件”的合法批次
        feasible_batches = []
        for batch_id in new_sol["batch_sequence"]:
            current_area = sum(problem_data["part_data"][p]['area'] for p in new_sol["batch_composition"][batch_id])
            if current_area + part_area <= problem_data["printer_params"]['A_max']:
                feasible_batches.append(batch_id)

        # 随机决定去向：
        # 即使有合法批次，也有 20% 的极小概率强制新开一个批次（彻底的随机扰动）
        # 否则，从合法批次中随机挑一个塞进去
        if feasible_batches and random.random() < 0.8:
            chosen_batch = random.choice(feasible_batches)
            new_sol["batch_composition"][chosen_batch].append(part_id)
        else:
            # 新开一个批次
            new_batch_id = f"B_rand_{part_id}_{random.randint(1000, 9999)}"
            new_sol["batch_composition"][new_batch_id] = [part_id]

            # 为了彻底的随机性，新批次不一定非要加在最后，而是随机插入到批次加工序列中
            batch_insert_pos = random.randint(0, len(new_sol["batch_sequence"]))
            new_sol["batch_sequence"].insert(batch_insert_pos, new_batch_id)

    #零件防丢
    all_parts_in_batches = {p for parts in new_sol["batch_composition"].values() for p in parts}
    required_parts = {p for c in new_sol["route"] if c != 0 for p in problem_data["customer_data"][c]['parts']}
    missing_parts = required_parts - all_parts_in_batches

    if missing_parts:
        for part_id in missing_parts:
            part_area = problem_data["part_data"][part_id]['area']
            inserted = False
            for batch_id in new_sol["batch_sequence"]:
                current_area = sum(problem_data["part_data"][p]['area'] for p in new_sol["batch_composition"][batch_id])
                if current_area + part_area <= problem_data["printer_params"]['A_max']:
                    new_sol["batch_composition"][batch_id].append(part_id)
                    inserted = True
                    break
            if not inserted:
                new_batch_id = f"B_fix_{part_id}_{random.randint(1000, 9999)}"
                new_sol["batch_composition"][new_batch_id] = [part_id]
                new_sol["batch_sequence"].append(new_batch_id)


    return new_sol




def greedy_repair(sol_partial, removed_customers, problem_data):
    """
    Regret-2 修复算子：优先插入“如果不插在最好位置，代价增加最大”的客户
    """
    new_sol = copy.deepcopy(sol_partial)
    unassigned_customers = copy.deepcopy(removed_customers)

    # 当还有客户没有被重新插入车辆路线时
    while unassigned_customers:
        regret_scores = {}
        best_insert_info = {}

        # 遍历所有还没安排的客户
        for cust_id in unassigned_customers:
            insertion_costs = []

            # 评估插入当前路线所有可能缝隙的成本增量
            for i in range(1, len(new_sol["route"])):
                prev_node = new_sol["route"][i - 1]
                next_node = new_sol["route"][i]

                # 获取旅行时间
                t_prev_cust = problem_data["travel_time_params"].get((prev_node, cust_id), {'value': float('inf')})[
                    'value']
                t_cust_next = problem_data["travel_time_params"].get((cust_id, next_node), {'value': float('inf')})[
                    'value']
                t_prev_next = problem_data["travel_time_params"].get((prev_node, next_node), {'value': 0})['value']

                # 计算增量
                cost_increase = t_prev_cust + t_cust_next - t_prev_next
                insertion_costs.append((cost_increase, i))

            # 如果找不到任何合法插入位置（通常不会发生，除非有硬约束），设为无穷大
            if not insertion_costs:
                continue

            # 按成本增量从小到大排序
            insertion_costs.sort(key=lambda x: x[0])

            best_cost, best_pos = insertion_costs[0]
            # 如果只有一个位置可插，懊悔值为 0
            if len(insertion_costs) > 1:
                second_best_cost = insertion_costs[1][0]
                regret_val = second_best_cost - best_cost
            else:
                regret_val = 0

            regret_scores[cust_id] = regret_val
            best_insert_info[cust_id] = best_pos

        # 找出懊悔值最大的客户
        # 也就是：如果不把他安排在 best_pos，改安排在次优位置，路线增加得最多的那个客户
        if not regret_scores:
            break

        target_customer = max(regret_scores.items(), key=lambda x: x[1])[0]
        target_pos = best_insert_info[target_customer]

        # 实施插入
        new_sol["route"].insert(target_pos, target_customer)
        unassigned_customers.remove(target_customer)

    parts_to_insert = []
    for cust_id in removed_customers:
        parts_to_insert.extend(problem_data["customer_data"][cust_id]['parts'])

    # 零件寻找最近能放得下的批次 插入
    for part_id in parts_to_insert:
        part_area = problem_data["part_data"][part_id]['area']
        inserted = False

        for batch_id in new_sol["batch_sequence"]:
            current_area = sum(problem_data["part_data"][p]['area'] for p in new_sol["batch_composition"][batch_id])
            if current_area + part_area <= problem_data["printer_params"]['A_max']:
                new_sol["batch_composition"][batch_id].append(part_id)
                inserted = True
                break

        if not inserted:
            new_batch_id = f"B_new_{part_id}_{random.randint(1000, 9999)}"
            new_sol["batch_composition"][new_batch_id] = [part_id]
            new_sol["batch_sequence"].append(new_batch_id)

    #零件防丢
    all_parts_in_batches = {p for parts in new_sol["batch_composition"].values() for p in parts}
    required_parts = {p for c in new_sol["route"] if c != 0 for p in problem_data["customer_data"][c]['parts']}
    missing_parts = required_parts - all_parts_in_batches

    if missing_parts:
        for part_id in missing_parts:
            part_area = problem_data["part_data"][part_id]['area']
            inserted = False
            for batch_id in new_sol["batch_sequence"]:
                current_area = sum(problem_data["part_data"][p]['area'] for p in new_sol["batch_composition"][batch_id])
                if current_area + part_area <= problem_data["printer_params"]['A_max']:
                    new_sol["batch_composition"][batch_id].append(part_id)
                    inserted = True
                    break
            if not inserted:
                new_batch_id = f"B_fix_{part_id}_{random.randint(1000, 9999)}"
                new_sol["batch_composition"][new_batch_id] = [part_id]
                new_sol["batch_sequence"].append(new_batch_id)

    return new_sol


def _calculate_batch_proc_time(part_list, problem_data):
    if not part_list: return 0
    total_volume = sum(problem_data["part_data"][j]['volume'] for j in part_list)
    max_height = max(problem_data["part_data"][j]['height'] for j in part_list)
    return problem_data["printer_params"]['alpha'] * total_volume + problem_data["printer_params"]['beta'] * max_height


def best_batch_insertion_repair(sol_partial, removed_customers, problem_data):
    new_sol = copy.deepcopy(sol_partial)

    for cust_id in removed_customers:
        best_pos, min_cost_increase = -1, float('inf')
        for i in range(1, len(new_sol["route"])):
            prev_node, next_node = new_sol["route"][i - 1], new_sol["route"][i]

            key1 = (prev_node, cust_id)
            key2 = (cust_id, next_node)
            key3 = (prev_node, next_node)

            if key1 in problem_data["travel_time_params"] and key2 in problem_data["travel_time_params"] and key3 in \
                    problem_data["travel_time_params"]:
                cost_increase = (problem_data["travel_time_params"][key1]['value'] +
                                 problem_data["travel_time_params"][key2]['value'] -
                                 problem_data["travel_time_params"][key3]['value'])
                if cost_increase < min_cost_increase:
                    min_cost_increase, best_pos = cost_increase, i
        if best_pos != -1:
            new_sol["route"].insert(best_pos, cust_id)

    parts_to_insert = [p for c in removed_customers for p in problem_data["customer_data"][c]['parts']]

    ##插入批次时，寻找导致打印时间增量最小的方式
    for part_id in parts_to_insert:
        best_option = {'type': 'new', 'batch_id': None, 'cost': float('inf')}
        part_area = problem_data["part_data"][part_id]['area']

        cost_new_batch = _calculate_batch_proc_time([part_id], problem_data)
        best_option['cost'] = cost_new_batch

        for batch_id in new_sol["batch_sequence"]:
            current_parts = new_sol["batch_composition"][batch_id]
            current_area = sum(problem_data["part_data"][p]['area'] for p in current_parts)

            if current_area + part_area <= problem_data["printer_params"]['A_max']:
                old_proc_time = _calculate_batch_proc_time(current_parts, problem_data)
                new_proc_time = _calculate_batch_proc_time(current_parts + [part_id], problem_data)
                cost_increase = new_proc_time - old_proc_time

                if cost_increase < best_option['cost']:
                    best_option = {'type': 'existing', 'batch_id': batch_id, 'cost': cost_increase}

        if best_option['type'] == 'new':
            new_batch_id = f"B_new_{part_id}_{random.randint(1000, 9999)}"
            new_sol["batch_composition"][new_batch_id] = [part_id]
            new_sol["batch_sequence"].append(new_batch_id)
        else:
            new_sol["batch_composition"][best_option['batch_id']].append(part_id)

    #零件防丢
    all_parts_in_batches = {p for parts in new_sol["batch_composition"].values() for p in parts}
    required_parts = {p for c in new_sol["route"] if c != 0 for p in problem_data["customer_data"][c]['parts']}
    missing_parts = required_parts - all_parts_in_batches

    if missing_parts:
        for part_id in missing_parts:
            part_area = problem_data["part_data"][part_id]['area']
            inserted = False
            for batch_id in new_sol["batch_sequence"]:
                current_area = sum(problem_data["part_data"][p]['area'] for p in new_sol["batch_composition"][batch_id])
                if current_area + part_area <= problem_data["printer_params"]['A_max']:
                    new_sol["batch_composition"][batch_id].append(part_id)
                    inserted = True
                    break
            if not inserted:
                new_batch_id = f"B_fix_{part_id}_{random.randint(1000, 9999)}"
                new_sol["batch_composition"][new_batch_id] = [part_id]
                new_sol["batch_sequence"].append(new_batch_id)


    return new_sol


# def slack_aware_batch_repair(sol_partial, removed_customers, problem_data):
#     new_sol = copy.deepcopy(sol_partial)
#
#     ####顾客插入方式不变
#     for cust_id in removed_customers:
#         best_pos, min_cost_increase = -1, float('inf')
#         for i in range(1, len(new_sol["route"])):
#             prev_node, next_node = new_sol["route"][i - 1], new_sol["route"][i]
#
#             key1 = (prev_node, cust_id)
#             key2 = (cust_id, next_node)
#             key3 = (prev_node, next_node)
#
#             if key1 in problem_data["travel_time_params"] and key2 in problem_data["travel_time_params"] and key3 in \
#                     problem_data["travel_time_params"]:
#                 cost_increase = (problem_data["travel_time_params"][key1]['value'] +
#                                  problem_data["travel_time_params"][key2]['value'] -
#                                  problem_data["travel_time_params"][key3]['value'])
#                 if cost_increase < min_cost_increase:
#                     min_cost_increase, best_pos = cost_increase, i
#         if best_pos != -1:
#             new_sol["route"].insert(best_pos, cust_id)
#
#     parts_to_insert = [p for c in removed_customers for p in problem_data["customer_data"][c]['parts']]
#
#     def calculate_batch_completion_times(solution, problem_data):
#         completion_times = {}
#         current_time = 0.0
#         for batch_id in solution["batch_sequence"]:
#             part_list = solution["batch_composition"].get(batch_id, [])
#             if not part_list: continue
#             proc_time = _calculate_batch_proc_time(part_list, problem_data)
#             current_time += proc_time
#             completion_times[batch_id] = current_time
#         return completion_times
#
#     def calculate_customer_tardiness(solution, problem_data, batch_completion_times):
#         customer_tardiness = {}
#         truck_time = 0.0
#         route = solution["route"]
#
#         for i in range(len(route) - 1):
#             s_node, n_node = route[i], route[i + 1]
#             if n_node == 0: break
#
#             key = (s_node, n_node)
#             travel_time = problem_data["travel_time_params"].get(key, {'value': 0})['value']
#             arrival_time = truck_time + travel_time
#
#             customer_parts = problem_data["customer_data"][n_node]['parts']
#             relevant_batches = {b for b, parts in solution["batch_composition"].items() if
#                                 any(p in parts for p in customer_parts)}
#             max_batch_completion = max([batch_completion_times.get(b, 0) for b in relevant_batches] + [0])
#
#             departure_time = max(arrival_time, max_batch_completion)
#             due_date = problem_data["customer_data"][n_node]['due_date']
#             customer_tardiness[n_node] = max(0, departure_time - due_date)
#             truck_time = departure_time
#
#         return customer_tardiness
#
#     #####计算零件插入后使客户延迟tardiness增加量
#     initial_batch_completion = calculate_batch_completion_times(new_sol, problem_data)
#     initial_customer_tardiness = calculate_customer_tardiness(new_sol, problem_data, initial_batch_completion)
#
#     for part_id in parts_to_insert:
#         best_batch = None
#         best_cost = float('inf')
#         part_area = problem_data["part_data"][part_id]['area']
#
#         part_customer = None
#         for cust_id, cust_info in problem_data["customer_data"].items():
#             if part_id in cust_info['parts']:
#                 part_customer = cust_id
#                 break
#
#         if not part_customer:
#             inserted = False
#             for batch_id in new_sol["batch_sequence"]:
#                 current_area = sum(problem_data["part_data"][p]['area'] for p in new_sol["batch_composition"][batch_id])
#                 if current_area + part_area <= problem_data["printer_params"]['A_max']:
#                     new_sol["batch_composition"][batch_id].append(part_id)
#                     inserted = True
#                     break
#             if not inserted:
#                 new_batch_id = f"B_new_{part_id}_{random.randint(1000, 9999)}"
#                 new_sol["batch_composition"][new_batch_id] = [part_id]
#                 new_sol["batch_sequence"].append(new_batch_id)
#             continue
#
#         for batch_id in new_sol["batch_sequence"]:
#             current_parts = new_sol["batch_composition"][batch_id]
#             current_area = sum(problem_data["part_data"][p]['area'] for p in current_parts)
#
#             if current_area + part_area <= problem_data["printer_params"]['A_max']:
#                 temp_parts = current_parts + [part_id]
#                 new_proc_time = _calculate_batch_proc_time(temp_parts, problem_data)
#
#                 temp_batch_completion = initial_batch_completion.copy()
#                 batch_index = new_sol["batch_sequence"].index(batch_id)
#                 current_time = 0.0
#                 for i, b_id in enumerate(new_sol["batch_sequence"]):
#                     if i < batch_index:
#                         current_time = temp_batch_completion.get(b_id, 0)
#                     elif i == batch_index:
#                         current_time += new_proc_time
#                         temp_batch_completion[b_id] = current_time
#                     else:
#                         proc_t = _calculate_batch_proc_time(new_sol["batch_composition"].get(b_id, []), problem_data)
#                         current_time += proc_t
#                         temp_batch_completion[b_id] = current_time
#
#                 temp_tardiness = calculate_customer_tardiness(new_sol, problem_data, temp_batch_completion)
#                 new_tardiness = temp_tardiness.get(part_customer, 0)
#                 original_tardiness = initial_customer_tardiness.get(part_customer, 0)
#                 delta_tard = new_tardiness - original_tardiness
#
#                 if delta_tard < best_cost:
#                     best_cost = delta_tard
#                     best_batch = batch_id
#
#         new_batch_proc_time = _calculate_batch_proc_time([part_id], problem_data)
#         temp_batch_completion = initial_batch_completion.copy()
#         new_batch_id = f"B_new_temp_{part_id}"
#         temp_batch_sequence = new_sol["batch_sequence"] + [new_batch_id]
#
#         current_time = 0.0
#         for b_id in temp_batch_sequence:
#             if b_id == new_batch_id:
#                 current_time += new_batch_proc_time
#                 temp_batch_completion[b_id] = current_time
#             else:
#                 current_time = temp_batch_completion.get(b_id, 0)
#
#         temp_tardiness = calculate_customer_tardiness(new_sol, problem_data, temp_batch_completion)
#         new_tardiness = temp_tardiness.get(part_customer, 0)
#         original_tardiness = initial_customer_tardiness.get(part_customer, 0)
#         delta_tard = new_tardiness - original_tardiness
#
#         if delta_tard < best_cost:
#             best_cost = delta_tard
#             best_batch = "new"
#
#         if best_batch == "new":
#             final_new_batch_id = f"B_new_{part_id}_{random.randint(1000, 9999)}"
#             new_sol["batch_composition"][final_new_batch_id] = [part_id]
#             new_sol["batch_sequence"].append(final_new_batch_id)
#         elif best_batch:
#             new_sol["batch_composition"][best_batch].append(part_id)
#         else:
#             inserted = False
#             for batch_id in new_sol["batch_sequence"]:
#                 current_area = sum(problem_data["part_data"][p]['area'] for p in new_sol["batch_composition"][batch_id])
#                 if current_area + part_area <= problem_data["printer_params"]['A_max']:
#                     new_sol["batch_composition"][batch_id].append(part_id)
#                     inserted = True
#                     break
#             if not inserted:
#                 new_batch_id = f"B_fallback_{part_id}_{random.randint(1000, 9999)}"
#                 new_sol["batch_composition"][new_batch_id] = [part_id]
#                 new_sol["batch_sequence"].append(new_batch_id)
#
#     return new_sol

def slack_aware_batch_repair(sol_partial, removed_customers, problem_data):
    new_sol = copy.deepcopy(sol_partial)

    # 【新增核心逻辑】：按客户的交货期 (due_date) 从小到大排序
    # 越紧急的客户排在越前面，优先插入车辆路线
    removed_customers.sort(key=lambda c: problem_data["customer_data"][c]['due_date'])

    for cust_id in removed_customers:
        best_pos, min_cost_increase = -1, float('inf')
        for i in range(1, len(new_sol["route"])):
            prev_node, next_node = new_sol["route"][i - 1], new_sol["route"][i]

            key1 = (prev_node, cust_id)
            key2 = (cust_id, next_node)
            key3 = (prev_node, next_node)

            if key1 in problem_data["travel_time_params"] and key2 in problem_data["travel_time_params"] and key3 in \
                    problem_data["travel_time_params"]:
                cost_increase = (problem_data["travel_time_params"][key1]['value'] +
                                 problem_data["travel_time_params"][key2]['value'] -
                                 problem_data["travel_time_params"][key3]['value'])
                if cost_increase < min_cost_increase:
                    min_cost_increase, best_pos = cost_increase, i
        if best_pos != -1:
            new_sol["route"].insert(best_pos, cust_id)

    parts_to_insert = [p for c in removed_customers for p in problem_data["customer_data"][c]['parts']]

    def calculate_batch_completion_times(solution, problem_data):
        completion_times = {}
        current_time = 0.0
        for batch_id in solution["batch_sequence"]:
            part_list = solution["batch_composition"].get(batch_id, [])
            if not part_list: continue
            proc_time = _calculate_batch_proc_time(part_list, problem_data)
            current_time += proc_time
            completion_times[batch_id] = current_time
        return completion_times

    def calculate_customer_tardiness(solution, problem_data, batch_completion_times):
        customer_tardiness = {}
        truck_time = 0.0
        route = solution["route"]

        for i in range(len(route) - 1):
            s_node, n_node = route[i], route[i + 1]
            if n_node == 0: break

            key = (s_node, n_node)
            travel_time = problem_data["travel_time_params"].get(key, {'value': 0})['value']
            arrival_time = truck_time + travel_time

            customer_parts = problem_data["customer_data"][n_node]['parts']
            relevant_batches = {b for b, parts in solution["batch_composition"].items() if
                                any(p in parts for p in customer_parts)}
            max_batch_completion = max([batch_completion_times.get(b, 0) for b in relevant_batches] + [0])

            departure_time = max(arrival_time, max_batch_completion)
            due_date = problem_data["customer_data"][n_node]['due_date']
            customer_tardiness[n_node] = max(0, departure_time - due_date)
            truck_time = departure_time

        return customer_tardiness

    #####计算零件插入后使客户延迟tardiness增加量
    initial_batch_completion = calculate_batch_completion_times(new_sol, problem_data)
    initial_customer_tardiness = calculate_customer_tardiness(new_sol, problem_data, initial_batch_completion)

    for part_id in parts_to_insert:
        best_batch = None
        best_cost = float('inf')
        part_area = problem_data["part_data"][part_id]['area']

        part_customer = None
        for cust_id, cust_info in problem_data["customer_data"].items():
            if part_id in cust_info['parts']:
                part_customer = cust_id
                break

        if not part_customer:
            inserted = False
            for batch_id in new_sol["batch_sequence"]:
                current_area = sum(problem_data["part_data"][p]['area'] for p in new_sol["batch_composition"][batch_id])
                if current_area + part_area <= problem_data["printer_params"]['A_max']:
                    new_sol["batch_composition"][batch_id].append(part_id)
                    inserted = True
                    break
            if not inserted:
                new_batch_id = f"B_new_{part_id}_{random.randint(1000, 9999)}"
                new_sol["batch_composition"][new_batch_id] = [part_id]
                new_sol["batch_sequence"].append(new_batch_id)
            continue

        for batch_id in new_sol["batch_sequence"]:
            current_parts = new_sol["batch_composition"][batch_id]
            current_area = sum(problem_data["part_data"][p]['area'] for p in current_parts)

            if current_area + part_area <= problem_data["printer_params"]['A_max']:
                temp_parts = current_parts + [part_id]
                new_proc_time = _calculate_batch_proc_time(temp_parts, problem_data)

                temp_batch_completion = initial_batch_completion.copy()
                batch_index = new_sol["batch_sequence"].index(batch_id)
                current_time = 0.0
                for i, b_id in enumerate(new_sol["batch_sequence"]):
                    if i < batch_index:
                        current_time = temp_batch_completion.get(b_id, 0)
                    elif i == batch_index:
                        current_time += new_proc_time
                        temp_batch_completion[b_id] = current_time
                    else:
                        proc_t = _calculate_batch_proc_time(new_sol["batch_composition"].get(b_id, []), problem_data)
                        current_time += proc_t
                        temp_batch_completion[b_id] = current_time

                temp_tardiness = calculate_customer_tardiness(new_sol, problem_data, temp_batch_completion)
                new_tardiness = temp_tardiness.get(part_customer, 0)
                original_tardiness = initial_customer_tardiness.get(part_customer, 0)
                delta_tard = new_tardiness - original_tardiness

                if delta_tard < best_cost:
                    best_cost = delta_tard
                    best_batch = batch_id

        new_batch_proc_time = _calculate_batch_proc_time([part_id], problem_data)
        temp_batch_completion = initial_batch_completion.copy()
        new_batch_id = f"B_new_temp_{part_id}"
        temp_batch_sequence = new_sol["batch_sequence"] + [new_batch_id]

        current_time = 0.0
        for b_id in temp_batch_sequence:
            if b_id == new_batch_id:
                current_time += new_batch_proc_time
                temp_batch_completion[b_id] = current_time
            else:
                current_time = temp_batch_completion.get(b_id, 0)

        temp_tardiness = calculate_customer_tardiness(new_sol, problem_data, temp_batch_completion)
        new_tardiness = temp_tardiness.get(part_customer, 0)
        original_tardiness = initial_customer_tardiness.get(part_customer, 0)
        delta_tard = new_tardiness - original_tardiness

        if delta_tard < best_cost:
            best_cost = delta_tard
            best_batch = "new"

        if best_batch == "new":
            final_new_batch_id = f"B_new_{part_id}_{random.randint(1000, 9999)}"
            new_sol["batch_composition"][final_new_batch_id] = [part_id]
            new_sol["batch_sequence"].append(final_new_batch_id)
        elif best_batch:
            new_sol["batch_composition"][best_batch].append(part_id)
        else:
            inserted = False
            for batch_id in new_sol["batch_sequence"]:
                current_area = sum(problem_data["part_data"][p]['area'] for p in new_sol["batch_composition"][batch_id])
                if current_area + part_area <= problem_data["printer_params"]['A_max']:
                    new_sol["batch_composition"][batch_id].append(part_id)
                    inserted = True
                    break
            if not inserted:
                new_batch_id = f"B_fallback_{part_id}_{random.randint(1000, 9999)}"
                new_sol["batch_composition"][new_batch_id] = [part_id]
                new_sol["batch_sequence"].append(new_batch_id)

    return new_sol

def risk_balancing_batch_repair(sol_partial, removed_customers, problem_data):
    new_sol = copy.deepcopy(sol_partial)

    for cust_id in removed_customers:
        best_pos, min_cost_increase = -1, float('inf')
        for i in range(1, len(new_sol["route"])):
            prev_node, next_node = new_sol["route"][i - 1], new_sol["route"][i]

            key1 = (prev_node, cust_id)
            key2 = (cust_id, next_node)
            key3 = (prev_node, next_node)

            if key1 in problem_data["travel_time_params"] and key2 in problem_data["travel_time_params"] and key3 in \
                    problem_data["travel_time_params"]:
                cost_increase = (problem_data["travel_time_params"][key1]['value'] +
                                 problem_data["travel_time_params"][key2]['value'] -
                                 problem_data["travel_time_params"][key3]['value'])
                # base_cost_increase = (problem_data["travel_time_params"][key1]['value'] +
                #                  problem_data["travel_time_params"][key2]['value'] -
                #                  problem_data["travel_time_params"][key3]['value'])
                # # 【新增】：引入 0.8 ~ 1.2 的随机扰动因子
                # noise_factor = random.uniform(0.8, 1.2)
                # cost_increase = base_cost_increase * noise_factor

                if cost_increase < min_cost_increase:
                    min_cost_increase, best_pos = cost_increase, i
        if best_pos != -1:
            new_sol["route"].insert(best_pos, cust_id)

    parts_to_insert = [p for c in removed_customers for p in problem_data["customer_data"][c]['parts']]

    def calculate_part_risk(part_id, problem_data):
        parts_data = problem_data["part_data"]
        if part_id not in parts_data: return 0
        return parts_data[part_id].get('volume', 0) * parts_data[part_id].get('height', 0)

    parts_with_risk = [(part_id, calculate_part_risk(part_id, problem_data)) for part_id in parts_to_insert]
    parts_with_risk.sort(key=lambda x: x[1], reverse=True)
    sorted_parts = [part[0] for part in parts_with_risk]

    for part_id in sorted_parts:
        best_batch = None
        lowest_risk = float('inf')
        part_area = problem_data["part_data"][part_id]['area']

        ###遍历现有批次
        for batch_id in new_sol["batch_sequence"]:
            current_parts = new_sol["batch_composition"][batch_id]
            current_area = sum(problem_data["part_data"][p]['area'] for p in current_parts)

            if current_area + part_area <= problem_data["printer_params"]['A_max']:
                temp_parts = current_parts + [part_id]
                batch_risk = sum(calculate_part_risk(p, problem_data) for p in temp_parts)  ###评估批次累积的总风险度
                if batch_risk < lowest_risk:  ###迫使高风险零件和低风险零件相匹配进同一批次
                    lowest_risk = batch_risk
                    best_batch = batch_id
        ###新建批次
        new_batch_risk = calculate_part_risk(part_id, problem_data)
        if new_batch_risk < lowest_risk:
            lowest_risk = new_batch_risk
            best_batch = "new"

        if best_batch == "new":
            new_batch_id = f"B_new_{part_id}_{random.randint(1000, 9999)}"
            new_sol["batch_composition"][new_batch_id] = [part_id]
            new_sol["batch_sequence"].append(new_batch_id)
        elif best_batch:
            new_sol["batch_composition"][best_batch].append(part_id)
        else:
            inserted = False
            for batch_id in new_sol["batch_sequence"]:
                current_area = sum(problem_data["part_data"][p]['area'] for p in new_sol["batch_composition"][batch_id])
                if current_area + part_area <= problem_data["printer_params"]['A_max']:
                    new_sol["batch_composition"][batch_id].append(part_id)
                    inserted = True
                    break
            if not inserted:
                new_batch_id = f"B_fallback_{part_id}_{random.randint(1000, 9999)}"
                new_sol["batch_composition"][new_batch_id] = [part_id]
                new_sol["batch_sequence"].append(new_batch_id)

    #零件防丢
    all_parts_in_batches = {p for parts in new_sol["batch_composition"].values() for p in parts}
    required_parts = {p for c in new_sol["route"] if c != 0 for p in problem_data["customer_data"][c]['parts']}
    missing_parts = required_parts - all_parts_in_batches

    if missing_parts:
        for part_id in missing_parts:
            part_area = problem_data["part_data"][part_id]['area']
            inserted = False
            for batch_id in new_sol["batch_sequence"]:
                current_area = sum(problem_data["part_data"][p]['area'] for p in new_sol["batch_composition"][batch_id])
                if current_area + part_area <= problem_data["printer_params"]['A_max']:
                    new_sol["batch_composition"][batch_id].append(part_id)
                    inserted = True
                    break
            if not inserted:
                new_batch_id = f"B_fix_{part_id}_{random.randint(1000, 9999)}"
                new_sol["batch_composition"][new_batch_id] = [part_id]
                new_sol["batch_sequence"].append(new_batch_id)

    return new_sol
#endregion

# ==============================================================================
# SECTION 4: ALGORITHM MAINS
# ==============================================================================

def exp_alns(problem_data, max_iter=1000, initial_temp=50, cool_rate=0.995):
    start_time = time.time()
    time_best = 0.0  # 🌟【新增】：初始化找到最优解的时间
    destroy_operators = [
        random_customer_removal, worst_tardiness_removal, related_removal,
        # batch_blocking_part_removal,
        exp_synchronization_critical_removal, exp_high_variance_exposure_removal
    ]
    repair_operators = [
        greedy_repair, best_batch_insertion_repair, slack_aware_batch_repair,
        risk_balancing_batch_repair,random_repair
    ]

    # 🌟【新增1】：初始化算子统计字典
    op_stats = {
        'destroy': {op.__name__: {'used': 0, 'accepted': 0, 'best': 0} for op in destroy_operators},
        'repair': {op.__name__: {'used': 0, 'accepted': 0, 'best': 0} for op in repair_operators}
    }

    # --- 动态计算 q 的取值范围 (基于客户数量的 10% ~ 40%) ---
    num_customers = len(problem_data["customer_data"])
    q_min = max(1, int(0.1 * num_customers))
    #TODO: 将原来的 0.4 改小一点，比如 0.2 或 0.25，防止破坏过猛导致修复后极难被接受
    q_max = max(2, int(0.25 * num_customers))
    # q_max = max(2, int(0.4 * num_customers))
    print(f"EXP-ALNS 动态 q 范围: [{q_min}, {q_max}] (基于 {num_customers} 个客户)")

    sol_current = generate_initial_solution(problem_data)
    sol_best = copy.deepcopy(sol_current)

    f_current = deterministic_mam_vprp(sol_current, problem_data)
    f_best = f_current
    evalu_sol_best = simulate_mam_vprp(sol_best, problem_data)

    print(f"EXP-ALNS Initial solution objective: {f_best}, Eval: {evalu_sol_best:.2f}")

    T = initial_temp
    history = [(0, evalu_sol_best, f_current)]

    # TODO:（权重衰减系数调整,新增初始温度变为动态计算）
    # 🌟【新增】：初始化轮盘赌权重与控制参数
    w_destroy = [1.0] * len(destroy_operators)
    w_repair = [1.0] * len(repair_operators)
    rho = 0.1  # 权重衰减系数 (通常取0.1~0.5)
    # 🌟【新增】：动态计算初始温度
    # 设定初期接受 5% 恶化解的概率为 50%

    # 防御性编程：如果初始解恰好为 0（虽然罕见），给一个最小温差防止除零报错
    worse_gap = f_current * 0.10  # 注：sim_alns 中这里用 f_det_current
    if worse_gap <= 1e-5:
        worse_gap = 10.0

    # 公式: T = - (恶化量) / ln(接受概率)
    T_start = -worse_gap / math.log(0.60)
    T = T_start
    initial_temp_record = T_start  # 🌟 必须记录下来！供后续停滞回火机制使用

    # 3. 动态计算完美冷却率 (cool_rate)
    T_end = 0.1  # 设定算法结束时的极低目标温度
    cool_rate = (T_end / T_start) ** (1.0 / max_iter)

    print(f"Auto-calculated T_start: {T_start:.2f}, Cool Rate: {cool_rate:.6f} for {max_iter} iters")


    # 🌟【新增】：初始化停滞机制参数
    stagnation_counter = 0
    # 设定停滞阈值：比如连续 50 代没有发现全局最优，就触发强制随机
    stagnation_threshold = max(150, int(max_iter * 0.05))


    # for i in range(max_iter):
    #     # 🌟【修改】：使用 random.choices 进行轮盘赌选择 (k=1表示抽1个)
    #     destroy_op = random.choices(destroy_operators, weights=w_destroy, k=1)[0]
    #     repair_op = random.choices(repair_operators, weights=w_repair, k=1)[0]



    for i in range(max_iter):
        # 🌟【修改】：停滞触发与正常轮盘赌的分支逻辑
        if stagnation_counter >= stagnation_threshold:
            #1：exp_alns 停滞时，以“历史最优解”作为新起点重启
            sol_current = copy.deepcopy(sol_best)
            f_current = f_best  # 同步更新当前分数

            # 2. 触发破局：强制使用纯随机算子
            destroy_op = random_customer_removal
            repair_op = random_repair
            current_q = random.randint(q_min, q_max)
            stagnation_counter = 0  # 触发后立即清零
            #TODO:回火
            T = min(initial_temp_record, max(T * 1.5, initial_temp_record * 0.3))
        else:
            # 正常情况：使用轮盘赌自适应选择，q 值随机波动
            destroy_op = random.choices(destroy_operators, weights=w_destroy, k=1)[0]
            repair_op = random.choices(repair_operators, weights=w_repair, k=1)[0]


        sol_temp = copy.deepcopy(sol_current)

        # 🌟【新增2】：记录算子被使用 1 次
        d_name = destroy_op.__name__
        r_name = repair_op.__name__
        op_stats['destroy'][d_name]['used'] += 1
        op_stats['repair'][r_name]['used'] += 1

        # 每次迭代随机抽取一个当前 q，增加搜索的多样性
        current_q = random.randint(q_min, q_max)
        sol_partial, removed_items = destroy_op(sol_temp, problem_data, current_q)
        sol_new = repair_op(sol_partial, removed_items, problem_data)

        f_new = deterministic_mam_vprp(sol_new, problem_data)

        # 🌟【新增3】：标记本次迭代的结果状态
        is_accepted = False
        is_new_best = False

        if f_new < f_current or random.random() < math.exp(-(f_new - f_current) / T):
            sol_current = copy.deepcopy(sol_new)
            f_current = f_new
            is_accepted = True  # 被接受了



        if f_current < f_best:
            sol_best = copy.deepcopy(sol_current)
            f_best = f_current
            is_new_best = True  # 打破记录了
            time_best = time.time() - start_time  # 🌟【新增】：记录打破记录时的耗时
            evalu_sol_best = simulate_mam_vprp(sol_best, problem_data)
            print(f"EXP-ALNS Iter {i + 1}/{max_iter} | New best: {f_best} | Eval: {evalu_sol_best:.2f}")

        # 🌟【新增4】：更新统计字典
        if is_accepted:
            op_stats['destroy'][d_name]['accepted'] += 1
            op_stats['repair'][r_name]['accepted'] += 1
        if is_new_best:
            op_stats['destroy'][d_name]['best'] += 1
            op_stats['repair'][r_name]['best'] += 1


        #TODO:
        # 🌟【修改】：结合得分机制，更新停滞计数器
        if is_new_best:
            score = 20.0
            stagnation_counter = 0  # 发现新大陆，打破停滞，清零！
        elif is_accepted and (f_new < f_current): # (注: sim_alns 中叫 f_det_new < f_det_current)
            score = 5.0
            stagnation_counter += 1 # 局部改善不算打破停滞，继续累加
        elif is_accepted:
            score = 2.0
            stagnation_counter += 1
        else:
            score = 0.0
            stagnation_counter += 1



        # 获取当前算子的索引
        d_idx = destroy_operators.index(destroy_op)
        r_idx = repair_operators.index(repair_op)

        # 🌟【关键修改】：引入 max(0.1, ...) 设定权重下界，防止算子“死亡”
        w_destroy[d_idx] = max(0.1, (1 - rho) * w_destroy[d_idx] + rho * score)
        w_repair[r_idx] = max(0.1, (1 - rho) * w_repair[r_idx] + rho * score)


        # 获取当前算子的索引
        d_idx = destroy_operators.index(destroy_op)
        r_idx = repair_operators.index(repair_op)

        # 运用公式动态更新权重： 新权重 = (1 - rho) * 老权重 + rho * 得分
        w_destroy[d_idx] = (1 - rho) * w_destroy[d_idx] + rho * score
        w_repair[r_idx] = (1 - rho) * w_repair[r_idx] + rho * score#TODO:新增轮盘赌得分权重



        T *= cool_rate
        history.append((i + 1, evalu_sol_best,f_current))

        if (i + 1) % 100 == 0:
            print(f"EXP-ALNS Iter {i + 1}/{max_iter} | Cur: {f_current} | Best: {f_best} | Eval: {evalu_sol_best:.2f}")

    print(f"\nEXP-ALNS finished in {time.time() - start_time:.2f} seconds.")
    return sol_best, evalu_sol_best, history, op_stats, time_best


def _run_simulation_internal(solution, problem_data, N_min, epsilon):
    """
    辅助函数：调用底层的收敛仿真，并强制加上未服务惩罚
    """
    # 1. 确保ID有效
    solution, problem_data = validate_and_filter_ids(problem_data, solution)

    # 2. 检查未服务客户并计算静态惩罚
    # (这一步至关重要，防止仿真函数只计算了Route里的点，而忽略了丢弃的点)
    valid_customers = set(problem_data["customer_data"].keys())
    visited_customers = set(node for node in solution["route"] if node != 0)
    unserved_count = len(valid_customers - visited_customers)
    penalty = unserved_count * 10000.0  # 巨大的惩罚值

    # 3. 运行仿真
    N_max = max(N_min * 20, 2000)  # 提高上限

    _, mean_tardiness, _ = simulate_mam_vprp_convergent(
        solution=solution,
        problem_data=problem_data,
        alpha=0.05,
        epsilon=epsilon,
        N_min=N_min,
        N_max=N_max
    )

    # 返回：仿真得到的延迟 + 未服务惩罚
    return mean_tardiness + penalty


def sim_alns(problem_data, max_iter=1000, initial_temp=50, cool_rate=0.99, pool_size=20):
    """
    Sim-ALNS 算法 (改进版：混合评估策略)

    改进逻辑：
    1. ALNS的接受准则(Acceptance)主要依赖'确定性评估'，因为确定性评估是平滑的，能引导算法快速收敛。
    2. 每当发现一个新的'确定性好解'，或者一定概率下，触发'仿真评估'。
    3. 仿真评估的结果用于更新 Stochastic Pool 和 Best Stochastic Solution。
    这避免了算法被随机噪声误导，同时保留了发现鲁棒解的能力。
    """
    start_time = time.time()
    time_best = 0.0  # 🌟【新增】：初始化找到最优解的时间

    # 定义算子
    destroy_operators = [
        random_customer_removal, worst_tardiness_removal, related_removal,
        # batch_blocking_part_removal,
        sim_synchronization_critical_removal, sim_high_variance_exposure_removal
    ]
    repair_operators = [
        greedy_repair, best_batch_insertion_repair, slack_aware_batch_repair,
        risk_balancing_batch_repair,random_repair
    ]

    # 🌟【新增1】：初始化算子统计字典
    op_stats = {
        'destroy': {op.__name__: {'used': 0, 'accepted': 0, 'best': 0} for op in destroy_operators},
        'repair': {op.__name__: {'used': 0, 'accepted': 0, 'best': 0} for op in repair_operators}
    }


    # 动态 q 范围
    num_customers = len(problem_data["customer_data"])
    q_min = max(1, int(0.1 * num_customers))
    #TODO:将原来的 0.4 改小一点，比如 0.2 或 0.25，防止破坏过猛导致修复后极难被接受
    q_max = max(2, int(0.25 * num_customers))
    # q_max = max(2, int(0.4 * num_customers))
    print(f"SIM-ALNS 动态 q 范围: [{q_min}, {q_max}]")

    # 初始化
    sol_current = generate_initial_solution(problem_data)
    sol_best_det = copy.deepcopy(sol_current)  # 确定性最优

    # 关键：初始化随机最优解（需要先仿真一次）
    # 提高初次仿真的精度，确保起点准确
    INITIAL_SIM_PARAMS = {'N_min': 100, 'epsilon': 0.05}
    f_sim_current = _run_simulation_internal(sol_current, problem_data, **INITIAL_SIM_PARAMS)

    sol_best_stochastic = copy.deepcopy(sol_current)
    f_best_stochastic = f_sim_current

    # 确定性分数
    f_det_current = deterministic_mam_vprp(sol_current, problem_data)
    f_det_best = f_det_current

    # 初始化解池
    stochastic_pool = []
    update_solution_pool(stochastic_pool, sol_current, f_sim_current, pool_size)

    print(f"SIM-ALNS Start | Initial Sim Best: {f_best_stochastic:.2f} | Det Best: {f_det_best:.2f}")

    T = initial_temp
    # 这里的 history 我们记录随机最优的变化
    history = [(0, f_best_stochastic,f_det_current)]

    # --- 搜索参数 ---
    # 在搜索过程中，仿真精度可以适中，只要能区分好坏即可
    SEARCH_SIM_PARAMS = {'N_min': 50, 'epsilon': 0.05}

    # TODO:（初温改为动态计算） 修改停滞阈值 降温速率 权重衰减系数
    # 🌟【新增】：初始化轮盘赌权重与控制参数
    w_destroy = [1.0] * len(destroy_operators)
    w_repair = [1.0] * len(repair_operators)
    rho = 0.1  # 权重衰减系数 (通常取0.1~0.5)
    # 🌟【新增】：动态计算初始温度
    # 设定初期接受 5% 恶化解的概率为 50%

    # 防御性编程：如果初始解恰好为 0（虽然罕见），给一个最小温差防止除零报错
    worse_gap = f_det_current * 0.10  # 注：sim_alns 中这里用 f_det_current
    if worse_gap <= 1e-5:
        worse_gap = 10.0

    # 公式: T = - (恶化量) / ln(接受概率)
    T_start = -worse_gap / math.log(0.60)
    T = T_start
    initial_temp_record = T_start  # 🌟 必须记录下来！供后续停滞回火机制使用

    # 3. 动态计算完美冷却率 (cool_rate)
    T_end = 0.1  # 设定算法结束时的极低目标温度
    cool_rate = (T_end / T_start) ** (1.0 / max_iter)

    print(f"Auto-calculated T_start: {T_start:.2f}, Cool Rate: {cool_rate:.6f} for {max_iter} iters")

    # 🌟【新增】：初始化停滞机制参数
    stagnation_counter = 0
    # 设定停滞阈值：比如连续 50 代没有发现全局最优，就触发强制随机
    stagnation_threshold = max(150, int(max_iter * 0.05))


    # for i in range(max_iter):
    #     # 🌟【修改】：使用 random.choices 进行轮盘赌选择 (k=1表示抽1个)
    #     destroy_op = random.choices(destroy_operators, weights=w_destroy, k=1)[0]
    #     repair_op = random.choices(repair_operators, weights=w_repair, k=1)[0]
    #     current_q = random.randint(q_min, q_max)


    for i in range(max_iter):
        # 🌟【修改】：停滞触发与正常轮盘赌的分支逻辑
        if stagnation_counter >= stagnation_threshold:

            # 【修改】：sim_alns 停滞时，以“随机最优解”作为新起点重启
            sol_current = copy.deepcopy(sol_best_stochastic)
            # 同步更新主轨道的当前确定性分数，否则 SA 逻辑会乱
            f_det_current = deterministic_mam_vprp(sol_current, problem_data)

            # 触发破局：强制使用纯随机算子，并施加最大破坏力度
            destroy_op = random_customer_removal
            repair_op = random_repair
            current_q = random.randint(q_min, q_max)
            stagnation_counter = 0  # 触发后立即清零
            # T = T * 1.5  # 🌟【新增】：回火机制！将温度翻倍，恢复搜索活力
            # 🌟【回火】：取“当前温度1.5倍”和“初始温度30%”的最大值，且绝不超过初始温度
            T = min(initial_temp_record, max(T * 1.5, initial_temp_record* 0.3))

        else:
            # 正常情况：使用轮盘赌自适应选择，q 值随机波动
            destroy_op = random.choices(destroy_operators, weights=w_destroy, k=1)[0]
            repair_op = random.choices(repair_operators, weights=w_repair, k=1)[0]
            current_q = random.randint(q_min, q_max)



        # 🌟【新增2】：记录算子被使用 1 次
        d_name = destroy_op.__name__
        r_name = repair_op.__name__
        op_stats['destroy'][d_name]['used'] += 1
        op_stats['repair'][r_name]['used'] += 1


        sol_temp = copy.deepcopy(sol_current)
        sol_partial, removed_items = destroy_op(sol_temp, problem_data, current_q)
        sol_new = repair_op(sol_partial, removed_items, problem_data)

        # 2. 计算确定性目标值 (用于接受准则，保证搜索方向稳定)
        f_det_new = deterministic_mam_vprp(sol_new, problem_data)

        # 3. 混合策略：何时进行仿真？
        # 情况A: 如果确定性解变好了，非常有必要仿真看一下是否鲁棒性也好了
        # 情况B: 即使确定性解没变好，也有一定概率(如10%)仿真一下，防止漏掉那些"确定性差但鲁棒性好"的解
        should_simulate = False
        force_accept = False  # 🌟【新增】：强制接受标记
        is_new_best = False  # 🌟 将标记位提前

        if f_det_new < f_det_best * 1.05:  # 放宽一点，接近最优的都值得仿真
            should_simulate = True
        elif random.random() < 0.1:  # 随机抽检
            should_simulate = True

        if should_simulate:
            f_sim_new = _run_simulation_internal(sol_new, problem_data, **SEARCH_SIM_PARAMS)##记录解的仿真结果

            # 更新解池
            update_solution_pool(stochastic_pool, sol_new, f_sim_new, pool_size)

            # 更新随机全局最优
            if f_sim_new < f_best_stochastic:
                sol_best_stochastic = copy.deepcopy(sol_new)
                f_best_stochastic = f_sim_new
                time_best = time.time() - start_time  # 🌟【新增】：记录打破记录时的耗时
                force_accept = True  # 🌟【核心桥梁】：仿真发现了好解，强制主轨道跟进
                is_new_best = True  # 奖励算子发现鲁棒解
                print(f"Iter {i + 1} | New Stochastic Best: {f_best_stochastic:.2f} (Det: {f_det_new:.2f})")


        # 4. 接受准则 (使用确定性值，避免噪声干扰搜索轨迹)
        is_accepted = False
        # 也可以尝试：如果 simulated，用 sim 值；否则用 det 值。但在对比实验中，用 Det 引导通常更稳。
        # TODO:这里的退火准则表达
        if force_accept or f_det_new < f_det_current or random.random() < math.exp(-(f_det_new - f_det_current) / max(T, 1e-5)):
            sol_current = copy.deepcopy(sol_new)
            f_det_current = f_det_new
            is_accepted = True  # 被接受了


        # 更新确定性最优 (辅助)
        if f_det_new < f_det_best:
            sol_best_det = copy.deepcopy(sol_new)
            f_det_best = f_det_new
            # 如果上面没有触发随机最优更新，这里也可以标记新发现，但权重可以略低于随机最优
            is_new_best = True  # 打破记录了


        # 🌟【新增4】：更新统计字典
        if is_accepted:
            op_stats['destroy'][d_name]['accepted'] += 1
            op_stats['repair'][r_name]['accepted'] += 1
        if is_new_best:
            op_stats['destroy'][d_name]['best'] += 1
            op_stats['repair'][r_name]['best'] += 1

        # TODO:
        # # # 🌟【新增】：计算本次得分并更新轮盘赌权重
        # # # 定义得分规则：打破记录得5分，被接受得1分，被拒绝得0分
        # # if is_new_best:
        # #     score = 5.0
        # # elif is_accepted:
        # #     score = 1.0
        # # else:
        # #     score = 0.0
        # # 🌟【修改】：细化得分层次，奖励探索行为
        # if is_new_best:
        #     score = 10.0  # 打破全局记录，重赏
        # elif is_accepted and (f_det_new < f_det_current):  # 注: sim_alns 中这里变量名叫 f_det_new < f_det_current
        #     score = 5.0   # 产生比当前更好的解
        # elif is_accepted:
        #     score = 2.0   # 变差了但被SA接受，奖励其跳出局部最优的能力
        # else:
        #     score = 0.0   # 完全被拒
        # 🌟【修改】：结合得分机制，更新停滞计数器
        if force_accept:
            # 1. 终极目标：发现了更优的鲁棒解 (Stochastic Best)
            # 这是 SIM-ALNS 最渴望的结果，给予最高级别重赏！
            score = 20.0
            stagnation_counter = 0  # 彻底打破停滞
        elif is_new_best:
            # 2. 次级目标：发现了更好的确定性解 (Deterministic Best)
            # 也很棒，保证了主轨道的推进，给予次高奖励
            score = 10.0
            stagnation_counter = 0  # 发现确定性新大陆，同样打破停滞
        elif is_accepted and (f_det_new < f_det_current): # (注: sim_alns 中叫 f_det_new < f_det_current)
            score = 5.0
            stagnation_counter += 1 # 局部改善不算打破停滞，继续累加
        elif is_accepted:
            score = 2.0
            stagnation_counter += 1
        else:
            score = 0.0
            stagnation_counter += 1


        # 获取当前算子的索引
        d_idx = destroy_operators.index(destroy_op)
        r_idx = repair_operators.index(repair_op)

        # 运用公式动态更新权重： 新权重 = (1 - rho) * 老权重 + rho * 得分
        w_destroy[d_idx] = (1 - rho) * w_destroy[d_idx] + rho * score
        w_repair[r_idx] = (1 - rho) * w_repair[r_idx] + rho * score


        T *= cool_rate
        history.append((i + 1, f_best_stochastic,f_det_current))

        if (i + 1) % 100 == 0:
            print(
                f"Iter {i + 1}/{max_iter} | Det Cur: {f_det_current:.0f} | Sim Best: {f_best_stochastic:.0f} | Pool: {len(stochastic_pool)}")

    # --- 阶段 2: Long Simulation (精细评估) ---
    print("\n" + "=" * 50)
    print("PHASE 2: LONG SIMULATION (Pool Refinement)")
    print("=" * 50)

    # 使用高精度参数
    LONG_SIM_PARAMS = {'N_min': 500, 'epsilon': 0.01}  # 1% 误差

    final_best_sol = None
    final_best_fit = float('inf')

    # 确保目前的随机最优和确定性最优都在池子里
    update_solution_pool(stochastic_pool, sol_best_stochastic, f_best_stochastic, pool_size)
    update_solution_pool(stochastic_pool, sol_best_det, 0, pool_size)  # fitness暂填0，会被重算

    for idx, item in enumerate(stochastic_pool):
        candidate_sol = item['solution']

        # 运行长仿真
        real_fit = _run_simulation_internal(candidate_sol, problem_data, **LONG_SIM_PARAMS)

        # 简单打印进度
        if (idx + 1) % 5 == 0:
            print(f"Refining Pool {idx + 1}/{len(stochastic_pool)}...")

        if real_fit < final_best_fit:
            final_best_fit = real_fit
            final_best_sol = copy.deepcopy(candidate_sol)

    print(f"\nSIM-ALNS Finished. Final Robust Best: {final_best_fit:.2f}")

    return final_best_sol, final_best_fit, history, op_stats, time_best


def simulate_results(solution, problem_data, num_replications=300):
    """
    通过直接调用底层仿真引擎，获得完整的各仿真回合延迟分布数组。
    保证评估逻辑和分布生成完全一致。
    """
    results, _, _ = simulate_mam_vprp_convergent(
        solution=solution,
        problem_data=problem_data,
        alpha=0.05,
        epsilon=0.05,
        N_min=num_replications,
        N_max=num_replications
    )
    return results


# ==============================================================================
# SECTION 5: COMPARISON & PLOTTING
# ==============================================================================

def plot_tardiness_distribution(solution, problem_data, title="Total Tardiness Distribution"):
    plt.figure(figsize=(10, 6))
    simulation_results = simulate_results(solution, problem_data, num_replications=300)
    plt.hist(simulation_results, bins=30, density=True, alpha=0.7, color='skyblue', edgecolor='black',
             label='_nolegend_')

    mean_value = np.mean(simulation_results)
    plt.axvline(mean_value, color='yellow', linestyle='dashed', linewidth=2,
                label=f'Simulation Mean Value: {mean_value:.2f}')

    deterministic_value = deterministic_mam_vprp(solution, problem_data)
    plt.axvline(deterministic_value, color='red', linestyle='dashed', linewidth=2,
                label=f'Deterministic Value: {deterministic_value:.2f}')

    plt.title(title)
    plt.xlabel("Total Tardiness")
    plt.ylabel("Probability Density")
    plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left')
    plt.grid(axis='y', alpha=0.75)
    plt.show()


def compare_algorithms(problem_data, max_iter=1000, num_replications=300):
    print("=" * 60)
    print("COMPARING SIM-ALNS vs EXP-ALNS ALGORITHMS")
    print("=" * 60)

    print("\n[1/2] Running SIM-ALNS...")
    sim_start = time.time()

    # 🌟【修改】：接收第 5 个返回值 sim_time_best
    sim_solution, sim_fitness, sim_history, sim_op_stats, sim_time_best = sim_alns(problem_data, max_iter, 50, 0.99)
    sim_time = time.time() - sim_start

    print("\n[2/2] Running EXP-ALNS...")
    exp_start = time.time()

    # 🌟【修改】：接收第 5 个返回值 exp_time_best
    exp_solution, exp_fitness, exp_history, exp_op_stats, exp_time_best = exp_alns(problem_data, max_iter, 50, 0.99)
    exp_time = time.time() - exp_start

    sim_results = simulate_results(sim_solution, problem_data, num_replications)
    exp_results = simulate_results(exp_solution, problem_data, num_replications)

    print("\n" + "=" * 60)
    print("COMPARISON RESULTS")
    print("=" * 60)
    print(f"{'Metric':<25} {'SIM-ALNS':<15} {'EXP-ALNS':<15} {'Difference':<15}")
    print("-" * 70)

    metrics = {
        'Best Fitness': [sim_fitness, exp_fitness, exp_fitness - sim_fitness],
        'Computation Time (s)': [sim_time, exp_time, exp_time - sim_time],
        'Time to Best (s)': [sim_time_best, exp_time_best, exp_time_best - sim_time_best],  # <--- 新增这行
        'Mean Tardiness': [np.mean(sim_results), np.mean(exp_results), np.mean(exp_results) - np.mean(sim_results)],
        'Std Dev': [np.std(sim_results), np.std(exp_results), np.std(exp_results) - np.std(sim_results)],
        '95th Percentile': [np.percentile(sim_results, 95), np.percentile(exp_results, 95),
                            np.percentile(exp_results, 95) - np.percentile(sim_results, 95)],
        'Worst Case': [np.max(sim_results), np.max(exp_results), np.max(exp_results) - np.max(sim_results)]
    }

    for metric, values in metrics.items():
        sim_val, exp_val, diff = values
        if 'Time' in metric:
            print(f"{metric:<25} {sim_val:<15.2f} {exp_val:<15.2f} {diff:<+15.2f}")
        else:
            print(f"{metric:<25} {sim_val:<15.4f} {exp_val:<15.4f} {diff:<+15.4f}")

    # 🌟【修改点 3】：把 op_stats 也存进字典里，传给主程序
    return {
        'sim_alns': {
            'solution': sim_solution, 'fitness': sim_fitness, 'history': sim_history,
            'time': sim_time, 'simulation_results': sim_results, 'op_stats': sim_op_stats
        },
        'exp_alns': {
            'solution': exp_solution, 'fitness': exp_fitness, 'history': exp_history,
            'time': exp_time, 'simulation_results': exp_results, 'op_stats': exp_op_stats
        }
    }


import textwrap  # 需要引入这个库用于文字换行

def calculate_batch_utilization(solution, problem_data):
    """
    计算批次利用率
    """
    parts_data = problem_data["part_data"]
    printer_params = problem_data["printer_params"]
    A_max = printer_params['A_max']

    batch_sizes = []
    total_utilization = 0

    for batch_id, part_list in solution["batch_composition"].items():
        if not part_list:
            continue
        batch_area = sum(parts_data[p]['area'] for p in part_list)
        utilization = (batch_area / A_max) * 100
        batch_sizes.append(batch_area)
        total_utilization += utilization

    avg_utilization = total_utilization / len(batch_sizes) if batch_sizes else 0

    return {
        'batch_sizes': batch_sizes,
        'avg_utilization': avg_utilization
    }

def calculate_solution_similarity(solution1, solution2):
    """
    计算两个解决方案的相似度 (基于客户节点位置差异)
    """
    route1 = solution1["route"]
    route2 = solution2["route"]

    # 只比较客户节点（排除仓库0）
    customers1 = [c for c in route1 if c != 0]
    customers2 = [c for c in route2 if c != 0]

    common_customers = set(customers1) & set(customers2)
    if not common_customers:
        return 0

    # 计算位置差异
    position_diff = 0
    for cust in common_customers:
        pos1 = customers1.index(cust) if cust in customers1 else len(customers1)
        pos2 = customers2.index(cust) if cust in customers2 else len(customers2)
        position_diff += abs(pos1 - pos2)

    # 归一化处理
    max_diff = len(common_customers) * max(len(customers1), len(customers2))
    similarity = 100 * (1 - position_diff / max_diff) if max_diff > 0 else 0

    return max(0, min(100, similarity))


import textwrap

def draw_summary_dashboard(results, problem_data, save_dir=None):
    """
    绘制美观、完整、自动换行的总结报告图（引入自适应高度，彻底解决截断问题）
    """
    if not MATPLOTLIB_AVAILABLE: return

    sim_data = results['sim_alns']
    exp_data = results['exp_alns']

    # 1. 计算所需数据
    sim_batch_info = calculate_batch_utilization(sim_data['solution'], problem_data)
    exp_batch_info = calculate_batch_utilization(exp_data['solution'], problem_data)
    similarity = calculate_solution_similarity(sim_data['solution'], exp_data['solution'])

    # 2. 格式化路径字符串
    def format_route(route):
        route_str = " -> ".join(map(str, route))
        return "\n          ".join(textwrap.wrap(route_str, width=70, break_long_words=False))

    # 格式化批次构成与顺序
    def format_batches(solution):
        seq = solution['batch_sequence']
        comp = solution['batch_composition']
        batch_strings = []

        for i, b_id in enumerate(seq):
            parts = comp.get(b_id, [])
            parts_str = ', '.join(parts)
            batch_strings.append(f"B{i + 1}[{parts_str}]")

        full_batch_str = " -> ".join(batch_strings)
        wrapped_lines = textwrap.wrap(full_batch_str, width=70, break_long_words=False)
        return "\n          ".join(wrapped_lines)

    sim_route_fmt = format_route(sim_data['solution']['route'])
    exp_route_fmt = format_route(exp_data['solution']['route'])

    sim_batch_fmt = format_batches(sim_data['solution'])
    exp_batch_fmt = format_batches(exp_data['solution'])

    # 3. 构建文本内容
    summary_text = (
        f"{'=' * 65}\n"
        f"MAM-VPRP SIMULATION OPTIMIZATION REPORT\n"
        f"{'=' * 65}\n\n"

        f"--- SIM-ALNS PERFORMANCE (Stochastic) ---\n"
        f"• Fitness (Exp. Tardiness) : {sim_data['fitness']:.4f}\n"
        f"• Computation Time         : {sim_data['time']:.2f} seconds\n"
        f"• Mean Tardiness (Sim)     : {np.mean(sim_data['simulation_results']):.2f} (Std: {np.std(sim_data['simulation_results']):.2f})\n"
        f"• Batch Utilization        : {sim_batch_info['avg_utilization']:.1f}% ({len(sim_data['solution']['batch_composition'])} batches)\n"
        f"• Route Structure          :\n          {sim_route_fmt}\n"
        f"• Batch Sequence & Comp    :\n          {sim_batch_fmt}\n\n"

        f"--- EXP-ALNS PERFORMANCE (Deterministic) ---\n"
        f"• Fitness (Det. Model)     : {exp_data['fitness']:.4f}\n"
        f"• Computation Time         : {exp_data['time']:.2f} seconds\n"
        f"• Mean Tardiness (Sim)     : {np.mean(exp_data['simulation_results']):.2f} (Std: {np.std(exp_data['simulation_results']):.2f})\n"
        f"• Batch Utilization        : {exp_batch_info['avg_utilization']:.1f}% ({len(exp_data['solution']['batch_composition'])} batches)\n"
        f"• Route Structure          :\n          {exp_route_fmt}\n"
        f"• Batch Sequence & Comp    :\n          {exp_batch_fmt}\n\n"

        f"{'-' * 65}\n"
        f"COMPARISON INSIGHTS\n"
        f"{'-' * 65}\n"
        f"1. Optimization Gap        : {exp_data['fitness'] - sim_data['fitness']:+.2f} "
        f"({(exp_data['fitness'] - sim_data['fitness']) / sim_data['fitness'] * 100 if sim_data['fitness'] > 0 else 0:+.1f}%)\n"
        f"2. Time Efficiency         : EXP-ALNS is {sim_data['time'] / exp_data['time']:.1f}x faster\n"
        f"3. Solution Similarity     : {similarity:.1f}% (Route Sequence Overlap)\n"
        f"4. Robustness Check        : \n"
        f"   SIM 95th Percentile: {np.percentile(sim_data['simulation_results'], 95):.2f}\n"
        f"   EXP 95th Percentile: {np.percentile(exp_data['simulation_results'], 95):.2f}\n"
    )

    # 🌟【绝杀修复点：动态计算画布高度】
    # 统计完整的文本一共有多少行
    total_lines = summary_text.count('\n') + 1
    # 每行分配约 0.35 英寸的物理高度。即使有 40 行，也会自动拉长到 14 英寸高。
    # 设定最小高度为 12.0，防止内容太少时图片被压扁。
    dynamic_height = max(12.0, total_lines * 0.35)

    # 4. 绘图设置 (使用算出来的动态高度)
    fig = plt.figure(figsize=(14, dynamic_height))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.axis('off')

    # 5. 绘制文本框
    props = dict(boxstyle='round', facecolor='lightyellow', alpha=0.9, edgecolor='gold', linewidth=2)

    # 🌟【微调样式】：字号调为 11，行距调为 1.4，起始坐标稍微往上挪 (y=0.98)，让排版更紧凑高级
    ax.text(0.05, 0.98, summary_text, transform=ax.transAxes, fontsize=11,
            fontfamily='monospace', verticalalignment='top', bbox=props, linespacing=1.4)

    # 6. 保存与显示
    if save_dir:
        ensure_directory_exists(save_dir)
        fig_path = os.path.join(save_dir, '05_final_summary_report.png')
        fig.savefig(fig_path, dpi=300, bbox_inches='tight', pad_inches=0.5)
        print(f"Saved Summary Report to: {fig_path}")

    plt.show()
    plt.close(fig)


def plot_operator_performance(op_stats, algo_name="SIM-ALNS", save_dir=None):
    """
    绘制算子表现统计图（分组柱状图 + 综合贡献得分）
    """
    if not MATPLOTLIB_AVAILABLE: return

    # 1. 绘图设置：一行两列（左破坏，右修复）
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    fig.suptitle(f'Operator Contribution Analysis ({algo_name})', fontsize=18, fontweight='bold', y=0.98)

    # 定义 ALNS 学术标准权重：打破记录得 5 分，被接受得 1 分
    SCORE_WEIGHTS = {'best': 5, 'accepted': 1}

    for i, op_type in enumerate(['destroy', 'repair']):
        ax = axes[i]
        stats = op_stats[op_type]

        # 获取算子名字，为了图表美观，去掉过长的后缀并自动换行
        raw_names = list(stats.keys())
        display_names = [n.replace('_removal', '').replace('_repair', '').replace('_', '\n').title() for n in raw_names]

        # 提取数据
        used = [stats[n]['used'] for n in raw_names]
        accepted = [stats[n]['accepted'] for n in raw_names]
        best = [stats[n]['best'] for n in raw_names]

        # 计算综合贡献得分 (Composite Score)
        scores = [stats[n]['best'] * SCORE_WEIGHTS['best'] + stats[n]['accepted'] * SCORE_WEIGHTS['accepted'] for n in
                  raw_names]

        # 设置柱状图的X轴位置
        x = np.arange(len(raw_names))
        width = 0.25  # 柱子宽度

        # 绘制三组柱子：Used, Accepted, New Best
        ax.bar(x - width, used, width, label='Times Used', color='#E0E0E0', edgecolor='black', hatch='//')
        rects_acc = ax.bar(x, accepted, width, label='Times Accepted', color='#87CEFA', edgecolor='black')
        rects_best = ax.bar(x + width, best, width, label='Times New Best', color='#FFB6C1', edgecolor='black')

        # 在“New Best”柱子上方标注“综合得分”
        for j, rect in enumerate(rects_best):
            height = max(rect.get_height(), rects_acc[j].get_height())  # 找同组最高的
            ax.text(rect.get_x() + rect.get_width() / 2., height + max(used) * 0.02,
                    f"Score\n{scores[j]}", ha='center', va='bottom', fontsize=10,
                    fontweight='bold', color='red',
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.7, edgecolor='red'))

        # 图表美化
        ax.set_title(f'{op_type.capitalize()} Operators Performance', fontsize=15, pad=15)
        ax.set_xticks(x)
        ax.set_xticklabels(display_names, fontsize=11)
        ax.set_ylabel('Frequency (Count)', fontsize=12)
        ax.legend(fontsize=11, loc='upper right')
        ax.grid(True, axis='y', alpha=0.4, linestyle='--')

        # 设置Y轴上限，留出写Score的空间
        ax.set_ylim(0, max(used) * 1.25)

    plt.tight_layout(rect=[0, 0, 1, 0.95])

    if save_dir:
        ensure_directory_exists(save_dir)
        fig_path = os.path.join(save_dir, f'08_operator_contribution_{algo_name}.png')
        plt.savefig(fig_path, dpi=300, bbox_inches='tight')
        print(f"Saved Operator Contribution Chart to: {fig_path}")

    plt.show()
    plt.close(fig)

def plot_comparison(results, problem_data, save_dir=None):
    if not MATPLOTLIB_AVAILABLE: return
    sim_data, exp_data = results['sim_alns'], results['exp_alns']
    if save_dir: ensure_directory_exists(save_dir)

    # ===========================================
    # Plot 1: Convergence (只保留完整收敛曲线，去除右侧放大图)
    # ===========================================
    # 变动1：figsize从(16, 6)改为(10, 6)，subplots从(1, 2)改为(1, 1)
    fig1, ax1 = plt.subplots(1, 1, figsize=(12, 6))

    sim_iters, sim_best = [h[0] for h in sim_data['history']], [h[1] for h in sim_data['history']]
    exp_iters, exp_best = [h[0] for h in exp_data['history']], [h[1] for h in exp_data['history']]
    sim_curr = [h[2] for h in sim_data['history']] # 🌟 提取 sim 当前解
    exp_curr = [h[2] for h in exp_data['history']] # 🌟 提取 exp 当前解

    # 2. 绘制底层：当前解的震荡轨迹 (使用高透明度和细线，表现 SA 的接受过程)
    ax1.plot(sim_iters, sim_curr, color='skyblue', alpha=0.4, linewidth=1.0, label='SIM-ALNS Current (Trajectory)')
    ax1.plot(exp_iters, exp_curr, color='lightcoral', alpha=0.4, linewidth=1.0, label='EXP-ALNS Current (Trajectory)')

    # 3. 绘制顶层：全局最优解的收敛阶梯 (使用实线和深色，突出最终效果)
    ax1.plot(sim_iters, sim_best, color='blue', linewidth=2.5, label='SIM-ALNS Best')
    ax1.plot(exp_iters, exp_best, color='red', linewidth=2.5, label='EXP-ALNS Best')

    ax1.set_title('Convergence Curve Comparison', fontsize=14)
    ax1.set_xlabel('Iteration', fontsize=12)
    ax1.set_ylabel('Best Found Expected Tardiness', fontsize=12)
    # 把图例放在右上角，避免遮挡曲线
    ax1.legend(fontsize=11, loc='upper right', framealpha=0.9)
    ax1.grid(True, alpha=0.3, linestyle='--')

    if save_dir:
        fig1.savefig(os.path.join(save_dir, '01_convergence.png'), dpi=300, bbox_inches='tight')
    plt.show()
    plt.close(fig1)

    # # Plot 2: Time & Final values
    # fig2, (ax3, ax4) = plt.subplots(1, 2, figsize=(16, 6))
    # time_vals = [sim_data['time'], exp_data['time']]
    # ax3.bar(['SIM-ALNS', 'EXP-ALNS'], time_vals, color=['blue', 'red'], alpha=0.8)
    # ax3.set_title('Computation Time Comparison')
    # ax3.set_ylabel('Time (seconds)')
    #
    # fitness_values = [sim_data['fitness'], exp_data['fitness']]
    # ax4.bar(['SIM-ALNS', 'EXP-ALNS'], fitness_values, color=['blue', 'red'], alpha=0.8)
    # ax4.set_title('Final Objective Value')
    # ax4.set_ylabel('Expected Total Tardiness')
    #
    # if save_dir: fig2.savefig(os.path.join(save_dir, '02_performance.png'), dpi=300, bbox_inches='tight')
    # plt.show()
    # plt.close(fig2)
    #
    #
    # # ===========================================
    # # Plot 3: Distribution histograms (Enhanced Style)
    # # ===========================================
    # fig3, (ax5, ax6) = plt.subplots(1, 2, figsize=(16, 6))
    #
    # # --- 左图：直方图 ---
    # # 计算统计数据
    # sim_res = np.array(sim_data['simulation_results'])
    # exp_res = np.array(exp_data['simulation_results'])
    # sim_mean = np.mean(sim_res)
    # exp_mean = np.mean(exp_res)
    #
    # # 绘制直方图
    # ax5.hist(sim_res, bins=30, alpha=0.6, density=True, color='blue',
    #          edgecolor='black', linewidth=0.8, label='SIM-ALNS')
    # ax5.hist(exp_res, bins=30, alpha=0.6, density=True, color='red',
    #          edgecolor='black', linewidth=0.8, label='EXP-ALNS')
    #
    # # 绘制均值虚线
    # ax5.axvline(sim_mean, color='blue', linestyle='--', linewidth=2, alpha=0.8)
    # ax5.axvline(exp_mean, color='red', linestyle='--', linewidth=2, alpha=0.8)
    #
    # # 添加右上角统计文本框 (保持在图内，直方图通常有空位)
    # diff_val = exp_mean - sim_mean
    # stats_text = (f"SIM-ALNS Mean: {sim_mean:.1f}\n"
    #               f"EXP-ALNS Mean: {exp_mean:.1f}\n"
    #               f"Difference: {diff_val:+.1f}")
    #
    # props = dict(boxstyle='round', facecolor='white', alpha=0.9, edgecolor='gray')
    # ax5.text(0.96, 0.96, stats_text, transform=ax5.transAxes, fontsize=10,
    #          verticalalignment='top', horizontalalignment='right', bbox=props)
    #
    # # 设置标题和标签
    # ax5.set_title('Tardiness Distribution Comparison', fontsize=14, fontweight='bold')
    # ax5.set_xlabel('Total Tardiness', fontsize=12)
    # ax5.set_ylabel('Probability Density', fontsize=12)
    # ax5.legend(loc='upper left', fontsize=11)
    # ax5.grid(True, alpha=0.3, linestyle='--')
    #
    # # --- 右图：箱线图 ---
    # # 准备数据和标签
    # data_to_plot = [sim_res, exp_res]
    # labels_list = ['SIM-ALNS', 'EXP-ALNS']
    #
    # # 绘制箱线图
    # bplot = ax6.boxplot(data_to_plot,
    #                     patch_artist=True,  # 允许填充颜色
    #                     tick_labels=labels_list,  # 修复报错
    #                     showfliers=False,  # 不显示异常值
    #                     showmeans=True,  # 显示均值
    #                     meanline=True,  # 均值显示为线
    #                     widths=0.3,
    #                     medianprops=dict(color='orange', linewidth=2.5),
    #                     meanprops=dict(color='black', linestyle='--', linewidth=2)
    #                     )
    #
    # # 填充颜色
    # colors = ['lightblue', 'lightcoral']
    # for patch, color in zip(bplot['boxes'], colors):
    #     patch.set_facecolor(color)
    #     patch.set_edgecolor('black')
    #     patch.set_linewidth(1.2)
    #
    # # 计算 IQR 并添加统计文本框
    # sim_q1, sim_q3 = np.percentile(sim_res, [25, 75])
    # exp_q1, exp_q3 = np.percentile(exp_res, [25, 75])
    # sim_iqr = sim_q3 - sim_q1
    # exp_iqr = exp_q3 - exp_q1
    #
    # iqr_text = (f"SIM-ALNS IQR: {sim_iqr:.1f}\n"
    #             f"EXP-ALNS IQR: {exp_iqr:.1f}")
    #
    # # [修改点] 将位置改到 1.03 (图外右侧)，对齐方式改为 left
    # ax6.text(1.03, 1.00, iqr_text, transform=ax6.transAxes, fontsize=10,
    #          verticalalignment='top', horizontalalignment='left', bbox=props)
    #
    # # 设置标题和标签
    # ax6.set_title('Statistical Distribution Box Plot', fontsize=14, fontweight='bold')
    # ax6.set_ylabel('Total Tardiness', fontsize=12)
    # ax6.tick_params(axis='x', labelsize=11)
    # ax6.grid(True, axis='y', alpha=0.3, linestyle='--')
    #
    # # 自动调整布局，防止外置文本被切掉
    # plt.tight_layout()
    #
    # # 保存
    # if save_dir:
    #     fig3.savefig(os.path.join(save_dir, '03_distribution.png'), dpi=300, bbox_inches='tight')
    # plt.show()
    # plt.close(fig3)

    # === 在这里插入新的总结图调用 ===

    print("\nGenerating Summary Dashboard...")
    draw_summary_dashboard(results, problem_data, save_dir)
    # ===============================
###TODO：先不显示甘特图
    # print("\nVisualizing individual solutions (Gantt Charts)...")
    #
    # # print("\nVisualizing solutions...")
    # visualize_solution(sim_data['solution'], problem_data, save_dir, '06_sim_alns_gantt')
    # visualize_solution(exp_data['solution'], problem_data, save_dir, '07_exp_alns_gantt')


def ensure_directory_exists(directory_path):
    if not os.path.exists(directory_path):
        os.makedirs(directory_path)
    return directory_path


# ==============================================================================
# SECTION 6: SOLUTION VISUALIZATION
# ==============================================================================
try:
    import matplotlib.pyplot as plt

    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False


def visualize_solution(solution, problem_data, save_dir=None, filename_prefix=None):
    if not MATPLOTLIB_AVAILABLE: return

    parts_data = problem_data["part_data"]
    customer_data = problem_data["customer_data"]

    det_proc_times = {}
    for batch_id, part_list in solution["batch_composition"].items():
        if not part_list: continue
        total_volume = sum(parts_data[j]['volume'] for j in part_list)
        max_height = max(parts_data[j]['height'] for j in part_list)
        det_proc_times[batch_id] = (problem_data["printer_params"]['alpha'] * total_volume +
                                    problem_data["printer_params"]['beta'] * max_height)

    part_completion_times = {}
    production_time_accumulator = 0.0
    for batch_id in solution["batch_sequence"]:
        proc_time = det_proc_times.get(batch_id, 0)
        production_time_accumulator += proc_time
        for part_id in solution["batch_composition"].get(batch_id, []):
            part_completion_times[part_id] = production_time_accumulator

    timeline = {'travel_segs': [], 'wait_segs': [], 'customer_arrivals': {}, 'customer_departures': {}}
    truck_time = 0.0
    route = solution["route"]

    for i in range(len(route) - 1):
        s_node, n_node = route[i], route[i + 1]
        key = (s_node, n_node)
        travel_duration = problem_data["travel_time_params"].get(key, {'value': 0})['value']

        timeline['travel_segs'].append(
            {'start': truck_time, 'end': truck_time + travel_duration, 'from': s_node, 'to': n_node})
        arrival_time_at_n = truck_time + travel_duration

        if n_node == 0: break

        timeline['customer_arrivals'][n_node] = arrival_time_at_n
        customer_parts = customer_data[n_node]['parts']
        last_part_completion_time = max([part_completion_times.get(p, 0) for p in customer_parts] + [0])
        departure_time_from_n = max(arrival_time_at_n, last_part_completion_time)

        wait_duration = departure_time_from_n - arrival_time_at_n
        if wait_duration > 1e-6:
            timeline['wait_segs'].append({'start': arrival_time_at_n, 'end': departure_time_from_n, 'at': n_node})

        timeline['customer_departures'][n_node] = departure_time_from_n
        truck_time = departure_time_from_n

    fig, ax2 = plt.subplots(1, 1, figsize=(14, 6))
    ax2.set_title('Production & Delivery Gantt Chart')
    ax2.set_yticks([1, 0])
    ax2.set_yticklabels(['3D Printing', 'Truck Activity'])

    print_time = 0
    batch_colors = plt.cm.viridis(np.linspace(0, 1, len(solution['batch_sequence'])))
    for i, batch_id in enumerate(solution['batch_sequence']):
        proc_time = det_proc_times.get(batch_id, 0)
        ax2.barh(1, proc_time, left=print_time, height=0.5, color=batch_colors[i], edgecolor='black')
        ax2.text(print_time + proc_time / 2, 1, batch_id, ha='center', va='center', color='white', weight='bold')
        print_time += proc_time

    for seg in timeline['travel_segs']:
        ax2.barh(0, seg['end'] - seg['start'], left=seg['start'], height=0.5, color='cornflowerblue', edgecolor='black')

    for seg in timeline['wait_segs']:
        ax2.barh(0, seg['end'] - seg['start'], left=seg['start'], height=0.5, color='salmon', edgecolor='black',
                 hatch='//')

    for cid, cdata in customer_data.items():
        due_date = cdata['due_date']
        ax2.axvline(x=due_date, color='red', linestyle='-', linewidth=1.5, ymin=0, ymax=0.5)
        ax2.text(due_date, -0.4, f'Due C{cid}', color='red', ha='center')

        last_part_completion = max([part_completion_times.get(p, 0) for p in cdata['parts']] + [0])
        departure_time = timeline['customer_departures'].get(cid)
        if departure_time and last_part_completion > 1e-6 and abs(departure_time - last_part_completion) < abs(
                departure_time - timeline['customer_arrivals'][cid]):
            ax2.plot([last_part_completion, departure_time], [0.75, 0.25], 'k--')

    ax2.set_xlabel('Time')
    ax2.set_ylim(-0.5, 1.5)
    ax2.grid(True, axis='x', linestyle='--', alpha=0.5)
    plt.tight_layout(rect=[0, 0, 1, 0.96])

    if save_dir and filename_prefix:
        ensure_directory_exists(save_dir)
        import datetime
        filepath = os.path.join(save_dir, f"{filename_prefix}_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.png")
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
    plt.show()



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='SIM-ALNS vs EXP-ALNS for MAM-VPRP')
    parser.add_argument('--data', type=str, default='instance/problem_instance_15cust_40parts.json', help='Problem data file')
    parser.add_argument('--iter', type=int, default=20000, help='Maximum iterations')
    parser.add_argument('--mode', type=str, default='compare', choices=['sim', 'exp', 'compare'])
    parser.add_argument('--replications', type=int, default=300, help='Number of simulation replications')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--output', type=str, default='comparison_results.json')
    parser.add_argument('--save_dir', type=str, default='convergence_curves')
    args = parser.parse_args()

    output_dir = ensure_directory_exists(args.save_dir)
    print("\n[1/4] Loading problem data...")

    # 使用 mam_simulation_environment 中的统一函数进行读取与格式转换
    example_problem_data = load_problem_data(args.data)
    if not example_problem_data:
        exit(1)

    random.seed(args.seed)
    np.random.seed(args.seed)

    if args.mode == 'sim':
        print("\nRunning SIM-ALNS only...")
        # best_sol, best_fit, hist = sim_alns(example_problem_data, args.iter, 20, 0.99)
        # TODO:
        best_sol, best_fit, hist, t_best = sim_alns(example_problem_data, args.iter)
    elif args.mode == 'exp':
        print("\nRunning EXP-ALNS only...")
        # best_sol, best_fit, hist = exp_alns(example_problem_data, args.iter, 20, 0.99)
        best_sol, best_fit, hist, t_best= exp_alns(example_problem_data, args.iter)
    else:
        print("\n[2/4] Running both SIM-ALNS and EXP-ALNS...")

        # 注意这里需要接住新增的第四个返回值
        sim_solution, sim_fitness, sim_history, sim_op_stats, sim_time_best = sim_alns(example_problem_data, args.iter)
        exp_solution, exp_fitness, exp_history, exp_op_stats, exp_time_best = exp_alns(example_problem_data, args.iter)
        # sim_solution, sim_fitness, sim_history, sim_op_stats = sim_alns(example_problem_data, args.iter, 20, 0.99)
        # exp_solution, exp_fitness, exp_history, exp_op_stats = exp_alns(example_problem_data, args.iter, 20, 0.99)

        results = compare_algorithms(example_problem_data, args.iter, args.replications)
        print("\n[3/4] Generating visualizations...")
        plot_comparison(results, example_problem_data, save_dir=output_dir)

        # 🌟【新增】：绘制算子贡献图
        print("\nGenerating Operator Contribution Charts...")
        sim_stats = results['sim_alns']['op_stats']
        exp_stats = results['exp_alns']['op_stats']
        plot_operator_performance(sim_op_stats, algo_name="SIM-ALNS", save_dir=output_dir)
        plot_operator_performance(exp_op_stats, algo_name="EXP-ALNS", save_dir=output_dir)

        print("\n[4/4] Saving results...")
        try:
            with open(os.path.join(output_dir, args.output), 'w') as f:
                json.dump(results, f, indent=2, default=str)
        except Exception as e:
            print(f"Warning: Could not save results: {e}")