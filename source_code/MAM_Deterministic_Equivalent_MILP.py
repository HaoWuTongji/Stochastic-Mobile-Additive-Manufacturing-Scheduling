# -*- coding: utf-8 -*-

import gurobipy as gp
from gurobipy import GRB
import json
import os
import numpy as np
from scipy.stats import norm
from generate_mam_plan import generate_mam_plan

def calculate_lognormal_quantile(base_value_mode, std_log, p_quantile):
    """
    计算对数正态分布的 p 分位数，基于众数 (Mode) 和底层正态分布的标准差 (std_log)。

    对数正态分布 LogN(mu, sigma^2) 的参数关系：
    1. 众数 (Mode) = exp(mu - sigma^2)
    2. p 分位数 (Q_p) = exp(mu + Z_p * sigma)

    其中 mu 和 sigma^2 是底层正态分布的均值和方差。
    我们使用 Mode 和 sigma^2 来计算 mu，再计算 Q_p。

    Args:
        base_value_mode (float): 输入的“基准值”，即 Log-Normal 分布的众数 (Mode)。
        std_log (float): 底层正态分布的标准差 sigma。
        p_quantile (float): 所需的分位数 p (例如 0.50, 0.95)。

    Returns:
        float: Log-Normal 分布的 p 分位数。
    """
    if base_value_mode <= 0 or std_log < 0 or p_quantile < 0 or p_quantile > 1:
        return float('inf')

    # 计算底层正态分布的均值 mu
    # Mode = exp(mu - sigma^2)  => mu = log(Mode) + sigma^2
    mu = np.log(base_value_mode) + std_log ** 2

    if p_quantile == 0:
        # p=0 的特殊情况，我们将其定义为 Mode (基准值) 本身
        return base_value_mode

    # 查找标准正态分布的 Z_p 值
    z_p = norm.ppf(p_quantile)

    # 计算 p 分位数
    # Q_p = exp(mu + Z_p * sigma)
    quantile_value = np.exp(mu + z_p * std_log)

    return quantile_value


# --- 辅助函数：加载 JSON 算例数据 ---
import json
import os



def load_problem_data(data_source):
    """
    加载并预处理问题数据

    参数：
        data_source: 可以是文件路径（字符串）或已加载的数据字典

    返回：
        预处理后的数据字典
    """
    # 如果是字符串，假设是文件路径
    if isinstance(data_source, str):
        if not os.path.exists(data_source):
            print(f"错误: 算例文件未找到: {data_source}")
            return None

        with open(data_source, 'r', encoding='utf-8') as f:
            data = json.load(f)
    else:
        # 假设已经是数据字典
        data = data_source

    # 转换 travel_time_params 中的键格式
    if "travel_time_params" in data:
        processed_travel_times = {}
        for key, params in data["travel_time_params"].items():
            try:
                s, n = map(int, key.split('_'))
                processed_travel_times[(s, n)] = params
            except ValueError:
                print(f"警告: 无法解析旅行时间键: {key}")

        data["travel_time_params"] = processed_travel_times

    # 转换 customer_data 中的键格式
    if "customer_data" in data:
        data["customer_data"] = {int(k): v for k, v in data["customer_data"].items()}

    return data


def create_and_solve_model(data, percentile=None, use_mean=False, model_name="Model"):
    """
    统一的模型求解函数。
    根据参数计算时间参数（分位数或期望值），然后构建并求解模型。

    :param data: 问题数据。
    :param percentile: 分位数 (0 <= p <= 1)。当 use_mean 为 False 时有效。
    :param use_mean: 如果为 True，则计算期望值作为参数。
    :param model_name: 模型名称。
    :return: 求解结果。
    """

    # --- 1. 初始化集合和参数 ---
    customer_data = data["customer_data"]
    part_data = data["part_data"]
    printer_params = data["printer_params"]
    travel_time_params = data["travel_time_params"]
    proc_uncertainty = data["proc_uncertainty"]##从data字典中取出数据加以使用

    J = list(part_data.keys())
    Nc = list(customer_data.keys())
    N = [0] + Nc  # 节点集合 N 包含仓库 0
    len_Nc = len(Nc)

    # 按照模型定义，批次索引从 1 开始，最大批次数为 |J|
    B = range(1, len(J) + 1)
    len_N = len(N)

    T = {}
    xi = 0

    if use_mean:
        print("  - 使用期望值作为参数 (MVP模型)")
        # 计算期望值
        for (s, n), params in travel_time_params.items():
            value = params["value"]
            std = params["std"]
            mu = np.log(value) + std**2
            T[(s, n)] = np.exp(mu + std**2 / 2) # E[X] = exp(mu + sigma^2/2)

        base_value = proc_uncertainty["base_value"]
        proc_std = proc_uncertainty["std"]
        mu = np.log(base_value) + proc_std**2
        xi = np.exp(mu + proc_std**2 / 2)
    else:
        if percentile is None:
            raise ValueError("必须提供 'percentile' 或设置 'use_mean=True'")
        print(f"  - 使用 {percentile*100}% 分位数作为参数")
        # 计算分位数
        for (s, n), params in travel_time_params.items():
            T[(s, n)] = calculate_lognormal_quantile(params["value"], params["std"], percentile)
        xi = calculate_lognormal_quantile(proc_uncertainty['base_value'], proc_uncertainty['std'], percentile)

    # # --- 2. 基于 Log-Normal 分位数计算确定性时间参数 ---
    #
    # # 2.1 旅行时间 P-分位数 (T_p)
    # T_p = {}
    # for (s, n), params in travel_time_params.items():
    #     base_value = params['value']
    #     std = params['std']
    #     T_p[(s, n)] = calculate_lognormal_quantile(base_value, std, percentile)
    #     # T_p[(s, n)] = int(T_p[(s, n)]) # 保持浮点数精度，不进行整数化
    #
    # # 2.2 生产乘数 P-分位数 (xi_p)
    # xi_p = calculate_lognormal_quantile(proc_uncertainty['base_value'], proc_uncertainty['std'], percentile)

    if use_mean:
        print(f"--- 正在构建模型: {model_name} (期望值模型) ---")
    else:
        print(f"--- 正在构建模型: {model_name} (p={percentile * 100:.0f}%) ---")
    # print(f"--- 正在构建模型: {model_name} (p={percentile * 100:.0f}%) ---")
    print(f"客户数 (|Nc|): {len_Nc}")
    print(f"生产乘数 xi_p: {xi:.4f}")
    print("---------------------------------")

    # --- 3. Big-M 值 ---
    A_max = printer_params["A_max"]
    alpha = printer_params["alpha"]
    beta = printer_params["beta"]

    M_h = max(p['height'] for p in part_data.values()) if J else 0

    # 粗略估计最大时间用于 Big-M
    max_travel_time_est = sum(T.values())
    max_base_proc_time = alpha * sum(p['volume'] for p in part_data.values()) + beta * M_h * len(J)
    M_L = max_travel_time_est + max_base_proc_time * xi * len(B)
    M_c = M_L
    M_subtour = len_Nc  # MTZ M value is usually |Nc| or |Nc|+1

    # --- 4. 创建 Gurobi 模型 ---
    m = gp.Model(model_name)
    m.setParam('TimeLimit', 600)
    m.Params.MIPGap = 0.005  # 0.5% gap
    # m.Params.OutputFlag = 0  # 隐藏Gurobi输出

    # --- 5. 定义决策变量 ---
    # 第一阶段变量 (Route, Batching)
    y_indices = [(s, n) for s in N for n in N if s != n]
    y = m.addVars(y_indices, vtype=GRB.BINARY, name="y")
    x = m.addVars(J, B, vtype=GRB.BINARY, name="x")
    u = m.addVars(N, vtype=GRB.CONTINUOUS, lb=0, name="u")

    # 连续变量 (对应第二阶段)
    H = m.addVars(B, vtype=GRB.CONTINUOUS, lb=0, name="H")
    P = m.addVars(B, vtype=GRB.CONTINUOUS, lb=0, name="P")
    C = m.addVars(B, vtype=GRB.CONTINUOUS, lb=0, name="C")
    A = m.addVars(Nc, vtype=GRB.CONTINUOUS, lb=0, name="A")
    L = m.addVars(N, vtype=GRB.CONTINUOUS, lb=0, name="L")
    Tard = m.addVars(Nc, vtype=GRB.CONTINUOUS, lb=0, name="Tard")

    # --- 6. 设置目标函数 ---
    # 最小化总延迟 (为保证数值稳定性，添加微小的到达时间惩罚)
    obj_penalty = 0.0001
    m.setObjective(gp.quicksum((Tard[n] + obj_penalty * A[n]) for n in Nc), GRB.MINIMIZE)

    # --- 7. 添加约束 ---

    # A. VRP 路径和 MTZ 约束 (与 MVP 相同)
    m.addConstrs((y.sum('*', n) == 1 for n in Nc), name="Flow_In")
    m.addConstrs((y.sum(s, '*') == 1 for s in Nc), name="Flow_Out")
    m.addConstr((y.sum(0, '*') == 1), name="Depot_Out")
    m.addConstr((y.sum('*', 0) == 1), name="Depot_In")

    # MTZ
    m.addConstr(u[0] == 0, name="MTZ_Depot")
    m.addConstrs((u[s] - u[n] + M_subtour * y[s, n] <= M_subtour - 1
                  for s in Nc for n in Nc if s != n), name="MTZ_SEC")

    # B. 生产和批次约束 (与 MVP 相同)
    m.addConstrs((x.sum(j, '*') == 1 for j in J), name="Part_Assign")
    m.addConstrs((gp.quicksum(part_data[j]['area'] * x[j, b] for j in J) <= A_max
                  for b in B), name="Capacity")
    m.addConstrs((x.sum('*', b + 1) <= len(J) * x.sum('*', b)
                  for b in B if b < len(J)), name="Contiguity")##批次连续性

    # 批次最大高度 H_b
    m.addConstrs((H[b] >= part_data[j]['height'] - M_h * (1 - x[j, b])
                  for j in J for b in B), name="Max_H_1")
    m.addConstrs((H[b] <= M_h * x.sum('*', b)
                  for b in B), name="Max_H_2")

    if use_mean:
        m.addConstrs(
            (P[b] == xi * (alpha * gp.quicksum(part_data[j]['volume'] * x[j, b] for j in J) + beta * H[b])
             for b in B), name=f"Proc_Time_mvp")
    else:
        m.addConstrs(
            (P[b] == xi * (alpha * gp.quicksum(part_data[j]['volume'] * x[j, b] for j in J) + beta * H[b])
             for b in B), name=f"Proc_Time_p{percentile * 100:.0f}")
    # # 确定性处理时间 P_b (使用 xi_p)
    # m.addConstrs(
    #     (P[b] == xi * (alpha * gp.quicksum(part_data[j]['volume'] * x[j, b] for j in J) + beta * H[b])
    #      for b in B), name=f"Proc_Time_p{percentile * 100:.0f}")

    # 批次完成时间 C_b
    m.addConstr(C[1] == P[1], name="Comp_Time_Start")
    m.addConstrs((C[b] == C[b - 1] + P[b]
                  for b in B if b > 1), name="Comp_Time_Seq")

    # C. 物流与同步约束
    m.addConstr(L[0] == 0, name="Depot_Depart")

    # 车辆到达时间 A_n (使用 T_p)
    for s in N:
        for n in Nc:
            if s != n:
                travel_t = T.get((s, n), 0)
                constr_name = "Arrival_Time_MVP" if use_mean else f"Arrival_Time_p{percentile * 100:.0f}_{s}_{n}"
                m.addConstr(
                    A[n] >= L[s] + travel_t - M_L * (1 - y[s, n]),
                    name=constr_name
                )
                # m.addConstr(A[n] >= L[s] + travel_t - M_L * (1 - y[s, n])
                #             # name=f"Arrival_Time_p{percentile * 100:.0f}_{s}_{n}")

    # 车辆离开时间 L_n (同步)
    m.addConstrs((L[n] >= A[n] for n in Nc), name="Depart_After_Arrival")

    # L_n >= Part Completion Time (由于没有明确的 Part Completion Time 变量 c_j, 我们使用 C_b 和 x_jb 间接约束)
    # L[n] >= C[b] if part j of customer n is in batch b (i.e., x[j,b]=1)
    for n in Nc:
        for j in customer_data[n]['parts']:
            for b in B:
                m.addConstr(L[n] >= C[b] - M_c * (1 - x[j, b]), name=f"Depart_After_Prod_{n}_{j}_{b}")

    # 延迟计算 Tard_n
    m.addConstrs((Tard[n] >= L[n] - customer_data[n]['due_date']
                  for n in Nc), name="Tardiness_Calc")

    # --- 8. 求解模型 ---
    m.optimize()

    # --- 9. 提取和格式化结果 ---
    # solution_export = {"percentile": percentile, "status": m.Status}
    solution_export = {
        "percentile": percentile,
        "use_mean": use_mean,  # 新增字段
        "status": m.Status
    }
    if m.Status == GRB.OPTIMAL or m.Status == GRB.TIME_LIMIT or m.Status == GRB.USER_OBJ_LIMIT:
        solution_export["objective"] = m.ObjVal - sum(obj_penalty * A[n].X for n in Nc)

        # 提取路径
        route = []
        curr_node = 0
        route.append(curr_node)
        while len(route) <= len_Nc:
            found_next = False
            for n in N:
                if n != curr_node and y.get((curr_node, n)) and y[curr_node, n].X > 0.5:
                    route.append(n)
                    curr_node = n
                    found_next = True
                    break
            if not found_next and curr_node != 0:
                route.append(0)  # 强制返回 depot
                break

        solution_export["route"] = route

        # ========== 调用 generate_mam_plan 函数 ==========
        generate_mam_plan(
            customer_data=customer_data,  # 传入客户数据
            route=route,  # 传入求解得到的路径
            output_path="plan_mam.txt"  # 输出方案文件
        )

        # 提取批次组成和序列
        batch_composition = {}
        batch_sequence = []
        b_idx = 1
        for b in B:
            parts_in_batch = [j for j in J if x[j, b].X > 0.5]
            if parts_in_batch:
                batch_name = f"Batch{b_idx}"
                batch_composition[batch_name] = parts_in_batch
                batch_sequence.append(batch_name)
                b_idx += 1

        solution_export["batch_composition"] = batch_composition
        solution_export["batch_sequence"] = batch_sequence

        # # 提取时间点
        # solution_export["times"] = {
        #     "ArrivalTimes": {n: A[n].X for n in Nc},
        #     "DepartureTimes": {n: L[n].X for n in Nc},
        #     "Tardiness": {n: Tard[n].X for n in Nc}
        # }

    print(solution_export)
    return solution_export


# =============================================================================
# --- 主执行程序 ---
# =============================================================================

if __name__ == "__main__":
            # 1. 加载数据
    PROBLEM_FILE = "instance/problem_instance_10cust_10parts.json"
    problem_data = load_problem_data(PROBLEM_FILE)

    if problem_data is None:
            exit()

            print("\n=========================================================")
            print(" 移动增材制造 (MAM) P-分位数确定性等价模型求解器 ")
            print("=========================================================\n")

    models_to_solve = [
            {"name": "BASE-MODE (p=0)", "percentile": 0.0, "use_mean": False},
            {"name": "MVP-MEDIAN (p=0.50)", "percentile": 0.50, "use_mean": False},
            {"name": "PDE-P95 (p=0.95)", "percentile": 0.95, "use_mean": False},
            # 新增：调用同一个函数，但设置 use_mean=True
            {"name": "PAPER-MVP (均值化模型)", "percentile": None, "use_mean": True},
        ]

    all_solutions = []
    for model_info in models_to_solve:
            solution = create_and_solve_model(
                problem_data,
                percentile=model_info["percentile"],
                use_mean=model_info["use_mean"],
                model_name=model_info["name"]
            )
            all_solutions.append(solution)
            if solution["status"] in [GRB.OPTIMAL, GRB.TIME_LIMIT, GRB.USER_OBJ_LIMIT]:
                print(f"[{model_info['name']}] 目标总延迟: {solution['objective']:.4f}")
                print(f"[{model_info['name']}] 路径: {' -> '.join(map(str, solution['route']))}")
                print(f"[{model_info['name']}] 批次数: {len(solution['batch_sequence'])}\n")
            else:
                print(f"[{model_info['name']}] 求解失败，状态: {solution['status']}\n")

            # ... (打印结果)
        # 打印摘要
    # if solution["status"] in [GRB.OPTIMAL, GRB.TIME_LIMIT, GRB.USER_OBJ_LIMIT]:
    #         print(f"[{name}] 目标总延迟: {solution['objective']:.4f}")
    #         print(f"[{name}] 路径: {' -> '.join(map(str, solution['route']))}")
    #         print(f"[{name}] 批次数: {len(solution['batch_sequence'])}\n")
    # else:
    #         print(f"[{name}] 求解失败，状态: {solution['status']}\n")


    print("=========================================================")
    print(" 所有确定性等价模型求解完成 ")
    print("=========================================================")


