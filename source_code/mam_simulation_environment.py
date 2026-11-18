

# -*- coding: utf-8 -*-
"""
MAM Stochastic Simulation Environment
Contains Log-Normal utilities, data loading, and the convergent Monte Carlo simulation.

Updates:
1. Integrated data loading logic to read problem_instance.txt.
2. Implemented Log-Normal quantile calculation based on Mode (base value).
3. Fixed Log-Normal sampling in simulation: mu = sigma^2 (Mode = 1.0).
4. Simulation environment now accepts travel_time_params and proc_uncertainty
   with 'value' (mode) and 'std' (sigma) keys.
"""
import numpy as np
import random
from scipy import stats  # For calculating t-distribution critical value
from scipy.stats import norm  # For norm.ppf
import os
import re
import json


# =============================================================================
# A. UTILITIES (Data Loading and Log-Normal Math)
# =============================================================================
def calculate_lognormal_quantile(base_value_mode, std_log, p_quantile):
    """
    Calculate the p-quantile of a Log-Normal distribution given its Mode and sigma.
    Used for calculating MVP (p=0.50) and PDE (p=0.95) time parameters.
    """
    if base_value_mode <= 0 or std_log < 0 or p_quantile < 0 or p_quantile > 1:
        return float('inf')

    # Calculate mu for the underlying Normal distribution
    # Mode = exp(mu - sigma^2) => mu = log(Mode) + sigma^2
    mu = np.log(base_value_mode) + std_log ** 2

    if p_quantile == 0:
        return base_value_mode

    # Calculate p-quantile: Q_p = exp(mu + Z_p * sigma)
    z_p = norm.ppf(p_quantile)
    quantile_value = np.exp(mu + z_p * std_log)

    return quantile_value


def load_problem_data(filepath="problem_instance.json"):
    """加载 JSON 算例文件并进行预处理。"""
    if not os.path.exists(filepath):
        print(f"错误: 算例文件未找到: {filepath}")
        return None

    with open(filepath, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # 转换 travel_time_params 中的键格式 (e.g., "0_1" -> (0, 1))
    processed_travel_times = {}
    for key, params in data["travel_time_params"].items():
        try:
            s, n = map(int, key.split('_'))
            processed_travel_times[(s, n)] = params
        except ValueError:
            print(f"警告: 无法解析旅行时间键: {key}")

    data["travel_time_params"] = processed_travel_times

    # 转换 customer_data 中的键格式 (e.g., "1" -> 1)
    data["customer_data"] = {int(k): v for k, v in data["customer_data"].items()}

    # 从旅行时间推算节点坐标
    try:
        from sklearn.manifold import MDS
        import numpy as np

        # 提取节点编号
        nodes = sorted(set([str(i) for (i, j) in processed_travel_times.keys()] +
                           [str(j) for (i, j) in processed_travel_times.keys()]))
        n = len(nodes)

        # 如果节点太少，使用简单布局
        if n <= 3:
            simple_coords = {
                '0': (0, 0),
                '1': (1, 0),
                '2': (0, 1),
                '3': (-1, 0),
                '4': (0, -1),
                '5': (1, 1),
                '6': (-1, -1)
            }
            node_coords = {int(node): simple_coords.get(node, (float(node), 0)) for node in nodes}
        else:
            # 构建距离矩阵
            index_map = {node: idx for idx, node in enumerate(nodes)}
            D = np.zeros((n, n))
            for (i, j), val in processed_travel_times.items():
                if i != j:
                    distance = val.get("mean", val.get("value", 1.0))
                    D[index_map[str(i)], index_map[str(j)]] = distance
                    D[index_map[str(j)], index_map[str(i)]] = distance

            # 检查距离矩阵是否有效
            if np.all(D == 0):
                default_coords = {
                    '0': (0, 0),
                    '1': (3.7, 5.7),
                    '2': (7.8, 3.6),
                    '3': (1.6, 0.3),
                    '4': (6.0, 4.2),
                    '5': (0.6, 4.3),
                    '6': (2.1, 1.1)
                }
                node_coords = {int(node): default_coords.get(node, (float(node), 0)) for node in nodes}
            else:
                # 执行 MDS 拟合二维坐标
                mds = MDS(n_components=2, dissimilarity="precomputed", random_state=42,
                          normalized_stress='auto', n_init=10)
                coords_2d = mds.fit_transform(D)

                # 转换为字典形式
                coords = {}
                for node in nodes:
                    idx = index_map[node]
                    x = coords_2d[idx, 0]
                    y = coords_2d[idx, 1]
                    coords[node] = (float(x), float(y))

                # 平移和缩放，使坐标更合理
                all_x = [coords[node][0] for node in nodes]
                all_y = [coords[node][1] for node in nodes]

                # 计算缩放因子
                max_coord = max(max(all_x) - min(all_x), max(all_y) - min(all_y))
                if max_coord > 0:
                    scale_factor = 15.0 / max_coord
                else:
                    scale_factor = 1.0

                # 应用缩放
                for node in nodes:
                    x, y = coords[node]
                    coords[node] = (x * scale_factor, y * scale_factor)

                # 平移，使仓库点 (0) 在原点附近
                if '0' in coords:
                    x0, y0 = coords['0']
                    for k in coords:
                        coords[k] = (coords[k][0] - x0, coords[k][1] - y0)

                node_coords = {int(node): coords[node] for node in nodes}

        data["node_coords"] = node_coords
        print(f"已生成节点坐标: {node_coords}")

    except Exception as e:
        print(f"警告: 无法生成节点坐标: {e}")
        # 动态生成默认坐标：根据实际节点数量生成
        # 提取所有节点编号
        all_nodes = set()
        for (i, j) in processed_travel_times.keys():
            all_nodes.add(i)
            all_nodes.add(j)
        all_nodes = sorted(all_nodes)

        # 动态生成坐标：仓库(0)在原点，其他节点随机分布在圆周上
        import math
        node_coords = {0: (0.0, 0.0)}
        num_customers = len(all_nodes) - 1  # 减去仓库0

        if num_customers > 0:
            for idx, node in enumerate(all_nodes):
                if node == 0:
                    continue
                # 将客户均匀分布在半径为10的圆周上
                angle = 2 * math.pi * idx / num_customers
                x = 10.0 * math.cos(angle)
                y = 10.0 * math.sin(angle)
                node_coords[node] = (x, y)

        data["node_coords"] = node_coords
        print(f"已生成默认节点坐标: {node_coords}")

    return data



# =============================================================================
# B. SIMULATION LOGIC
# =============================================================================

def simulate_mam_vprp_convergent(solution, problem_data,
                                 alpha=0.05, epsilon=0.05,
                                 N_min=50, N_max=1000):
    """
    Adaptive Monte Carlo Simulation with convergence control based on confidence interval.
    """
    # random.seed(42)
    # np.random.seed(42)

    tardiness_results = []
    N = 0

    parts_data = problem_data["part_data"]  # Corrected key
    customer_data = problem_data["customer_data"]
    printer_params = problem_data["printer_params"]
    travel_time_params = problem_data["travel_time_params"]
    proc_uncertainty = problem_data["proc_uncertainty"]  # Corrected key

    print(f"Starting convergent simulation: alpha={alpha}, epsilon={epsilon}, N_min={N_min}, N_max={N_max}")

    # Pre-calculate base production time (deterministic part)
    base_proc_times = {}
    for batch_id, part_list in solution["batch_composition"].items():
        if not part_list:
            continue
        total_volume = sum(parts_data[j]['volume'] for j in part_list)
        max_height = max(parts_data[j]['height'] for j in part_list)
        base_proc_times[batch_id] = (printer_params['alpha'] * total_volume + printer_params['beta'] * max_height)

    # Main simulation loop
    while N < N_max:
        current_rep_tardiness = 0.0

        # 1. Stochastic Sampling##生产、旅行时间的随机采样

        # Production Multiplier Sampling: LogN(mu=sigma^2, sigma) -> Mode = 1.0
        proc_std = proc_uncertainty['std']
        mu_proc = proc_std ** 2
        proc_time_multipliers = {}
        for b_id in solution["batch_composition"]:
            # Sample from LogN(mu=sigma^2, sigma)
            val = np.random.lognormal(mean=mu_proc, sigma=proc_std)  ##val自定义临时变量
            proc_time_multipliers[b_id] = min(max(val, 0.5), 2.0)  # Clamp multiplier

        # Travel Time Sampling: LogN(mu=sigma^2, sigma) * Base_Value
        sampled_travel_times = {}
        for (s, n), params in travel_time_params.items():
            base_value = params['value']
            travel_std = params['std']
            mu_travel = travel_std ** 2

            # Sample multiplier
            multiplier = np.random.lognormal(mean=mu_travel, sigma=travel_std)
            # Sampled Time = Multiplier * Base_Value (Mode)
            sampled_travel_times[(s, n)] = min(max(multiplier, 0.5), 2.0) * base_value  # Clamp multiplier * base_value

        # 2. Simulate Production Timeline##计算采样后的实际生产时间
        part_completion_times = {}
        production_time_accumulator = 0.0
        for batch_id in solution["batch_sequence"]:
            if batch_id not in solution["batch_composition"]:  ##批次不存在则跳过
                continue

            proc_mult = proc_time_multipliers.get(batch_id, 1.0)  ##获取当前批次（batch_id）对应的加工时间乘数，如果该批次在字典中不存在，则使用默认值 1.0
            sampled_proc_time = base_proc_times.get(batch_id, 0) * proc_mult

            production_time_accumulator += min(max(sampled_proc_time, 1e-3), 1e5)  # 累加采样后的实际生产时间
            for part_id in solution["batch_composition"][batch_id]:
                part_completion_times[part_id] = production_time_accumulator

        # 3. Simulate Vehicle Route & Synchronization
        truck_time = 0.0
        route = solution["route"]
        for i in range(len(route) - 1):
            s_node, n_node = route[i], route[i + 1]

            travel_duration = sampled_travel_times.get((s_node, n_node), 0.0)
            arrival_time_at_n = truck_time + travel_duration

            if n_node == 0:
                break

            # ====== ★ 关键防御性检查（新加） ======
            if n_node not in customer_data:
                raise ValueError(
                    f"[Simulation Error] Route contains invalid node: {n_node}. "
                    f"Route = {route}"
                )
            # =============================================

            customer_parts = customer_data.get(n_node, {}).get('parts',
                                                               [])  ##尝试从 customer_data 中获取键为 n_node 的客户数据，不存在则返回默认值{}，然后进一步获取键为 parts 的值（即该客户需要的零件列表）
            if not customer_parts:
                last_part_completion_time = 0.0
            else:
                last_part_completion_time = max(part_completion_times.get(p, 0.0) for p in customer_parts)

            departure_time_from_n = max(arrival_time_at_n, last_part_completion_time)
            tardiness = max(0, departure_time_from_n - customer_data[n_node]['due_date'])
            current_rep_tardiness += min(tardiness, 1e6)
            truck_time = departure_time_from_n

        tardiness_results.append(current_rep_tardiness)
        N += 1

        # 4. Convergence Check
        if N >= N_min:
            Y_bar = np.mean(tardiness_results)
            if Y_bar > 1e-3:  # Avoid division by zero/near zero mean
                sigma2_hat = np.var(tardiness_results, ddof=1)  ##ddof=1无偏估计
                t_critical = stats.t.ppf(1 - alpha / 2,
                                         df=N - 1)  ##自由度为N-1，t分布的百分点函数，根据给定的 “累积概率” 和 “自由度”，返回对应的 t 分布临界值
                W_half = t_critical * np.sqrt(sigma2_hat) / np.sqrt(N)

                if W_half <= epsilon * Y_bar:
                    print(f"   Simulation converged at N={N} (Rel. Error: {W_half / Y_bar * 100:.1f}%)")
                    convergence_info = {
                        'final_N': N, 'mean_tardiness': Y_bar, 'confidence_half_width': W_half,
                        'relative_error': W_half / Y_bar, 'variance': sigma2_hat, 'converged': True
                    }
                    return tardiness_results, Y_bar, convergence_info

            # Progress log
            if N % 500 == 0:
                print(f"Progress: N={N}, Current Mean: {Y_bar:.2f}")  ##:.2f 表示保留 2 位小数

    # Max iteration reached
    Y_bar = np.mean(tardiness_results)
    sigma2_hat = np.var(tardiness_results, ddof=1)
    t_critical = stats.t.ppf(1 - alpha / 2, df=N - 1)
    W_half = t_critical * np.sqrt(sigma2_hat) / np.sqrt(N)

    if Y_bar > 1e-3:
        rel_error = W_half / Y_bar
        print(f"   Reached N={N_max}, not fully converged (Rel. Error: {rel_error * 100:.1f}%)")
    else:
        rel_error = float('nan')
        print(f"   Reached N={N_max}, not fully converged (Mean tardiness too small)")

    convergence_info = {
        'final_N': N, 'mean_tardiness': Y_bar, 'confidence_half_width': W_half,
        'relative_error': rel_error, 'variance': sigma2_hat, 'converged': False
    }
    return tardiness_results, Y_bar, convergence_info



def deterministic_mam_vprp_evaluation(solution, problem_data, model_type="mode", percentile=None):
    parts_data = problem_data["part_data"]
    customer_data = problem_data["customer_data"]
    printer_params = problem_data["printer_params"]
    travel_time_params = problem_data["travel_time_params"]
    proc_uncertainty = problem_data["proc_uncertainty"]

    # 辅助函数：根据模型类型获取确定性值
    def get_deterministic_value(base_value, sigma, model_type, percentile):
        """获取对数正态分布的确定性值"""
        if sigma <= 0:
            return base_value  # 无不确定性

        if model_type == "mean":
            # 均值: exp(mu + sigma^2/2)
            # 注意：这里的base_value是众数，而计算均值需要转换
            mu = np.log(base_value) + sigma ** 2  # 因为众数 = exp(mu - sigma^2)
            return np.exp(mu + sigma ** 2 / 2)
        else:
            # 确定百分位
            if model_type == "mode":
                p = 0.0
            elif model_type == "median":
                p = 0.50
            elif model_type == "percentile" and percentile is not None:
                p = percentile
            else:
                p = 0.0  # 默认众数

            # 使用现有的 calculate_lognormal_quantile 函数
            return calculate_lognormal_quantile(base_value, sigma, p)

    # 1. 计算基础生产时间
    base_proc_times = {}
    for batch_id, part_list in solution["batch_composition"].items():
        if not part_list:
            continue
        total_volume = sum(parts_data[j]['volume'] for j in part_list)
        max_height = max(parts_data[j]['height'] for j in part_list)
        base_proc_times[batch_id] = (printer_params['alpha'] * total_volume +
                                     printer_params['beta'] * max_height)

    # 2. 生产时间线计算
    part_completion_times = {}
    production_time_accumulator = 0.0

    # 生产时间的乘数（众数=1）
    proc_multiplier = get_deterministic_value(
        base_value=1.0,  # 生产时间乘数的众数为1
        sigma=proc_uncertainty['std'],
        model_type=model_type,
        percentile=percentile
    )

    for batch_id in solution["batch_sequence"]:
        if batch_id not in solution["batch_composition"]:
            continue

        deterministic_proc_time = base_proc_times.get(batch_id, 0) * proc_multiplier
        production_time_accumulator += deterministic_proc_time

        for part_id in solution["batch_composition"][batch_id]:
            part_completion_times[part_id] = production_time_accumulator

    # 3. 车辆路线计算
    total_tardiness = 0.0
    truck_time = 0.0
    route = solution["route"]
    
    # 获取所有需要服务的客户ID
    all_customers = set(customer_data.keys())
    # 获取route中实际访问的客户（排除depot 0）
    visited_customers = set([n for n in route if n != 0])
    # 计算未服务的客户
    unserved_customers = all_customers - visited_customers
    
    # 添加对未服务客户的惩罚（每个未服务客户惩罚1000）
    penalty_per_unserved = 1000.0
    total_tardiness += len(unserved_customers) * penalty_per_unserved

    for i in range(len(route) - 1):
        s_node, n_node = route[i], route[i + 1]

        # 获取旅行时间参数
        travel_params = travel_time_params.get((s_node, n_node), {'value': 0, 'std': 0})
        travel_base = travel_params['value']
        travel_sigma = travel_params.get('std', 0)

        # 旅行时间
        travel_time = get_deterministic_value(
            base_value=travel_base,
            sigma=travel_sigma,
            model_type=model_type,
            percentile=percentile
        )

        arrival_time_at_n = truck_time + travel_time

        if n_node == 0:
            break

        customer_parts = customer_data.get(n_node, {}).get('parts', [])
        if customer_parts:
            last_part_completion_time = max(part_completion_times.get(p, 0) for p in customer_parts)
        else:
            last_part_completion_time = 0.0

        departure_time_from_n = max(arrival_time_at_n, last_part_completion_time)
        tardiness = max(0, departure_time_from_n - customer_data[n_node]['due_date'])
        total_tardiness += min(tardiness, 1e6)
        truck_time = departure_time_from_n

    return total_tardiness


if __name__ == '__main__':
    # 示例运行 (假设 problem_instance.txt 存在)
    data = load_problem_data("instance/problem_instance_5cust_10parts_std01.json")
    if data:

        test_solution = {
            'route': [0, 1, 2, 3, 4, 5, 0],  # 5个客户的合法路径
            'batch_composition': {
                'Batch1': ['p1', 'p2'],
                'Batch2': ['p3', 'p4'],
                'Batch3': ['p5', 'p6'],
                'Batch4': ['p7', 'p8'],
                'Batch5': ['p9', 'p10']
            },
            'batch_sequence': ['Batch1', 'Batch2', 'Batch3', 'Batch4', 'Batch5']
        }

        print("Testing Deterministic Evaluation (BASE-MODE):")

        # 为了防止还有命名不匹配，加上 try-except 保护
        det_val = deterministic_mam_vprp_evaluation(test_solution, data)
        print(f"Base Deterministic Value: {det_val:.2f}\n")

        print("Testing Convergent Simulation:")
        results, mean, info = simulate_mam_vprp_convergent(test_solution, data, N_max=3000)

        print("\nSimulation Test Summary:")
        print(f"Mean Expected Tardiness: {mean:.2f}")
        print(f"Final N: {info['final_N']}")
        print(f"Converged: {info['converged']}")

        import matplotlib.pyplot as plt

        # 设置字体
        plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
        plt.rcParams['axes.unicode_minus'] = False

        plt.figure(figsize=(10, 6))
        plt.hist(results, bins=30, density=True, alpha=0.7, color='skyblue', edgecolor='black')
        plt.axvline(mean, color='yellow', linestyle='dashed', linewidth=2,
                    label=f'Simulation Mean: {mean:.2f}')
        plt.axvline(det_val, color='red', linestyle='dashed', linewidth=2,
                    label=f'Deterministic Value: {det_val:.2f}')
        plt.title(f"Total Tardiness Distribution")
        plt.xlabel("Total Tardiness")
        plt.ylabel("Probability Density")
        plt.legend()
        plt.grid(axis='y', alpha=0.75)
        plt.tight_layout()
        plt.show()
    else:
        print("Skipping simulation test due to data loading failure.")