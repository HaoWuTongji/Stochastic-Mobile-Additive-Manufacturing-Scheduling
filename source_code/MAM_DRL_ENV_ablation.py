import gymnasium as gym
from gymnasium import spaces
import numpy as np
import heapq
import math
import random
from collections import defaultdict
import json
import os
#TODO:环境版本号
ENV_VERSION = "MAM_DRL_ENV_16obs_active_actions_ablation_v01"
# ==========================================
# 常量：离散事件类型 (按优先级排序，确保同一时间点优先处理到达事件)
# ==========================================
EV_NEW_ORDER = 1  # 突发新订单到达
EV_TRUCK_ARRIVE = 2  # 卡车到达节点
EV_PRINT_DONE = 3  # 3D打印机完成当前批次

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



# ==========================================
# 实体类定义
# ==========================================
class Order:
    def __init__(self, order_id, node_id, area, volume, height, due_date,
                 customer_id=None, part_name=None):
        self.id = order_id
        self.customer_id = customer_id if customer_id is not None else node_id
        self.node_id = node_id
        self.part_name = part_name
        self.area = area
        self.volume = volume
        self.height = height
        self.due_date = due_date
        self.status = 'unassigned'


class Truck:
    def __init__(self):
        self.from_node = 0
        self.to_node = 0
        self.depart_time = 0.0
        self.expected_arrival = 0.0
        self.route_version = 0  # 路线版本号，用于废弃幽灵事件
        self.state = 'idle'  # 'idle'(闲置/空等), 'moving'(移动中)
        self.to_node = 0
        self.planned_target = None  # 👈 新增：预计划的下一站


class Printer:
    def __init__(self, max_area):
        self.max_area = max_area
        self.is_printing = False
        self.current_batch = []  # 当前炉正在打的订单对象列表
        self.finish_time = 0.0


# ==========================================
# 强化学习主环境
# ==========================================
class MobileAMEnv(gym.Env):
    """移动增材制造动态调度 DRL 环境"""

    def __init__(self, instance_path=None, config=None,active_actions=None):
        super(MobileAMEnv, self).__init__()

        self.debug = False

        if instance_path is None:
            # 这种写法会自动定位到当前文件所在目录下的 instance/problem_instance.json
            current_dir = os.path.dirname(os.path.abspath(__file__))
            self.instance_path = os.path.join(current_dir, "instance", "problem_instance.json")
        else:
            self.instance_path = instance_path

        # # 动作空间：假设定义了 6 种规则组合 (如 0: NN+EDD, 1: NN+SPT等)
        # self.action_space = spaces.Discrete(6)
        # self.action_meanings = {
        #     0: "nearest + EDD",
        #     1: "urgent + EDD",
        #     2: "nearest + SPT",
        #     3: "risk + SPT",
        #     4: "batch_gain + CR",
        #     5: "nearest + same_node_cluster",
        # }

        # 默认使用完整六动作空间
        if active_actions is None:
            active_actions = [0, 1, 2, 3, 4, 5]

        self.active_actions = list(active_actions)

        if len(self.active_actions) == 0:
            raise ValueError("active_actions cannot be empty.")

        for global_id in self.active_actions:
            if global_id not in ACTION_RULES:
                raise ValueError(f"Unknown global action id in active_actions: {global_id}")

        # PPO 只看到 local action: 0, 1, ..., len(active_actions)-1
        self.action_space = spaces.Discrete(len(self.active_actions))

        # 用于打印和分析
        self.action_meanings = {
            local_id: ACTION_RULES[global_id]["name"]
            for local_id, global_id in enumerate(self.active_actions)
        }

        self.global_action_meanings = {
            global_id: ACTION_RULES[global_id]["name"]
            for global_id in self.active_actions
        }

        # TODO:状态空间：提取 _ 个维度的宏观特征 (均归一化到 0~1 之间)
        # self.observation_space = spaces.Box(low=0.0, high=1.0, shape=(22,), dtype=np.float32)

        # 配套修改 1：定义 15 维非对称观察空间
        # 我们让索引为 6, 10, 13 的维度（涉及交期松弛的维度）下限为 -1.0
        low_bounds = np.zeros(16, dtype=np.float32)
        low_bounds[6] = -1.0  # 现货最紧交期
        low_bounds[10] = -1.0  # 待产最紧交期
        low_bounds[13] = -1.0  # 全局最紧交期

        self.observation_space = spaces.Box(
            low=low_bounds,
            high=np.ones(16, dtype=np.float32),
            shape=(16,),
            dtype=np.float32
        )


        # 物理参数设定 (根据你的 JSON 算例抽象)
        self.alpha = 0.5
        self.beta = 1.0

        self.max_batch_area = 130
        self.max_batch_height = 200.0
        self.max_batch_size = 5
        self.printer_area = self.max_batch_area

        # 节点坐标库 (0为车场，其余为客户。这里暂用随机生成的坐标代替)
        # self.node_coords = {i: (random.uniform(0, 100), random.uniform(0, 100)) for i in range(self.num_initial_nodes)}
        self.node_coords = {}  # 先初始化为空字典，具体内容等加载算例时再填

        # 内部系统变量
        self.time = 0.0
        self.event_queue = []
        self.orders = {}
        self.truck = Truck()
        self.printer = Printer(self.printer_area)
        self.total_tardiness = 0.0

        self.initial_order_ratio = 0.3  # 30% 初始订单，70% 动态到达
        self.dynamic_time_range = (50.0, 500.0)  # 动态订单出现的随机时间区间


    def _debug_print(self, *args, **kwargs):
        if self.debug:
            print(*args, **kwargs)


    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        if seed is not None:
            np.random.seed(seed)
            random.seed(seed)

        self.time = 0.0
        self.orders = {}
        self.event_queue = []

        # 初始化资源
        self.printer.is_printing = False
        self.printer.current_batch = []
        self.printer.finish_time = 0.0

        self.truck.state = 'idle'
        self.truck.from_node = 0
        self.truck.to_node = 0
        self.truck.depart_time = 0.0
        self.truck.expected_arrival = 0.0
        self.truck.route_version = 0
        self.truck.planned_target = None

        self._load_instance(self.instance_path)
        # # ✅ 关键：加载算例
        # self._load_instance(r"D:\PythonProjects\pythonProject1\.idea\instance\problem_instance.json")

        # ✅ 构建初始订单（全部存在）
        self._build_orders_from_instance()

        # TODO:动态订单数
        # self._generate_dynamic_order_events(start_id=100, num_dynamic_orders=10)

        obs = self._get_observation()

        return obs, {
            'time': self.time,
            'tardiness': self._calculate_total_tardiness()
        }

    def _load_instance(self, file_path):
        with open(file_path, 'r') as f:
            data = json.load(f)
        self.instance_data = data

        # ✅ 核心逻辑：动态识别所有节点并分配坐标
        # 1. 提取所有客户 ID 并转为整数
        customer_ids = [int(k) for k in data["customer_data"].keys()]

        # 2. 节点全集 = 0号仓库 + 所有客户ID
        all_node_ids = [0] + customer_ids

        # 3. 动态生成坐标（0-100范围内）
        # 固定种子是为了保证同一个算例每次运行坐标是一致的，方便复现结果
        temp_rng = random.Random(42)
        self.node_coords = {
            nid: (temp_rng.uniform(0, 100), temp_rng.uniform(0, 100))
            for nid in all_node_ids
        }

        self._debug_print(f"[System] 算例加载成功：检测到 {len(all_node_ids)} 个节点，坐标库已更新。")


        # 更新节点总数计数（可选，用于特征工程归一化）
        self.num_nodes = len(all_node_ids)

        self.max_observed_orders = max(20, len(self.orders) * 2)  # 留出动态订单余量
        self.max_obs_dist = 300.0  # 大算例地图可能更空旷，适当调大距离分母

    def _build_orders_from_instance(self):
        customer_data = self.instance_data["customer_data"]
        part_data = self.instance_data["part_data"]

        self.orders = {}
        next_order_id = 1

        # 建立一个待分流的订单总表
        temp_all_orders_data = []
        for customer_id_str, cinfo in customer_data.items():
            customer_id = int(customer_id_str)
            for part_name in cinfo["parts"]:
                p = part_data[part_name]
                order_info = {
                    "order_id": next_order_id,
                    "node_id": customer_id,
                    "area": p["area"],
                    "volume": p["volume"],
                    "height": p["height"],
                    "due_date": cinfo["due_date"]  # 这是基准交期
                }
                temp_all_orders_data.append(order_info)
                next_order_id += 1

        # 随机打乱订单顺序
        random.shuffle(temp_all_orders_data)

        # 计算切分点
        split_idx = int(len(temp_all_orders_data) * self.initial_order_ratio)

        # --- A. 处理初始订单 ---
        initial_orders = temp_all_orders_data[:split_idx]
        for o_info in initial_orders:
            order = Order(**o_info)
            self.orders[order.id] = order

        # --- B. 将剩余订单转化为动态到达事件 ---
        dynamic_orders = temp_all_orders_data[split_idx:]
        for o_info in dynamic_orders:
            # 随机生成一个到达时间
            arrival_time = random.uniform(*self.dynamic_time_range)

            # 修正交期：既然订单晚到了，交期也应该相应顺延，或者保持原样（增加压力）
            # 建议顺延：o_info["due_date"] += arrival_time

            # 将订单信息封装进事件数据
            event_data = o_info.copy()
            event_data["due_date"] = o_info["due_date"] + arrival_time
            # 注意：EV_NEW_ORDER 处理逻辑会根据这个 data 创建 Order 对象
            heapq.heappush(self.event_queue, (arrival_time, EV_NEW_ORDER, event_data))

        self._debug_print(f"[System] 算例解析完成：初始订单 {len(initial_orders)} 个，后续动态到达 {len(dynamic_orders)} 个。")


    # def _generate_dynamic_order_events(self, start_id=100, num_dynamic_orders=3):
    #     """
    #     动态生成 part 级订单请求：
    #     - 客户节点必须来自已有 customer_data
    #     - part 必须从该客户原本允许的 parts 中选
    #     - 与静态订单保持相同的 customer / part 结构
    #     """
    #     customer_data = self.instance_data["customer_data"]
    #     part_data = self.instance_data["part_data"]
    #
    #     customer_ids = [int(k) for k in customer_data.keys()]
    #     next_order_id = max(max(self.orders.keys(), default=0) + 1, start_id)
    #
    #     for _ in range(num_dynamic_orders):
    #         arrival_time = random.uniform(50.0, 300.0)
    #
    #         customer_id = random.choice(customer_ids)
    #         candidate_parts = customer_data[str(customer_id)]["parts"]
    #         part_name = random.choice(candidate_parts)
    #         p = part_data[part_name]
    #
    #         dyn_order = {
    #             "order_id": next_order_id,
    #             "customer_id": customer_id,
    #             "node_id": customer_id,
    #             "part_name": part_name,
    #             "area": p["area"],
    #             "volume": p["volume"],
    #             "height": p["height"],
    #             "due_date": arrival_time + random.randint(200, 500)
    #         }
    #
    #         heapq.heappush(self.event_queue, (arrival_time, EV_NEW_ORDER, dyn_order))
    #         next_order_id += 1

    def _get_candidate_nodes(self):
        deliverable_nodes = set()
        future_nodes = set()

        for o in self.orders.values():
            if o.status == 'printed':
                deliverable_nodes.add(o.node_id)
            elif o.status in ('unassigned', 'printing'):
                future_nodes.add(o.node_id)

        return list(deliverable_nodes), list(future_nodes)

    def _get_modal_travel_time(self, i, j):
        if i == j:
            return 0.0

        key = f"{i}_{j}"
        if key in self.instance_data["travel_time_params"]:
            return self.instance_data["travel_time_params"][key]["value"]

        reverse_key = f"{j}_{i}"
        if reverse_key in self.instance_data["travel_time_params"]:
            return self.instance_data["travel_time_params"][reverse_key]["value"]

        return 1e6

    # ==========================================
    # 模块二：启发式规则与动作执行
    # ==========================================
    def _node_distance_from_truck(self, node_id):
        """
        用 modal travel time 代替原来的随机几何距离。
        如果车在移动中，就近似看成：
        剩余到当前目标点的时间 + 从目标点再到候选节点的时间
        """
        if self.truck.state == 'idle':
            current_node = self.truck.to_node
            return self._get_modal_travel_time(current_node, node_id)

        remaining_time = max(self.truck.expected_arrival - self.time, 0.0)

        if node_id == self.truck.to_node:
            return remaining_time

        return remaining_time + self._get_modal_travel_time(self.truck.to_node, node_id)

    def _estimate_print_time(self, order):
        """
        用于规则排序的“估计打印时间”。
        如果你后面已经有更真实的打印时间公式，直接替换这里就行。
        这里先用 area / volume / height 做一个相对稳定的估计。
        """
        return 0.2 * order.area + 0.5 * order.volume + 3.0 * order.height

    def _build_batch_from_sorted(self, sorted_orders):
        """
        通用批处理装载函数。
        这版默认优先兼容你当前代码：
        - 如果定义了 max_batch_size / max_batch_area / max_batch_height，就按约束装
        - 如果没定义，就默认最多装 3 个
        """
        max_batch_size = getattr(self, "max_batch_size", 3)
        max_batch_area = getattr(self, "max_batch_area", None)
        max_batch_height = getattr(self, "max_batch_height", None)

        batch = []
        used_area = 0.0
        current_max_height = 0.0

        for o in sorted_orders:
            if len(batch) >= max_batch_size:
                break

            next_area = used_area + getattr(o, "area", 0.0)
            next_height = max(current_max_height, getattr(o, "height", 0.0))

            if max_batch_area is not None and next_area > max_batch_area:
                continue

            if max_batch_height is not None and next_height > max_batch_height:
                continue

            batch.append(o)
            used_area = next_area
            current_max_height = next_height

        return batch

    def _route_nearest_neighbor(self, candidate_nodes):
        if not candidate_nodes:
            return None
        return min(candidate_nodes, key=lambda nid: self._node_distance_from_truck(nid))

    def _route_earliest_due_node(self, deliverable_orders):
        """
        严格按“最早交期订单所在节点”选点
        """
        if not deliverable_orders:
            return None
        return min(deliverable_orders, key=lambda o: o.due_date).node_id

    def _route_most_urgent(self, deliverable_orders):
        """
        按节点的“最小剩余松弛时间”选点：
        slack = due_date - (当前时间 + 预计运输时间)
        谁最危险先去谁
        """
        if not deliverable_orders:
            return None

        node_best_slack = {}
        for o in deliverable_orders:
            dist = self._node_distance_from_truck(o.node_id)
            slack = o.due_date - (self.time + dist)
            if o.node_id not in node_best_slack:
                node_best_slack[o.node_id] = slack
            else:
                node_best_slack[o.node_id] = min(node_best_slack[o.node_id], slack)

        return min(node_best_slack, key=node_best_slack.get)

    def _route_max_risk(self, deliverable_orders):
        """
        按“预计到达时的延误风险”选点：
        risk = (当前时间 + 运输时间) - 该节点最早交期
        risk 越大，越危险
        """
        if not deliverable_orders:
            return None

        node_earliest_due = {}
        for o in deliverable_orders:
            if o.node_id not in node_earliest_due:
                node_earliest_due[o.node_id] = o.due_date
            else:
                node_earliest_due[o.node_id] = min(node_earliest_due[o.node_id], o.due_date)

        best_node = None
        best_risk = -float("inf")

        for node_id, earliest_due in node_earliest_due.items():
            dist = self._node_distance_from_truck(node_id)
            risk = (self.time + dist) - earliest_due
            if risk > best_risk:
                best_risk = risk
                best_node = node_id

        return best_node

    def _route_best_batch_gain(self, deliverable_orders):
        """
        综合考虑：
        - 该节点可交付订单数
        - 距离成本
        - 交期紧急度
        分数越高越优先
        """
        if not deliverable_orders:
            return None

        node_orders = defaultdict(list)
        for o in deliverable_orders:
            node_orders[o.node_id].append(o)

        best_node = None
        best_score = -float("inf")

        for node_id, orders in node_orders.items():
            dist = self._node_distance_from_truck(node_id)
            count = len(orders)
            earliest_due = min(o.due_date for o in orders)

            urgency = max(0.0, 200.0 - (earliest_due - self.time)) / 200.0
            score = count / (dist + 1.0) + urgency

            if score > best_score:
                best_score = score
                best_node = node_id

        return best_node


    ###打印规则


    def _batch_edd_greedy(self, orders):
        """
        Earliest Due Date：最早交期优先
        """
        if not orders:
            return []

        sorted_orders = sorted(
            orders,
            key=lambda o: (o.due_date, self._estimate_print_time(o))
        )
        return self._build_batch_from_sorted(sorted_orders)

    def _batch_spt_greedy(self, orders):
        """
        Shortest Processing Time：最短加工时间优先
        """
        if not orders:
            return []

        sorted_orders = sorted(
            orders,
            key=lambda o: (self._estimate_print_time(o), o.due_date)
        )
        return self._build_batch_from_sorted(sorted_orders)

    def _batch_cr_greedy(self, orders):
        """
        Critical Ratio：
        CR = (due_date - current_time) / processing_time
        CR 越小越紧急
        """
        if not orders:
            return []

        def cr(o):
            p = max(self._estimate_print_time(o), 1e-6)
            return (o.due_date - self.time) / p

        sorted_orders = sorted(
            orders,
            key=lambda o: (cr(o), o.due_date)
        )
        return self._build_batch_from_sorted(sorted_orders)

    def _batch_same_node_cluster_greedy(self, orders):
        """
        优先把同一个节点的订单集中打出来，
        有利于后面整批配送。
        """
        if not orders:
            return []

        node_groups = defaultdict(list)
        for o in orders:
            node_groups[o.node_id].append(o)

        # 先选“数量多且交期更紧”的节点
        best_node = max(
            node_groups.keys(),
            key=lambda nid: (
                len(node_groups[nid]),
                -min(o.due_date for o in node_groups[nid])
            )
        )

        primary_orders = sorted(
            node_groups[best_node],
            key=lambda o: (o.due_date, self._estimate_print_time(o))
        )

        other_orders = sorted(
            [o for o in orders if o.node_id != best_node],
            key=lambda o: (o.due_date, self._estimate_print_time(o))
        )

        return self._build_batch_from_sorted(primary_orders + other_orders)

    # def _apply_action(self, action_id):
    #     """
    #     重构版动作执行：非抢占式物流 + 预约制决策
    #     1. 决策当前批次打印什么
    #     2. 决策卡车的“当前目标”或“预约目标”
    #     """
    #
    #     # ---------- A. 准备候选集 ----------
    #     unassigned = [o for o in self.orders.values() if o.status == "unassigned"]
    #     printed_orders = [o for o in self.orders.values() if o.status == "printed"]
    #     printed_nodes = list(set(o.node_id for o in printed_orders))
    #
    #     current_batch_orders = list(self.printer.current_batch) if self.printer.is_printing else []
    #     current_batch_nodes = list(set(o.node_id for o in current_batch_orders))
    #
    #     # ---------- B. 决策：下一批打什么 (Batching) ----------
    #     next_batch = []
    #     if action_id == 0:
    #         next_batch = self._batch_edd_greedy(unassigned)
    #     elif action_id == 1:
    #         next_batch = self._batch_edd_greedy(unassigned)
    #     elif action_id == 2:
    #         next_batch = self._batch_spt_greedy(unassigned)
    #     elif action_id == 3:
    #         next_batch = self._batch_spt_greedy(unassigned)
    #     elif action_id == 4:
    #         next_batch = self._batch_cr_greedy(unassigned)
    #     elif action_id == 5:
    #         next_batch = self._batch_same_node_cluster_greedy(unassigned)
    #     else:
    #         next_batch = self._batch_edd_greedy(unassigned)
    #
    #
    #     # TODO:逻辑顺序
    #     # 启动打印机（如果空闲）
    #     if (not self.printer.is_printing) and next_batch:
    #         self._start_printing(next_batch)
    #         current_batch_orders = next_batch
    #         current_batch_nodes = list(set(o.node_id for o in current_batch_orders))
    #     else:
    #         current_batch_orders = list(self.printer.current_batch) if self.printer.is_printing else []
    #         current_batch_nodes = list(set(o.node_id for o in current_batch_orders))
    #
    #     # ---------- C. 决策：卡车去哪 (Routing) ----------
    #     next_node = None
    #
    #     # 优先级逻辑：
    #     # 第一优先级：去送现货 (Printed)
    #     if printed_orders:
    #         if action_id in [0, 2, 5]:
    #             next_node = self._route_nearest_neighbor(printed_nodes)
    #         elif action_id == 1:
    #             next_node = self._route_most_urgent(printed_orders)
    #         elif action_id == 3:
    #             next_node = self._route_max_risk(printed_orders)
    #         elif action_id == 4:
    #             next_node = self._route_best_batch_gain(printed_orders)
    #         else:
    #             next_node = self._route_nearest_neighbor(printed_nodes)
    #
    #     # 第二优先级：去等正在打的货 (Current Batch)
    #     elif current_batch_orders:
    #         if action_id in [0, 2, 5]:
    #             next_node = self._route_nearest_neighbor(current_batch_nodes)
    #         elif action_id == 1:
    #             next_node = self._route_most_urgent(current_batch_orders)
    #         elif action_id == 3:
    #             next_node = self._route_max_risk(current_batch_orders)
    #         elif action_id == 4:
    #             next_node = self._route_best_batch_gain(current_batch_orders)
    #         else:
    #             next_node = self._route_nearest_neighbor(current_batch_nodes)
    #
    #
    #
    #     # ---------- D. 执行：分流处理 (Non-preemptive Logic) ----------
    #     if self.truck.state == "idle":
    #         # 状态 1: 卡车在节点处。直接出发去“当前目标”
    #         if next_node is not None and next_node != self.truck.to_node:
    #             self._dispatch_truck(next_node)
    #             # 既然已经出发了，清除之前的预约
    #             self.truck.planned_target = None
    #     else:
    #         # 状态 2: 卡车在派送途中。不改变当前航向，但更新“预约目标”
    #         # 这就是你说的：不改变“当前目标节点”，只决策“下一目标节点”
    #         if next_node is not None:
    #             self.truck.planned_target = next_node
    #
    #     # 调试打印
    #     status = "IDLE-Dispatch" if self.truck.state == "idle" else "MOVING-Reserve"
    #     self._debug_print(f"[{status}] Action={action_id}, NextNode={next_node}, PlannedNext={self.truck.planned_target}")

    ##TODO：
        # =========================================================
        # D. 打印调度：打印机空闲时启动下一批
        # 当前批次一旦开始，不中途撤销
        # =========================================================
        # if (not self.printer.is_printing) and next_batch:
        #     self._start_printing(next_batch)

    def _apply_action(self, action):
        """
        动作执行逻辑：规则选择型动作 + 非抢占式车辆调度 + 预约目标机制

        local_action:
            PPO 实际输出的动作编号，范围为 0 ~ len(active_actions)-1

        global_action:
            原始六动作空间中的动作编号，范围为 0 ~ 5
        """

        # ---------- 1. local action -> global action ----------
        local_action = int(action)

        if local_action < 0 or local_action >= len(self.active_actions):
            raise ValueError(f"Invalid local action: {local_action}")

        global_action = self.active_actions[local_action]
        rule = ACTION_RULES[global_action]

        route_rule = rule["route_rule"]
        print_rule = rule["print_rule"]

        truck_state_before = self.truck.state

        # ---------- 2. 根据打印规则选择下一批次 ----------
        next_batch = self._select_batch_by_rule(print_rule)

        # 如果打印机空闲，则启动打印
        if (not self.printer.is_printing) and next_batch:
            self._start_printing(next_batch)

        # ---------- 3. 根据车辆规则选择下一目标节点 ----------
        # 注意：这里不要再把 next_node 重置为 None
        next_node = self._select_next_node_by_rule(route_rule)

        # ---------- 4. 非抢占式车辆调度 ----------
        if self.truck.state == "idle":
            # 车辆空闲：可以直接派车
            if next_node is not None and next_node != self.truck.to_node:
                self._dispatch_truck(next_node)
                self.truck.planned_target = None

            # 如果 next_node 就是当前所在节点，不需要派车
            elif next_node == self.truck.to_node:
                self.truck.planned_target = None

        else:
            # 车辆正在移动：不改变当前目标，只更新预约目标
            if next_node is not None:
                # 避免把当前正在前往的节点重复设为预约目标
                if next_node != self.truck.to_node:
                    self.truck.planned_target = next_node

        # ---------- 5. 记录动作信息，方便日志分析 ----------
        self.last_local_action = local_action
        self.last_global_action = global_action
        self.last_action_name = rule["name"]

        # ---------- 6. 调试打印 ----------
        if truck_state_before == "idle":
            status = "IDLE-Decision"
        else:
            status = "MOVING-Reserve"

        self._debug_print(
            f"[{status}] "
            f"LocalAction={local_action}, "
            f"GlobalAction={global_action}, "
            f"Rule={rule['name']}, "
            f"RouteRule={route_rule}, "
            f"PrintRule={print_rule}, "
            f"NextNode={next_node}, "
            f"PlannedNext={self.truck.planned_target}"
        )

    def _select_next_node_by_rule(self, route_rule):
        """
        根据车辆路径规则选择下一目标节点。

        注意：
        - PPO 输出的是 local action；
        - _apply_action() 已经把 local action 映射为 global action；
        - 这里不再关心动作编号，只根据 route_rule 调用具体路径规则。

        候选优先级沿用原始环境逻辑：
        1. 优先配送已经打印完成的订单 printed；
        2. 如果没有 printed，则可以前瞻当前正在打印的批次 current_batch；
        3. 如果二者都没有，则车辆不派发，返回 None。
        """
        printed_orders = [
            o for o in self.orders.values()
            if o.status == "printed"
        ]

        current_batch_orders = (
            list(self.printer.current_batch)
            if self.printer.is_printing
            else []
        )

        if printed_orders:
            route_orders = printed_orders
        elif current_batch_orders:
            route_orders = current_batch_orders
        else:
            return None

        candidate_nodes = list(set(o.node_id for o in route_orders))

        if route_rule == "nearest":
            return self._route_nearest_neighbor(candidate_nodes)

        elif route_rule == "urgent":
            return self._route_most_urgent(route_orders)

        elif route_rule == "risk":
            return self._route_max_risk(route_orders)

        elif route_rule == "batch_gain":
            return self._route_best_batch_gain(route_orders)

        else:
            raise ValueError(f"Unknown route rule: {route_rule}")

    def _select_batch_by_rule(self, print_rule):
        """
        根据打印批次规则选择下一批订单。

        这里调用的是本文件已有的 greedy 批处理函数：
        - _batch_edd_greedy
        - _batch_spt_greedy
        - _batch_cr_greedy
        - _batch_same_node_cluster_greedy

        因此不会再出现 _select_batch_spt 不存在的问题。
        """
        unassigned = [
            o for o in self.orders.values()
            if o.status == "unassigned"
        ]

        if print_rule == "EDD":
            return self._batch_edd_greedy(unassigned)

        elif print_rule == "SPT":
            return self._batch_spt_greedy(unassigned)

        elif print_rule == "CR":
            return self._batch_cr_greedy(unassigned)

        elif print_rule == "same_node_cluster":
            return self._batch_same_node_cluster_greedy(unassigned)

        else:
            raise ValueError(f"Unknown print rule: {print_rule}")


    def _get_travel_time(self, i, j):
        if i == j:
            return 0.0

        key = f"{i}_{j}"
        reverse_key = f"{j}_{i}"

        if key in self.instance_data["travel_time_params"]:
            param = self.instance_data["travel_time_params"][key]
        elif reverse_key in self.instance_data["travel_time_params"]:
            param = self.instance_data["travel_time_params"][reverse_key]
        else:
            raise KeyError(f"Travel time parameter missing for edge {i}_{j} and {j}_{i}")

        mean = param["value"]
        std = param["std"]

        return max(0.1, np.random.lognormal(mean=np.log(mean), sigma=std))

    def _dispatch_truck(self, target_node):
        current_node = self.truck.to_node

        if target_node == current_node:
            self.truck.state = 'idle'
            self.truck.from_node = current_node
            self.truck.to_node = current_node
            return

        self.truck.state = 'moving'
        self.truck.from_node = current_node
        self.truck.to_node = target_node
        self.truck.depart_time = self.time
        self.truck.route_version += 1

        real_time = self._get_travel_time(current_node, target_node)

        self.truck.expected_arrival = self.time + real_time
        self._debug_print(
            f"[dispatch] time={self.time:.1f}, from={current_node}, to={target_node}, version={self.truck.route_version}")
        self._debug_print(f"[dispatch] expected_arrival={self.truck.expected_arrival:.1f}")

        heapq.heappush(
            self.event_queue,
            (self.truck.expected_arrival, EV_TRUCK_ARRIVE, {'version': self.truck.route_version})
        )


    def _start_printing(self, batch):
        """启动打印机，计算加工时间并生成打印完成事件"""
        self.printer.is_printing = True
        self.printer.current_batch = batch
        for o in batch:
            o.status = 'printing'  # 状态翻转

        # 根据你的确定性模型等式计算时间
        max_h = max([o.height for o in batch])
        tot_v = sum([o.volume for o in batch])
        base_time = self.alpha * max_h + self.beta * tot_v

        # 加入加工时间的不确定性 (算例中的 std: 0.15)
        real_time = max(0.1, np.random.lognormal(mean=math.log(base_time), sigma=0.15))

        self.printer.finish_time = self.time + real_time
        heapq.heappush(self.event_queue, (self.printer.finish_time, EV_PRINT_DONE, None))

        self._debug_print(f"[print_start] time={self.time:.1f}, batch={[o.id for o in batch]}, finish={self.printer.finish_time:.1f}")

    # ==========================================
    # 模块三/四：离散事件引擎与 Step 交互
    # ==========================================
    def step(self, action):
        """强化学习核心交互步"""
        # --- 记录执行动作前的状态 ---
        old_time = self.time
        # 记录当前的预约目标，用于后续对比
        old_planned_target = self.truck.planned_target

        # 1. 执行动作
        self._apply_action(action)

        # old_tardiness = self._calculate_total_tardiness()
        old_weighted_tardiness = self._calculate_weighted_tardiness()
        decision_needed = False
        step_delivered_count = 0  # ✅ 新增：用于统计本步交付的订单数
        step_on_time_count = 0

        # 2. 事件推进
        while self.event_queue and not decision_needed:
            self._debug_print(f"[before_pop] time={self.time:.1f}, queue_len={len(self.event_queue)}")

            event_time, event_type, data = heapq.heappop(self.event_queue)
            self._debug_print(f"[pop] event_time={event_time:.1f}, event_type={event_type}, data={data}")

            self.time = event_time
            self._debug_print(f"[time_update] self.time={self.time:.1f}")

            if event_type == EV_NEW_ORDER:
                new_o = Order(**data)
                self.orders[new_o.id] = new_o
                self._debug_print(f"[NEW_ORDER] time={self.time:.1f}, new_order_id={new_o.id}")

                # 删除了 decision_needed = True
                # 新订单仅被加入系统列表，系统会继续推进时间直到下一个物理事件（如卡车到达或打印完成）
                #TODO:如果系统当时卡车和打印机都空闲，新订单到达后让它马上决策并不属于“途中改派”，而是“空闲资源接新任务”
                if self.truck.state == "idle" and not self.printer.is_printing:
                    decision_needed = True
                    self._debug_print(f"[decision] triggered because all resources are idle")


            elif event_type == EV_TRUCK_ARRIVE:
                if data["version"] != self.truck.route_version:
                    continue

                self.truck.state = "idle"
                count, on_time = self._attempt_delivery()
                step_delivered_count += count
                step_on_time_count += on_time
                self._debug_print(f"[TRUCK_ARRIVE] time={self.time:.1f}, node={self.truck.to_node}")

                # ✅ 关键逻辑：到达后自动检查并执行“预约目标”
                if self.truck.planned_target is not None:
                    target = self.truck.planned_target
                    self.truck.planned_target = None  # 弹出预约

                    # 如果预约的节点不是当前所在地，立即出发
                    if target != self.truck.to_node:
                        self._debug_print(f"[Auto-Connect] 执行预约目标: {target}")
                        self._dispatch_truck(target)

                decision_needed = True


            elif event_type == EV_PRINT_DONE:
                self.printer.is_printing = False
                for o in self.printer.current_batch:
                    o.status = "printed"
                self.printer.current_batch = []

                # self._attempt_delivery()
                count, on_time = self._attempt_delivery()
                step_delivered_count += count
                step_on_time_count += on_time
                self._debug_print(f"[PRINT_DONE] time={self.time:.1f}")

                decision_needed = True
                self._debug_print(f"[decision] triggered by PRINT_DONE")

        done = all(o.status == "delivered" for o in self.orders.values())


        # --- 2. 计算稳定性奖励 (Consistency Penalty) ---
        new_planned_target = self.truck.planned_target
        stability_penalty = 0.0
        # # 逻辑：只有当卡车在移动中，且预约目标从一个节点变成了另一个节点时，才惩罚
        # # 如果从 None 变成某个节点（建立预约），或者从节点变成 None（取消预约），通常不视为恶性跳变
        # if self.truck.state == 'moving':
        #     if old_planned_target is not None and new_planned_target is not None:
        #         if old_planned_target != new_planned_target:
        #             stability_penalty = -5.0  # “违约金”，强制 Agent 保持决策连贯性
        #             # print(f"[Penalty] 预约目标变动: {old_planned_target} -> {new_planned_target}")

        # # ✅ (4) 新增：空驶距离惩罚 (Deadhead Penalty)
        # # 逻辑：只要这段时间内卡车在跑，就按时间扣分
        # travel_time_in_step = self.time - old_time
        # dist_penalty = 0.0
        # if self.truck.state == 'moving':
        #     # 这里的系数 0.5 意味着卡车空跑 1 分钟，相当于额外增加 10 分钟的订单延误
        #     dist_penalty = -travel_time_in_step * 0.05

        # current_tardiness = self._calculate_total_tardiness()
        #TODO:奖励
        current_weighted_tardiness = self._calculate_weighted_tardiness()

        # 基础延误惩罚 (注意缩放系数)
        tardiness_penalty = -(current_weighted_tardiness - old_weighted_tardiness) * 0.01

        # 里程碑奖励：每交付一个订单给 +10 的奖励 (这个值不进行 0.01 缩放，给以强刺激)
        delivery_bonus = step_delivered_count * 0.1

        # 3.准时奖：每准时交付一个给 +100 (不缩放)
        on_time_bonus = step_on_time_count * 0.2

        reward = tardiness_penalty + delivery_bonus + on_time_bonus

        # 终局惩罚：让模型更关心最终总延误
        if done:
            reward -= 0.005 * current_weighted_tardiness

        obs = self._get_observation()

        self._debug_print(f"[step_end] time={self.time:.1f}, done={done}, reward={reward:.2f}, queue_len={len(self.event_queue)}")

        return obs, float(reward), done, False, {
            "time": self.time,
            # "tardiness": current_tardiness
            "tardiness": current_weighted_tardiness
        }


    def _calculate_weighted_tardiness(self):
        """
        ✅ 非线性延误计算：
        如果订单延误超过 500 分钟，超出部分按 2 倍惩罚
        """
        weighted_total = 0.0
        for o in self.orders.values():
            delay = 0.0
            if o.status == 'delivered':
                delay = max(o.delivery_time - o.due_date, 0)
            else:
                delay = max(self.time - o.due_date, 0)

            # 非线性加重逻辑
            if delay > 500:
                weighted_total += 500 + (delay - 500) * 2.0
            else:
                weighted_total += delay
        return weighted_total


    def _attempt_delivery(self):
        current_node = self.truck.to_node
        delivered_count = 0###TODO:送达计数初始化_为后面作为送完货之后进行重调度的判断标准之一（？）
        on_time_count = 0  # ✅ 新增：准时交付计数器

        for o in self.orders.values():
            if o.node_id == current_node and o.status == 'printed':
                o.delivery_time = self.time
                o.status = 'delivered'
                delivered_count += 1
                self._debug_print(f"[Time {self.time:.1f}] 订单 {o.id} 在节点 {current_node} 成功交付!")

                # ✅ 判定逻辑：如果交付时间早于或等于交期，计为准时
                if o.delivery_time <= o.due_date:
                    on_time_count += 1
                    self._debug_print(f"🌟 [On-Time] 订单 {o.id} 准时送达！")

        return delivered_count, on_time_count

    # ==========================================
    # 辅助函数与特征工程
    # ==========================================
    def _get_truck_realtime_coords(self):
        """计算卡车实时的二维欧氏坐标 (包括行进中的虚拟坐标)"""
        if self.truck.state == 'idle' or self.truck.depart_time == self.truck.expected_arrival:
            return self.node_coords[self.truck.to_node]

        # 线性插值计算进度
        total_time = self.truck.expected_arrival - self.truck.depart_time
        progress = (self.time - self.truck.depart_time) / total_time
        progress = min(max(progress, 0.0), 1.0)  # 约束在 0~1

        x_from, y_from = self.node_coords[self.truck.from_node]
        x_to, y_to = self.node_coords[self.truck.to_node]

        v_x = x_from + progress * (x_to - x_from)
        v_y = y_from + progress * (y_to - y_from)
        return (v_x, v_y)

    def _calculate_total_tardiness(self):
        total = 0.0
        for o in self.orders.values():
            if o.status == 'delivered':
                total += max(o.delivery_time - o.due_date, 0)
            else:
                total += max(self.time - o.due_date, 0)
        return total

    # def _get_observation(self):
    #     # 初始化 22 维向量
    #     obs = np.zeros(22, dtype=np.float32)
    #
    #     # --- 1. 基础数据准备 (必须保留这些定义) ---
    #     all_orders = list(self.orders.values())
    #     unassigned = [o for o in all_orders if o.status == 'unassigned']
    #     printed = [o for o in all_orders if o.status == 'printed']
    #     printing = [o for o in all_orders if o.status == 'printing']
    #     undelivered = [o for o in all_orders if o.status != 'delivered']  # 修正拼写
    #
    #     # 如果所有订单都送达了，直接返回全 0（防止计算出错）
    #     if not undelivered:
    #         return obs
    #
    #     # 归一化常量 #TODO:根据算例范围调整
    #     MAX_DIST = self.max_obs_dist
    #     MAX_TIME = 2000.0
    #     MAX_ORDERS = self.max_observed_orders #改为动态调整了
    #     GRID_SIZE = 100.0
    #
    #     # --- 2. 特征填充 ---
    #
    #     # A. 卡车时空特征 (6维: 0-5)
    #     curr_x, curr_y = self._get_truck_realtime_coords()
    #     target_x, target_y = self.node_coords[self.truck.to_node]
    #
    #     obs[0] = curr_x / GRID_SIZE
    #     obs[1] = curr_y / GRID_SIZE
    #     obs[2] = target_x / GRID_SIZE
    #     obs[3] = target_y / GRID_SIZE
    #     obs[4] = 1.0 if self.truck.state == 'moving' else 0.0
    #
    #     remain_travel = max(0.0, self.truck.expected_arrival - self.time)
    #     obs[5] = min(1.0, remain_travel / MAX_DIST)
    #
    #     # B. 打印机状态特征 (4维: 6-9)
    #     obs[6] = 1.0 if self.printer.is_printing else 0.0
    #     remain_print = max(0.0, self.printer.finish_time - self.time)
    #     obs[7] = min(1.0, remain_print / MAX_DIST)
    #
    #     current_area = sum(o.area for o in self.printer.current_batch)
    #     ##TODO： 使用 min(1.0, ...) 确保永远不会超过边界
    #     obs[8] = min(1.0, (current_area / self.printer.max_area)) if self.printer.max_area > 0 else 0.0
    #     obs[9] = len(self.printer.current_batch) / 5.0  # 假设一炉最多5个零件
    #
    #     # C. 现货 (Printed) 分布特征 (4维: 10-13)
    #     if printed:
    #         dist_to_printed = [self._node_distance_from_truck(o.node_id) for o in printed]
    #         obs[10] = min(dist_to_printed) / MAX_DIST
    #
    #         slacks = [(o.due_date - self.time) for o in printed]
    #         obs[11] = max(-1.0, min(slacks) / MAX_TIME)  # 允许负值，表示已延误
    #
    #         obs[12] = len(printed) / MAX_ORDERS
    #         at_target = any(o.node_id == self.truck.to_node for o in printed)
    #         obs[13] = 1.0 if at_target else 0.0
    #     else:
    #         obs[10], obs[11], obs[12], obs[13] = 1.0, 1.0, 0.0, 0.0
    #
    #     # D. 未分配订单 (Unassigned) 全局特征 (4维: 14-17)
    #     if unassigned:
    #         dist_to_un = [self._node_distance_from_truck(o.node_id) for o in unassigned]
    #         obs[14] = min(dist_to_un) / MAX_DIST
    #
    #         u_slacks = [(o.due_date - self.time) for o in unassigned]
    #         obs[15] = max(-1.0, min(u_slacks) / MAX_TIME)
    #
    #         obs[16] = len(unassigned) / MAX_ORDERS
    #         avg_area = (sum(o.area for o in unassigned) / len(unassigned)) if len(unassigned) > 0 else 0
    #         obs[17] = avg_area / self.printer.max_area
    #     else:
    #         obs[14], obs[15], obs[16], obs[17] = 1.0, 1.0, 0.0, 0.0
    #
    #     # E. 系统宏观与协同特征 (4维: 18-21)
    #     obs[18] = len(undelivered) / MAX_ORDERS  # 总未完成数
    #
    #     all_slacks = [(o.due_date - self.time) for o in undelivered]
    #     obs[19] = max(-1.0, min(all_slacks) / MAX_TIME)  # 全局最紧急交期
    #
    #     # 协同度：当前目标点关联的订单总数（打印中+已完成）
    #     target_related = [o for o in undelivered if o.node_id == self.truck.to_node]
    #     obs[20] = len(target_related) / self.max_observed_orders
    #
    #     obs[21] = len(self.event_queue) / (self.max_observed_orders + 10.0)
    #
    #     # 这一步防止任何由于浮点数计算（如 1.0000001）导致的报错
    #     obs = np.clip(obs, 0.0, 1.0)
    #
    #     # 特殊处理：涉及交期（Slack）的维度如果允许负值（延误），可以单独处理
    #     # 比如 obs[11], obs[15], obs[19] 允许在 [-1, 1] 之间
    #     # 但为了通过 check_env，如果你的 low 设的是 0，就必须 clip 到 [0, 1]
    #
    #     return obs.astype(np.float32)

    def _get_observation(self):
        obs = np.zeros(16, dtype=np.float32)

        # --- 1. 数据准备 ---
        all_orders = list(self.orders.values())
        unassigned = [o for o in all_orders if o.status == 'unassigned']
        printed = [o for o in all_orders if o.status == 'printed']
        undelivered = [o for o in all_orders if o.status != 'delivered']

        if not undelivered:
            return obs

        # 使用动态分母，防止大算例爆表 (已经在 _load_instance 中定义)
        MAX_DIST = getattr(self, "max_obs_dist", 400.0)
        MAX_TIME = 2000.0
        MAX_ORDERS = getattr(self, "max_observed_orders", 30.0)

        # --- 2. 模块 A: 卡车状态 (2维) ---
        # 1. 状态位：0 为 idle, 1 为 moving
        obs[0] = 1.0 if self.truck.state == 'moving' else 0.0
        # 2. 剩余行驶时间占比 (非抢占式核心：还要多久才能自由决策)
        remain_travel = max(0.0, self.truck.expected_arrival - self.time)
        obs[1] = min(1.0, remain_travel / MAX_DIST)

        # --- 3. 模块 B: 打印机状态 (3维) ---
        # 3. 状态位：0 为 idle, 1 为 printing
        obs[2] = 1.0 if self.printer.is_printing else 0.0
        # 4. 剩余打印时间占比
        remain_print = max(0.0, self.printer.finish_time - self.time)
        obs[3] = min(1.0, remain_print / MAX_DIST)
        # 5. 打印面积占用率
        curr_area = sum(o.area for o in self.printer.current_batch)
        obs[4] = (curr_area / self.printer.max_area) if self.printer.max_area > 0 else 0.0

        # --- 4. 模块 C: 现货 (Printed) 分布 (4维) ---
        if printed:
            # 6. 最近现货距离
            dists_p = [self._node_distance_from_truck(o.node_id) for o in printed]
            obs[5] = min(dists_p) / MAX_DIST
            # 7. 现货最紧交期 (允许为负)
            slacks_p = [(o.due_date - self.time) / MAX_TIME for o in printed]
            obs[6] = min(slacks_p)
            # 8. 现货订单数占比
            obs[7] = len(printed) / MAX_ORDERS
            # 9. 当前目标点是否有货可卸 (决定当前的 trip 是否有效)
            at_target = any(o.node_id == self.truck.to_node for o in printed)
            obs[8] = 1.0 if at_target else 0.0
        else:
            obs[5], obs[6], obs[7], obs[8] = 1.0, 1.0, 0.0, 0.0

        # --- 5. 模块 D: 待产 (Unassigned) 分布 (3维) ---
        if unassigned:
            # 10. 最近待产距离
            dists_u = [self._node_distance_from_truck(o.node_id) for o in unassigned]
            obs[9] = min(dists_u) / MAX_DIST
            # 11. 待产最紧交期 (允许为负)
            slacks_u = [(o.due_date - self.time) / MAX_TIME for o in unassigned]
            obs[10] = min(slacks_u)
            # 12. 待产订单数占比
            obs[11] = len(unassigned) / MAX_ORDERS
        else:
            obs[9], obs[10], obs[11] = 1.0, 1.0, 0.0

        # --- 6. 模块 E: 全局协同监控 (3维) ---
        # 13. 总未交付订单数占比 (系统总进度)
        obs[12] = len(undelivered) / MAX_ORDERS
        # 14. 全局最紧交期 (系统的最高预警)
        all_slacks = [(o.due_date - self.time) / MAX_TIME for o in undelivered]
        obs[13] = min(all_slacks)
        # 15. 当前目标点关联密度 (当前目标点还有多少订单没搞定：包括打印中和未分配)
        target_tasks = [o for o in undelivered if o.node_id == self.truck.to_node]
        obs[14] = len(target_tasks) / 5.0  # 假设单节点超过5个订单即为高密度

        # 索引 15：当前预约目标距离占比(归一化)
        if self.truck.planned_target is not None:
            obs[15] = min(1.0, self._node_distance_from_truck(self.truck.planned_target) / MAX_DIST)
        else:
            obs[15] = 0.0

        # # ✅ 终极护甲：确保数值不溢出（注意：low 是我们定义的下限向量）
        # obs = np.clip(obs, self.observation_space.low, self.observation_space.high)

        # 保存裁剪前的原始状态，方便诊断归一化是否合理
        raw_obs = obs.copy()

        obs = np.clip(
            raw_obs,
            self.observation_space.low,
            self.observation_space.high
        )

        self.last_raw_obs = raw_obs
        self.last_obs = obs
        self.last_obs_low = raw_obs < self.observation_space.low
        self.last_obs_high = raw_obs > self.observation_space.high

        return obs.astype(np.float32)


def run_fixed_action_policy(action_id, instance_path=None, seed=42, max_steps=200, verbose=False, debug=False):
    """
    用固定动作 action_id 跑完整个 episode
    返回该动作在当前 seed 下的表现
    """
    env = MobileAMEnv(instance_path=instance_path)
    obs, info = env.reset(seed=seed)
    env.debug = debug

    done = False
    step_count = 0
    total_reward = 0.0

    if verbose:
        print(f"\n===== 测试动作 {action_id}: {env.action_meanings.get(action_id, 'unknown')} =====")
        print(f"初始状态: {obs}")

    while not done and step_count < max_steps:
        step_count += 1

        obs, reward, done, truncated, info = env.step(action_id)
        total_reward += reward

        if verbose:
            print(
                f"step={step_count:02d}, "
                f"reward={reward:.2f}, "
                f"tardiness={info['tardiness']:.2f}, "
                f"time={info['time']:.1f}, "
                f"done={done}"
            )

    result = {
        "action_id": action_id,
        "action_name": env.action_meanings.get(action_id, "unknown"),
        "seed": seed,
        "done": done,
        "steps": step_count,
        "total_reward": total_reward,
        "final_tardiness": info["tardiness"],
        "final_time": info["time"],
    }

    return result


def benchmark_all_actions(instance_path=None, seed=42, max_steps=200, verbose=False, debug=False):
    """
    在同一个 seed 下，依次测试所有动作
    """
    results = []

    for action_id in range(6):
        result = run_fixed_action_policy(
            action_id=action_id,
            instance_path=instance_path,
            seed=seed,
            max_steps=max_steps,
            verbose=verbose,
            debug=debug
        )
        results.append(result)

    # 按最终总延误从小到大排序（越小越好）
    results = sorted(results, key=lambda x: x["final_tardiness"])

    print("\n================= 单种子动作对比结果 =================")
    print(f"{'action':<6} {'name':<24} {'done':<6} {'steps':<6} {'reward':<12} {'tardiness':<12} {'final_time':<12}")
    for r in results:
        print(
            f"{r['action_id']:<6} "
            f"{r['action_name']:<24} "
            f"{str(r['done']):<6} "
            f"{r['steps']:<6} "
            f"{r['total_reward']:<12.2f} "
            f"{r['final_tardiness']:<12.2f} "
            f"{r['final_time']:<12.1f}"
        )

    best = results[0]
    print("\n最优动作（按 final_tardiness 最小）:")
    print(best)

    return results


# # ==========================================
# # 本地简单测试运行代码
# # ==========================================
# if __name__ == "__main__":
#     print("初始化移动增材制造环境...")
#     env = MobileAMEnv()
#     obs, info = env.reset()
#
#     print(f"初始状态向量: {obs}")
#
#     done = False
#     step_count = 0
#     total_reward = 0
#
#     while not done and step_count < 50:
#         step_count += 1
#         print(f"\n--- 第 {step_count} 步决策 ---")
#
#         action = 0
#
#         obs, reward, done, truncated, info = env.step(action)
#         total_reward += reward
#
#         print(f"获得奖励: {reward:.2f}, 当前总延误: {info['tardiness']:.2f}, 当前时间: {info['time']:.1f}")
#
#     print(f"\n测试结束! 总步数: {step_count}, 最终系统时间: {env.time:.1f}")

if __name__ == "__main__":
    print("初始化移动增材制造环境...")

    current_dir = os.path.dirname(os.path.abspath(__file__))
    json_path = os.path.join(current_dir, "instance", "problem_instance_10cust_30parts.json")
    print(f"正在从以下路径加载算例: {json_path}")

    # 实例化时传入路径
    # env = MobileAMEnv(instance_path=json_path)
    one_result = run_fixed_action_policy(action_id=0, instance_path=json_path, seed=42, max_steps=200, verbose=True, debug=False)
    all_results = benchmark_all_actions(instance_path=json_path,seed=42, max_steps=200, verbose=False, debug=False)
