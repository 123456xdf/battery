import numpy as np
# 可能需要导入 objective_functions 和 battery/charger 相关的类
from .objective_functions import calculate_charging_time, calculate_soh_degradation, calculate_energy_loss, calculate_temperature_rise
from battery_model.battery import Battery
from charging_strategy.msccctcv import MSCCCTCVStrategy
from charging_strategy.charger import Charger

class QOGA:
    def __init__(self, population_size, n_generations, strategy_param_bounds, battery_params, charger_params,mutation_rate=0.1,crossover_rate=0.9, early_stopping_patience=None, diversity_threshold=None):
        # QOGA 参数
        self.population_size = population_size # 种群大小
        self.n_generations = n_generations # 迭代代数
        self.strategy_param_bounds = strategy_param_bounds # 充电策略参数的取值范围，例如: {'CC1_current': (0.5, 5.0), ...}

        # 电池和充电器参数 (用于评估每个个体)
        self.battery_params = battery_params
        self.charger_params = charger_params

        # 目标函数 (引用外部函数)
        self.objective_functions = [
            calculate_charging_time,
            calculate_soh_degradation,
            calculate_energy_loss,
            calculate_temperature_rise
        ]

    def initialize_population(self):
        # 初始化种群
        # 每个个体是一组充电策略参数，确保参数在定义范围内
        population = []
        for _ in range(self.population_size):
            individual_params = {}
            for param, bounds in self.strategy_param_bounds.items():
                # 随机生成参数值 (在指定范围内均匀分布)
                individual_params[param] = np.random.uniform(bounds[0], bounds[1])
            population.append(individual_params)
        return population

    def evaluate_population(self, population):
        # 评估种群中每个个体的适应度
        # 适应度是根据目标函数计算的一组目标值
        fitness_scores = [] # 存储每个个体的目标值列表

        for individual_params in population:
            # 1. 使用这组参数创建充电策略
            # 需要将 battery_params 中的 nominal_capacity 传递给策略，以便计算电流值
            strategy_params_with_capacity = individual_params.copy()
            strategy_params_with_capacity['nominal_capacity'] = self.battery_params.get('nominal_capacity', 2.6)
            strategy = MSCCCTCVStrategy(strategy_params_with_capacity)

            # 2. 创建电池和充电器实例
            # 确保电池参数被正确传递，尤其是时间步长 dt
            battery = Battery(
                initial_soc=self.battery_params.get('initial_soc', 0.0),
                nominal_capacity=self.battery_params.get('nominal_capacity', 2.6),
                dt=self.battery_params.get('dt', 1.0), # 确保 dt 从 battery_params 获取
                ecm_params=self.battery_params['ecm'],
                thermal_params=self.battery_params['thermal'],
                aging_params=self.battery_params['aging']
            )
            charger = Charger(
                battery=battery,
                strategy=strategy,
                dt=self.charger_params.get('dt', 1.0) # 确保 dt 从 charger_params 获取
            )

            # 3. 运行充电模拟
            charging_data = [] # 初始化模拟数据列表
            try:
                # 模拟直到充电完成或达到最大时间
                # 将最大模拟时间从 charger_params 传递给 start_charging 方法
                charger.start_charging(
                    target_soc=self.charger_params.get('target_soc', 1.0),
                    max_time_s=self.charger_params.get('max_time_s', 10000.0) # 示例最大时间
                )
                # 4. 获取模拟数据
                charging_data = charger.get_charging_data()

                # 5. 计算目标值
                if charging_data:
                    objectives = [
                        self.objective_functions[0](charging_data), # 充电时间
                        self.objective_functions[1](charging_data, initial_soh=self.battery_params['aging'].get('initial_soh', 1.0)), # SoH 退化
                        # 能量损耗函数需要额定容量
                        self.objective_functions[2](charging_data), # 能量损耗
                        self.objective_functions[3](charging_data, ambient_temperature=self.battery_params['thermal'].get('ambient_temperature', 298.15)) # 温度升高
                    ]
                else:
                    # 如果模拟失败或无数据，赋予极差的目标值 (最大化目标)
                    objectives = [float('inf')] * len(self.objective_functions)

            except Exception as e:
                print(f"Error during simulation for individual {individual_params}: {e}")
                # 模拟出错时，赋予极差的目标值
                objectives = [float('inf')] * len(self.objective_functions)

            fitness_scores.append(objectives)

        return np.array(fitness_scores) # 返回 numpy 数组方便后续计算

    def non_dominated_sort(self, population, fitness_scores):
        # 非劣排序 (实现 NSGA-II 的排序部分)
        # 这是一个复杂的过程，需要比较每一对个体，确定支配关系。
        # 返回一个列表，每个元素是一个非劣前沿的个体索引列表。
        
        # Placeholder: 简单地返回所有个体都在同一个前沿
        # 实际需要根据目标值实现非劣排序算法
        # 例如，可以参考 NSGA-II 的非劣排序实现。
        
        n_individuals = len(population)
        # 初始化支配关系和被支配计数
        dominates_list = [[] for _ in range(n_individuals)] # dominates_list[i] 存储个体 i 支配的个体索引列表
        dominated_by_count = [0] * n_individuals # dominated_by_count[i] 存储支配个体 i 的个体数量
        fronts = [[]] # 存储非劣前沿，fronts[k] 是第 k 个非劣前沿的个体索引列表

        # 计算支配关系和被支配计数
        for i in range(n_individuals):
            for j in range(n_individuals):
                if i == j: continue

                # 检查个体 i 是否支配个体 j (所有目标都小于等于 j，且至少一个目标严格小于 j)
                # 假设所有目标都是要最小化
                is_i_dominating_j = True
                is_i_strictly_dominating_j = False
                for obj_idx in range(fitness_scores.shape[1]):
                    if fitness_scores[i, obj_idx] > fitness_scores[j, obj_idx]:
                        is_i_dominating_j = False
                        break
                    if fitness_scores[i, obj_idx] < fitness_scores[j, obj_idx]:
                        is_i_strictly_dominating_j = True

                if is_i_dominating_j and is_i_strictly_dominating_j:
                    dominates_list[i].append(j)
                elif is_i_dominating_j and not is_i_strictly_dominating_j: # i 和 j 互不支配 (相同目标值)
                     pass # 或者根据多目标 GA 的其他规则处理相等情况
                else: # i 不支配 j，检查 j 是否支配 i
                    # 检查个体 j 是否支配个体 i (所有目标都小于等于 i，且至少一个目标严格小于 i)
                    is_j_dominating_i = True
                    is_j_strictly_dominating_i = False
                    for obj_idx in range(fitness_scores.shape[1]):
                         if fitness_scores[j, obj_idx] > fitness_scores[i, obj_idx]:
                             is_j_dominating_i = False
                             break
                         if fitness_scores[j, obj_idx] < fitness_scores[i, obj_idx]:
                              is_j_strictly_dominating_i = True

                    if is_j_dominating_i and is_j_strictly_dominating_i:
                        dominated_by_count[i] += 1

            # 如果个体 i 不被任何个体支配，则它属于第一个非劣前沿 (Rank 0)
            if dominated_by_count[i] == 0:
                fronts[0].append(i)

        # 构建后续的非劣前沿
        k = 0 # 当前处理的前沿索引
        while fronts[k]:
            next_front = []
            for i in fronts[k]:
                for j in dominates_list[i]:
                    dominated_by_count[j] -= 1
                    if dominated_by_count[j] == 0:
                        next_front.append(j)
            k += 1
            if next_front:
                 fronts.append(next_front)

        return fronts

    def calculate_crowding_distance(self, front_indices, fitness_scores):
        # 计算一个前沿中个体的拥挤距离
        # 拥挤距离用于在同一前沿的个体之间进行区分，优先选择距离大的（在目标空间中分布更均匀）
        # 对于边界个体，拥挤距离为无穷大。
        # 对于中间个体，拥挤距离是其在每个目标上左右相邻个体目标值差值的总和。

        n_individuals_in_front = len(front_indices)
        if n_individuals_in_front == 0:
            return []

        # 初始化拥挤距离
        distances = [0.0] * n_individuals_in_front

        # 如果前沿只有一个个体，拥挤距离设为无穷大 (或一个很大的值)
        if n_individuals_in_front == 1:
            return [float('inf')]

        n_objectives = fitness_scores.shape[1]

        # 对每个目标计算拥挤距离
        for obj_idx in range(n_objectives):
            # 获取当前前沿个体在该目标上的值，以及对应的原始索引
            # 例如: [(obj_value1, original_index1), (obj_value2, original_index2), ...]
            front_obj_values = [(fitness_scores[idx, obj_idx], idx) for idx in front_indices]

            # 按当前目标值升序排序
            sorted_front_obj_values = sorted(front_obj_values)

            # 边界个体的拥挤距离设为无穷大
            distances[front_indices.index(sorted_front_obj_values[0][1])] = float('inf')
            distances[front_indices.index(sorted_front_obj_values[-1][1])] = float('inf')

            # 计算中间个体的拥挤距离
            # 归一化因子：当前目标上的最大值和最小值之差
            obj_min = sorted_front_obj_values[0][0]
            obj_max = sorted_front_obj_values[-1][0]
            
            # 防止最大值和最小值相等导致除以零
            if obj_max - obj_min > 1e-9:
                for i in range(1, n_individuals_in_front - 1):
                    # 获取当前个体、前一个和后一个的原始索引
                    current_original_idx = sorted_front_obj_values[i][1]
                    previous_original_idx = sorted_front_obj_values[i-1][1]
                    next_original_idx = sorted_front_obj_values[i+1][1]

                    # 获取前一个和后一个在该目标上的值
                    previous_obj_value = fitness_scores[previous_original_idx, obj_idx]
                    next_obj_value = fitness_scores[next_original_idx, obj_idx]

                    # 累加到当前个体的拥挤距离 (需要找到当前个体在原始 front_indices 列表中的位置)
                    current_index_in_front_indices = front_indices.index(current_original_idx)
                    distances[current_index_in_front_indices] += (next_obj_value - previous_obj_value) / (obj_max - obj_min)

        return distances


    def select(self, population, fitness_scores):
        # 选择操作 (基于多目标非劣排序和拥挤距离)
        # 将父代和子代合并，进行非劣排序，然后根据排序和拥挤距离选择下一代种群
        # 在 run_optimization 中，我们会将父代和子代合并，然后调用此方法进行选择。
        # 这里的输入 population 和 fitness_scores 应该是合并后的种群和对应的适应度。

        # 1. 非劣排序
        fronts = self.non_dominated_sort(population, fitness_scores)

        next_population = []
        next_fitness_scores = []
        current_population_size = 0

        # 2. 逐个非劣前沿选择个体，直到下一代种群大小达到 population_size
        for front_indices in fronts:
            if current_population_size + len(front_indices) <= self.population_size:
                for idx in front_indices:
                    next_population.append(population[idx])
                    next_fitness_scores.append(fitness_scores[idx])
                    current_population_size += 1
            else:
                # 如果当前前沿的部分个体可以加入，需要根据拥挤距离选择
                remaining_space = self.population_size - current_population_size
                # 计算当前前沿个体的拥挤距离
                distances = self.calculate_crowding_distance(front_indices, fitness_scores)
                # 将个体索引和拥挤距离配对，并按拥挤距离降序排序
                sorted_front_with_distance = sorted(zip(distances, front_indices), reverse=True)
                # 选择拥挤距离最大的前 remaining_space 个个体
                for i in range(remaining_space):
                    original_idx = sorted_front_with_distance[i][1]
                    next_population.append(population[original_idx])
                    next_fitness_scores.append(fitness_scores[original_idx])
                    current_population_size += 1
                break # 种群已满，停止选择

        return next_population, np.array(next_fitness_scores)

    def crossover(self, parents):
        # 交叉操作
        # 将父代个体的参数进行组合生成子代
        # 实现交叉逻辑，例如单点交叉、两点交叉或均匀交叉
        offspring = []
        crossover_rate = 0.9 # 交叉概率 (示例值)

        # 确保父代数量是偶数，如果不是，最后一个父代复制一份
        parents_copy = parents[:] # 创建副本以避免修改原始列表
        if len(parents_copy) % 2 != 0:
            parents_copy.append(parents_copy[-1])

        for i in range(0, len(parents_copy), 2):
            parent1_params = parents_copy[i]
            parent2_params = parents_copy[i+1]

            child1_params = parent1_params.copy()
            child2_params = parent2_params.copy()

            if np.random.rand() < crossover_rate:
                # 示例: 均匀交叉 (对于字典参数)
                for param in self.strategy_param_bounds.keys():
                    if np.random.rand() < 0.5: # 以 0.5 的概率交换参数值
                        child1_params[param], child2_params[param] = child2_params[param], child1_params[param]

            offspring.append(child1_params)
            offspring.append(child2_params)

        return offspring

    def mutate(self, offspring):
        # 变异操作
        # 随机改变子代个体的参数
        # 实现变异逻辑，例如对参数进行随机小范围改动
        mutated_offspring = []
        mutation_rate = 0.1 # 变异概率 (示例值)
        mutation_strength = 0.05 # 变异强度 (控制变异范围，示例值)

        for individual_params in offspring:
            mutated_params = individual_params.copy()
            # 对参数进行变异 (根据变异概率和变异强度)
            for param, bounds in self.strategy_param_bounds.items():
                if np.random.rand() < mutation_rate:
                    # 在参数范围内进行随机变异
                    delta = np.random.normal(0, mutation_strength * (bounds[1] - bounds[0])) # 正态分布变异
                    mutated_params[param] += delta
                    # 确保变异后的参数仍在范围内
                    mutated_params[param] = max(bounds[0], min(bounds[1], mutated_params[param]))
            mutated_offspring.append(mutated_params)

        return mutated_offspring

    def run_optimization(self, test_cases=None, initial_params=None):
        # 运行 QOGA 优化过程
        try:
            # 初始化种群
            population = self.initialize_population()

            # 迭代多代
            for generation in range(self.n_generations):
                print(f"Generation {generation+1}/{self.n_generations}") # 示例输出

                # 评估父代种群
                parent_fitness = self.evaluate_population(population)

                # 选择父代进行交叉和变异 (通常基于非劣排序和拥挤距离选择)
                # 这里简化处理，直接将当前种群作为父代进行交叉和变异，
                # 更标准的做法是在父代和子代合并后再进行选择。
                # selected_parents, _ = self.select(population, parent_fitness) # 如果需要更复杂的父代选择

                # 交叉生成子代
                offspring_population = self.crossover(population)

                # 变异子代
                mutated_offspring = self.mutate(offspring_population)

                # 评估子代种群
                offspring_fitness = self.evaluate_population(mutated_offspring)

                # 合并父代和子代种群及其适应度
                combined_population = population + mutated_offspring
                combined_fitness = np.vstack((parent_fitness, offspring_fitness))

                # 从合并后的种群中选择下一代种群 (基于非劣排序和拥挤距离)
                population, fitness_scores = self.select(combined_population, combined_fitness)

            # 优化完成后，评估最终种群，获取非劣解集
            # final_fitness_scores = self.evaluate_population(population) # 已经有最终种群的适应度 fitness_scores
            non_dominated_solutions_indices = self.non_dominated_sort(population, fitness_scores)[0] # 获取第一个非劣前沿的索引
            non_dominated_solutions = [population[i] for i in non_dominated_solutions_indices]
            non_dominated_fitness = [fitness_scores[i] for i in non_dominated_solutions_indices]

            print("优化完成。") # 示例输出

            # 返回非劣解集和对应的目标值
            return non_dominated_solutions, non_dominated_fitness 
        except Exception as e:
            print(f"优化过程发生异常: {e}")
            return [], float('inf') 