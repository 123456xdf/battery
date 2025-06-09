import numpy as np
from scipy.optimize import minimize
import json
import os
from datetime import datetime
from battery_model.battery import Battery
import logging
import random
from optimization.qoga import QOGA
import copy
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass

@dataclass
class OptimizationResult:
    """优化结果数据类"""
    success: bool
    message: str
    optimized_params: Optional[Dict[str, Any]] = None
    objective_value: Optional[float] = None
    optimization_result: Optional[Any] = None
    metrics: Optional[Dict[str, float]] = None
    simulated_voltage: Optional[List[float]] = None
    simulated_temperature: Optional[List[float]] = None

class ParameterOptimizer:
    def __init__(self, output_dir: str = "output"):
        """
        初始化参数优化器
        :param output_dir: 输出目录
        """
        self.output_dir = output_dir
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.INFO)
        
        # 创建输出目录
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
            
        # 创建文件处理器
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = os.path.join(output_dir, f"optimization_{timestamp}.log")
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.INFO)
        
        # 创建控制台处理器
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        
        # 设置日志格式
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)
        
        # 添加处理器
        self.logger.addHandler(file_handler)
        self.logger.addHandler(console_handler)
        
        # 设置默认参数范围
        self.default_param_ranges = {
            'R0': (0.001, 0.1),  # 欧姆内阻范围
            'R1': (0.001, 0.1),  # 极化内阻范围
            'C1': (100, 10000),  # 极化电容范围
            'convection_coefficient': (5, 50),  # 对流换热系数范围
            'heat_capacity': (500, 5000)  # 热容范围
        }
        
        self.logger.info(f"ParameterOptimizer 初始化，输出目录: {output_dir}")

    def validate_battery_params(self, params: Dict[str, Any]) -> Tuple[bool, str]:
        """
        验证电池参数
        :param params: 电池参数字典
        :return: (是否有效, 错误信息)
        """
        required_params = ['R0', 'R1', 'C1', 'convection_coefficient', 'heat_capacity']
        if not all(param in params for param in required_params):
            missing = [param for param in required_params if param not in params]
            return False, f"缺少必要参数: {missing}"
            
        # 验证参数值范围和类型
        for param in required_params:
            value = params[param]
            if not isinstance(value, (int, float)):
                return False, f"参数 {param} 必须是数值类型"
                
            min_val, max_val = self.default_param_ranges[param]
            if not min_val <= value <= max_val:
                return False, f"参数 {param} 的值必须在 [{min_val}, {max_val}] 范围内"
                
        return True, ""

    def validate_experimental_data(self, data: Dict[str, Any]) -> Tuple[bool, str]:
        """
        验证实验数据
        :param data: 实验数据字典
        :return: (是否有效, 错误信息)
        """
        required_fields = ['time', 'voltage', 'current', 'soc', 'temperature']
        if not all(field in data for field in required_fields):
            missing = [field for field in required_fields if field not in data]
            return False, f"缺少必要字段: {missing}"
            
        # 验证数据长度
        data_lengths = {field: len(data[field]) for field in required_fields}
        if len(set(data_lengths.values())) != 1:
            return False, f"数据长度不一致: {data_lengths}"
            
        # 验证数据类型和范围
        for field in required_fields:
            values = data[field]
            if not all(isinstance(x, (int, float)) for x in values):
                return False, f"字段 {field} 包含非数值数据"
                
            if field == 'soc' and not all(0 <= x <= 1 for x in values):
                return False, f"字段 {field} 的值必须在 [0,1] 范围内"
                
            if field == 'temperature' and not all(x > 0 for x in values):
                return False, f"字段 {field} 的值必须为正数"
                
        return True, ""

    def optimize_battery_parameters(self, experimental_data: Dict[str, Any], initial_params: Optional[Dict[str, Any]] = None, callback=None) -> OptimizationResult:
        """
        优化电池模型参数
        :param experimental_data: 实验数据
        :param initial_params: 初始参数
        :return: 优化结果
        """
        self.logger.info("开始优化电池模型参数")

        # 验证实验数据
        is_valid, error_msg = self.validate_experimental_data(experimental_data)
        if not is_valid:
            return OptimizationResult(success=False, message=f"实验数据验证失败: {error_msg}")

        # 验证初始参数
        if initial_params:
            is_valid, error_msg = self.validate_battery_params(initial_params.get('battery_params', {}))
            if not is_valid:
                return OptimizationResult(success=False, message=f"初始参数验证失败: {error_msg}")
            # 补充 OCV 曲线
            if 'ocv_soc_curve_data' not in initial_params['battery_params'] or not initial_params['battery_params']['ocv_soc_curve_data']:
                initial_params['battery_params']['ocv_soc_curve_data'] = [
                    [0.0, 3.0], [0.1, 3.2], [0.2, 3.3], [0.3, 3.4], [0.4, 3.5],
                    [0.5, 3.6], [0.6, 3.7], [0.7, 3.8], [0.8, 3.9], [0.9, 4.0], [1.0, 4.2]
                ]
        else:
            # 使用默认参数
            initial_params = {
                'battery_params': {
                    'R0': 0.01,
                    'R1': 0.01,
                    'C1': 1000,
                    'convection_coefficient': 10,
                    'heat_capacity': 1000,
                    'ocv_soc_curve_data': [
                        [0.0, 3.0], [0.1, 3.2], [0.2, 3.3], [0.3, 3.4], [0.4, 3.5],
                        [0.5, 3.6], [0.6, 3.7], [0.7, 3.8], [0.8, 3.9], [0.9, 4.0], [1.0, 4.2]
                    ]
                }
            }

        # 提取初始值 x0
        battery_params = initial_params['battery_params']
        x0 = [
            battery_params['R0'],
            battery_params['R1'],
            battery_params['C1'],
            battery_params['convection_coefficient'],
            battery_params['heat_capacity']
        ]

        # 定义参数边界
        bounds = [
            self.default_param_ranges['R0'],
            self.default_param_ranges['R1'],
            self.default_param_ranges['C1'],
            self.default_param_ranges['convection_coefficient'],
            self.default_param_ranges['heat_capacity']
        ]

        self.logger.info(f"电池参数优化初始值: {x0}")
        self.logger.info(f"电池参数优化边界: {bounds}")

        try:
            # 使用 SciPy 的 minimize 函数进行优化
            result = minimize(
                self._battery_objective_function,
                x0,
                args=(experimental_data, battery_params),
                method='L-BFGS-B',
                bounds=bounds,
                options={
                    'maxiter': 200,
                    'disp': True,
                    'ftol': 1e-8,  # 更严格的收敛条件
                    'gtol': 1e-8,
                    'maxls': 50  # 增加线搜索次数
                }
            )

            if result.success:
                optimized_params_list = result.x
                optimized_params = {#设置优化过程中各种参数的字典：
                    'battery_params': {
                        'R0': optimized_params_list[0],
                        'R1': optimized_params_list[1],
                        'C1': optimized_params_list[2],
                        'convection_coefficient': optimized_params_list[3],
                        'heat_capacity': optimized_params_list[4]
                    },
                    'algorithm_params': initial_params.get('algorithm_params', {}),
                    'objective_weights': initial_params.get('objective_weights', {})
                }
                
                self.logger.info(f"电池参数优化成功，优化后的参数: {optimized_params}")
                # 在 optimize_battery_parameters 结尾处，运行一次仿真，获取模拟电压和温度
                sim_voltage, sim_temp = self.simulate_battery(experimental_data, optimized_params['battery_params'])
                return OptimizationResult(
                    success=True,
                    message="优化成功",
                    optimized_params=optimized_params,
                    objective_value=result.fun,
                    optimization_result=result,
                    simulated_voltage=sim_voltage,
                    simulated_temperature=sim_temp,
                    metrics=self._calculate_validation_metrics(optimized_params, experimental_data)
                )
            else:
                self.logger.warning(f"电池参数优化未成功完成: {result.message}")
                return OptimizationResult(
                    success=False,
                    message=result.message,
                    optimization_result=result
                )

        except Exception as e:
            self.logger.error("电池参数优化过程发生错误", exc_info=True)
            return OptimizationResult(
                success=False,
                message=f"优化过程异常: {str(e)}"
            )

    def optimize_algorithm_parameters(self, test_cases: List[Dict], initial_params: Dict) -> Optional[Dict]:#表示函数返回的是一个字典（如果优化成功），如果优化未成功则返回 None。
        """
        使用QOGA优化算法参数
        :param test_cases: 测试用例列表
        :param initial_params: 初始参数
        :return: 优化后的参数
        """
        self.logger.info("开始优化算法参数")

        try:
            # 设置QOGA参数范围
            strategy_param_bounds = {
                'cc_currents': (0.8, 1.8),  # 恒流充电电流范围
                'ct_temps': (305, 312),  # 恒温充电温度范围
                'cv_voltage': (4.1, 4.25),  # 恒压充电电压范围
                'cc_to_ct_temp_threshold': (305, 312),  # CC到CT转换温度阈值
                'ct_to_cv_soc_threshold': (0.75, 0.95),  # CT到CV转换SOC阈值
                'cv_cut_off_current_rate': (0.01, 0.07)  # CV截止电流率
            }
            if 'charger_params' not in initial_params:
                initial_params['charger_params'] = {
                    "dt": 1.0,  # 时间步长（秒）
                    "target_soc": 1.0,  # 目标充电状态（0~1）
                    "max_time_s": 10000.0  # 最大充电时间（秒）
                }

            # 初始化QOGA优化器
            qoga = QOGA(
                population_size=initial_params.get('population_size', 100),  # 增加种群大小
                n_generations=initial_params.get('n_generations', 50),  # 增加迭代次数
                mutation_rate=initial_params.get('mutation_rate', 0.1),
                crossover_rate=initial_params.get('crossover_rate', 0.8),
                strategy_param_bounds=strategy_param_bounds,
                early_stopping_patience=10,  # 添加早停机制
                diversity_threshold=0.1, # 添加多样性保护
                battery_params = initial_params['battery_params'],
                charger_params = initial_params['charger_params'] # 确保这里已存在
            )
            
            # 运行优化
            optimized_params = qoga.run_optimization(test_cases, initial_params)
            
            if optimized_params:
                self.logger.info("算法参数优化成功")
                return optimized_params
            else:
                self.logger.warning("算法参数优化未找到有效解")
                return None
                
        except Exception as e:
            self.logger.error("算法参数优化过程发生错误", exc_info=True)
            return None

    def validate_parameters(self, params: Dict, validation_data: Dict) -> OptimizationResult:
        """
        验证参数
        :param params: 待验证的参数
        :param validation_data: 验证数据
        :return: 验证结果
        """
        self.logger.info("开始验证参数")
        
        try:
            # 验证参数
            is_valid, error_msg = self.validate_battery_params(params.get('battery_params', {}))
            if not is_valid:
                return OptimizationResult(success=False, message=f"参数验证失败: {error_msg}")
                
            # 验证数据
            is_valid, error_msg = self.validate_experimental_data(validation_data)
            if not is_valid:
                return OptimizationResult(success=False, message=f"验证数据无效: {error_msg}")
                
            # 计算验证指标
            metrics = self._calculate_validation_metrics(params, validation_data)
            
            return OptimizationResult(
                success=True,
                message='验证成功',
                metrics=metrics
            )
            
        except Exception as e:
            self.logger.error("参数验证过程发生错误", exc_info=True)
            return OptimizationResult(
                success=False,
                message=f"验证过程异常: {str(e)}"
            )

    def _calculate_validation_metrics(self, params: Dict, validation_data: Dict) -> Dict[str, float]:
        """
        计算验证指标
        :param params: 参数
        :param validation_data: 验证数据
        :return: 验证指标字典
        """
        try:
            # 创建电池模型实例
            ecm_params = params['battery_params'].copy()
            if 'ocv_soc_curve_data' not in ecm_params:
                ecm_params['ocv_soc_curve_data'] = [
                    [0.0, 3.0], [0.1, 3.2], [0.2, 3.3], [0.3, 3.4], [0.4, 3.5],
                    [0.5, 3.6], [0.6, 3.7], [0.7, 3.8], [0.8, 3.9], [0.9, 4.0], [1.0, 4.2]
                ]
            battery = Battery(
                initial_soc=validation_data['soc'][0],
                nominal_capacity=params.get('nominal_capacity', 2.6),#从字典 params 中获取 'nominal_capacity' 键的值。如果 params 中不存在该键，它会返回默认值 2.6
                dt=params.get('dt', 1.0),
                ecm_params=ecm_params,
                thermal_params={
                    'ambient_temperature': validation_data['temperature'][0],
                    'convection_coefficient': ecm_params['convection_coefficient'],
                    'heat_capacity': ecm_params['heat_capacity']
                },
                aging_params={'initial_soh': 1.0}#直接生成一个键值对
            )
            
            # 模拟充电过程
            simulated_voltage = []
            sim_dt = battery.dt
            experimental_time = validation_data['time']
            experimental_voltage = validation_data['voltage']
            experimental_current = validation_data['current']
            
            current_time = 0
            total_experimental_time = experimental_time[-1]
            
            while current_time <= total_experimental_time + sim_dt/2:
                closest_time_index = np.argmin(np.abs(np.array(experimental_time) - current_time))
                input_current = experimental_current[closest_time_index]
                
                battery.set_current(input_current)
                battery.update()
                simulated_voltage.append(battery.terminal_voltage)
                
                current_time += sim_dt
                
            # 计算验证指标
            min_len = min(len(simulated_voltage), len(experimental_voltage))
            simulated_voltage_matched = np.array(simulated_voltage[:min_len])
            experimental_voltage_matched = np.array(experimental_voltage[:min_len])
            
            # RMSE
            rmse = np.sqrt(np.mean((simulated_voltage_matched - experimental_voltage_matched)**2))
            
            # MAE
            mae = np.mean(np.abs(simulated_voltage_matched - experimental_voltage_matched))
            
            # 最大误差
            max_error = np.max(np.abs(simulated_voltage_matched - experimental_voltage_matched))
            
            # 相关系数
            correlation = np.corrcoef(simulated_voltage_matched, experimental_voltage_matched)[0,1]
            
            return {
                'rmse': rmse,
                'mae': mae,
                'max_error': max_error,
                'correlation': correlation
            }
            
        except Exception as e:
            self.logger.error("计算验证指标时发生错误", exc_info=True)
            raise

    def _save_optimization_results(self, results, objective_value=None, param_type=None, simulated_voltage=None, simulated_temperature=None, filename_suffix=None):
        """
        保存优化或验证结果到JSON文件，包含模拟数据
        """
        output = {
            'timestamp': datetime.now().strftime('%Y%m%d_%H%M%S'),
            'param_type': param_type,
            'results': results,
            'objective_value': objective_value
        }
        if simulated_voltage is not None:
            output['simulated_voltage'] = simulated_voltage
        if simulated_temperature is not None:
            output['simulated_temperature'] = simulated_temperature
        filename = f"{param_type or 'result'}_{output['timestamp']}{('_' + filename_suffix) if filename_suffix else ''}.json"
        filepath = os.path.join(self.output_dir, filename)
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(output, f, ensure_ascii=False, indent=4)
        return filepath

    def _battery_objective_function(self, params: np.ndarray, experimental_data: Dict[str, Any], fixed_battery_params: Dict[str, Any]) -> float:
        """
        电池参数优化的目标函数
        :param params: 当前待评估的电池参数
        :param experimental_data: 实验数据
        :param fixed_battery_params: 固定参数
        :return: 误差值
        """
        try:
            # 构建完整的电池参数
            ocv_curve = fixed_battery_params.get('ocv_soc_curve_data')
            if not ocv_curve:
                ocv_curve = [
                    [0.0, 3.0], [0.1, 3.2], [0.2, 3.3], [0.3, 3.4], [0.4, 3.5],
                    [0.5, 3.6], [0.6, 3.7], [0.7, 3.8], [0.8, 3.9], [0.9, 4.0], [1.0, 4.2]
                ]
            current_battery_full_params = {
                'nominal_capacity': fixed_battery_params.get('nominal_capacity', 2.6),
                'dt': fixed_battery_params.get('dt', 1.0),
                'ecm': {
                    'R0': params[0],
                    'R1': params[1],
                    'C1': params[2],
                    'Z_W_params': fixed_battery_params.get('Z_W_params'),
                    'ocv_soc_curve_data': ocv_curve
                },
                'thermal': {
                    'ambient_temperature': fixed_battery_params.get('ambient_temperature', 298.15),
                    'convection_coefficient': params[3],
                    'heat_capacity': params[4]
                },
                'aging': fixed_battery_params.get('aging', {'initial_soh': 1.0, 'capacity_loss_params': {'A': 1e-5, 'Ea': 50000.0}}),
                'initial_soh': fixed_battery_params.get('initial_soh', 1.0)
            }

            # 创建电池模型实例
            initial_soc = experimental_data['soc'][0]
            initial_temp = experimental_data['temperature'][0]

            if initial_soc is None or initial_temp is None:
                self.logger.error("实验数据中缺少初始SOC或温度")
                return float('inf')

            battery = Battery(
                initial_soc=initial_soc,
                nominal_capacity=current_battery_full_params['nominal_capacity'],
                dt=current_battery_full_params['dt'],
                ecm_params=current_battery_full_params['ecm'],
                thermal_params={
                    'ambient_temperature': initial_temp,
                    'convection_coefficient': params[3],
                    'heat_capacity': params[4]
                },
                aging_params=current_battery_full_params['aging']
            )

            # 模拟充电过程
            simulated_voltage = []
            simulated_temp = []  # 添加温度模拟
            sim_dt = battery.dt
            experimental_time = experimental_data['time']
            experimental_voltage = experimental_data['voltage']
            experimental_current = experimental_data['current']
            experimental_temp = experimental_data['temperature']  # 添加实验温度数据

            current_time = 0
            total_experimental_time = experimental_time[-1]

            while current_time <= total_experimental_time + sim_dt/2:
                closest_time_index = np.argmin(np.abs(np.array(experimental_time) - current_time))
                input_current = experimental_current[closest_time_index]

                battery.set_current(input_current)
                battery.update()
                simulated_voltage.append(battery.terminal_voltage)
                simulated_temp.append(battery.temperature)  # 记录温度

                current_time += sim_dt

            # 计算多目标误差
            min_len = min(len(simulated_voltage), len(experimental_voltage))
            if min_len == 0:
                self.logger.warning("模拟或实验数据长度为零")
                return float('inf')

            simulated_voltage_matched = np.array(simulated_voltage[:min_len])
            experimental_voltage_matched = np.array(experimental_voltage[:min_len])
            simulated_temp_matched = np.array(simulated_temp[:min_len])
            experimental_temp_matched = np.array(experimental_temp[:min_len])

            # 计算电压误差
            voltage_error = np.sqrt(np.mean((simulated_voltage_matched - experimental_voltage_matched)**2))
            # 计算温度误差
            temp_error = np.sqrt(np.mean((simulated_temp_matched - experimental_temp_matched)**2))
            # 加权组合误差
            total_error = 0.7 * voltage_error + 0.3 * temp_error  # 电压误差权重更大
            self.logger.debug(f"参数评估误差 - 电压RMSE: {voltage_error}, 温度RMSE: {temp_error}, 总误差: {total_error}")
            return total_error
        except Exception as e:
            self.logger.error("电池参数优化目标函数执行错误", exc_info=True)
            return float('inf')

    def simulate_battery(self, experimental_data, battery_params):
        """
        根据实验数据和电池参数，逐步仿真，返回模拟电压和温度序列
        """
        from battery_model.battery import Battery
        # 假设实验数据有 time, current, soc, temperature
        time_seq = experimental_data['time']
        current_seq = experimental_data['current']
        soc_seq = experimental_data['soc']
        temperature_seq = experimental_data['temperature']
        dt = time_seq[1] - time_seq[0] if len(time_seq) > 1 else 1.0
        initial_soc = soc_seq[0]
        initial_temp = temperature_seq[0]
        # 构造参数
        ecm_params = battery_params.copy()
        if 'ocv_soc_curve_data' not in ecm_params:
            ecm_params['ocv_soc_curve_data'] = [
                [0.0, 3.0], [0.1, 3.2], [0.2, 3.3], [0.3, 3.4], [0.4, 3.5],
                [0.5, 3.6], [0.6, 3.7], [0.7, 3.8], [0.8, 3.9], [0.9, 4.0], [1.0, 4.2]
            ]
        thermal_params = {
            'ambient_temperature': initial_temp,
            'convection_coefficient': battery_params.get('convection_coefficient', 10),
            'heat_capacity': battery_params.get('heat_capacity', 1000)
        }
        aging_params = {'initial_soh': 1.0}
        battery = Battery(
            initial_soc=initial_soc,
            nominal_capacity=1.0,
            dt=dt,
            ecm_params=ecm_params,
            thermal_params=thermal_params,
            aging_params=aging_params
        )
        simulated_voltage = []
        simulated_temperature = []
        for i in range(len(time_seq)):
            battery.set_current(current_seq[i])
            state = battery.update()
            simulated_voltage.append(state['terminal_voltage'])
            simulated_temperature.append(state['temperature'])
        return simulated_voltage, simulated_temperature

# End of file 