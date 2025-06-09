# main.py - 项目入口点，用于运行模拟和优化

import numpy as np
import os
from datetime import datetime
import json
import logging

# 导入项目中定义的类
from battery_model.battery import Battery
# from charging_strategy.msccctcv import MSCCCTCVStrategy
# from charging_strategy.charger import Charger
# from optimization.qoga import QOGA
from optimization.objective_functions import (
    calculate_charging_time,
    calculate_soh_degradation,
    calculate_energy_loss,
    calculate_temperature_rise
)
# from visualization.plotter import Plotter
# from visualization.interactive_plotter import InteractivePlotter
from utils.data_exporter import DataExporter
from utils.parameter_optimizer import ParameterOptimizer

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def main():
    logging.info("程序启动")

    # 创建输出目录
    output_dir = "output"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # 创建时间戳目录
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    result_dir = os.path.join(output_dir, timestamp)
    os.makedirs(result_dir)
    logging.info(f"输出目录已创建: {result_dir}")

    # 创建数据导出器
    exporter = DataExporter(result_dir)

    # --- 准备实验数据 ---
    # 这里应该加载实际的实验数据，例如从data/experimental_data.json读取
    experimental_data_path = os.path.join('data', 'experimental_data.json')
    if not os.path.exists(experimental_data_path):
         logging.error(f"实验数据文件未找到: {experimental_data_path}")
         print(f"错误：实验数据文件未找到: {experimental_data_path}")
         return
    
    try:
        with open(experimental_data_path, 'r', encoding='utf-8') as f:
            experimental_data = json.load(f)
        logging.info(f"实验数据加载成功: {experimental_data_path}")
    except Exception as e:
        logging.error(f"加载实验数据失败: {experimental_data_path}", exc_info=True)
        print(f"错误：加载实验数据失败: {e}")
        return

    # --- 准备电池模型初始参数 ---
    # 这应该是一个直接包含电池模型参数的字典，例如从data/initial_params.json的battery_params部分读取
    initial_params_path = os.path.join('data', 'initial_params.json')
    if not os.path.exists(initial_params_path):
        logging.warning(f"初始参数文件未找到: {initial_params_path}，将使用硬编码的默认电池参数部分。")
        # 如果文件不存在，使用硬编码的默认电池参数部分
        battery_model_initial_params = {
             "R0": 0.01, "R1": 0.005, "C1": 100.0, "convection_coefficient": 5.0, "heat_capacity": 1000.0,
             # 添加其他默认的电池参数，如果需要，例如 nominal_capacity, dt, ocv_soc_curve_data等
             'nominal_capacity': 2.6,
             'dt': 1.0,
             'ecm': {'Z_W_params': None, 'ocv_soc_curve_data': {'soc': np.linspace(0, 1, 10), 'ocv': np.linspace(3.0, 4.2, 10)}},
             'thermal': {'ambient_temperature': 298.15},
             'aging': {'initial_soh': 1.0, 'capacity_loss_params': {'A': 1e-5, 'Ea': 50000.0}}

        }
    else:
        try:
            with open(initial_params_path, 'r', encoding='utf-8') as f:
                full_initial_params = json.load(f)
                # 提取 battery_params 部分
                battery_model_initial_params = full_initial_params.get('battery_params', {})
                # 检查是否成功提取电池参数
                required_battery_keys = ['R0', 'R1', 'C1', 'convection_coefficient', 'heat_capacity']
                if not all(key in battery_model_initial_params for key in required_battery_keys):
                    missing = [key for key in required_battery_keys if key not in battery_model_initial_params]
                    logging.error(f"从 {initial_params_path} 加载的参数缺少必要的电池模型参数: {missing}，将使用硬编码的默认值。")
                    # 如果文件存在但缺少必要的电池参数，也使用硬编码的默认值
                    battery_model_initial_params = {
                         "R0": 0.01, "R1": 0.005, "C1": 100.0, "convection_coefficient": 5.0, "heat_capacity": 1000.0,
                         'nominal_capacity': 2.6,
                         'dt': 1.0,
                         'ecm': {'Z_W_params': None, 'ocv_soc_curve_data': {'soc': np.linspace(0, 1, 10), 'ocv': np.linspace(3.0, 4.2, 10)}},
                         'thermal': {'ambient_temperature': 298.15},
                         'aging': {'initial_soh': 1.0, 'capacity_loss_params': {'A': 1e-5, 'Ea': 50000.0}}
                    }

                logging.info(f"电池模型初始参数加载成功: {initial_params_path}")
                logging.debug(f"加载的电池模型初始参数: {battery_model_initial_params}")

        except Exception as e:
            logging.error(f"加载初始参数文件失败: {initial_params_path}，将使用硬编码的默认电池参数部分。", exc_info=True)
            print(f"警告：加载初始参数文件失败: {e}，将使用硬编码的默认电池参数部分。")
            # 加载失败时也使用硬编码的默认电池参数部分
            battery_model_initial_params = {
                 "R0": 0.01, "R1": 0.005, "C1": 100.0, "convection_coefficient": 5.0, "heat_capacity": 1000.0,
                 'nominal_capacity': 2.6,
                 'dt': 1.0,
                 'ecm': {'Z_W_params': None, 'ocv_soc_curve_data': {'soc': np.linspace(0, 1, 10), 'ocv': np.linspace(3.0, 4.2, 10)}},
                 'thermal': {'ambient_temperature': 298.15},
                 'aging': {'initial_soh': 1.0, 'capacity_loss_params': {'A': 1e-5, 'Ea': 50000.0}}
            }


    # --- 实例化 ParameterOptimizer ---
    optimizer = ParameterOptimizer(result_dir)

    # --- 运行电池模型参数优化 ---
    logging.info("开始运行电池模型参数优化")
    # 将 battery_model_initial_params 直接传递给 optimize_battery_parameters
    optimization_result = optimizer.optimize_battery_parameters(
        experimental_data=experimental_data,
        initial_params=battery_model_initial_params
    )
    logging.info("电池模型参数优化过程结束")

    # --- 处理优化结果 ---
    if optimization_result and optimization_result.get('success'):
        optimized_params = optimization_result.get('optimized_params')
        print("\n电池模型参数优化成功！")
        print("优化后的参数:")
        for param, value in optimized_params.items():
            print(f"  {param}: {value}")
        
        # 新增：保存优化结果到output目录
        optimizer._save_optimization_results(
            results=optimized_params,
            objective_value=optimization_result.get('optimization_result').fun if optimization_result.get('optimization_result') else None,
            param_type='battery',
            filename_suffix='main'
        )

    else:
        print("\n电池模型参数优化未成功完成。")
        if optimization_result and optimization_result.get('message'):
            print("原因:", optimization_result.get('message'))
        logging.error("电池模型参数优化未成功。")

    logging.info(f"优化结果已保存到: {result_dir} 目录下的相关文件")
    print(f"\n详细日志和结果已保存到 {result_dir} 目录。")

if __name__ == "__main__":
    main() 