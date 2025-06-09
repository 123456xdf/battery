import numpy as np
from utils.parameter_optimizer import ParameterOptimizer
import logging

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def generate_test_data():
    """生成测试用的实验数据"""
    # 生成时间序列 (0-3600秒，步长1秒)
    time = np.arange(0, 3600, 1)
    
    # 生成模拟的电压数据 (使用简单的模型)
    soc = np.linspace(0.2, 0.9, len(time))  # SOC从0.2到0.9
    voltage = 3.7 + 0.5 * soc + 0.1 * np.sin(time/100)  # 基础电压加上SOC影响和波动
    
    # 生成电流数据 (恒流充电)
    current = np.ones_like(time) * 1.0  # 1A恒流充电
    
    # 生成温度数据
    temperature = 298.15 + 5 * (1 - np.exp(-time/1000))  # 从环境温度开始，逐渐上升
    
    return {
        'time': time.tolist(),
        'voltage': voltage.tolist(),
        'current': current.tolist(),
        'temperature': temperature.tolist(),
        'soc': soc.tolist()
    }

def main():
    # 创建参数优化器实例
    optimizer = ParameterOptimizer(output_dir="output")
    
    # 生成测试数据
    experimental_data = generate_test_data()
    
    # 设置初始参数
    initial_params = {
        'battery_params': {
            'R0': 0.01,
            'R1': 0.005,
            'C1': 1000.0,
            'convection_coefficient': 0.5,
            'heat_capacity': 1000.0,
            'nominal_capacity': 2.6,
            'dt': 1.0,
            'ecm': {
                'Z_W_params': None,
                'ocv_soc_curve_data': {
                    'soc': np.linspace(0, 1, 10),
                    'ocv': np.linspace(3.0, 4.2, 10)
                }
            },
            'thermal': {
                'ambient_temperature': 298.15
            },
            'aging': {
                'initial_soh': 1.0,
                'capacity_loss_params': {'A': 1e-5, 'Ea': 50000.0}
            }
        },
        'algorithm_params': {
            'population_size': 50,
            'n_generations': 20,
            'mutation_rate': 0.1,
            'crossover_rate': 0.8
        },
        'objective_weights': {
            'time_weight': 0.3,
            'soh_weight': 0.3,
            'energy_weight': 0.2,
            'temp_weight': 0.2
        }
    }
    
    # 运行参数优化
    logging.info("开始运行参数优化...")
    result = optimizer.optimize_battery_parameters(experimental_data, initial_params)
    
    # 输出结果
    if result['success']:
        logging.info("参数优化成功！")
        logging.info(f"优化后的参数: {result['optimized_params']}")
    else:
        logging.error(f"参数优化失败: {result['message']}")

if __name__ == "__main__":
    main() 