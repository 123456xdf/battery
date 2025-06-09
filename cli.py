import argparse
import json
import os
from datetime import datetime
import logging
from typing import Dict, Any, List
import matplotlib.pyplot as plt
import numpy as np

from utils.parameter_optimizer import ParameterOptimizer, OptimizationResult
from utils.data_exporter import DataExporter
from visualization.paper_figs import plot_all_paper_figs

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def validate_json_file(filepath: str, required_fields: list = None) -> Dict[str, Any]:
    """
    验证并加载JSON文件
    :param filepath: JSON文件路径
    :param required_fields: 必需字段列表
    :return: 加载的数据字典
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"文件不存在: {filepath}")
    if not os.access(filepath, os.R_OK):
        raise PermissionError(f"没有读取权限: {filepath}")
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        if required_fields:
            missing_fields = [field for field in required_fields if field not in data]
            if missing_fields:
                raise ValueError(f"数据缺少必要字段: {missing_fields}")
        return data
    except json.JSONDecodeError:
        raise ValueError(f"JSON文件格式错误: {filepath}")
    except Exception as e:
        raise RuntimeError(f"加载文件时发生错误: {str(e)}")

def load_experimental_data(filepath: str) -> Dict[str, Any]:
    """
    加载并验证实验数据
    :param filepath: 实验数据文件路径
    :return: 验证后的实验数据字典
    """
    required_fields = ['time', 'voltage', 'current', 'soc', 'temperature']
    data = validate_json_file(filepath, required_fields)
    
    # 验证数据长度
    data_lengths = {field: len(data[field]) for field in required_fields}
    if len(set(data_lengths.values())) != 1:
        raise ValueError(f"数据长度不一致: {data_lengths}")
    #set(data_lengths.values())：获取data_lengths字典中所有值（即各字段的长度）的集合。集合会自动去重。
    #len(set(...)) != 1：如果长度集合的元素个数不为1，说明各字段的长度不一致。
    #raise ValueError(...)：如果数据长度不一致，抛出ValueError异常，并显示错误信息。
        
    # 验证数据类型
    for field in required_fields:
        if not all(isinstance(x, (int, float)) for x in data[field]):
            raise ValueError(f"字段 {field} 包含非数值数据")
            
    return data

def ensure_output_directory(output_dir: str) -> str:
    """
    确保输出目录存在并可写
    :param output_dir: 输出目录路径
    :return: 时间戳子目录路径
    """
    try:
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        if not os.access(output_dir, os.W_OK):
            raise PermissionError(f"没有写入权限: {output_dir}")
            
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        result_dir = os.path.join(output_dir, timestamp)
        os.makedirs(result_dir)
        return result_dir
        
    except Exception as e:
        raise RuntimeError(f"创建输出目录失败: {str(e)}")

def plot_results(experimental_data: Dict[str, Any], simulated_voltage: List[float], 
                simulated_temperature: List[float], output_dir: str, filename_suffix: str = ''):
    """Plot comparison between experimental and simulated data and generate detailed single cycle plot"""
    try:
        experimental_time = experimental_data['time']
        experimental_voltage = experimental_data['voltage']
        experimental_temperature = experimental_data['temperature']
        experimental_current = experimental_data['current']
        experimental_soc = experimental_data['soc']

        # Ensure data lengths match
        min_len = min(len(experimental_time), len(simulated_voltage), 
                     len(simulated_temperature), len(experimental_current))
        if min_len == 0:
            logger.warning("Data length is zero, cannot generate plots")
            return

        time_points = experimental_time[:min_len]
        experimental_voltage_matched = experimental_voltage[:min_len]
        simulated_voltage_matched = simulated_voltage[:min_len]
        experimental_temperature_matched = experimental_temperature[:min_len]
        simulated_temperature_matched = simulated_temperature[:min_len]
        experimental_current_matched = experimental_current[:min_len]
        experimental_soc_matched = experimental_soc[:min_len]

        # Plot voltage comparison
        plt.figure(figsize=(12, 6))
        plt.plot(time_points, experimental_voltage_matched, 'r-', label='Experimental Voltage (V)')
        plt.plot(time_points, simulated_voltage_matched, 'k--', label='Simulated Voltage (V)')
        plt.xlabel('Time (s)')
        plt.ylabel('Voltage (V)')
        plt.title('Voltage Comparison Over Time')
        plt.legend()
        plt.grid(True)
        voltage_plot_path = os.path.join(output_dir, f"voltage_comparison{filename_suffix}.png")
        plt.savefig(voltage_plot_path, dpi=300, bbox_inches='tight')
        plt.close()

        # Plot temperature comparison
        plt.figure(figsize=(12, 6))
        plt.plot(time_points, experimental_temperature_matched, 'r-', label='Experimental Temperature (K)')
        plt.plot(time_points, simulated_temperature_matched, 'k--', label='Simulated Temperature (K)')
        plt.xlabel('Time (s)')
        plt.ylabel('Temperature (K)')
        plt.title('Temperature Comparison Over Time')
        plt.legend()
        plt.grid(True)
        temperature_plot_path = os.path.join(output_dir, f"temperature_comparison{filename_suffix}.png")
        plt.savefig(temperature_plot_path, dpi=300, bbox_inches='tight')
        plt.close()

        # Plot current
        plt.figure(figsize=(12, 6))
        plt.plot(time_points, experimental_current_matched, 'b-', label='Experimental Current (A)')
        plt.xlabel('Time (s)')
        plt.ylabel('Current (A)')
        plt.title('Current Over Time')
        plt.legend()
        plt.grid(True)
        current_plot_path = os.path.join(output_dir, f"current{filename_suffix}.png")
        plt.savefig(current_plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        # --- Add detailed single cycle plot ---        
        fig, axes = plt.subplots(4, 1, figsize=(10, 12), sharex=True)

        # Plot Voltage
        axes[0].plot(time_points, experimental_voltage_matched, label='Experimental Voltage (V)')
        axes[0].plot(time_points, simulated_voltage_matched, label='Simulated Voltage (V)', linestyle='--')
        axes[0].set_ylabel('Voltage (V)')
        axes[0].legend()
        axes[0].grid(True)

        # Plot Current
        axes[1].plot(time_points, experimental_current_matched, label='Experimental Current (A)', color='red')
        axes[1].set_ylabel('Current (A)')
        axes[1].legend()
        axes[1].grid(True)

        # Plot Capacity (calculated from experimental current)
        # Assuming initial capacity is available or starting from 0
        # This requires integrating the current over time (I*dt). 
        # We'll use the time difference between consecutive points as dt.
        capacity = [0] # Start with 0 capacity or initial capacity if available
        # Need to ensure time_points has more than one element for diff
        if len(time_points) > 1:
            dts = np.diff(time_points)
            current_steps = experimental_current_matched[:-1] # Use current values corresponding to the start of each time step
            capacity_increase = current_steps * dts / 3600 # Convert As to Ah (divide by 3600)
            capacity.extend(np.cumsum(capacity_increase))
        # Ensure capacity has the same length as time_points
        capacity_matched = capacity[:min_len]
        
        axes[2].plot(time_points, capacity_matched, label='Experimental Capacity (Ah)', color='green')
        # If you have simulated capacity data, plot it here as well.
        # For now, we only have simulated V and T, so we plot experimental capacity.
        axes[2].set_ylabel('Capacity (Ah)')
        axes[2].legend()
        axes[2].grid(True)

        # Plot Temperature
        axes[3].plot(time_points, experimental_temperature_matched, label='Experimental Temperature (K)', color='purple')
        axes[3].plot(time_points, simulated_temperature_matched, label='Simulated Temperature (K)', linestyle='--', color='orange')
        axes[3].set_ylabel('Temperature (K)')
        axes[3].set_xlabel('Time (s)')
        axes[3].legend()
        axes[3].grid(True)

        plt.suptitle('Detailed Single Cycle Comparison', y=0.95)
        plt.tight_layout(rect=[0, 0.03, 1, 0.92]) # Adjust layout to prevent title overlap
        single_cycle_plot_path = os.path.join(output_dir, f"detailed_single_cycle{filename_suffix}.png")
        plt.savefig(single_cycle_plot_path, dpi=300, bbox_inches='tight')
        plt.close(fig) # Close the figure to free memory

        logger.info(f"Plots have been saved to directory: {output_dir}")

    except Exception as e:
        logger.error(f"Error generating plots: {str(e)}", exc_info=True)

def generate_and_plot_synthetic_cycle_life(output_dir: str, filename_suffix: str = ''):
    """Generates synthetic cycle life data and plots capacity vs. cycle number"""
    logger.info("Generating synthetic cycle life plot...")

    # Synthetic data mimicking capacity degradation for different strategies
    cycle_numbers = np.arange(1, 801, 10) # Cycles up to 800
    initial_capacity = 2.6 # Ah

    strategies = {
        '0.5C CC-CV': initial_capacity * (1 - 0.0002 * cycle_numbers - 1e-6 * cycle_numbers**2), # Slower degradation
        '1C CC-CV': initial_capacity * (1 - 0.0003 * cycle_numbers - 2e-6 * cycle_numbers**2),   # Moderate degradation
        '1.25C CC-CV': initial_capacity * (1 - 0.0004 * cycle_numbers - 4e-6 * cycle_numbers**2), # Faster degradation
        'O-MCC': initial_capacity * (1 - 0.00015 * cycle_numbers - 0.8e-6 * cycle_numbers**2), # Better degradation
        'PC': initial_capacity * (1 - 0.00022 * cycle_numbers - 1.5e-6 * cycle_numbers**2), # Slightly better than 0.5C
        'SRC': initial_capacity * (1 - 0.00035 * cycle_numbers - 3e-6 * cycle_numbers**2), # Worse degradation
        'SS-CCCTCV': initial_capacity * (1 - 0.00018 * cycle_numbers - 0.9e-6 * cycle_numbers**2), # Good degradation
        'DS-CCCTCV': initial_capacity * (1 - 0.00016 * cycle_numbers - 0.85e-6 * cycle_numbers**2), # Slightly better
        'TS-CCCTCV': initial_capacity * (1 - 0.00014 * cycle_numbers - 0.8e-6 * cycle_numbers**2)  # Best degradation (synthetic)
    }

    plt.figure(figsize=(12, 8))

    for strategy, capacity_data in strategies.items():
        # Ensure capacity does not go below a reasonable limit, e.g., 1.8 Ah based on Fig 10
        capacity_data[capacity_data < 1.8] = np.nan # Use NaN to stop the line when capacity drops too low
        plt.plot(cycle_numbers, capacity_data, label=strategy)

    plt.xlabel('Cycle Number')
    plt.ylabel('Capacity (Ah)')
    plt.title('Synthetic Cell Cycle Life Comparison')
    plt.legend()
    plt.grid(True)

    cycle_life_plot_path = os.path.join(output_dir, f"synthetic_cycle_life{filename_suffix}.png")
    plt.savefig(cycle_life_plot_path, dpi=300, bbox_inches='tight')
    plt.close()

    logger.info(f"Synthetic cycle life plot saved to: {cycle_life_plot_path}")

def main():
    parser = argparse.ArgumentParser(description='电池充电策略优化工具')
    
    # 创建子命令
    subparsers = parser.add_subparsers(dest='command', help='可用命令')
    
    # 优化电池参数命令
    optimize_battery_parser = subparsers.add_parser('optimize-battery',
        help='优化电池模型参数')
    optimize_battery_parser.add_argument('--data', required=True,
        help='实验数据文件路径')
    optimize_battery_parser.add_argument('--output', default='output',
        help='输出目录')
    optimize_battery_parser.add_argument('--initial-params',
        help='初始参数文件路径')
    
    # 优化算法参数命令
    optimize_algo_parser = subparsers.add_parser('optimize-algo',
        help='优化算法参数')
    optimize_algo_parser.add_argument('--test-cases', required=True,
        help='测试用例文件路径')
    optimize_algo_parser.add_argument('--output', default='output',
        help='输出目录')
    optimize_algo_parser.add_argument('--initial-params',
        help='初始参数文件路径')
    
    # 验证参数命令
    validate_parser = subparsers.add_parser('validate',
        help='验证参数')
    validate_parser.add_argument('--params', required=True,
        help='参数文件路径')
    validate_parser.add_argument('--validation-data', required=True,
        help='验证数据文件路径')
    validate_parser.add_argument('--output', default='output',
        help='输出目录')
    
    # 添加绘图命令
    plot_parser = subparsers.add_parser('plot',
        help='绘制实验数据与模拟结果对比图')
    plot_parser.add_argument('--data', required=True,
        help='原始实验数据文件路径')
    plot_parser.add_argument('--sim-results', required=True,
        help='包含模拟结果的优化或验证结果文件路径')
    plot_parser.add_argument('--output', default='output',
        help='图表输出目录')

    # Add plot cycle life command
    plot_cycle_life_parser = subparsers.add_parser('plot-cycle-life',
        help='Generate and plot synthetic cell cycle life comparison')
    plot_cycle_life_parser.add_argument('--output', default='output',
        help='Output directory for the cycle life plot')
    
    # 添加论文风格画图命令
    plot_paper_figs_parser = subparsers.add_parser('plot-paper-figs',
        help='生成所有论文风格的模拟图片（无数据也可用）')
    plot_paper_figs_parser.add_argument('--output', default='output_plots/paper_figs',
        help='输出目录')
    
    args = parser.parse_args()
    
    try:
        if args.command in ['plot', 'plot-cycle-life', 'plot-paper-figs']:
            output_dir = args.output
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)
        else:
            output_dir = ensure_output_directory(args.output)
        optimizer = ParameterOptimizer(output_dir)
    
        if args.command == 'optimize-battery':
            experimental_data = load_experimental_data(args.data)
            initial_params = None
            if args.initial_params:
                initial_params = validate_json_file(args.initial_params)
            logger.info("开始优化电池参数...")
            result = optimizer.optimize_battery_parameters(
                experimental_data,
                initial_params
            )
            if not result.success:
                logger.error(f"优化失败: {result.message}")
                return 1
            logger.info("\n优化完成！")
            logger.info("优化后的参数：")
            for param, value in result.optimized_params['battery_params'].items():
                if param != 'ocv_soc_curve_data':
                    logger.info(f"  {param}: {value}")
            try:
                optimizer._save_optimization_results(
                    results=result.optimized_params,
                    objective_value=result.optimization_result.fun if result.optimization_result else None,
                    param_type='battery',
                    simulated_voltage=result.simulated_voltage,
                    simulated_temperature=result.simulated_temperature,
                    filename_suffix='cli'
                )
                logger.info(f"\n优化结果已保存到目录: {output_dir}")
            except Exception as e:
                logger.warning(f"保存优化结果时发生错误: {str(e)}")
        
        elif args.command == 'optimize-algo':
            test_cases_data = validate_json_file(args.test_cases)
            initial_params_data = validate_json_file(args.initial_params)
            logger.info("开始优化算法参数...")
            test_cases_list = test_cases_data['test_cases'] if isinstance(test_cases_data, dict) and 'test_cases' in test_cases_data else test_cases_data
            result = optimizer.optimize_algorithm_parameters(test_cases_list, initial_params_data)
            if result:
                logger.info("\n优化完成！")
                logger.info("优化后的参数：")
                logger.info(json.dumps(result, indent=4, ensure_ascii=False))
                try:
                    optimizer._save_optimization_results(
                        results=result,
                        objective_value=None,
                        param_type='algorithm',
                        filename_suffix='cli'
                    )
                    logger.info(f"\n优化结果已保存到目录: {output_dir}")
                except Exception as e:
                    logger.warning(f"保存优化结果时发生错误: {str(e)}")
            else:
                logger.error("优化失败，未能找到有效的参数组合")
        
        elif args.command == 'validate':
            params = validate_json_file(args.params)
            validation_data = load_experimental_data(args.validation_data)
            logger.info("开始验证参数...")
            result = optimizer.validate_parameters(params, validation_data)
            if not result['success']:
                logger.error(f"验证失败: {result['message']}")
                return 1
            logger.info("\n验证完成！")
            logger.info("验证指标：")
            for metric, value in result['metrics'].items():
                logger.info(f"  {metric}: {value}")
            try:
                optimizer._save_optimization_results(
                    results=result['metrics'],
                    objective_value=None,
                    param_type='validation',
                    filename_suffix='cli'
                )
                logger.info(f"\n验证结果已保存到目录: {output_dir}")
            except Exception as e:
                logger.warning(f"保存验证结果时发生错误: {str(e)}")

        elif args.command == 'plot':
            experimental_data = load_experimental_data(args.data)
            sim_results_data = validate_json_file(args.sim_results)
            simulated_voltage = sim_results_data.get('simulated_voltage')
            simulated_temperature = sim_results_data.get('simulated_temperature')
            if not simulated_voltage or not simulated_temperature:
                results_content = sim_results_data.get('results')
                if isinstance(results_content, dict):
                    simulated_voltage = results_content.get('simulated_voltage')
                    simulated_temperature = results_content.get('simulated_temperature')
            if not simulated_voltage or not simulated_temperature:
                logger.error(f"错误: 指定的结果文件 {args.sim_results} 不包含模拟电压或温度数据。")
                return 1
            logger.info("开始生成图表...")
            plot_results(experimental_data, simulated_voltage, simulated_temperature, output_dir, filename_suffix='_comparison')
            logger.info(f"图表已保存到目录: {output_dir}")

        elif args.command == 'plot-cycle-life':
            generate_and_plot_synthetic_cycle_life(output_dir)

        elif args.command == 'plot-paper-figs':
            plot_all_paper_figs()
            return 0

        else:
            logger.error("错误: 未知命令")
            return 1
        return 0
    except FileNotFoundError as e:
        logger.error(f"文件错误: {str(e)}")
        return 1
    except PermissionError as e:
        logger.error(f"权限错误: {str(e)}")
        return 1
    except ValueError as e:
        logger.error(f"数据验证错误: {str(e)}")
        return 1
    except RuntimeError as e:
        logger.error(f"运行时错误: {str(e)}")
        return 1
    except Exception as e:
        logger.error(f"执行过程中发生未预期的错误: {str(e)}", exc_info=True)
        return 1

if __name__ == '__main__':
    exit(main()) 