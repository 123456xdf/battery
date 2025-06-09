import csv
import json
import os
import numpy as np
from datetime import datetime

class DataExporter:
    def __init__(self, output_dir="output"):
        """
        初始化数据导出器
        output_dir: 输出目录
        """
        self.output_dir = output_dir
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

    def export_charging_data_to_csv(self, charging_data, filename=None):
        """
        将充电数据导出为CSV格式
        charging_data: 充电过程数据列表
        filename: 输出文件名，如果为None则使用时间戳
        """
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"charging_data_{timestamp}.csv"

        filepath = os.path.join(self.output_dir, filename)
        
        # 准备CSV表头
        headers = [
            '时间(s)', 'SoC(%)', 'SoH(%)', '电压(V)', '电流(A)',
            '温度(K)', '功率(W)', '累积能量(J)'
        ]
        
        with open(filepath, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(headers)
            
            for data in charging_data:
                row = [
                    data['time'],
                    data['soc'] * 100,  # 转换为百分比
                    data['soh'] * 100,  # 转换为百分比
                    data['terminal_voltage'],
                    data['current'],
                    data['temperature'],
                    data['terminal_voltage'] * data['current'],  # 功率
                    data.get('cumulative_energy', 0)  # 累积能量
                ]
                writer.writerow(row)
        
        return filepath

    def export_optimization_results_to_csv(self, solutions, fitness_values, filename=None):
        """
        将优化结果导出为CSV格式
        solutions: 非支配解列表
        fitness_values: 对应的目标值列表
        filename: 输出文件名，如果为None则使用时间戳
        """
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"optimization_results_{timestamp}.csv"

        filepath = os.path.join(self.output_dir, filename)
        
        # 准备CSV表头
        headers = [
            '解编号',
            'CC电流倍率', 'CT目标温度(K)', 'CV目标电压(V)',
            'CC到CT温度阈值(K)', 'CT到CV SoC阈值', 'CV截止电流倍率',
            '充电时间(s)', 'SoH退化', '能量损耗(J)', '温度升高(K)'
        ]
        
        with open(filepath, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(headers)
            
            for i, (solution, fitness) in enumerate(zip(solutions, fitness_values)):
                row = [
                    i + 1,
                    solution['cc_currents'],
                    solution['ct_temps'],
                    solution['cv_voltage'],
                    solution['cc_to_ct_temp_threshold'],
                    solution['ct_to_cv_soc_threshold'],
                    solution['cv_cut_off_current_rate'],
                    fitness[0],  # 充电时间
                    fitness[1],  # SoH退化
                    fitness[2],  # 能量损耗
                    fitness[3]   # 温度升高
                ]
                writer.writerow(row)
        
        return filepath

    def export_to_json(self, data, filename=None):
        """
        将数据导出为JSON格式
        data: 要导出的数据
        filename: 输出文件名，如果为None则使用时间戳
        """
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"data_{timestamp}.json"

        filepath = os.path.join(self.output_dir, filename)
        
        # 处理numpy数组
        def numpy_to_list(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, dict):
                return {k: numpy_to_list(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [numpy_to_list(item) for item in obj]
            return obj
        
        json_data = numpy_to_list(data)
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(json_data, f, indent=4, ensure_ascii=False)
        
        return filepath

    def export_statistics(self, charging_data, filename=None):
        """
        导出充电数据的统计信息
        charging_data: 充电过程数据列表
        filename: 输出文件名，如果为None则使用时间戳
        """
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"statistics_{timestamp}.csv"

        filepath = os.path.join(self.output_dir, filename)
        
        # 提取数据
        times = [data['time'] for data in charging_data]
        socs = [data['soc'] * 100 for data in charging_data]
        sohs = [data['soh'] * 100 for data in charging_data]
        voltages = [data['terminal_voltage'] for data in charging_data]
        currents = [data['current'] for data in charging_data]
        temperatures = [data['temperature'] for data in charging_data]
        powers = [v * i for v, i in zip(voltages, currents)]
        
        # 计算统计信息
        stats = {
            '总充电时间(s)': max(times),
            'SoC变化范围(%)': f"{min(socs):.2f} - {max(socs):.2f}",
            'SoH变化范围(%)': f"{min(sohs):.2f} - {max(sohs):.2f}",
            '电压范围(V)': f"{min(voltages):.2f} - {max(voltages):.2f}",
            '电流范围(A)': f"{min(currents):.2f} - {max(currents):.2f}",
            '温度范围(K)': f"{min(temperatures):.2f} - {max(temperatures):.2f}",
            '平均功率(W)': f"{np.mean(powers):.2f}",
            '最大功率(W)': f"{max(powers):.2f}",
            '总能量(J)': f"{sum(p * (t2 - t1) for p, t1, t2 in zip(powers, times[:-1], times[1:])):.2f}"
        }
        
        # 导出为CSV
        with open(filepath, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(['指标', '值'])
            for key, value in stats.items():
                writer.writerow([key, value])
        
        return filepath 