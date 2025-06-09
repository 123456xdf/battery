import numpy as np

def calculate_charging_time(charging_data):
    """
    计算充电时间（从开始到结束的总时间）
    charging_data: 包含时间序列的字典列表
    """
    if not charging_data:
        return float('inf')
    return charging_data[-1]['time'] - charging_data[0]['time']

def calculate_soh_degradation(charging_data, initial_soh=1.0):
    """
    计算充电过程中的SoH退化
    charging_data: 包含SoH序列的字典列表
    initial_soh: 初始SoH值
    """
    if not charging_data:
        return float('inf')
    # SoH退化 = 初始SoH - 最终SoH
    final_soh = charging_data[-1]['soh']
    return initial_soh - final_soh

def calculate_energy_loss(charging_data):
    """
    计算充电过程中的能量损耗
    能量损耗 = 输入能量 - 存储能量
    charging_data: 包含电压、电流和时间序列的字典列表
    """
    if not charging_data:
        return float('inf')
    
    total_input_energy = 0.0
    total_stored_energy = 0.0
    
    for i in range(1, len(charging_data)):
        dt = charging_data[i]['time'] - charging_data[i-1]['time']
        current = charging_data[i]['current']
        voltage = charging_data[i]['terminal_voltage']
        
        # 输入能量 = V * I * dt
        input_energy = voltage * current * dt
        total_input_energy += input_energy
        
        # 存储能量 = 电压变化 * 电流 * dt
        voltage_change = charging_data[i]['terminal_voltage'] - charging_data[i-1]['terminal_voltage']
        stored_energy = voltage_change * current * dt
        total_stored_energy += stored_energy
    
    # 能量损耗 = 输入能量 - 存储能量
    return total_input_energy - total_stored_energy

def calculate_temperature_rise(charging_data, ambient_temperature=298.15):
    """
    计算充电过程中的最大温度升高
    charging_data: 包含温度序列的字典列表
    ambient_temperature: 环境温度（K）
    """
    if not charging_data:
        return float('inf')
    
    # 计算最大温度升高 = 最高温度 - 环境温度
    max_temperature = max(data['temperature'] for data in charging_data)
    return max_temperature - ambient_temperature

# 这些函数将作为 QOGA 的目标函数，QOGA 将尝试最小化它们。 