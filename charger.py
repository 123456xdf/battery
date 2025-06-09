import time # 可能需要用于模拟实时过程或记录时间
from .msccctcv import MSCCCTCVStrategy
# from ..battery_model.battery import Battery # 导入 Battery 类
import numpy as np

class Charger:
    def __init__(self, battery, strategy, dt=1.0):
        """
        初始化充电器
        battery: 电池对象
        strategy: 充电策略对象
        dt: 时间步长 (s)
        """
        self.battery = battery
        self.strategy = strategy
        self.dt = dt
        self.charging_data = [] # 存储充电过程数据
        self.current_time = 0.0 # 当前充电时间
        self.is_charging = False # 充电状态标志

    def start_charging(self, target_soc=1.0, max_time_s=3600.0):
        """
        开始充电过程
        target_soc: 目标SoC
        max_time_s: 最大充电时间 (s)
        """
        self.is_charging = True
        self.current_time = 0.0
        self.charging_data = []

        # 记录初始状态
        initial_state = self.battery.get_state()
        initial_state['time'] = self.current_time
        self.charging_data.append(initial_state)

        # 充电循环
        while self.is_charging and self.current_time < max_time_s:
            # 获取当前电池状态
            current_state = self.battery.get_state()
            
            # 检查是否达到目标SoC
            if current_state['soc'] >= target_soc:
                print(f"达到目标SoC: {target_soc}")
                break

            # 获取充电策略的下一步动作
            action = self.strategy.determine_charging_action(current_state)
            
            # 根据控制模式执行相应动作
            if action['control_mode'] == 'finish':
                print("充电策略指示结束充电")
                break
            elif action['control_mode'] == 'CC':
                # 恒流充电
                self.battery.set_current(action['value'])
            elif action['control_mode'] == 'CV':
                # 恒压充电
                target_voltage = action['value']
                current_voltage = current_state['terminal_voltage']
                # 简单的PI控制逻辑（示例）
                voltage_error = target_voltage - current_voltage
                # 根据电压误差调整电流（简化处理）
                current = self.battery._current + 0.1 * voltage_error
                self.battery.set_current(current)
            elif action['control_mode'] == 'CT':
                # 恒温充电
                target_temperature = action['value']
                current_temperature = current_state['temperature']
                # 简单的PI控制逻辑（示例）
                temperature_error = target_temperature - current_temperature
                # 根据温度误差调整电流（简化处理）
                current = self.battery._current + 0.05 * temperature_error
                self.battery.set_current(current)

            # 更新电池状态
            updated_state = self.battery.update()
            
            # 更新时间
            self.current_time += self.dt
            updated_state['time'] = self.current_time
            
            # 记录状态
            self.charging_data.append(updated_state)

        # 充电结束，记录最终状态
        final_state = self.battery.get_state()
        final_state['time'] = self.current_time
        self.charging_data.append(final_state)
        
        self.is_charging = False
        return self.charging_data

    def get_charging_data(self):
        """
        获取充电过程数据
        """
        return self.charging_data

    def stop_charging(self):
        """
        停止充电
        """
        self.is_charging = False 