from .ecm import EquivalentCircuitModel
from .thermal import ThermalModel
from .aging import AgingModel
import numpy as np

class Battery:
    def __init__(self, initial_soc, nominal_capacity, dt, ecm_params, thermal_params, aging_params):
        # 电池基本参数
        self.nominal_capacity = nominal_capacity # 标称容量 (Ah)
        self.dt = dt # 模拟时间步长 (s)

        # 电池状态
        self.soc = initial_soc # 当前荷电状态 (0-1)
        self.soh = aging_params.get('initial_soh', 1.0) # 当前健康状态 (0-1)
        # 从热模型参数获取初始温度，确保是浮点数
        self.temperature = float(thermal_params.get('ambient_temperature', 298.15)) # 当前温度 (K)
        self._current = 0.0 # 当前电流 (A) - 内部状态，由充电策略设置
        self.terminal_voltage = 0.0 # 当前端电压 (V)

        # 集成各个模型
        # 实例化 ECM 模型
        self.ecm = EquivalentCircuitModel(
            R0=ecm_params['R0'],
            R1=ecm_params['R1'],
            C1=ecm_params['C1'],
            Z_W_params=ecm_params.get('Z_W_params'), # Warburg 参数可能为 None
            ocv_soc_curve_data=ecm_params['ocv_soc_curve_data']
        )
        # 实例化热模型
        self.thermal = ThermalModel(
            ambient_temperature=thermal_params.get('ambient_temperature', 298.15),
            convection_coefficient=thermal_params['convection_coefficient'],
            heat_capacity=thermal_params['heat_capacity']
        )
        # 实例化老化模型，需要传入额定容量
        self.aging = AgingModel(
            nominal_capacity=self.nominal_capacity,
            initial_soh=aging_params.get('initial_soh', 1.0),
            capacity_loss_params=aging_params.get('capacity_loss_params', None)
        )

        # 初始化温度状态在热模型中
        self.thermal.battery_temperature = self.temperature

    def set_current(self, current):
        # 设置电池当前电流 (充电为正，放电为负)
        self._current = current

    def update(self):
        # 更新电池状态 (在一个时间步长 dt 内)

        # 1. 更新老化状态 (SoH 和实际容量)
        # 在更新老化前，先获取当前的温度和电流
        current_temperature = self.thermal.get_temperature()
        self.aging.update_aging_state(self._current, current_temperature, self.dt)
        self.soh = self.aging.get_soh()
        actual_capacity = self.aging.get_actual_capacity() # 获取更新后的实际容量

        # 2. 更新 SoC
        # dSoC/dt = -I / (Capacity * 3600)
        # SoC(t+dt) = SoC(t) + dt * (-I(t)) / (Actual_Capacity * 3600)
        # 防止实际容量为零
        if actual_capacity > 0:
             self.soc += self.dt * (-self._current) / (actual_capacity * 3600.0)
        # 确保 SoC 在合理范围内 (例如 0 到 1)
        self.soc = max(0.0, min(1.0, self.soc))

        # 3. 计算并更新温度
        # 在计算热量前，需要获取当前的等效内阻和 dOCV/dT
        # 等效内阻 (简化处理): 使用 ECM 的 R0 + R1。更精确应考虑老化对内阻的影响。
        # 如果 AgingModel 提供了内阻增加量，应加到这里。
        equivalent_internal_resistance = self.ecm.R0 + self.ecm.R1
        # Placeholder: 获取 dOCV/dT。这需要 OCV 随温度变化的数据或模型。
        # 如果 ECM 模型没有提供这个，这里需要一个估算或查找表。
        ocv_temperature_coefficient = 0.0 # 示例值，需要根据论文或数据获取

        heat_generation = self.thermal.calculate_heat_generation(
            self._current,
            equivalent_internal_resistance, # 传递等效内阻
            ocv_temperature_coefficient # 传递 dOCV/dT
        )
        heat_dissipation = self.thermal.calculate_heat_dissipation()

        # 更新热模型内部的温度状态
        self.thermal.update_temperature(heat_generation, heat_dissipation, self.dt)
        # 更新电池对象的温度状态，与热模型同步
        self.temperature = self.thermal.get_temperature()


        # 4. 计算当前时间步的端电压
        # 在计算电压时，使用当前更新后的状态（SoC, 温度等）
        self.terminal_voltage = self.ecm.calculate_terminal_voltage(
            self._current,
            self.soc, # 使用更新后的 SoC
            self.dt
        )

        # 返回更新后的主要状态
        return {
            'time': None, # 时间在 Charger 中管理
            'soc': self.soc,
            'soh': self.soh,
            'temperature': self.temperature,
            'terminal_voltage': self.terminal_voltage,
            'current': self._current # 返回当前电流，方便外部检查
        }

    def get_state(self):
        # 获取电池当前状态
        return {
            'soc': self.soc,
            'soh': self.soh,
            'temperature': self.temperature,
            'terminal_voltage': self.terminal_voltage,
            'current': self._current
        } 