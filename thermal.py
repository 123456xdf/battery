class ThermalModel:
    def __init__(self, ambient_temperature, convection_coefficient, heat_capacity):
        # 热模型参数
        self.ambient_temperature = ambient_temperature  # 环境温度 (K)
        # 论文中使用 K 表示散热系数 (包含 h*A)
        self.convection_coefficient = convection_coefficient # 对流换热系数 K (W/K)
        self.heat_capacity = heat_capacity              # 电池热容 C (J/K)

        # 电池温度状态变量
        self.battery_temperature = ambient_temperature # 初始温度设为环境温度 (K)

    def calculate_heat_generation(self, current, internal_resistance, ocv_temperature_coefficient):
        # 计算产热 (基于论文公式 4 和 5)
        # Q_gen = Q_irr + Q_rev

        # 不可逆热 (欧姆热): Q_irr = I^2 * R (公式 4)
        # internal_resistance 应该从电池模型中获取当前的等效内阻
        irreversible_heat = current**2 * internal_resistance

        # 可逆热: Q_rev = -I * T * dOCV/dT (公式 5)
        # ocv_temperature_coefficient 是 dOCV/dT，需要根据 OCV 随温度变化的数据或模型获得
        # 这里的 self.battery_temperature 是当前的电池温度 T
        reversible_heat = -current * self.battery_temperature * ocv_temperature_coefficient

        total_heat_generation = irreversible_heat + reversible_heat
        return total_heat_generation

    def calculate_heat_dissipation(self):
        # 计算散热 (基于论文公式 6，使用系数 K)
        # Q_loss = K * (T - T_amb)
        heat_dissipation = self.convection_coefficient * (self.battery_temperature - self.ambient_temperature)
        return heat_dissipation

    def update_temperature(self, heat_generation, heat_dissipation, dt):
        # 更新电池温度 (基于热平衡方程 公式 7)
        # C * dT/dt = Q_gen - Q_loss
        # T(t+dt) = T(t) + dt * (Q_gen - Q_loss) / C
        # 防止热容为零
        if self.heat_capacity != 0:
            temperature_change = dt * (heat_generation - heat_dissipation) / self.heat_capacity
            self.battery_temperature += temperature_change
        # 返回更新后的温度 (转换为摄氏度可能更直观，但模型内部使用开尔文)
        return self.battery_temperature

    def get_temperature(self):
        # 获取当前电池温度 (K)
        return self.battery_temperature

    # 可能需要的方法：根据温度更新热模型参数 (如果参数随温度变化)
    # def update_parameters(self, temperature):
    #     pass 