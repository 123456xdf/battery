import numpy as np
# 可能需要科学计算库 Scipy 进行更精确的插值，如果可用的话
# from scipy.interpolate import interp1d

class EquivalentCircuitModel:
    def __init__(self, R0, R1, C1, Z_W_params, ocv_soc_curve_data):
        # ECM 参数
        self.R0 = R0  # 欧姆内阻 (Ω)
        self.R1 = R1  # 极化电阻 (Ω)
        self.C1 = C1  # 极化电容 (F)
        # 确保 C1 不为零，避免除零错误
        self.tau1 = R1 * C1 if C1 != 0 else float('inf') # 时间常数 (s)'inf'表示无穷大


        self.Z_W_params = Z_W_params  # Warburg 阻抗参数 (如果使用)
        # OCV-SoC 曲线数据: [(soc1, ocv1), (soc2, ocv2), ...]
        self.ocv_soc_curve_data = sorted(ocv_soc_curve_data, key=lambda item: item[0]) # 按 SoC 排序

        # 使用 Scipy 的插值函数会更方便和精确，这里先用 numpy 实现简单线性插值
        self._soc_points = np.array([point[0] for point in self.ocv_soc_curve_data])
        self._ocv_points = np.array([point[1] for point in self.ocv_soc_curve_data])
        
        # 如果 scipy 可用，可以使用:
        # self._ocv_interp = interp1d(self._soc_points, self._ocv_points, kind='linear', bounds_error=False, fill_value="extrapolate")

        # RC 电路的状态变量 (极化电压)
        self._V1 = 0.0  # 电压跨越 R1C1 并联 (V)

    def get_ocv(self, soc):
        # 根据 SoC 计算开路电压 (OCV)
        # 使用线性插值或其他方法根据 OCV-SoC 曲线数据获取 OCV
        soc = float(soc)  # 确保soc是浮点数
        soc = max(float(self._soc_points[0]), min(float(self._soc_points[-1]), soc)) # 将 SoC 限制在数据范围内

        # 使用 numpy 进行线性插值
        return np.interp(soc, self._soc_points, self._ocv_points)
        
        # 如果 scipy 可用，可以使用:
        # return self._ocv_interp(soc).item() # .item() 将 numpy array 转换为标量


    def calculate_terminal_voltage(self, current, soc, dt):
        # 根据电流、SoC 和时间步长计算端电压

        # 更新 RC 网络电压 (使用前向欧拉法)
        # dV1/dt = (I*R1 - V1) / (R1*C1)
        # V1(t+dt) = V1(t) + dt * (I*R1 - V1(t)) / tau1
        # 防止除零
        if self.tau1 != 0 and not np.isinf(self.tau1):
             self._V1 = self._V1 + dt * (current * self.R1 - self._V1) / self.tau1
        elif current != 0: # 如果 tau1 趋近于无穷大（C1 接近 0），行为类似纯电阻 R1
             self._V1 = current * self.R1
        else:
             self._V1 = 0.0

        # 计算 Warburg 阻抗压降 (简化处理)
        # 实际的 Warburg 阻抗模型复杂，与频率相关。
        # 在时域模拟中通常采用近似方法或状态空间模型。
        # 为了简化，这里暂时忽略 Warburg 阻抗的压降，或者使用一个非常简化的模型。
        warburg_voltage_drop = 0.0 # Placeholder for Warburg voltage drop
        # 如果论文提供了 Warburg 阻抗的具体时域模型，应在此实现。

        # 计算极化电压 (包括欧姆压降、RC 网络压降和 Warburg 压降)
        # 根据论文公式 (2) 的结构: V_p(t) = I(t)R0 + V_RC(t) + V_W(t)
        polarization_voltage = current * self.R0 + self._V1 + warburg_voltage_drop

        # 获取开路电压 (OCV)
        ocv = self.get_ocv(soc)

        # 计算端电压
        # V_t(t) = OCV(SoC(t)) - V_p(t)
        terminal_voltage = ocv - polarization_voltage

        return terminal_voltage

    # 方法：根据温度和老化更新 ECM 参数
    # 如果论文提供了 R0, R1, C1, Warburg 参数随温度和 SoH 的变化关系，应在此实现。
    # def update_parameters(self, temperature, soh):
    #     # Implement logic to update R0, R1, C1, Z_W_params based on temp and SoH
    #     pass 