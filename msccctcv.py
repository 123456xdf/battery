class MSCCCTCVStrategy:
    def __init__(self, strategy_params):
        # MSCCCTCV 策略参数
        # 示例参数 (需要根据论文详细定义和优化):
        # 'cc_currents': [C1_rate, C2_rate, ...], # 各个 CC 阶段的电流倍率
        # 'ct_temps': [T1_target_K, T2_target_K, ...], # 各个 CT 阶段的目标温度 (K)
        # 'cv_voltage': V_target_V, # CV 阶段的目标电压 (V)
        # 'cc_to_ct_temp_threshold': T_threshold_K, # 从 CC 到 CT 的温度阈值
        # 'ct_to_cc_voltage_threshold': V_threshold_V, # 从 CT 回到 CC 的电压阈值 (如果存在)
        # 'ct_to_cv_soc_threshold': SoC_threshold, # 从 CT 到 CV 的 SoC 阈值 (或时间)
        # 'cv_cut_off_current_rate': Cut_off_rate, # CV 阶段结束的电流倍率阈值
        # 'max_charging_time_s': Max_Time_s # 最大充电时间 (s)
        
        self.strategy_params = strategy_params
        self._current_stage = "INIT" # 初始阶段，在第一次 determine_charging_action 时进入第一个 CC 阶段
        self._stage_start_time = 0.0 # 当前阶段开始的时间
        self._stage_start_soc = 0.0 # 当前阶段开始的 SoC

        # 定义策略阶段的顺序 (示例，需要根据论文调整)
        self._stage_order = ["CC1", "CT1", "CV"]
        self._stage_index = 0 # 当前阶段在 stage_order 中的索引


    def determine_charging_action(self, battery_state):
        # 根据当前电池状态决定下一步的充电行动 (电流或电压)
        # battery_state 是从 Battery 对象获取的状态字典，例如: {'soc': ..., 'soh': ..., 'temperature': ..., 'terminal_voltage': ..., 'current': ...}

        current_time = battery_state.get('time', 0.0) # 假设 battery_state 中包含 'time'
        current_soc = battery_state['soc']
        current_voltage = battery_state['terminal_voltage']
        current_temperature = battery_state['temperature']
        current_current = battery_state['current']

        # 初始化阶段处理
        if self._current_stage == "INIT":
            self._current_stage = self._stage_order[0] # 进入第一个阶段
            self._stage_start_time = current_time
            self._stage_start_soc = current_soc
            print(f"进入充电阶段: {self._current_stage}") # 示例输出

        # 检查充电完成条件 (例如达到目标 SoC 且电流低于阈值)
        # 最终的充电完成判断通常在 CV 阶段进行
        # if self._current_stage == "CV" and abs(current_current) < self.strategy_params.get('cv_cut_off_current_rate', 0.05) * self.strategy_params.get('nominal_capacity', 2.6):
        #     print("充电完成。")
        #     return {'control_mode': 'finish'}
        
        # 检查是否达到最大充电时间
        # if current_time > self.strategy_params.get('max_charging_time_s', float('inf')):
        #      print("达到最大充电时间，停止充电。")
        #      return {'control_mode': 'finish'}


        # 根据当前阶段执行逻辑并检查切换条件
        if self._current_stage.startswith("CC"):
            # 恒流阶段 (CC)
            # 获取当前 CC 阶段的目标电流
            stage_number = int(self._current_stage[2:]) # 从 "CC1" 提取 1
            target_current_rate = self.strategy_params.get('cc_currents', [1.0])[stage_number - 1]
            # 假设 strategy_params 中提供了标称容量来计算电流值
            nominal_capacity = self.strategy_params.get('nominal_capacity', 2.6) # 额定容量 (Ah)
            target_current = target_current_rate * nominal_capacity # 目标电流 (A)

            # 检查从当前 CC 阶段切换到下一个阶段的条件
            # 例如: 达到电压阈值 或 温度阈值
            # 切换到 CT 阶段的条件示例 (需要根据论文详细实现):
            # if current_voltage >= self.strategy_params.get('cc_to_ct_voltage_threshold', float('inf')) or \
            #    current_temperature >= self.strategy_params.get('cc_to_ct_temp_threshold', float('inf')):
            #     self._stage_index += 1
            #     if self._stage_index < len(self._stage_order):
            #         self._current_stage = self._stage_order[self._stage_index]
            #         self._stage_start_time = current_time
            #         self._stage_start_soc = current_soc
            #         print(f"进入充电阶段: {self._current_stage}") # 示例输出
            #     else:
            #          # 如果没有下一个预设阶段，可能直接进入 CV 或结束
            #          print("CC 阶段结束，没有预设的下一阶段。")
            #          return {'control_mode': 'finish'} # 示例: 直接结束

            # 返回当前 CC 阶段的控制指令
            return {'control_mode': 'CC', 'value': target_current}

        elif self._current_stage.startswith("CT"):
            # 恒温阶段 (CT)
            # 获取当前 CT 阶段的目标温度
            stage_number = int(self._current_stage[2:]) # 从 "CT1" 提取 1
            target_temperature = self.strategy_params.get('ct_temps', [308.15])[stage_number - 1] # 示例温度 35°C

            # 在 CT 阶段，充电器需要调节电流以维持目标温度。
            # 这里的策略只返回目标温度，具体的电流控制逻辑应在 Charger 类中实现 (例如使用 PI 控制)。

            # 检查从当前 CT 阶段切换到下一个阶段的条件
            # 例如: 达到 SoC 阈值 或 时间阈值，或者达到电压阈值切换到 CV
            # 切换到 CV 阶段的条件示例 (需要根据论文详细实现):
            # if current_voltage >= self.strategy_params.get('ct_to_cv_voltage_threshold', self.strategy_params.get('cv_voltage', float('inf'))):
            #     self._stage_index += 1
            #     if self._stage_index < len(self._stage_order):
            #         self._current_stage = self._stage_order[self._stage_index]
            #         self._stage_start_time = current_time
            #         self._stage_start_soc = current_soc
            #         print(f"进入充电阶段: {self._current_stage}") # 示例输出
            #     else:
            #          # 如果没有下一个预设阶段，可能直接结束
            #          print("CT 阶段结束，没有预设的下一阶段。")
            #          return {'control_mode': 'finish'} # 示例: 直接结束

            # 返回当前 CT 阶段的控制指令
            return {'control_mode': 'CT', 'value': target_temperature}

        elif self._current_stage == "CV":
            # 恒压阶段 (CV)
            # 获取 CV 阶段的目标电压
            target_voltage = self.strategy_params.get('cv_voltage', 4.2) # 示例电压 4.2V

            # 在 CV 阶段，充电器需要调节电流以维持目标电压。
            # 这里的策略只返回目标电压，具体的电流控制逻辑应在 Charger 类中实现。

            # 检查 CV 阶段的结束条件 (电流下降到阈值以下)
            # if abs(current_current) < self.strategy_params.get('cv_cut_off_current_rate', 0.05) * self.strategy_params.get('nominal_capacity', 2.6):
            #     print("CV 阶段结束，充电完成。")
            #     return {'control_mode': 'finish'}

            # 返回当前 CV 阶段的控制指令
            return {'control_mode': 'CV', 'value': target_voltage}

        # 如果当前阶段未知或策略已完成
        return {'control_mode': 'finish'}

    def get_current_stage(self):
        # 获取当前充电阶段
        return self._current_stage 