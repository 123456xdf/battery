from PyQt5.QtWidgets import (QDialog, QVBoxLayout, QHBoxLayout, QLabel,
                           QLineEdit, QPushButton, QGroupBox, QFormLayout,
                           QDoubleSpinBox, QSpinBox, QTabWidget, QWidget)
from PyQt5.QtCore import Qt

class SettingsDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("参数设置")
        self.setMinimumWidth(600)
        
        # 初始化UI
        self.init_ui()
        
        # 加载默认值
        self.load_default_values()

    def init_ui(self):
        """初始化用户界面"""
        layout = QVBoxLayout(self)
        
        # 创建标签页
        tab_widget = QTabWidget()
        layout.addWidget(tab_widget)
        
        # 添加电池模型参数标签页
        tab_widget.addTab(self.create_battery_params_tab(), "电池模型参数")
        
        # 添加算法参数标签页
        tab_widget.addTab(self.create_algorithm_params_tab(), "算法参数")
        
        # 添加优化目标标签页
        tab_widget.addTab(self.create_objectives_tab(), "优化目标")
        
        # 按钮
        button_layout = QHBoxLayout()
        
        save_btn = QPushButton("保存")
        save_btn.clicked.connect(self.accept)
        button_layout.addWidget(save_btn)
        
        cancel_btn = QPushButton("取消")
        cancel_btn.clicked.connect(self.reject)
        button_layout.addWidget(cancel_btn)
        
        layout.addLayout(button_layout)

    def create_battery_params_tab(self):
        """创建电池模型参数标签页"""
        widget = QWidget()
        layout = QFormLayout(widget)
        
        # 等效电路模型参数
        ecm_group = QGroupBox("等效电路模型参数")
        ecm_layout = QFormLayout()
        
        self.r0_spin = QDoubleSpinBox()
        self.r0_spin.setRange(0.001, 0.1)
        self.r0_spin.setDecimals(4)
        self.r0_spin.setSingleStep(0.001)
        ecm_layout.addRow("R0 (Ω):", self.r0_spin)
        
        self.r1_spin = QDoubleSpinBox()
        self.r1_spin.setRange(0.001, 0.1)
        self.r1_spin.setDecimals(4)
        self.r1_spin.setSingleStep(0.001)
        ecm_layout.addRow("R1 (Ω):", self.r1_spin)
        
        self.c1_spin = QDoubleSpinBox()
        self.c1_spin.setRange(100, 10000)
        self.c1_spin.setDecimals(1)
        self.c1_spin.setSingleStep(100)
        ecm_layout.addRow("C1 (F):", self.c1_spin)
        
        ecm_group.setLayout(ecm_layout)
        layout.addRow(ecm_group)
        
        # 热模型参数
        thermal_group = QGroupBox("热模型参数")
        thermal_layout = QFormLayout()
        
        self.conv_coef_spin = QDoubleSpinBox()
        self.conv_coef_spin.setRange(0.1, 10.0)
        self.conv_coef_spin.setDecimals(2)
        self.conv_coef_spin.setSingleStep(0.1)
        thermal_layout.addRow("对流换热系数 (W/K):", self.conv_coef_spin)
        
        self.heat_cap_spin = QDoubleSpinBox()
        self.heat_cap_spin.setRange(500, 2000)
        self.heat_cap_spin.setDecimals(1)
        self.heat_cap_spin.setSingleStep(50)
        thermal_layout.addRow("热容 (J/K):", self.heat_cap_spin)
        
        thermal_group.setLayout(thermal_layout)
        layout.addRow(thermal_group)
        
        return widget

    def create_algorithm_params_tab(self):
        """创建算法参数标签页"""
        widget = QWidget()
        layout = QFormLayout(widget)
        
        # 遗传算法参数
        ga_group = QGroupBox("遗传算法参数")
        ga_layout = QFormLayout()
        
        self.pop_size_spin = QSpinBox()
        self.pop_size_spin.setRange(20, 200)
        self.pop_size_spin.setSingleStep(10)
        ga_layout.addRow("种群大小:", self.pop_size_spin)
        
        self.n_gen_spin = QSpinBox()
        self.n_gen_spin.setRange(10, 100)
        self.n_gen_spin.setSingleStep(5)
        ga_layout.addRow("迭代次数:", self.n_gen_spin)
        
        self.mut_rate_spin = QDoubleSpinBox()
        self.mut_rate_spin.setRange(0.01, 0.5)
        self.mut_rate_spin.setDecimals(2)
        self.mut_rate_spin.setSingleStep(0.01)
        ga_layout.addRow("变异率:", self.mut_rate_spin)
        
        self.cross_rate_spin = QDoubleSpinBox()
        self.cross_rate_spin.setRange(0.5, 1.0)
        self.cross_rate_spin.setDecimals(2)
        self.cross_rate_spin.setSingleStep(0.05)
        ga_layout.addRow("交叉率:", self.cross_rate_spin)
        
        self.tour_size_spin = QSpinBox()
        self.tour_size_spin.setRange(2, 10)
        self.tour_size_spin.setSingleStep(1)
        ga_layout.addRow("锦标赛规模:", self.tour_size_spin)
        
        ga_group.setLayout(ga_layout)
        layout.addRow(ga_group)
        
        return widget

    def create_objectives_tab(self):
        """创建优化目标标签页"""
        widget = QWidget()
        layout = QFormLayout(widget)
        
        # 目标权重
        weights_group = QGroupBox("目标权重")
        weights_layout = QFormLayout()
        
        self.time_weight_spin = QDoubleSpinBox()
        self.time_weight_spin.setRange(0, 1)
        self.time_weight_spin.setDecimals(2)
        self.time_weight_spin.setSingleStep(0.1)
        weights_layout.addRow("充电时间权重:", self.time_weight_spin)
        
        self.soh_weight_spin = QDoubleSpinBox()
        self.soh_weight_spin.setRange(0, 1)
        self.soh_weight_spin.setDecimals(2)
        self.soh_weight_spin.setSingleStep(0.1)
        weights_layout.addRow("电池寿命权重:", self.soh_weight_spin)
        
        self.energy_weight_spin = QDoubleSpinBox()
        self.energy_weight_spin.setRange(0, 1)
        self.energy_weight_spin.setDecimals(2)
        self.energy_weight_spin.setSingleStep(0.1)
        weights_layout.addRow("能量损耗权重:", self.energy_weight_spin)
        
        self.temp_weight_spin = QDoubleSpinBox()
        self.temp_weight_spin.setRange(0, 1)
        self.temp_weight_spin.setDecimals(2)
        self.temp_weight_spin.setSingleStep(0.1)
        weights_layout.addRow("温度上升权重:", self.temp_weight_spin)
        
        weights_group.setLayout(weights_layout)
        layout.addRow(weights_group)
        
        return widget

    def load_default_values(self):
        """加载默认参数值"""
        # 电池模型参数
        self.r0_spin.setValue(0.01)
        self.r1_spin.setValue(0.005)
        self.c1_spin.setValue(1000.0)
        self.conv_coef_spin.setValue(0.5)
        self.heat_cap_spin.setValue(1000.0)
        
        # 算法参数
        self.pop_size_spin.setValue(50)
        self.n_gen_spin.setValue(20)
        self.mut_rate_spin.setValue(0.1)
        self.cross_rate_spin.setValue(0.8)
        self.tour_size_spin.setValue(3)
        
        # 目标权重
        self.time_weight_spin.setValue(0.25)
        self.soh_weight_spin.setValue(0.25)
        self.energy_weight_spin.setValue(0.25)
        self.temp_weight_spin.setValue(0.25)

    def get_battery_params(self):
        """获取电池模型参数"""
        return {
            'R0': self.r0_spin.value(),
            'R1': self.r1_spin.value(),
            'C1': self.c1_spin.value(),
            'convection_coefficient': self.conv_coef_spin.value(),
            'heat_capacity': self.heat_cap_spin.value()
        }

    def get_algorithm_params(self):
        """获取算法参数"""
        return {
            'population_size': self.pop_size_spin.value(),
            'n_generations': self.n_gen_spin.value(),
            'mutation_rate': self.mut_rate_spin.value(),
            'crossover_rate': self.cross_rate_spin.value(),
            'tournament_size': self.tour_size_spin.value()
        }

    def get_objective_weights(self):
        """获取目标权重"""
        return {
            'time_weight': self.time_weight_spin.value(),
            'soh_weight': self.soh_weight_spin.value(),
            'energy_weight': self.energy_weight_spin.value(),
            'temp_weight': self.temp_weight_spin.value()
        }

    def get_all_params(self):
        """获取所有参数"""
        return {
            'battery_params': self.get_battery_params(),
            'algorithm_params': self.get_algorithm_params(),
            'objective_weights': self.get_objective_weights()
        } 