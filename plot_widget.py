from PyQt5.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLabel
from PyQt5.QtCore import Qt, pyqtSignal
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np

class PlotWidget(QWidget):
    """图表显示组件"""
    def __init__(self, parent=None):
        super().__init__(parent)
        self.init_ui()
        
        # 数据存储
        self.charging_data = []
        self.optimization_data = {
            'generations': [],
            'best_fitness': [],
            'avg_fitness': [],
            'pareto_front': []
        }

    def init_ui(self):
        """初始化用户界面"""
        layout = QVBoxLayout(self)
        
        # 图表显示区域
        self.plot_label = QLabel("暂无图表")
        self.plot_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.plot_label)
        
        # 控制按钮
        control_layout = QHBoxLayout()
        
        self.refresh_btn = QPushButton("刷新")
        self.refresh_btn.clicked.connect(self.refresh_plot)
        control_layout.addWidget(self.refresh_btn)
        
        self.export_btn = QPushButton("导出")
        self.export_btn.clicked.connect(self.export_plot)
        control_layout.addWidget(self.export_btn)
        
        layout.addLayout(control_layout)

    def update_charging_data(self, data):
        """更新充电数据"""
        self.charging_data = data
        self.refresh_plot()

    def update_optimization_data(self, generation, best_fitness, avg_fitness, pareto_front):
        """更新优化数据"""
        self.optimization_data['generations'].append(generation)
        self.optimization_data['best_fitness'].append(best_fitness)
        self.optimization_data['avg_fitness'].append(avg_fitness)
        self.optimization_data['pareto_front'].append(pareto_front)
        self.refresh_plot()

    def refresh_plot(self):
        """刷新图表"""
        if not self.charging_data and not self.optimization_data['generations']:
            self.plot_label.setText("暂无数据")
            return
        
        # 创建子图
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=("充电过程", "优化进度", "Pareto前沿", "目标值分布")
        )
        
        # 绘制充电过程
        if self.charging_data:
            time = [d['time'] for d in self.charging_data]
            soc = [d['soc'] for d in self.charging_data]
            voltage = [d['voltage'] for d in self.charging_data]
            current = [d['current'] for d in self.charging_data]
            temperature = [d['temperature'] for d in self.charging_data]
            
            fig.add_trace(
                go.Scatter(x=time, y=soc, name="SoC"),
                row=1, col=1
            )
            fig.add_trace(
                go.Scatter(x=time, y=voltage, name="电压"),
                row=1, col=1
            )
            fig.add_trace(
                go.Scatter(x=time, y=current, name="电流"),
                row=1, col=1
            )
            fig.add_trace(
                go.Scatter(x=time, y=temperature, name="温度"),
                row=1, col=1
            )
        
        # 绘制优化进度
        if self.optimization_data['generations']:
            fig.add_trace(
                go.Scatter(
                    x=self.optimization_data['generations'],
                    y=self.optimization_data['best_fitness'],
                    name="最优适应度"
                ),
                row=1, col=2
            )
            fig.add_trace(
                go.Scatter(
                    x=self.optimization_data['generations'],
                    y=self.optimization_data['avg_fitness'],
                    name="平均适应度"
                ),
                row=1, col=2
            )
        
        # 绘制Pareto前沿
        if self.optimization_data['pareto_front']:
            latest_front = self.optimization_data['pareto_front'][-1]
            fig.add_trace(
                go.Scatter3d(
                    x=[s[0] for s in latest_front],
                    y=[s[1] for s in latest_front],
                    z=[s[2] for s in latest_front],
                    mode='markers',
                    name="Pareto前沿"
                ),
                row=2, col=1
            )
        
        # 绘制目标值分布
        if self.optimization_data['best_fitness']:
            latest_fitness = self.optimization_data['best_fitness'][-1]
            fig.add_trace(
                go.Bar(
                    x=['充电时间', '电池寿命', '能量损耗', '温度上升'],
                    y=latest_fitness,
                    name="目标值"
                ),
                row=2, col=2
            )
        
        # 更新布局
        fig.update_layout(
            height=800,
            showlegend=True,
            title_text="实时优化过程"
        )
        
        # 保存图表
        self.current_fig = fig
        fig.write_html("temp_plot.html")
        
        # 显示图表
        self.plot_label.setText("图表已更新")

    def export_plot(self):
        """导出图表"""
        if hasattr(self, 'current_fig'):
            self.current_fig.write_html("optimization_process.html")
            self.plot_label.setText("图表已导出") 