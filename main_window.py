############ 添加 #################
import sys
import os
# 确保项目根目录在 Python 路径中
root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, root_dir)

# 检查 PyQt5 插件路径
from PyQt5.QtCore import QLibraryInfo

# 设置 Qt 插件路径（如果自动检测失败）
os.environ["QT_PLUGIN_PATH"] = os.path.join(
    os.path.dirname(os.path.dirname(__file__)),
    ".venv", "Lib", "site-packages", "PyQt5", "Qt", "plugins"
)

##################################

import sys
import os
import json
from datetime import datetime
import logging
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout,
                           QHBoxLayout, QTabWidget, QPushButton, QLabel,
                           QFileDialog, QMessageBox, QProgressBar, QDialog)
from PyQt5.QtCore import Qt, QThread, pyqtSignal

from utils.parameter_optimizer import ParameterOptimizer
from utils.data_exporter import DataExporter
from visualization.interactive_plotter import InteractivePlotter
from gui.settings_dialog import SettingsDialog
from gui.plot_widget import PlotWidget

################# 添加 ##################
import sys
import os
# 确保项目根目录（D:\battery\073）在 Python 路径中
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
#####################################


# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class OptimizationThread(QThread):
    """优化线程"""
    progress = pyqtSignal(int)
    data_update = pyqtSignal(dict)
    finished = pyqtSignal(dict)
    error = pyqtSignal(str)

    def __init__(self, optimizer, data, params):
        super().__init__()
        self.optimizer = optimizer
        self.data = data
        self.params = params
        self.is_running = True
        logging.info("优化线程初始化")


    def run(self):
        logging.info("优化线程开始运行")

        try:
            # 设置回调函数

            def callback(generation, best_fitness, avg_fitness, pareto_front, charging_data):
                if not self.is_running:
                    return False
                
                # 发送数据更新信号
                self.data_update.emit({
                    'generation': generation,
                    'best_fitness': best_fitness,
                    'avg_fitness': avg_fitness,
                    'pareto_front': pareto_front,
                    'charging_data': charging_data
                })
                
                # 发送进度信号
                progress = int((generation / self.params['algorithm_params']['n_generations']) * 100)
                self.progress.emit(progress)
                
                logging.debug(f"优化回调: 代数={generation}, 进度={progress}%")
                return True
            
            # 运行优化
            raw_result = self.optimizer.optimize_battery_parameters(
                self.data,
                self.params,
                callback=callback
            )
            # 处理不同类型的结果
            if hasattr(raw_result, '__dict__'):  # 如果是对象
                # 转换numpy类型为Python原生类型
                optimized_params = {}
                for k, v in getattr(raw_result, 'optimized_params', {}).items():
                    if 'numpy' in str(type(v)):
                        optimized_params[k] = float(v)
                    else:
                        optimized_params[k] = v

                result = {
                    'status': 'success',
                    'message': getattr(raw_result, 'message', ''),
                    'solutions': [getattr(raw_result, 'optimized_params', {})],
                    'fitness_values': [[float(getattr(raw_result, 'objective_value', 0))]],  # 确保是二维数组
                    'parameters': {
                        'battery_params': getattr(raw_result, 'battery_params', {}),
                        'algorithm_params': getattr(raw_result, 'algorithm_params', {}),
                    },
                    'metrics': getattr(raw_result, 'metrics', {}),
                    'raw_data': str(raw_result)
                }
            elif isinstance(raw_result, dict):  # 如果是字典
                result = {
                    'status': raw_result.get('status', 'success'),
                    'message': raw_result.get('message', ''),
                    'solutions': raw_result.get('solutions', []),
                    'fitness_values': raw_result.get('fitness_values', []),
                    'parameters': raw_result.get('parameters', {}),
                    'raw_data': raw_result
                }
            else:  # 其他未知类型
                result = {
                    'status': 'error',
                    'message': 'Unknown result format',
                    'solutions': [],
                    'fitness_values': [],
                    'parameters': {},
                    'raw_data': str(raw_result)
                }

            if self.is_running:
                self.finished.emit(result)
        except Exception as e:
            logging.error("优化过程中发生错误", exc_info=True)
            self.error.emit(str(e))
            self.finished.emit({'status': 'error',
                                'message': str(e),
                                'solutions': [],
                                'fitness_values': [],
                                'parameters': {}})

    def stop(self):
        """停止优化"""
        self.is_running = False
        logging.info("收到停止优化请求")

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("电池充电策略优化工具")
        self.setMinimumSize(1200, 800)
        
        # 初始化组件
        self.init_ui()
        
        # 创建输出目录
        self.output_dir = os.path.join("output", datetime.now().strftime("%Y%m%d_%H%M%S"))
        os.makedirs(self.output_dir, exist_ok=True)
        
        # 初始化工具类
        self.optimizer = ParameterOptimizer(self.output_dir)
        self.exporter = DataExporter(self.output_dir)
        self.plotter = InteractivePlotter()
        
        # 数据存储
        self.experimental_data = None
        self.optimization_results = None
        self.current_params = None

    def init_ui(self):
        """初始化用户界面"""
        # 创建中央部件
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        # 创建主布局
        main_layout = QVBoxLayout(central_widget)
        
        # 创建标签页
        tab_widget = QTabWidget()
        main_layout.addWidget(tab_widget)
        
        # 添加各个标签页
        tab_widget.addTab(self.create_optimization_tab(), "优化")
        tab_widget.addTab(self.create_visualization_tab(), "可视化")
        tab_widget.addTab(self.create_settings_tab(), "设置")
        
        # 创建状态栏
        self.statusBar().showMessage("就绪")

    def create_optimization_tab(self):
        """创建优化标签页"""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        
        # 数据导入部分
        data_group = QWidget()
        data_layout = QHBoxLayout(data_group)
        
        self.data_label = QLabel("未加载数据")
        data_layout.addWidget(self.data_label)
        
        load_data_btn = QPushButton("加载数据")
        load_data_btn.clicked.connect(self.load_data)
        data_layout.addWidget(load_data_btn)
        
        layout.addWidget(data_group)
        
        # 优化控制部分
        control_group = QWidget()
        control_layout = QHBoxLayout(control_group)
        
        self.start_btn = QPushButton("开始优化")
        self.start_btn.clicked.connect(self.start_optimization)
        control_layout.addWidget(self.start_btn)
        
        self.stop_btn = QPushButton("停止")
        self.stop_btn.clicked.connect(self.stop_optimization)
        self.stop_btn.setEnabled(False)
        control_layout.addWidget(self.stop_btn)
        
        layout.addWidget(control_group)
        
        # 进度条
        self.progress_bar = QProgressBar()
        layout.addWidget(self.progress_bar)
        
        # 实时图表
        self.plot_widget = PlotWidget()
        layout.addWidget(self.plot_widget)
        
        # 结果显示
        self.result_label = QLabel("等待优化...")
        layout.addWidget(self.result_label)
        
        return widget

    def create_visualization_tab(self):
        """创建可视化标签页"""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        
        # 图表显示区域
        self.plot_label = QLabel("暂无图表")
        self.plot_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.plot_label)
        
        # 控制按钮
        control_group = QWidget()
        control_layout = QHBoxLayout(control_group)
        
        plot_btn = QPushButton("显示Pareto前沿")
        plot_btn.clicked.connect(self.show_pareto_front)
        control_layout.addWidget(plot_btn)
        
        compare_btn = QPushButton("比较解")
        compare_btn.clicked.connect(self.show_solution_comparison)
        control_layout.addWidget(compare_btn)
        
        export_btn = QPushButton("导出图表")
        export_btn.clicked.connect(self.export_plots)
        control_layout.addWidget(export_btn)
        
        layout.addWidget(control_group)
        
        return widget

    def create_settings_tab(self):
        """创建设置标签页"""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        
        # 参数设置部分
        settings_group = QWidget()
        settings_layout = QVBoxLayout(settings_group)
        
        # 加载/保存设置按钮
        button_layout = QHBoxLayout()
        
        load_settings_btn = QPushButton("加载设置")
        load_settings_btn.clicked.connect(self.load_settings)
        button_layout.addWidget(load_settings_btn)
        
        save_settings_btn = QPushButton("保存设置")
        save_settings_btn.clicked.connect(self.save_settings)
        button_layout.addWidget(save_settings_btn)
        
        edit_settings_btn = QPushButton("编辑设置")
        edit_settings_btn.clicked.connect(self.show_settings_dialog)
        button_layout.addWidget(edit_settings_btn)
        
        settings_layout.addLayout(button_layout)
        
        # 当前设置显示
        self.settings_label = QLabel("当前设置：\n未加载")
        settings_layout.addWidget(self.settings_label)
        
        layout.addWidget(settings_group)
        
        return widget

    def load_data(self):
        """加载数据文件"""
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "选择数据文件",
            "",
            "JSON文件 (*.json);;所有文件 (*.*)"
        )
        
        if file_path:
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    self.experimental_data = json.load(f)
                self.data_label.setText(f"已加载数据: {os.path.basename(file_path)}")
                self.statusBar().showMessage("数据加载成功")
                logging.info(f"实验数据加载成功: {file_path}")
            except Exception as e:
                QMessageBox.critical(self, "错误", f"加载数据失败: {str(e)}")
                logging.error(f"加载数据失败: {file_path}", exc_info=True)
                self.experimental_data = None
                self.data_label.setText("未加载数据")

    def start_optimization(self):
        """开始优化过程"""
        if not self.experimental_data:
            QMessageBox.warning(self, "警告", "请先加载数据")
            self.statusBar().showMessage("请加载实验数据后开始优化。")
            return
        
        if not self.current_params:
            QMessageBox.warning(self, "警告", "请先设置参数")
            return
        
        self.start_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)
        self.progress_bar.setValue(0)
        
        # 创建并启动优化线程
        self.optimization_thread = OptimizationThread(
            self.optimizer,
            self.experimental_data,
            self.current_params
        )
        self.optimization_thread.progress.connect(self.update_progress)
        self.optimization_thread.data_update.connect(self.update_optimization_data)
        self.optimization_thread.finished.connect(self.optimization_finished)
        self.optimization_thread.error.connect(self.optimization_error)
        self.optimization_thread.start()

    def stop_optimization(self):
        """停止优化过程"""
        if hasattr(self, 'optimization_thread'):
            self.optimization_thread.stop()
            self.optimization_thread.wait()
        
        self.start_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        self.statusBar().showMessage("优化已停止")

    def update_progress(self, value):
        """更新进度条"""
        self.progress_bar.setValue(value)

    def update_optimization_data(self, data):
        """更新优化数据"""
        # 更新图表
        self.plot_widget.update_optimization_data(
            data['generation'],
            data['best_fitness'],
            data['avg_fitness'],
            data['pareto_front']
        )
        
        # 更新充电数据
        if data['charging_data']:
            self.plot_widget.update_charging_data(data['charging_data'])

    def optimization_finished(self, results):
        """优化完成处理"""
        self.optimization_results = results
        self.start_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)

        try:
            # 显示结果（添加健壮性检查）
            result_text = f"优化状态: {results.get('status', 'unknown')}\n"
            result_text += f"消息: {results.get('message', '')}\n"
            # 显示优化参数
            if results.get('parameters'):
                result_text += "\n优化参数:\n"
                for category, params in results['parameters'].items():
                    result_text += f"{category}:\n"
                    for k, v in params.items():
                        # 处理numpy.float64类型
                        if 'numpy.float64' in str(type(v)):
                            v = float(v)
                        result_text += f"  {k}: {v}\n"

            # 显示评估指标
            if results.get('metrics'):
                result_text += "\n评估指标:\n"
                for k, v in results['metrics'].items():
                    result_text += f"  {k}: {v:.4f}\n"

            self.result_label.setText(result_text)

            # 仅当有解时才保存
            if results.get('solutions')and results.get('fitness_values'):
                # 确保所有numpy类型转换为Python原生类型
                processed_results = self._convert_numpy_types(results)
                self.save_optimization_results(results)
            else:
                logging.warning("无有效解可保存")
                self.result_label.setText(f"{result_text}\n警告: 无有效解或适应度值")

        except Exception as e:
            logging.error(f"结果显示或保存失败: {str(e)}")
            self.result_label.setText("优化完成但结果显示失败")

        self.statusBar().showMessage("优化完成")
        logging.info("优化线程完成")

    def _convert_numpy_types(self, data):
        """递归转换numpy类型为Python原生类型"""
        if isinstance(data, dict):
            return {k: self._convert_numpy_types(v) for k, v in data.items()}
        elif isinstance(data, list):
            return [self._convert_numpy_types(item) for item in data]
        elif 'numpy' in str(type(data)):
            return float(data)  # 或其他适当的转换
        else:
            return data

    def save_optimization_results(self, results):
        """保存优化结果"""
        try:
            # 检查必要字段是否存在
            if not all(key in results for key in ['solutions', 'fitness_values']):
                missing = [k for k in ['solutions', 'fitness_values'] if k not in results]
                raise ValueError(f"结果缺少必要字段: {missing}")

            # 保存到JSON文件
            filepath = os.path.join(self.output_dir, "optimization_results.json")
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(results, f, indent=4, ensure_ascii=False)

            # 保存到CSV文件
            self.exporter.export_optimization_results_to_csv(
                results['solutions'],
                results['fitness_values']
            )
            logging.info(f"优化结果已导出到CSV: {filepath}")

        except Exception as e:
            logging.error(f"保存优化结果失败: {str(e)}")
            QMessageBox.warning(self, "警告", f"保存结果时出错: {str(e)}")

    def optimization_error(self, error_msg):
        """优化错误处理"""
        QMessageBox.critical(self, "错误", f"优化过程出错: {error_msg}")
        self.start_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        self.statusBar().showMessage("优化失败")
        logging.error(f"优化过程中发生错误: {error_msg}")

    def show_pareto_front(self):
        """显示Pareto前沿图"""
        if not self.optimization_results or not self.optimization_results.get('solutions') or not self.optimization_results.get('fitness_values'):
            QMessageBox.warning(self, "警告", "请先完成优化或优化结果无效")
            return
        try:
            # 确保fitness_values是二维数组
            fitness_values = self.optimization_results['fitness_values']
            if not isinstance(fitness_values, list) or not all(isinstance(f, list) for f in fitness_values):
                if isinstance(fitness_values[0], (float, int)):
                    fitness_values = [fitness_values]  # 转换为二维数组
                else:
                    raise ValueError("fitness_values格式不正确")
            # 创建Pareto前沿图
            fig = self.plotter.plot_pareto_front(
                self.optimization_results['solutions'],
                self.optimization_results['fitness_values'],
                "Pareto前沿"
            )

            # 保存图表
            self.plotter.save_figure(fig, os.path.join(self.output_dir, "pareto_front.html"))

            # 显示图表
            self.plot_label.setText("Pareto前沿图已生成")
        except Exception as e:
            QMessageBox.critical(self, "错误", f"生成Pareto前沿图失败: {str(e)}")
            logging.error(f"生成Pareto前沿图失败: {str(e)}")
    def show_solution_comparison(self):
        """显示解的比较图"""
        if not self.optimization_results or not self.optimization_results.get('solutions') or not self.optimization_results.get('fitness_values'):
            QMessageBox.warning(self, "警告", "请先完成优化")
            return
        try:
            # 确保solutions和fitness_values格式正确
            solutions = self.optimization_results['solutions']
            fitness_values = self.optimization_results['fitness_values']

            # 如果fitness_values是一维数组，转换为二维
            if isinstance(fitness_values[0], (float, int)):
                fitness_values = [fitness_values]

            # 创建解的比较图
            fig = self.plotter.plot_solution_comparison(
                self.optimization_results['solutions'],
                self.optimization_results['fitness_values'],
                "解的比较"
            )

            # 保存图表
            self.plotter.save_figure(fig, os.path.join(self.output_dir, "solution_comparison.html"))

            # 显示图表
            self.plot_label.setText("解的比较图已生成")
        except Exception as e:
            QMessageBox.critical(self, "错误", f"生成解比较图失败: {str(e)}")
            logging.error(f"生成解比较图失败: {str(e)}")

    def export_plots(self):
        """导出图表"""
        if not self.optimization_results:
            QMessageBox.warning(self, "警告", "请先完成优化")
            return
        
        # 导出所有图表
        self.show_pareto_front()
        self.show_solution_comparison()
        
        QMessageBox.information(self, "提示", "图表已导出到输出目录")

    def show_settings_dialog(self):
        """显示设置对话框"""
        dialog = SettingsDialog(self)
        if dialog.exec_() == QDialog.Accepted:
            self.current_params = dialog.get_all_params()
            self.update_settings_display()

    def load_settings(self):
        """加载设置"""
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "选择设置文件",
            "",
            "JSON文件 (*.json);;所有文件 (*.*)"
        )
        
        if file_path:
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    self.current_params = json.load(f)
                self.update_settings_display()
                self.statusBar().showMessage("设置加载成功")
                logging.info(f"设置加载成功: {file_path}")
            except Exception as e:
                QMessageBox.critical(self, "错误", f"加载设置失败: {str(e)}")
                logging.error(f"加载设置失败: {file_path}", exc_info=True)

    def save_settings(self):
        """保存设置"""
        if not self.current_params:
            QMessageBox.warning(self, "警告", "没有可保存的设置")
            return
        
        file_path, _ = QFileDialog.getSaveFileName(
            self,
            "保存设置",
            "",
            "JSON文件 (*.json);;所有文件 (*.*)"
        )
        
        if file_path:
            try:
                with open(file_path, 'w', encoding='utf-8') as f:
                    json.dump(self.current_params, f, indent=4, ensure_ascii=False)
                self.statusBar().showMessage("设置保存成功")
                logging.info(f"设置保存成功: {file_path}")
            except Exception as e:
                QMessageBox.critical(self, "错误", f"保存设置失败: {str(e)}")
                logging.error(f"保存设置失败: {file_path}", exc_info=True)

    def update_settings_display(self):
        """更新设置显示"""
        if not self.current_params:
            self.settings_label.setText("当前设置：\n未加载")
            return
        
        settings_text = "当前设置：\n"
        for category, params in self.current_params.items():
            settings_text += f"\n{category}:\n"
            for param, value in params.items():
                settings_text += f"  {param}: {value}\n"
        
        self.settings_label.setText(settings_text)

    def save_optimization_results(self, results):
        """保存优化结果"""
        try:
            # 检查必要字段是否存在
            if not all(key in results for key in ['solutions', 'fitness_values']):
                missing = [k for k in ['solutions', 'fitness_values'] if k not in results]
                raise ValueError(f"结果缺少必要字段: {missing}")

            # 转换numpy类型为Python原生类型
            processed_results = self._convert_numpy_types(results)

            # 保存到JSON文件
            filepath = os.path.join(self.output_dir, "optimization_results.json")
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(processed_results, f, indent=4, ensure_ascii=False)

            # 保存到CSV文件
            self.exporter.export_optimization_results_to_csv(
                processed_results['solutions'],
                processed_results['fitness_values']
            )
            logging.info(f"优化结果已导出到CSV: {filepath}")

        except Exception as e:
            logging.error(f"保存优化结果失败: {str(e)}")
            QMessageBox.warning(self, "警告", f"保存结果时出错: {str(e)}")
def main():
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())

if __name__ == '__main__':
    main() 