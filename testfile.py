from PyQt5.QtCore import QLibraryInfo
print("Qt 插件路径:", QLibraryInfo.location(QLibraryInfo.PluginsPath))