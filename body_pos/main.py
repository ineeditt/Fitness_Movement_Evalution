import sys
from PyQt5.QtWidgets import QApplication
from UI_logic import MainWin

if __name__ == "__main__":
    # 创建一个app主体
    app = QApplication(sys.argv)
    # 创建一个主窗口
    win = MainWin()
    # 显示程序窗口
    win.show()
    # 启动主循环，开始程序的运行
    sys.exit(app.exec_())

