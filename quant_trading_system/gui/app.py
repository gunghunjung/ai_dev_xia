# gui/app.py — 애플리케이션 진입점
import tkinter as tk
from .main_window import MainWindow


def run_app():
    """애플리케이션 실행"""
    root = tk.Tk()
    app = MainWindow(root)
    root.mainloop()
