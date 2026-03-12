"""
stock_widget_ai/__main__.py
`python stock_widget_ai` 또는 `python -m stock_widget_ai` 로 실행 가능하게 해주는 진입점
"""
import sys
import os

# 이 파일이 있는 디렉토리(stock_widget_ai 패키지 폴더)를 sys.path 최앞에 추가
_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from main import main

if __name__ == "__main__":
    main()
