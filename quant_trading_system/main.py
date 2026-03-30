#!/usr/bin/env python3
# main.py — 퀀트 트레이딩 시스템 진입점
"""
퀀트 트레이딩 시스템 — CNN + Transformer 하이브리드

실행 방법:
    python main.py          # GUI 실행 (기본)
    python main.py --cli    # CLI 모드 (데이터 다운로드만)
    python -m quant_trading_system  # 모듈 모드

요구사항:
    pip install -r requirements.txt
    pip install torch --index-url https://download.pytorch.org/whl/cu121  # GPU
"""
from __future__ import annotations
import os
import sys
import warnings
import argparse
import logging

# torch 내부 pynvml 경고 억제 (nvidia-ml-py 설치 권장이지만 기능은 동일)
warnings.filterwarnings("ignore", category=FutureWarning, module="torch")

# 프로젝트 루트를 Python 경로에 추가
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE_DIR)


def check_dependencies() -> bool:
    """필수 패키지 확인"""
    required = ["numpy", "pandas", "yfinance"]
    optional = ["torch", "scipy", "pyarrow", "pynvml"]
    missing_required = []
    missing_optional = []

    for pkg in required:
        try:
            __import__(pkg)
        except ImportError:
            missing_required.append(pkg)

    for pkg in optional:
        try:
            __import__(pkg)
        except ImportError:
            missing_optional.append(pkg)

    if missing_required:
        print("❌ 필수 패키지 미설치:")
        for pkg in missing_required:
            print(f"   pip install {pkg}")
        return False

    if missing_optional:
        print("⚠️  선택적 패키지 미설치 (기능 제한):")
        for pkg in missing_optional:
            if pkg == "torch":
                print(f"   CPU: pip install torch")
                print(f"   GPU: pip install torch --index-url https://download.pytorch.org/whl/cu121")
            elif pkg == "pynvml":
                print(f"   GPU 모니터: pip install pynvml")
            else:
                print(f"   pip install {pkg}")

    return True


def run_gui():
    """GUI 모드 실행"""
    import tkinter as tk
    from gui import run_app
    from config import load_settings
    from utils.logger import setup_logger

    settings = load_settings()
    setup_logger("quant", settings.log_level,
                 os.path.join(BASE_DIR, settings.output_dir))

    print("퀀트 트레이딩 시스템 시작...")
    run_app()


def run_cli(args):
    """CLI 모드 (데이터 다운로드 테스트)"""
    from config import load_settings
    from data import DataLoader
    from utils.logger import setup_logger

    settings = load_settings()
    logger = setup_logger("quant", settings.log_level,
                          os.path.join(BASE_DIR, settings.output_dir))

    print("=== 퀀트 트레이딩 시스템 CLI 모드 ===")
    print(f"종목: {settings.data.symbols}")
    print(f"기간: {settings.data.period}")

    loader = DataLoader(
        os.path.join(BASE_DIR, settings.data.cache_dir),
        settings.data.cache_ttl_hours
    )

    data = loader.load_multi(settings.data.symbols, settings.data.period)
    print(f"\n로드 완료: {len(data)}개 종목")
    for sym, df in data.items():
        print(f"  {sym}: {len(df)}행 ({df.index[0].date()} ~ {df.index[-1].date()})")


def main():
    parser = argparse.ArgumentParser(
        description="퀀트 트레이딩 시스템 — CNN+Transformer 하이브리드"
    )
    parser.add_argument("--cli", action="store_true",
                        help="CLI 모드 (GUI 없이 실행)")
    parser.add_argument("--no-check", action="store_true",
                        help="의존성 확인 건너뜀")
    args = parser.parse_args()

    if not args.no_check:
        if not check_dependencies():
            sys.exit(1)

    os.makedirs(os.path.join(BASE_DIR, "outputs", "models"), exist_ok=True)
    os.makedirs(os.path.join(BASE_DIR, "cache"), exist_ok=True)
    os.makedirs(os.path.join(BASE_DIR, "outputs"), exist_ok=True)

    if args.cli:
        run_cli(args)
    else:
        run_gui()


if __name__ == "__main__":
    main()
