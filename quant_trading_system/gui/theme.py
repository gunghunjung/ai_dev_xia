# gui/theme.py — 통합 색상 테마 (Catppuccin Mocha 기반)
"""
프로그램 전체 공통 색상 상수.
각 패널에서 개별 정의하던 중복을 하나로 통합.

사용법:
    from gui.theme import BG, FG, ACCENT, GREEN, RED, YELLOW, PURPLE, DIM, PANEL_BG
"""

BG        = "#1e1e2e"   # 주 배경
PANEL_BG  = "#181825"   # 패널 배경
BG2       = "#181825"   # 보조 배경 (PANEL_BG 와 동일)
BG3       = "#11111b"   # 깊은 배경

FG        = "#cdd6f4"   # 기본 텍스트
DIM       = "#9399b2"   # 흐린 텍스트
BORDER    = "#313244"   # 테두리

ACCENT    = "#89b4fa"   # 강조 (파랑)
PURPLE    = "#cba6f7"   # 보라
CYAN      = "#89dceb"   # 청록

GREEN     = "#a6e3a1"   # 양수 / 성공 / 호재
RED       = "#f38ba8"   # 음수 / 오류 / 악재
YELLOW    = "#f9e2af"   # 경고 / 불확실
ORANGE    = "#fab387"   # 주의

# 호재/악재 강도별
BULL         = "#89dceb"
BULL_STRONG  = "#74c7ec"
BEAR         = "#f38ba8"
BEAR_STRONG  = "#e64553"
NEUTRAL      = "#6c7086"
