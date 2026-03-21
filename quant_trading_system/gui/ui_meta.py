# gui/ui_meta.py — UI 메타데이터 레지스트리
"""
모든 컨트롤의 레이블·도움말·기본값·검증 규칙을 한 곳에서 관리합니다.

사용 예:
    from gui.ui_meta import META, PRESETS, get_help, get_default

    help_text = get_help("model.d_model")
    default   = get_default("backtest.wf_train_days", mode="beginner")
"""
from __future__ import annotations
from typing import Any, Dict, Optional

# ─────────────────────────────────────────────────────────────────────────────
# 1. 컨트롤 메타데이터
#    key          : "section.field_name"  (settings.py 경로와 동일)
#    label        : UI 표시 이름 (초보자 눈높이)
#    help         : 한 줄 설명 (초보자 언어)
#    detail       : 더 자세한 설명 (툴팁에 표시)
#    default      : 일반 기본값
#    beginner     : 초보자 추천 기본값 (없으면 default 사용)
#    unit         : 단위 문자열 (표시용)
#    min_val      : 최소값 (검증용)
#    max_val      : 최대값 (검증용)
#    choices      : 콤보박스 선택지 목록
#    choice_labels: 선택지의 초보자 친화적 레이블
# ─────────────────────────────────────────────────────────────────────────────

META: Dict[str, Dict[str, Any]] = {

    # ── 데이터 설정 ──────────────────────────────────────────────────────────
    "data.period": {
        "label":    "데이터 기간",
        "help":     "과거 몇 년치 주가 데이터를 가져올지 설정합니다",
        "detail":   (
            "기간이 길수록 AI가 더 많은 데이터로 학습할 수 있습니다.\n"
            "너무 짧으면 결과가 불안정할 수 있습니다.\n\n"
            "초보자 추천: 5y (5년)\n"
            "단기 전략:   2y (2년)\n"
            "장기 분석:   10y (10년)"
        ),
        "default":  "5y",
        "beginner": "5y",
        "choices":  ["1y", "2y", "3y", "5y", "10y", "max"],
        "choice_labels": ["1년", "2년", "3년", "5년 (추천)", "10년", "전체"],
    },
    "data.interval": {
        "label":    "데이터 주기",
        "help":     "하루 단위(일봉)로 받을지, 일주일 단위(주봉)로 받을지 설정합니다",
        "detail":   (
            "일봉: 매일의 시가/고가/저가/종가 데이터\n"
            "주봉: 매주 한 번의 데이터 (변동이 적어 장기 분석에 적합)\n\n"
            "초보자 추천: 일봉 (1d)"
        ),
        "default":  "1d",
        "beginner": "1d",
        "choices":  ["1d", "1wk"],
        "choice_labels": ["일봉 — 매일 데이터 (추천)", "주봉 — 매주 데이터"],
    },
    "data.cache_ttl_hours": {
        "label":    "캐시 유효 시간 (시간)",
        "help":     "다운로드한 데이터를 다시 사용할 수 있는 시간입니다",
        "detail":   (
            "한 번 다운로드한 데이터를 저장해 두고 재사용합니다.\n"
            "이 시간이 지나면 새로 다운로드합니다.\n\n"
            "추천: 23시간 (하루 한 번 갱신)"
        ),
        "default":  23,
        "beginner": 23,
        "min_val":  1,
        "max_val":  168,
        "unit":     "시간",
    },
    "data.benchmark": {
        "label":    "비교 기준 지수",
        "help":     "전략 성과를 비교할 시장 지수입니다",
        "detail":   (
            "내 전략이 시장 전체보다 좋은지 나쁜지 비교하는 기준입니다.\n\n"
            "^KS11 = 코스피 지수 (한국 대형주 전체)\n"
            "^KQ11 = 코스닥 지수 (한국 중소형주)\n"
            "^GSPC = S&P 500 (미국 대형주 전체)"
        ),
        "default":  "^KS11",
        "beginner": "^KS11",
    },

    # ── ROI / 피처 설정 ──────────────────────────────────────────────────────
    "roi.segment_length": {
        "label":    "분석 구간 길이 (일)",
        "help":     "AI가 한 번에 분석하는 주가 데이터의 길이입니다",
        "detail":   (
            "AI는 주가 데이터를 이 길이만큼 잘라서 패턴을 찾습니다.\n"
            "너무 짧으면 패턴이 불분명하고,\n"
            "너무 길면 학습이 오래 걸립니다.\n\n"
            "초보자 추천: 30일\n"
            "단기 분석:   20일\n"
            "장기 패턴:   60일"
        ),
        "default":  30,
        "beginner": 30,
        "min_val":  10,
        "max_val":  120,
        "unit":     "거래일",
    },
    "roi.lookahead": {
        "label":    "예측 기간 (일)",
        "help":     "몇 거래일 뒤의 주가 방향을 예측할지 설정합니다",
        "detail":   (
            "AI가 '앞으로 며칠 뒤 주가가 오를지'를 예측하는 기간입니다.\n\n"
            "5일   = 1주일 뒤 예측 (단기, 초보자 추천)\n"
            "10일  = 2주일 뒤 예측\n"
            "20일  = 1개월 뒤 예측 (중기)\n\n"
            "⚠️ 이 값을 늘릴수록 예측이 어려워집니다."
        ),
        "default":  5,
        "beginner": 5,
        "min_val":  1,
        "max_val":  60,
        "unit":     "거래일",
    },
    "roi.vol_z_threshold": {
        "label":    "변동성 감지 민감도",
        "help":     "주가 변동이 얼마나 클 때 중요 구간으로 인식할지 조절합니다",
        "detail":   (
            "숫자가 클수록 큰 변동만 선택하고 (엄격),\n"
            "숫자가 작을수록 작은 변동도 포함합니다 (민감).\n\n"
            "초보자 추천: 1.5\n"
            "범위: 0.5 ~ 3.0"
        ),
        "default":  1.5,
        "beginner": 1.5,
        "min_val":  0.5,
        "max_val":  3.0,
    },

    # ── 모델 설정 ─────────────────────────────────────────────────────────────
    "model.d_model": {
        "label":    "모델 크기 (d_model)",
        "help":     "AI 모델의 내부 기억 용량입니다. 클수록 복잡한 패턴을 배울 수 있습니다",
        "detail":   (
            "AI의 '두뇌 용량'과 비슷한 개념입니다.\n"
            "값이 클수록 더 복잡한 패턴을 학습하지만\n"
            "학습 시간이 오래 걸리고, GPU 메모리를 더 사용합니다.\n\n"
            "⚠️ 반드시 '어텐션 헤드 수'의 배수여야 합니다\n"
            "   (예: d_model=256, 헤드=8 → 256÷8=32, 정수 → OK)\n\n"
            "초보자 추천: 128 (빠른 학습)\n"
            "고급 사용자: 256 이상"
        ),
        "default":  256,
        "beginner": 128,
        "min_val":  64,
        "max_val":  1024,
    },
    "model.nhead": {
        "label":    "어텐션 헤드 수 (nhead)",
        "help":     "AI가 동시에 몇 방향으로 주의를 기울일지 결정합니다",
        "detail":   (
            "AI가 '동시에 여러 관점으로 데이터를 분석'하는 수입니다.\n"
            "많을수록 다양한 패턴을 파악하지만 복잡해집니다.\n\n"
            "⚠️ 반드시 d_model의 약수여야 합니다\n"
            "   (d_model=256일 때: 1, 2, 4, 8, 16, 32, 64, 128, 256)\n\n"
            "초보자 추천: 4 (d_model=128과 함께)\n"
            "고급 사용자: 8"
        ),
        "default":  8,
        "beginner": 4,
        "min_val":  1,
        "max_val":  32,
    },
    "model.num_encoder_layers": {
        "label":    "Transformer 레이어 수",
        "help":     "AI의 분석 단계 수입니다. 많을수록 깊게 분석하지만 느려집니다",
        "detail":   (
            "AI가 데이터를 몇 번 반복해서 분석하는지입니다.\n"
            "층이 많을수록 더 정교하지만 학습 시간이 길어집니다.\n\n"
            "초보자 추천: 2 (빠른 학습)\n"
            "고급 사용자: 4"
        ),
        "default":  4,
        "beginner": 2,
        "min_val":  1,
        "max_val":  12,
    },
    "model.dropout": {
        "label":    "드롭아웃 (과적합 방지)",
        "help":     "학습 중 일부 뉴런을 무작위로 끄는 비율입니다. 과적합을 방지합니다",
        "detail":   (
            "과적합 = AI가 학습 데이터만 잘 맞추고 실제 상황에서 틀리는 현상.\n"
            "드롭아웃은 이를 방지하는 '규제' 기법입니다.\n\n"
            "0.1 = 10% 뉴런을 랜덤으로 비활성화 (추천)\n"
            "0.0 = 꺼짐 (과적합 위험)\n"
            "0.5 = 50% 비활성화 (너무 강한 규제)"
        ),
        "default":  0.1,
        "beginner": 0.1,
        "min_val":  0.0,
        "max_val":  0.5,
    },
    "model.learning_rate": {
        "label":    "학습률",
        "help":     "AI가 학습할 때 한 번에 얼마나 크게 가중치를 바꿀지 결정합니다",
        "detail":   (
            "학습률이 크면 빠르게 학습하지만 불안정하고,\n"
            "작으면 안정적이지만 오래 걸립니다.\n\n"
            "초보자 추천: 0.0001 (1e-4)\n"
            "빠른 학습:   0.001  (1e-3, 불안정할 수 있음)"
        ),
        "default":  1e-4,
        "beginner": 1e-4,
        "min_val":  1e-6,
        "max_val":  1e-1,
    },
    "model.batch_size": {
        "label":    "배치 크기",
        "help":     "한 번에 몇 개의 데이터를 묶어서 학습할지 결정합니다",
        "detail":   (
            "배치가 크면 GPU 메모리가 많이 필요하지만 안정적입니다.\n"
            "배치가 작으면 메모리를 덜 쓰지만 불안정할 수 있습니다.\n\n"
            "초보자 추천: 32\n"
            "GPU 메모리 부족 시: 16"
        ),
        "default":  32,
        "beginner": 32,
        "min_val":  4,
        "max_val":  256,
    },
    "model.epochs": {
        "label":    "최대 학습 횟수 (에폭)",
        "help":     "전체 데이터를 최대 몇 번 반복해서 학습할지 설정합니다",
        "detail":   (
            "AI가 전체 데이터를 한 번 처음부터 끝까지 보는 것이 1 에폭입니다.\n"
            "Early Stopping(조기 종료)이 설정되어 있어 개선이 없으면 자동으로 멈춥니다.\n\n"
            "초보자 추천: 50 (빠른 실험)\n"
            "정밀 학습:   100~200"
        ),
        "default":  100,
        "beginner": 50,
        "min_val":  5,
        "max_val":  1000,
        "unit":     "회",
    },
    "model.patience": {
        "label":    "조기 종료 인내 횟수",
        "help":     "성능 개선이 없을 때 몇 번까지 기다리다 학습을 멈출지 설정합니다",
        "detail":   (
            "Early Stopping: 일정 횟수 동안 검증 성능이 좋아지지 않으면\n"
            "자동으로 학습을 멈추는 기법입니다.\n"
            "과적합을 방지하고 학습 시간을 절약합니다.\n\n"
            "초보자 추천: 10\n"
            "안정적 학습: 15~20"
        ),
        "default":  15,
        "beginner": 10,
        "min_val":  3,
        "max_val":  50,
        "unit":     "에폭",
    },
    "model.image_size": {
        "label":    "이미지 크기 (GAF 변환)",
        "help":     "주가 데이터를 이미지로 변환할 때의 해상도입니다",
        "detail":   (
            "AI는 주가 데이터를 이미지로 변환해서 분석합니다.\n"
            "크기가 클수록 세밀하지만 느려집니다.\n\n"
            "초보자 추천: 48 (빠른 학습)\n"
            "고해상도:   64"
        ),
        "default":  64,
        "beginner": 48,
        "min_val":  16,
        "max_val":  128,
        "unit":     "px",
    },

    # ── 포트폴리오 설정 ───────────────────────────────────────────────────────
    "portfolio.method": {
        "label":    "포트폴리오 구성 방법",
        "help":     "각 종목에 얼마씩 투자할 비중을 어떻게 결정할지 선택합니다",
        "detail":   (
            "risk_parity  (위험 균등): 각 종목의 위험도에 따라 비중 조절 → 안전함 (추천)\n"
            "equal_weight (균등 분배): 모든 종목에 같은 비중 → 단순함\n"
            "mean_variance(기대수익 최적화): 수익 최대·위험 최소 계산 → 고급 기법\n"
            "vol_scaling  (변동성 조정): 변동성이 낮은 종목에 더 투자"
        ),
        "default":  "risk_parity",
        "beginner": "equal_weight",
        "choices":  ["risk_parity", "equal_weight", "mean_variance", "vol_scaling"],
        "choice_labels": [
            "위험 균등 (Risk Parity) — 안정적",
            "균등 분배 — 단순함 (초보자 추천)",
            "기대수익 최적화 — 고급",
            "변동성 비례 조정 — 고급",
        ],
    },
    "portfolio.max_weight": {
        "label":    "최대 비중 (한 종목 최대)",
        "help":     "한 종목에 최대 몇 %까지 투자할 수 있는지 제한합니다",
        "detail":   (
            "한 종목에 너무 많이 투자하면 그 종목이 폭락할 때 큰 손실이 납니다.\n"
            "이 값으로 분산 투자를 강제할 수 있습니다.\n\n"
            "초보자 추천: 0.35 (최대 35%)\n"
            "보수적:     0.20 (최대 20%)"
        ),
        "default":  0.35,
        "beginner": 0.35,
        "min_val":  0.05,
        "max_val":  1.0,
        "unit":     "(0~1, 예: 0.35 = 35%)",
    },
    "portfolio.rebalance_freq": {
        "label":    "리밸런싱 주기",
        "help":     "포트폴리오 비중을 다시 조정하는 주기입니다",
        "detail":   (
            "시간이 지나면 종목별 비중이 변합니다.\n"
            "리밸런싱은 목표 비중으로 다시 맞추는 작업입니다.\n\n"
            "daily   (매일):   거래 비용이 많이 발생할 수 있음\n"
            "weekly  (매주):   초보자 추천 — 적절한 균형\n"
            "monthly (매월):   장기 투자에 적합, 비용 절약"
        ),
        "default":  "weekly",
        "beginner": "weekly",
        "choices":  ["daily", "weekly", "monthly"],
        "choice_labels": ["매일 — 비용 많음", "매주 — 추천", "매월 — 장기 투자"],
    },

    # ── 백테스트 설정 ──────────────────────────────────────────────────────
    "backtest.initial_capital": {
        "label":    "초기 자본금 (원)",
        "help":     "백테스트를 시작할 때의 가상 자본금입니다",
        "detail":   (
            "실제 투자금이 아닌 가상 금액으로 전략을 테스트합니다.\n\n"
            "기본: 100,000,000원 (1억원)\n"
            "소규모 테스트: 10,000,000원 (1천만원)"
        ),
        "default":  100_000_000,
        "beginner": 100_000_000,
        "min_val":  1_000_000,
        "unit":     "원",
    },
    "backtest.transaction_cost": {
        "label":    "거래 비용 (%)",
        "help":     "주식을 사고팔 때 발생하는 수수료 비율입니다",
        "detail":   (
            "실제 증권사 수수료 + 세금을 반영합니다.\n\n"
            "한국 주식: 약 0.15% (수수료 0.015% + 증권거래세 0.1~0.2%)\n"
            "0.15 입력 = 0.15%\n\n"
            "이 값을 너무 낮게 설정하면 실제보다 좋은 결과가 나옵니다."
        ),
        "default":  0.15,
        "beginner": 0.15,
        "min_val":  0.0,
        "max_val":  2.0,
        "unit":     "%",
    },
    "backtest.slippage": {
        "label":    "슬리피지 (%)",
        "help":     "주문한 가격과 실제 체결 가격의 차이입니다",
        "detail":   (
            "실제 매수/매도 시 원하는 가격에 정확히 체결되지 않습니다.\n"
            "이 차이(슬리피지)를 백테스트에 반영합니다.\n\n"
            "한국 주식 추천: 0.05%\n"
            "거래량 적은 소형주: 0.1~0.3%"
        ),
        "default":  0.05,
        "beginner": 0.05,
        "min_val":  0.0,
        "max_val":  1.0,
        "unit":     "%",
    },
    "backtest.wf_train_days": {
        "label":    "학습 윈도우 (거래일)",
        "help":     "전략을 학습할 때 사용하는 과거 데이터 기간입니다",
        "detail":   (
            "Walk-Forward 백테스트에서 전략을 학습하는 구간입니다.\n\n"
            "504일 ≈ 2년치 거래일\n"
            "초보자 추천: 504일 (2년)\n\n"
            "너무 짧으면 학습 데이터가 부족하고,\n"
            "너무 길면 오래된 패턴에 의존할 수 있습니다."
        ),
        "default":  504,
        "beginner": 504,
        "min_val":  120,
        "max_val":  2520,
        "unit":     "거래일 (약 504일 = 2년)",
    },
    "backtest.wf_test_days": {
        "label":    "테스트 윈도우 (거래일)",
        "help":     "학습한 전략을 실제로 테스트하는 기간입니다",
        "detail":   (
            "Walk-Forward에서 학습 구간 이후의 '미래 데이터'로 성과를 검증합니다.\n"
            "이 구간은 학습에 사용하지 않으므로 공정한 평가가 됩니다.\n\n"
            "126일 ≈ 반년치 거래일\n"
            "초보자 추천: 126일"
        ),
        "default":  126,
        "beginner": 126,
        "min_val":  20,
        "max_val":  504,
        "unit":     "거래일 (약 126일 = 반년)",
    },
    "backtest.wf_step_days": {
        "label":    "슬라이딩 간격 (거래일)",
        "help":     "각 학습-테스트 구간을 얼마씩 이동하면서 반복할지 결정합니다",
        "detail":   (
            "학습→테스트 사이클을 이 간격만큼 앞으로 이동하며 반복합니다.\n"
            "더 많은 검증 구간을 확보할 수 있습니다.\n\n"
            "63일 ≈ 분기(3개월)\n"
            "초보자 추천: 63일"
        ),
        "default":  63,
        "beginner": 63,
        "min_val":  10,
        "max_val":  252,
        "unit":     "거래일 (약 63일 = 분기)",
    },

    # ── 리스크 설정 ───────────────────────────────────────────────────────────
    "risk.vol_target": {
        "label":    "목표 변동성",
        "help":     "포트폴리오의 연간 변동성 목표값입니다",
        "detail":   (
            "포트폴리오 전체의 '흔들림 정도'를 이 수준으로 유지합니다.\n\n"
            "0.15 = 연간 15% 변동성 목표 (중간 정도 위험)\n"
            "0.10 = 연간 10% (보수적)\n"
            "0.20 = 연간 20% (공격적)"
        ),
        "default":  0.15,
        "beginner": 0.15,
        "min_val":  0.01,
        "max_val":  0.50,
        "unit":     "(0~1, 예: 0.15 = 15%)",
    },
    "risk.max_drawdown_limit": {
        "label":    "최대 낙폭 한도",
        "help":     "포트폴리오 손실이 이 비율을 넘으면 거래를 멈춥니다",
        "detail":   (
            "최대 낙폭(MDD) = 최고점에서 최저점까지 떨어진 비율.\n"
            "이 값 이상 손실이 나면 자동으로 포지션을 줄입니다.\n\n"
            "0.20 = 20% 이상 손실 시 안전 모드 진입\n"
            "초보자 추천: 0.20"
        ),
        "default":  0.20,
        "beginner": 0.20,
        "min_val":  0.05,
        "max_val":  0.80,
        "unit":     "(0~1, 예: 0.20 = 20%)",
    },
}


# ─────────────────────────────────────────────────────────────────────────────
# 2. 프리셋 템플릿
#    각 프리셋은 settings 키에 매핑되는 값의 딕셔너리
# ─────────────────────────────────────────────────────────────────────────────

PRESETS: Dict[str, Dict] = {
    "beginner": {
        "name":        "🟢 초보자 추천",
        "description": (
            "처음 시작하는 분을 위한 설정입니다.\n"
            "빠른 학습, 안전한 기본값, 낮은 복잡도로 결과를 쉽게 확인할 수 있습니다."
        ),
        "settings": {
            "data.period":                  "5y",
            "data.interval":                "1d",
            "roi.segment_length":           30,
            "roi.lookahead":                5,
            "model.d_model":                128,
            "model.nhead":                  4,
            "model.num_encoder_layers":     2,
            "model.dropout":                0.1,
            "model.learning_rate":          1e-4,
            "model.batch_size":             32,
            "model.epochs":                 50,
            "model.patience":               10,
            "model.image_size":             48,
            "portfolio.method":             "equal_weight",
            "portfolio.max_weight":         0.35,
            "portfolio.rebalance_freq":     "weekly",
            "backtest.initial_capital":     100_000_000,
            "backtest.transaction_cost":    0.0015,
            "backtest.slippage":            0.0005,
            "backtest.wf_train_days":       504,
            "backtest.wf_test_days":        126,
            "backtest.wf_step_days":        63,
            "risk.max_drawdown_limit":      0.20,
        },
    },
    "short_term": {
        "name":        "⚡ 단기 전략",
        "description": (
            "1~2주 단위의 단기 예측에 최적화된 설정입니다.\n"
            "빠른 반응, 짧은 예측 기간, 자주 리밸런싱합니다."
        ),
        "settings": {
            "data.period":              "2y",
            "roi.segment_length":       20,
            "roi.lookahead":            3,
            "model.d_model":            128,
            "model.nhead":              4,
            "model.num_encoder_layers": 2,
            "model.epochs":             50,
            "portfolio.rebalance_freq": "daily",
            "backtest.wf_train_days":   252,
            "backtest.wf_test_days":    63,
            "backtest.wf_step_days":    21,
        },
    },
    "mid_term": {
        "name":        "📈 중기 전략",
        "description": (
            "1개월 단위의 중기 예측에 적합한 설정입니다.\n"
            "균형 잡힌 설정으로 안정적인 결과를 기대할 수 있습니다."
        ),
        "settings": {
            "data.period":              "5y",
            "roi.segment_length":       30,
            "roi.lookahead":            10,
            "model.d_model":            256,
            "model.nhead":              8,
            "model.num_encoder_layers": 4,
            "model.epochs":             100,
            "portfolio.rebalance_freq": "weekly",
            "backtest.wf_train_days":   504,
            "backtest.wf_test_days":    126,
            "backtest.wf_step_days":    63,
        },
    },
    "stable": {
        "name":        "🛡️ 안정형 테스트",
        "description": (
            "위험을 최소화하고 안정성을 최우선으로 하는 설정입니다.\n"
            "MDD(최대낙폭)와 변동성 제한을 강하게 설정합니다."
        ),
        "settings": {
            "data.period":                  "5y",
            "roi.lookahead":                5,
            "portfolio.method":             "risk_parity",
            "portfolio.max_weight":         0.20,
            "portfolio.rebalance_freq":     "monthly",
            "risk.vol_target":              0.10,
            "risk.max_drawdown_limit":      0.15,
            "backtest.transaction_cost":    0.0015,
            "backtest.slippage":            0.001,
        },
    },
    "advanced": {
        "name":        "🔬 고급 사용자",
        "description": (
            "전문 퀀트를 위한 고성능 설정입니다.\n"
            "더 큰 모델, 더 긴 학습, 더 정밀한 백테스트를 수행합니다.\n"
            "⚠️ GPU 및 고성능 PC가 필요합니다."
        ),
        "settings": {
            "data.period":              "10y",
            "roi.segment_length":       60,
            "roi.lookahead":            20,
            "model.d_model":            512,
            "model.nhead":              8,
            "model.num_encoder_layers": 6,
            "model.dropout":            0.1,
            "model.epochs":             200,
            "model.patience":           20,
            "model.image_size":         64,
            "portfolio.method":         "mean_variance",
            "backtest.wf_train_days":   756,
            "backtest.wf_test_days":    252,
            "backtest.wf_step_days":    63,
        },
    },
}


# ─────────────────────────────────────────────────────────────────────────────
# 3. 편의 함수
# ─────────────────────────────────────────────────────────────────────────────

def get_meta(key: str) -> Dict[str, Any]:
    """메타데이터 딕셔너리 반환 (없으면 빈 dict)"""
    return META.get(key, {})


def get_help(key: str) -> str:
    """짧은 도움말 문자열 반환"""
    return META.get(key, {}).get("help", "")


def get_detail(key: str) -> str:
    """툴팁용 상세 설명 반환"""
    return META.get(key, {}).get("detail", get_help(key))


def get_label(key: str, fallback: str = "") -> str:
    """라벨 문자열 반환"""
    return META.get(key, {}).get("label", fallback)


def get_default(key: str, mode: str = "normal") -> Any:
    """
    기본값 반환.
    mode='beginner' 이면 초보자 추천값 우선 반환.
    """
    m = META.get(key, {})
    if mode == "beginner" and "beginner" in m:
        return m["beginner"]
    return m.get("default", "")


def get_choice_labels(key: str) -> Optional[list]:
    """콤보박스용 친화적 레이블 목록 (없으면 None)"""
    return META.get(key, {}).get("choice_labels", None)


def apply_preset(preset_name: str, settings_obj) -> None:
    """
    프리셋을 settings 객체에 적용합니다.

    Parameters
    ----------
    preset_name  : PRESETS 딕셔너리의 키 (예: "beginner")
    settings_obj : AppSettings 인스턴스
    """
    preset = PRESETS.get(preset_name)
    if not preset:
        return

    for dotted_key, value in preset["settings"].items():
        parts = dotted_key.split(".", 1)
        if len(parts) != 2:
            continue
        section_name, field_name = parts
        section = getattr(settings_obj, section_name, None)
        if section is not None and hasattr(section, field_name):
            setattr(section, field_name, value)


# 백테스트 결과 지표 설명 (결과 화면에서 사용)
METRIC_DESCRIPTIONS: Dict[str, Dict[str, str]] = {
    "total_return": {
        "label": "총 수익률",
        "simple": "처음 투자금 대비 최종 수익 비율",
        "good":   "높을수록 좋음",
        "warn":   "너무 높으면 과적합일 수 있음",
    },
    "cagr": {
        "label":  "연평균 수익률 (CAGR)",
        "simple": "1년 평균 몇 % 수익이 났는지",
        "good":   "코스피 연평균(약 8~10%)보다 높으면 시장 이기기 성공",
        "warn":   None,
    },
    "annual_volatility": {
        "label":  "연간 변동성",
        "simple": "1년 동안 수익률이 얼마나 흔들렸는지",
        "good":   "낮을수록 안정적",
        "warn":   "20% 이상이면 꽤 위험한 전략",
    },
    "max_drawdown": {
        "label":  "최대 낙폭 (MDD)",
        "simple": "가장 많이 손실이 났을 때 얼마나 떨어졌는지",
        "good":   "작을수록(절댓값) 안전",
        "warn":   "-30% 이하면 매우 위험",
    },
    "sharpe_ratio": {
        "label":  "샤프 지수 (Sharpe Ratio)",
        "simple": "위험 대비 수익이 얼마나 효율적인지",
        "good":   "1.0 이상이면 양호, 2.0 이상이면 훌륭",
        "warn":   "0.5 미만이면 위험 대비 수익이 낮음",
    },
    "sortino_ratio": {
        "label":  "소르티노 지수 (Sortino)",
        "simple": "하락 위험만 고려한 효율성 지수",
        "good":   "샤프 지수보다 실용적인 지표, 1.0 이상 권장",
        "warn":   None,
    },
    "calmar_ratio": {
        "label":  "칼마 지수 (Calmar)",
        "simple": "최대 낙폭 대비 수익률",
        "good":   "1.0 이상이면 낙폭 대비 수익 양호",
        "warn":   None,
    },
    "win_rate": {
        "label":  "승률",
        "simple": "매매 중 수익이 난 비율",
        "good":   "50% 이상이면 수익 거래가 더 많음",
        "warn":   "승률이 낮아도 손익비가 높으면 전체 수익은 플러스일 수 있음",
    },
    "payoff_ratio": {
        "label":  "손익비",
        "simple": "이길 때 평균 수익 vs 질 때 평균 손실의 비율",
        "good":   "1.5 이상이면 이길 때 손실의 1.5배 이상 번다는 의미",
        "warn":   None,
    },
    "var_95": {
        "label":  "VaR 95% (최대 예상 손실)",
        "simple": "95% 확률로 하루 손실이 이 값 이내",
        "good":   "절댓값이 작을수록 안전",
        "warn":   None,
    },
    "n_windows": {
        "label":  "검증 구간 수",
        "simple": "워크포워드 검증에서 사용된 독립 구간 수",
        "good":   "많을수록 결과를 더 신뢰할 수 있음",
        "warn":   "3개 미만이면 신뢰도 낮음",
    },
}
