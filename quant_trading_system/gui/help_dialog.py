# gui/help_dialog.py — 상세 도움말 다이얼로그 (한국어)
import tkinter as tk
from tkinter import ttk


HELP_CONTENT = {
    "시스템 개요": """
퀀트 트레이딩 시스템 — CNN + Transformer 하이브리드

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

【핵심 철학】
• 가격 예측보다 포트폴리오 구성이 알파의 원천
• 과적합 방지 최우선 (Walk-Forward 검증, L2 정규화, Dropout)
• 확률적 출력 (결정론적 신호 금지)
• 미래 데이터 누수 완전 차단 (모든 피처는 과거 데이터만 사용)

【시스템 아키텍처 (8단계)】

  1. 데이터 레이어
     └─ yfinance로 OHLCV 수집 → Parquet 캐시 (23시간 TTL)

  2. ROI 감지 레이어  [핵심]
     └─ 변동성 급증 / 가격 돌파 / 거래량 급증 구간 자동 감지
     └─ 고정 길이 세그먼트 추출 → 정규화

  3. 멀티모달 피처 추출
     ├─ CV 피처: GAF(Gramian Angular Field) 이미지 변환
     └─ TS 피처: 30+ 기술적 지표 (모멘텀, 변동성, 거래량, RSI 등)

  4. CNN + Transformer 하이브리드 모델
     ├─ CNN 인코더: 이미지 공간 패턴 추출 (ResNet-lite)
     ├─ Transformer 인코더: 시간 의존성 포착 (멀티헤드 어텐션)
     └─ 출력: (μ, σ) — 기댓값 + 불확실성 (가우시안 NLL 손실)

  5. 신호 생성 레이어
     └─ 횡단면 랭킹 / 임계값 / 확률가중 방식 중 선택
     └─ μ/σ = 신뢰도 점수 → 신호 강도 결정

  6. 포트폴리오 구성 레이어  [핵심 알파]
     ├─ Risk Parity: 위험 동등 배분
     ├─ Mean-Variance: Sharpe 최대화 (MVO)
     └─ Volatility Scaling: 목표 변동성 달성

  7. Walk-Forward 백테스트  [과적합 방지]
     └─ 비중첩 확장 윈도우 → 미래 누수 완전 차단
     └─ 거래비용 0.15% + 슬리피지 0.05% + 1일 지연

  8. 리스크 관리 엔진
     ├─ 변동성 타겟팅: 목표 변동성 대비 레버리지 자동 조정
     ├─ 최대낙폭 제한: 한도 초과 시 익스포저 자동 축소
     ├─ 레짐 감지: 강세/약세/고변동성 구분 → 노출도 조정
     └─ 킬스위치: 60일 롤링 Sharpe < -0.5 시 전체 청산
""",

    "ROI 기반 피처 설계": """
ROI (Region of Interest) 기반 학습 시스템

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

【ROI란?】
전체 시계열에서 학습에 의미 있는 구간만 추출하는 기법.
주식 데이터는 대부분 "노이즈"이며,
패턴이 명확한 구간(ROI)에서 학습하면 성능이 향상됩니다.

【ROI 감지 조건 (OR 결합)】

  1. 변동성 급증 (Rolling Volatility Z-score)
     - 롤링 변동성의 z-score > 임계값 (기본: 1.5)
     - 시장 충격, 뉴스, 이벤트로 인한 변동성 폭발 감지

  2. 가격 돌파 (Price Breakout Z-score)
     - 롤링 평균 대비 가격 z-score > 임계값 (기본: 2.0)
     - 지지/저항선 돌파, 추세 전환 감지

  3. 거래량 급증 (Volume Spike Z-score)
     - 롤링 log-거래량의 z-score > 임계값 (기본: 2.0)
     - 기관 매매, 작전, 뉴스 반응 감지

【미래 누수 방지 (CRITICAL)】
  ✅ ROI 감지: 현재 시점까지의 과거 롤링 통계만 사용
  ✅ 레이블: ROI 감지 시점 이후 N일 수익률 (ex-post)
  ✅ 정규화: 세그먼트 내부 데이터만으로 계산
  ❌ 금지: 미래 종가 참조, 미래 변동성 사용

【세그먼트 추출】
  - 고정 길이 T=30 캔들 (설정 가능)
  - [t-29, t] 구간 → (T, 5) OHLCV 배열
  - 가격 정규화: 첫 종가 대비 상대 수익률 스케일
  - 거래량 정규화: log 변환 + z-score

【레이블 생성】
  - 연속 레이블: log(Price[t+lookahead] / Price[t])
  - 학습 손실: Gaussian NLL (기댓값 + 불확실성 동시 학습)
""",

    "모델 아키텍처": """
CNN + Transformer 하이브리드 모델

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

【전체 구조】

  ROI 이미지 (B, C, H, W)
      │
      ▼
  ┌─────────────────────────────┐
  │       CNN Encoder           │  ResNet-lite
  │  Stem → [ResBlock, Down]×3  │  Residual + BatchNorm + GELU
  │  GlobalAvgPool → Linear     │  출력: (B, 128)
  └─────────────────────────────┘
      │ emb_cnn
      │
  TS 피처 시퀀스 (B, T, F)
      │
      ▼
  ┌─────────────────────────────┐
  │    Transformer Encoder      │
  │  [CLS] 토큰 + PosEnc        │  사인 코사인 포지셔널 인코딩
  │  MultiHead Attention × N    │  Pre-Norm (안정적 학습)
  │  CLS 토큰 출력              │  출력: (B, 256)
  └─────────────────────────────┘
      │ emb_ts
      │
  emb_cnn || emb_ts → (B, 384)
      │
      ▼
  ┌─────────────────────────────┐
  │       Fusion MLP            │
  │  Linear → LayerNorm → GELU  │
  │  → Dropout                  │
  └─────────────────────────────┘
      │
      ▼
  ┌─────────────────────────────┐
  │    Prediction Head          │
  │  Linear → GELU → Linear     │
  │  μ 헤드: 기댓값 (수익률)    │
  │  σ 헤드: 불확실성 (Softplus)│
  └─────────────────────────────┘
      │
      ▼
   (μ, σ) — 확률적 출력

【GAF (Gramian Angular Field)】
  1. 종가 시계열 → [-1, 1] 정규화
  2. arccos 변환 → 각도 시퀀스 φ
  3. GASF = cos(φᵢ + φⱼ)  [대칭 행렬]
  4. GADF = sin(φᵢ - φⱼ)  [반대칭 행렬]

  장점:
  - 시계열 전역 의존성 유지
  - 시간 가역성 인코딩
  - CNN으로 공간 패턴 추출 가능

【손실 함수: Gaussian NLL】
  L = 0.5 × [log(2πσ²) + (y - μ)² / σ²]

  σ를 함께 학습 → 불확실성 자동 추정
  고변동성 구간: σ 증가 → 포지션 크기 감소

【정규화 기법】
  - Dropout (0.1): 과적합 방지
  - Weight Decay (1e-5): L2 정규화
  - Gradient Clipping (1.0): 기울기 폭발 방지
  - BatchNorm / LayerNorm: 학습 안정화
  - Early Stopping (patience=15): 최적 epoch 자동 선택
  - Pre-Norm Transformer: 깊은 레이어도 안정적 학습
""",

    "포트폴리오 구성": """
포트폴리오 구성 레이어 — 핵심 알파 소스

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

【왜 포트폴리오가 핵심인가?】
  단일 종목 예측 정확도 = 55% (실현 불가)
  포트폴리오 최적화로 평균 알파 추출 가능
  "Prediction is weak → Portfolio construction is alpha"

【방법 1: Risk Parity (위험 동등 배분)】
  - 모든 종목의 리스크 기여분을 동등하게 배분
  - 역변동성 가중치 (근사): w_i ∝ 1/σ_i
  - 고변동성 종목 자동 축소, 저변동성 종목 확대
  - 장점: 직관적, 분산 효과 최대화

【방법 2: Mean-Variance (MVO)】
  - Harry Markowitz 포트폴리오 이론
  - max Sharpe = (μᵀw - rf) / √(wᵀΣw)
  - scipy 최적화 (SLSQP)
  - 단점: 공분산 추정 오류에 민감

【방법 3: Volatility Scaling】
  - 각 종목 목표 변동성으로 비중 결정
  - w_i = σ_target / σ_i (최대 비중 제한)
  - 모델 신호를 방향으로만 사용

【방법 4: Equal Weight】
  - 단순 1/N 배분
  - 벤치마크용

【제약 조건】
  - 종목 최대 비중: 35% (집중 리스크 방지)
  - 종목 최소 비중: 0% (숏 포지션 금지)
  - 회전율 한도: 30% (과도한 매매 방지)
  - 상관관계 상한: 70% (집중 위험 제거)

【포트폴리오 조정 순서】
  신호 생성 → 비중 계산 → 제약 적용 → 회전율 제한
  → 레짐 조정 → 변동성 타겟팅 → 최종 비중

【알파 소스】
  1. 모델 신호의 정보 비율 (IR)
  2. 리밸런싱 주기 최적화
  3. 변동성 타겟팅 (변동성 역추종)
  4. 상관관계 제어
  5. 레짐 전환 포착
""",

    "백테스트 방법론": """
Walk-Forward 백테스트 — 기관급 검증

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

【Walk-Forward란?】
  과거 데이터를 시간 순서대로 학습/테스트 반복
  → 실제 투자와 동일한 환경 시뮬레이션
  → 미래 데이터 누수 완전 차단

【설정값】
  학습 윈도우: 504일 (2년) ← 충분한 학습 데이터
  테스트 윈도우: 126일 (6개월) ← 성과 검증 기간
  슬라이딩 스텝: 63일 (3개월) ← 갱신 주기

【검증 과정】

  [Window 1]
  학습: 2019-01-01 ~ 2020-12-31
  테스트: 2021-01-01 ~ 2021-06-30

  [Window 2]
  학습: 2019-04-01 ~ 2021-03-31
  테스트: 2021-07-01 ~ 2021-12-31

  ... 반복 ...

【현실적 비용 모델】
  거래비용: 0.15% 편도 (수수료 + 세금)
  슬리피지: 0.05% (매수 시 +0.05%, 매도 시 -0.05%)
  체결 지연: 1영업일 (당일 신호 → 익일 체결)

  총 왕복 비용 ≈ 0.4% (연간 리밸런싱 50회면 약 20%)

【주의사항 (Look-Ahead Bias 방지)】
  ✅ 비중 계산 시 현재까지의 데이터만 사용
  ✅ 체결은 항상 미래 날짜 가격으로 (지연 반영)
  ✅ 학습 데이터에 테스트 기간 포함 금지
  ❌ 금지: 테스트 기간 데이터로 학습
  ❌ 금지: 완벽한 리밸런싱 타이밍 가정

【성과 지표 해석】

  CAGR > 10%: 우수 (KOSPI 장기 평균 ≈ 8%)
  Sharpe > 1.0: 우수 (헤지펀드 목표)
  MDD < 20%: 양호
  Sortino > 1.5: 하방 리스크 대비 우수
  Calmar > 0.5: 양호

【과적합 경고 기준】
  OOS Sharpe / IS Sharpe < 0.5 → 과적합 의심
  조치: 피처 수 감소, 모델 단순화, 정규화 강화
""",

    "리스크 관리": """
리스크 관리 엔진

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

【변동성 타겟팅】
  - 목표: 포트폴리오 연간 변동성 = 15%
  - 방법: Exposure = σ_target / σ_realized
  - σ_realized > σ_target → Exposure < 1 (레버리지 축소)
  - σ_realized < σ_target → Exposure = 1 (레버리지 ≤ 100%)
  - 효과: 고변동성 환경에서 자동으로 포지션 축소

【최대낙폭 제한】
  - 현재 낙폭 > 20% → 익스포저 비례 축소
  - 낙폭 20% 초과분의 2배만큼 추가 축소
  - 낙폭 회복 시 자동으로 익스포저 복원

【시장 레짐 감지 (3-State HMM-lite)】

  강세장 (Bull):
    - 최근 20일 평균 수익률 > 5% (연환산)
    - 최근 60일 평균 수익률 > 0%
    - 익스포저 유지 (×1.0)

  중립 (Neutral):
    - 그 외 상황
    - 익스포저 소폭 축소 (×0.9)

  고변동성 (High-Vol):
    - 최근 20일 변동성 > 35% (연환산)
    - 익스포저 대폭 축소 (×0.7)

  약세장 (Bear):
    - 최근 20일 평균 수익률 < -5% (연환산)
    - 최근 60일 평균 수익률 < 0%
    - 익스포저 절반 축소 (×0.5)

【킬스위치 (Kill-Switch)】
  트리거: 최근 60일 롤링 Sharpe < -0.5
  동작: 전체 포지션 즉시 현금화
  해제: 수동 (GUI "킬스위치 해제" 버튼)
  목적: 전략 붕괴 시 추가 손실 방지

【VaR / CVaR (역사적 시뮬레이션)】
  VaR 95%: 95% 확률로 하루 손실 이 이하
  CVaR 95%: VaR 초과 손실의 평균 (Tail Risk)

  예: VaR = -2%, CVaR = -3.5%
  → 하루 95% 확률로 2% 이상 손실 없음
  → 손실 2% 초과 시 평균 3.5% 손실

【상관관계 제어】
  - 상관관계 > 70% 쌍 발견 시
  - 더 작은 비중의 종목 비중 50% 축소
  - 목적: 집중 리스크 방지 (한 섹터 집중 금지)
""",

    "GPU 사용 가이드": """
GPU (CUDA) 사용 가이드

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

【GPU 사용 이점】
  CPU 대비 학습 속도: 10~50배 향상
  VRAM 크기에 따라 배치 크기 조정 가능

【CUDA 요구사항】
  1. NVIDIA GPU (GeForce/Quadro/Tesla)
  2. CUDA Toolkit 11.8 이상
  3. PyTorch GPU 버전 (pip install torch --index-url ...)

【GPU 모니터 항목】
  GPU 사용률 (%): CUDA 코어 사용률
    0~30%: 대기/비학습
    30~80%: 정상 학습
    80~100%: GPU 풀 활용

  VRAM 사용량 (MB):
    학습 중 점진 증가
    배치 크기 조정으로 OOM 방지

  온도 (°C):
    < 70°C: 정상
    70~85°C: 고부하
    > 85°C: 열 제한 가능 (쿨링 개선 필요)

  전력 (W):
    최대 TDP의 70~90%가 효율적

【GPU 메모리 최적화】
  배치 크기 × 이미지 크기² × 채널수 ≤ VRAM × 0.7

  VRAM 4GB: 배치 16, 이미지 32×32
  VRAM 8GB: 배치 32, 이미지 64×64
  VRAM 16GB+: 배치 64+, 이미지 64×64+

【OOM (Out of Memory) 발생 시】
  1. 배치 크기를 절반으로 감소
  2. 이미지 크기를 32로 감소
  3. d_model을 128로 감소
  4. Transformer 레이어를 2로 감소

【Multi-GPU 지원】
  현재 버전: 단일 GPU (cuda:0)
  향후: DataParallel/DDP 지원 예정

【pynvml 설치】
  pip install pynvml
  (NVIDIA 드라이버 설치 시 자동 포함되는 경우도 있음)

【PyTorch CUDA 설치】
  pip install torch torchvision --index-url \\
    https://download.pytorch.org/whl/cu121
""",

    "모델 저장/불러오기": """
모델 저장 및 불러오기

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

【저장 위치】
  outputs/models/{종목코드}/
  ├── latest.pt          ← 최신 모델 (항상 덮어씀)
  ├── model_latest_{timestamp}.pt  ← 버전 백업
  ├── checkpoint_epoch{N}.pt       ← 에폭별 체크포인트
  └── meta.json          ← 성능 지표 + 히스토리

【저장 내용】
  - model_state_dict: 가중치 파라미터
  - timestamp: 저장 시각
  - symbol: 종목 코드
  - metrics: {best_val_loss, best_epoch, train_time_s}
  - config: 모델 설정 딕셔너리

【자동 저장 시점】
  1. 각 에폭에서 검증 손실 개선 시 → checkpoint 저장
  2. 학습 완료 후 → latest.pt 저장

【불러오기 우선순위】
  포트폴리오 계산 시:
  1. outputs/models/{종목}/latest.pt 존재 → 모델 예측 사용
  2. 없으면 → 역변동성 Fallback (단순 통계 신호)

【수동 관리】
  학습 탭 → "모델 목록 새로고침": 저장된 모델 확인
  meta.json 직접 조회: 성능 히스토리 확인

【히스토리 관리】
  최대 20개 버전 자동 보관
  오래된 버전 자동 삭제 (meta.json 기준)

【체크포인트에서 재개】
  ModelStore.load_checkpoint(model, optimizer, symbol, epoch)
  → 학습 중단 후 이어서 학습 가능

【포맷】
  PyTorch .pt 파일 (pickle 기반)
  weights_only=False로 로드 (torch.load)
""",

    "설치 및 요구사항": """
설치 가이드 및 요구사항

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

【Python 버전】
  Python 3.9 이상 권장 (3.11 최적)

【필수 패키지】
  pip install -r requirements.txt

  핵심:
    numpy>=1.24.0
    pandas>=2.0.0
    scipy>=1.11.0
    scikit-learn>=1.3.0
    matplotlib>=3.7.0
    yfinance>=0.2.36
    pyarrow>=12.0.0     ← Parquet 캐시용

  딥러닝 (CPU):
    torch>=2.1.0
    torchvision>=0.16.0

  딥러닝 (GPU/CUDA):
    pip install torch torchvision --index-url \\
      https://download.pytorch.org/whl/cu121

  GPU 모니터:
    pynvml>=11.0.0

【선택적 패키지】
  xgboost>=2.0.0        ← GBM 앙상블용
  lightgbm>=4.0.0       ← GBM 앙상블용
  shap>=0.42.0          ← 피처 중요도 설명
  optuna>=3.4.0         ← 하이퍼파라미터 최적화

【시스템 요구사항】
  RAM: 최소 8GB (16GB 권장)
  GPU VRAM: 4GB 이상 (8GB 권장)
  디스크: 2GB 이상 (데이터 캐시 포함)
  OS: Windows 10/11, Linux, macOS

【실행 방법】
  python main.py
  또는
  python -m quant_trading_system

【디렉토리 구조】
  quant_trading_system/
  ├── main.py              ← 진입점
  ├── settings.json        ← 자동 생성 설정 파일
  ├── config/              ← 설정 데이터클래스
  ├── data/                ← 데이터 수집/캐시
  ├── features/            ← ROI, CV, TS 피처
  ├── models/              ← CNN, Transformer, 하이브리드
  ├── signals/             ← 신호 생성
  ├── portfolio/           ← 포트폴리오 구성
  ├── backtest/            ← Walk-Forward 백테스트
  ├── risk/                ← 리스크 관리
  ├── evaluation/          ← 성과 평가
  ├── utils/               ← GPU 모니터, 로거
  ├── gui/                 ← Tkinter GUI
  ├── cache/               ← 데이터 캐시 (자동 생성)
  └── outputs/
      ├── models/          ← 학습 모델 저장
      └── {날짜}.log       ← 로그 파일
""",
}


class HelpDialog(tk.Toplevel):
    """상세 도움말 다이얼로그"""

    def __init__(self, parent):
        super().__init__(parent)
        self.title("📖 퀀트 트레이딩 시스템 — 상세 도움말")
        self.geometry("1000x720")
        self.configure(bg="#1e1e2e")
        self.resizable(True, True)
        self._build()

    def _build(self):
        # 제목
        tk.Label(
            self, text="📖 퀀트 트레이딩 시스템 상세 도움말",
            font=("맑은 고딕", 14, "bold"),
            bg="#11111b", fg="#cba6f7",
        ).pack(fill="x", pady=(0, 0))

        # 메인 분할
        main_fr = tk.Frame(self, bg="#1e1e2e")
        main_fr.pack(fill="both", expand=True, padx=8, pady=8)

        # 좌측: 목차
        toc_fr = tk.Frame(main_fr, bg="#181825", width=200)
        toc_fr.pack(side="left", fill="y", padx=(0, 8))
        toc_fr.pack_propagate(False)

        tk.Label(toc_fr, text="목차", font=("맑은 고딕", 11, "bold"),
                 bg="#181825", fg="#89b4fa").pack(pady=8)

        self._toc_buttons = []
        for i, topic in enumerate(HELP_CONTENT.keys()):
            btn = tk.Button(
                toc_fr, text=f"  {topic}",
                font=("맑은 고딕", 10),
                bg="#181825", fg="#cdd6f4",
                activebackground="#313244", activeforeground="#cba6f7",
                relief="flat", anchor="w",
                command=lambda t=topic: self._show_topic(t),
            )
            btn.pack(fill="x", padx=4, pady=2)
            self._toc_buttons.append((topic, btn))

        # 우측: 내용
        content_fr = tk.Frame(main_fr, bg="#1e1e2e")
        content_fr.pack(side="left", fill="both", expand=True)

        self.topic_label = tk.Label(
            content_fr, text="",
            font=("맑은 고딕", 12, "bold"),
            bg="#1e1e2e", fg="#cba6f7", anchor="w",
        )
        self.topic_label.pack(fill="x", padx=4)

        self.text = tk.Text(
            content_fr,
            font=("Consolas", 10),
            bg="#181825", fg="#cdd6f4",
            insertbackground="#cdd6f4",
            relief="flat", wrap="word",
            padx=12, pady=8,
            state="disabled",
        )
        vsb = ttk.Scrollbar(content_fr, orient="vertical",
                             command=self.text.yview)
        self.text.configure(yscrollcommand=vsb.set)
        vsb.pack(side="right", fill="y")
        self.text.pack(fill="both", expand=True)

        # 색상 태그
        self.text.tag_configure("header",
                                foreground="#89b4fa", font=("Consolas", 10, "bold"))
        self.text.tag_configure("check",
                                foreground="#a6e3a1")
        self.text.tag_configure("cross",
                                foreground="#f38ba8")
        self.text.tag_configure("note",
                                foreground="#f9e2af")

        # 첫 항목 표시
        self._show_topic(list(HELP_CONTENT.keys())[0])

        # 닫기 버튼
        tk.Button(
            self, text="✕ 닫기",
            command=self.destroy,
            font=("맑은 고딕", 10),
            bg="#45475a", fg="#cdd6f4",
            activebackground="#f38ba8",
            relief="flat", pady=6,
        ).pack(fill="x", padx=8, pady=6)

    def _show_topic(self, topic: str):
        """항목 내용 표시"""
        self.topic_label.config(text=f"  {topic}")
        content = HELP_CONTENT.get(topic, "")

        self.text.config(state="normal")
        self.text.delete("1.0", "end")

        # 구문 하이라이트
        for line in content.split("\n"):
            if line.startswith("【") or "━" in line:
                self.text.insert("end", line + "\n", "header")
            elif "✅" in line:
                self.text.insert("end", line + "\n", "check")
            elif "❌" in line:
                self.text.insert("end", line + "\n", "cross")
            elif line.startswith("  ⚠") or line.startswith("  주"):
                self.text.insert("end", line + "\n", "note")
            else:
                self.text.insert("end", line + "\n")

        self.text.config(state="disabled")
        self.text.see("1.0")

        # 활성 버튼 하이라이트
        for t, btn in self._toc_buttons:
            if t == topic:
                btn.config(bg="#313244", fg="#cba6f7")
            else:
                btn.config(bg="#181825", fg="#cdd6f4")
