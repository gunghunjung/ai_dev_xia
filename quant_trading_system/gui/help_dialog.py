# gui/help_dialog.py — 상세 도움말 다이얼로그 (한국어, 전체 탭/기능 포함)
import tkinter as tk
from tkinter import ttk


HELP_CONTENT = {

# ══════════════════════════════════════════════════════════════════════
"시스템 개요": """
퀀트 트레이딩 시스템 v2.0 — 뉴스 AI 통합 완전판

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

【이 프로그램이란?】
주가 예측 + 포트폴리오 최적화 + 리스크 관리를 한 곳에서 수행하는
기관급 퀀트 트레이딩 시스템입니다.
특히 뉴스/외부환경 정보를 AI로 분석해 예측 모델에 직접 반영합니다.

【핵심 철학】
• 가격 예측보다 포트폴리오 구성이 알파의 원천
• 과적합 방지 최우선 (Walk-Forward 검증, L2 정규화, Dropout)
• 확률적 출력 (μ ± σ) — 결정론적 신호 금지
• 미래 데이터 누수 완전 차단 (모든 피처는 과거 데이터만 사용)
• 뉴스 AI: 뉴스를 읽고 이해해 예측에 실질적으로 반영

【전체 처리 흐름】

  ┌──────────────────────────────────────────────────────────┐
  │  1. 데이터 수집                                          │
  │     OHLCV (yfinance) + 뉴스 (21개 RSS 소스 병렬 수집)  │
  │              ↓                                          │
  │  2. 피처 추출                                            │
  │     가격 피처: GAF 이미지 + 기술지표 시계열             │
  │     뉴스 피처: 12카테고리 × 6윈도우 → 40D 벡터          │
  │              ↓                                          │
  │  3. 하이브리드 모델 학습                                 │
  │     CNN (이미지) + Transformer (시계열) + NewsEncoder   │
  │              ↓                                          │
  │  4. 예측 → 신호 생성                                    │
  │     (μ, σ) → 확률 → UP/DOWN/HOLD                       │
  │              ↓                                          │
  │  5. 포트폴리오 최적화                                    │
  │     Risk Parity / MVO / Volatility Scaling             │
  │              ↓                                          │
  │  6. 리스크 관리 → 최종 비중                             │
  │     변동성 타겟팅 + 레짐 감지 + 킬스위치                │
  └──────────────────────────────────────────────────────────┘

【탭별 역할 한눈에 보기】
  데이터 탭    → 종목 등록, 가격 데이터 다운로드, 뉴스 수집 시작
  학습 탭      → 딥러닝 모델 학습 (뉴스 피처 자동 통합)
  예측 탭      → 등록 종목 전체 예측 + 포트폴리오 비중 산출
  백테스트 탭  → Walk-Forward 백테스트 성과 검증
  포트폴리오 탭→ 현재 권고 포트폴리오 + 리밸런싱
  외부환경 탭  → 뉴스 AI 실시간 현황 + 카테고리 분석 + 시나리오
  GPU 탭       → GPU 사용률/온도/VRAM 실시간 모니터
  히스토리 탭  → 예측 이력 및 실적 검증
  실패분석 탭  → 틀린 예측 원인 분석
  요구사항 탭  → 패키지 설치 현황 확인
  설정 탭      → 전체 파라미터 조정
""",

# ══════════════════════════════════════════════════════════════════════
"데이터 탭": """
데이터 탭 — 종목 등록 및 데이터 관리

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

【기능 목록】
  ① 종목 추가/삭제
  ② 전체 OHLCV 데이터 다운로드
  ③ 캐시 상태 확인
  ④ 뉴스 수집 제어 (신규)

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

【① 종목 추가/삭제】

  종목 코드 입력 형식:
    한국 주식: 005930.KS  (삼성전자 .KS 접미사 필수)
    미국 주식: AAPL       (심볼 그대로)
    KOSPI 지수: ^KS11
    S&P 500:   ^GSPC

  ▶ 종목 추가 버튼 또는 검색 버튼으로 종목명 검색 후 추가
  ▶ 삭제: 목록에서 선택 후 삭제 버튼
  ▶ 최대 추천 종목 수: 20개 (이상이면 학습 시간 증가)

【② OHLCV 데이터 다운로드】

  ▶ "전체 다운로드" 버튼 클릭
  ▶ 데이터 기간: 설정 탭 > 데이터 기간 설정 (기본: 5년)
  ▶ 저장 위치: cache/{종목코드}.parquet
  ▶ 캐시 유효시간: 23시간 (이후 자동 갱신)

  다운로드 소요 시간 (예상):
    5종목 / 5년: 약 10초
    20종목 / 5년: 약 40초

  ⚠ 주의: 한국 종목은 장 마감 후(16:00 이후) 당일 데이터 수집 가능

【③ 캐시 상태 확인】

  각 종목별 표시:
    ✅ 최신 (23시간 이내 갱신)
    ⚠ 오래됨 (갱신 필요)
    ❌ 없음 (다운로드 필요)

  "캐시 새로고침" 버튼: 만료된 캐시 즉시 갱신

【④ 뉴스 수집 제어】

  ▶ 뉴스는 별도 학습이 필요 없습니다.
  ▶ 수집만 해두면 학습/예측 시 자동으로 활용됩니다.

  수집 흐름:
    "뉴스 수집 시작" 클릭
      → 21개 RSS 소스 병렬 수집 (글로벌 + 한국)
      → AI 분류 → 12카테고리 이벤트 변환
      → outputs/news.db 저장
      → 15분마다 자동 갱신 (백그라운드)

  수집 소스 목록 (21개):
    글로벌: Yahoo Finance, MarketWatch, Reuters, CNBC,
            Investing.com, FedReserve Press, ECB, Oil Price...
    한국:   연합뉴스 경제, 한국경제, 매일경제, 조선비즈,
            이데일리, 뉴시스, 머니투데이, 서울경제...

  DB 현황 확인:
    "뉴스 DB 현황" 버튼 → 수집 건수, 이벤트 수, 기간 표시

  ⚠ 뉴스 수집 없이도 예측은 가능 (뉴스 피처 = 0으로 처리됨)
  ✅ 뉴스 수집 후 학습하면 더 정확한 예측 가능
""",

# ══════════════════════════════════════════════════════════════════════
"학습 탭": """
학습 탭 — 딥러닝 모델 학습

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

【뉴스 학습은 별도로 하지 않아도 됩니다!】

  기존 학습 버튼 그대로 → 뉴스 피처 자동 포함
  설정 탭에서 "뉴스 피처 사용" 체크만 하면 끝.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

【학습 절차 (5단계)】

  STEP 1. 종목 선택
    목록에서 학습할 종목 클릭 (복수 선택 가능)
    ⚠ 먼저 데이터 탭에서 해당 종목 데이터를 다운로드해야 합니다

  STEP 2. 프리셋 선택 (빠른 설정)
    빠른 학습:   Epoch=30,  Patience=5  → 테스트/확인용
    균형:        Epoch=100, Patience=15 → 기본 권장
    정밀:        Epoch=200, Patience=25 → 최고 성능
    커스텀:      직접 설정

  STEP 3. 세부 설정 조정
    ROI 세그먼트 길이: 30 (20~60)
      → 한 번에 학습할 캔들 수
    예측 기간(lookahead): 5 (1~20)
      → 몇 일 후 수익률을 맞출지
    배치 크기: 32 (GPU VRAM에 맞게 조정)
    학습률: 0.0001 (낮을수록 안정, 높을수록 빠름)

  STEP 4. 뉴스 피처 설정 (신규)
    "뉴스 피처 사용" 체크박스:
      □ OFF: 기존 가격/기술지표만으로 학습
      ☑ ON:  뉴스 AI 피처 40D 자동 추가 후 학습
               → outputs/news.db 에서 자동으로 뉴스 데이터 로드

    뉴스 피처가 추가되면 모델 구조:
      기존: CNN(128) + Transformer(256) = 384D
      신규: CNN(128) + Transformer(256) + NewsEncoder(64) = 448D

  STEP 5. 학습 시작
    "학습 시작" 버튼 클릭
    → 실시간 손실 그래프 표시
    → GPU 자동 사용 (CUDA 감지 시)
    → 조기 종료 (Patience epoch 연속 개선 없으면 자동 중단)

【학습 중 표시 항목】
  Epoch X/Y | Train Loss: 0.xxxx | Val Loss: 0.xxxx
  현재 최고 Val Loss: 0.xxxx (Epoch Z)
  경과 시간: Xs

【학습 완료 후 자동 저장】
  outputs/models/{종목코드}/latest.pt
  → 예측 탭/백테스트 탭에서 즉시 사용 가능

【Walk-Forward 검증 (고급)】
  "Walk-Forward CV" 버튼:
    → 5개 폴드로 시간 순서 교차 검증
    → 과적합 여부 확인
    → 각 폴드 Val Loss 표시

【주의사항】
  ⚠ 뉴스 피처 ON으로 학습한 모델은 뉴스 DB가 있어야 예측 가능
  ⚠ 뉴스 피처 OFF로 학습한 모델에 뉴스 피처를 추가할 수 없음
  ✅ 뉴스 DB가 없을 경우 피처 = 0으로 처리 (예측은 가능)
  ✅ GPU 없어도 학습 가능 (CPU 모드 자동 전환, 단 5~10배 느림)

【학습 시간 기준 (GTX 1660 Super 기준)】
  종목 1개, Epoch 100, 데이터 5년:
    뉴스 OFF: 약 2~5분
    뉴스 ON:  약 3~6분 (피처 로드 시간 추가)
""",

# ══════════════════════════════════════════════════════════════════════
"예측 탭": """
예측 탭 — 주가 방향/수익률 예측

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

【기능 개요】
  등록된 종목 전체에 대해 현재 시점 기준 예측 수행.
  뉴스 피처가 포함된 모델은 뉴스 영향을 자동 반영합니다.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

【예측 결과 항목 해설】

  예상 수익률 (μ):
    모델이 예측한 lookahead 기간 동안의 기대 수익률
    예: +2.3% → N일 후 평균 2.3% 상승 예측

  불확실성 (σ):
    예측의 신뢰 폭. 클수록 불확실
    예: σ=5% → 실제 수익률이 μ ± 5% 범위 내일 확률 높음

  상승 확률 (P↑):
    모델 기준 상승 확률 (0~100%)
    ≥ 60%: 매수 신호 (UP)
    ≤ 40%: 매도 신호 (DOWN)
    40~60%: 중립 (HOLD)

  SNR (신호 대 잡음비):
    |μ| / σ — 클수록 신뢰도 높음
    ≥ 1.0: HIGH confidence
    ≥ 0.5: MEDIUM confidence
    < 0.5: LOW confidence

  방향 (Direction):
    UP   🟢 — 상승 예측
    DOWN 🔴 — 하락 예측
    HOLD 🟡 — 중립

  행동 권고 (Action):
    BUY  → 포트폴리오 비중 확대 권고
    SELL → 포트폴리오 비중 축소 권고
    HOLD → 현상 유지

【뉴스 반영 예측 (신규)】

  모델 학습 시 뉴스 피처 ON으로 학습된 경우:
  → 예측 결과에 현재 뉴스 상황이 자동 반영됨

  표시 내용:
    "뉴스 반영됨" 표시 + 영향 뉴스 상위 3개 표시
    예:
      [+] 연준 금리 동결 결정 → 통화정책 카테고리 +0.42
      [-] 중동 긴장 고조 → 지정학 카테고리 -0.38
      [+] 삼성전자 실적 호조 → 기업 카테고리 +0.31

  "가격만 본 예측" vs "뉴스 반영 예측" 비교 버튼:
    → 뉴스 효과를 수치로 확인 가능

【예측 기준 시점】
  기본: 현재 시점 (최신 데이터 기준)
  과거 시점 지정: as_of_date 입력 → 특정 날짜 기준 예측

  ✅ 과거 시점 예측 시 해당 날짜 이후 뉴스는 자동 차단
  ✅ 미래 정보 누수 없음 보장

【예측 한계 및 주의사항】
  ⚠ 예측은 확률적 신호이며 투자 조언이 아닙니다
  ⚠ SNR < 0.5 이면 신호 신뢰도가 낮음
  ⚠ 뉴스 DB가 오래될수록 뉴스 피처 품질 저하
  ✅ 정기적으로 뉴스 수집을 실행하면 성능 유지
""",

# ══════════════════════════════════════════════════════════════════════
"뉴스 AI / 외부환경 탭": """
외부환경 탭 — 뉴스 AI 실시간 분석

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

【이 탭의 역할】
  수집된 뉴스를 AI가 분석한 결과를 시각화합니다.
  예측 모델이 뉴스를 어떻게 해석하는지 투명하게 보여줍니다.
  직접 시나리오를 설정해 "만약 이런 뉴스가 나온다면?" 을 시뮬레이션할 수 있습니다.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

【탭 1: 뉴스 이벤트 】

  상단 정보:
    외부환경 종합 스코어 게이지:
      -1.0 (극단 악재) ←─── 0.0 (중립) ───→ +1.0 (극단 호재)
    호재 건수 / 악재 건수 / 충격 이벤트 건수

  좌측 — 카테고리 히트맵 (12칸 4×3):
    각 카테고리의 현재 점수를 색상으로 표현
      파란색 계열: 호재 (진할수록 강한 호재)
      빨간색 계열: 악재 (진할수록 강한 악재)
      회색: 중립

    12개 카테고리:
      📊 거시경제   🏦 통화정책   ⚔ 지정학
      🏭 산업       📈 기업       🏛 정부/규제
      💰 수급       📉 시장이벤트 💻 기술
      🛢 원자재     💳 금융시장   😱 심리

  중앙 — 뉴스 이벤트 목록:
    최근 수집된 이벤트 목록 (시간 역순)
    각 줄: [방향] [카테고리] [강도] [제목]
    색상:
      🟦 파란 배경: 호재 뉴스
      🟥 빨간 배경: 악재 뉴스
      ⬜ 흰 배경: 중립 뉴스

    클릭 시 우측 상세 패널 업데이트

  우측 — 이벤트 상세 (NLP 분석 결과):
    제목/발행시각/출처
    감성 점수: -1.0 ~ +1.0
    영향 방향: 호재/악재/중립
    영향 강도: 0.0 ~ 1.0
    신뢰도:    0.0 ~ 1.0
    지속 기간: short/mid/long
    관련 종목: [005930.KS, ...]
    관련 섹터: [Technology, ...]
    키워드: [금리, 동결, 연준, ...]
    반복보도: X건 (같은 이슈 보도 수)

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

【탭 2: 카테고리 분석 】

  카테고리 바 차트:
    12개 카테고리 각각의 점수를 막대로 표시
    파란 막대: 해당 카테고리 호재 강도
    빨간 막대: 해당 카테고리 악재 강도

  32D 특징 벡터 요약:
    모델에 실제로 입력되는 숫자들을 보여줍니다
    [00] 외부환경총점  +0.312  ▓▓▓
    [01] 호재강도      +0.445  ▓▓▓▓▓
    [02] 악재강도      -0.233  ▓▓
    ...
    이 숫자들이 실제로 예측 모델에 들어갑니다.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

【탭 3: 시나리오 분석 】

  "만약 이런 뉴스가 터진다면?" 시뮬레이션

  설정 항목:
    이벤트 유형: 금리인상/금리인하/전쟁발발/실적발표 등 15종
    방향:        호재 / 악재 / 중립
    강도 슬라이더: 0.0 (미미) ~ 1.0 (충격적)
    기본 수익률: 현재 모델 예측 기본값 (%)
    이벤트 수:   1 ~ 5건

  시뮬레이션 실행:
    "▶ 시나리오 시뮬레이션 실행" 버튼

  결과 표시:
    기본 예측:  +1.2%
    이벤트 영향: -3.8%  ← 악재 이벤트 효과
    조정 예측:  -2.6%  ← 실제 모델이 볼 예상값

  "⟳ 현재 이벤트 기반 재설정":
    현재 뉴스 DB 상황으로 시나리오 초기화

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

【뉴스 AI 학습 원리 (중요!)】

  질문: "뉴스 학습을 별도로 시켜야 하나요?"
  답변: 아닙니다. 별도 학습 불필요합니다.

  작동 원리:
    ① 뉴스 수집 → outputs/news.db 자동 저장
    ② 학습 탭에서 "뉴스 피처 사용" 체크 후 일반 학습
    ③ 학습 시 각 ROI 샘플의 날짜에 맞는 뉴스 피처 자동 로드
    ④ 뉴스 피처가 모델 입력에 자동으로 포함되어 학습

  즉, 뉴스 수집 → 일반 학습 순서만 지키면 자동으로 통합됩니다.

  뉴스 피처 40D 구성:
    [0~11]  12개 카테고리별 시간감쇠 점수 (1일 기준)
    [12~17] 6개 시간창 종합 점수 (1h/4h/1d/3d/5d/20d)
    [18~23] 6개 시간창 이벤트 밀도
    [24~29] 6개 시간창 감성 평균
    [30]    종목 직접 관련 뉴스 점수
    [31]    섹터 관련 뉴스 점수
    [32]    시장 전체 리스크 점수
    [33]    정책 리스크 점수
    [34]    충격 이벤트 플래그 (0 또는 1)
    [35]    반복보도 강도
    [36]    정보 신선도 (마지막 뉴스 이후 경과)
    [37]    감성 변화 속도
    [38]    뉴스 볼륨 z-score
    [39]    누적 충격 점수 (20일치)

  시간 감쇠 수식:
    점수 = Σ(이벤트점수 × exp(-0.1 × 경과시간_h))
    → 최신 뉴스일수록 강한 영향, 오래된 뉴스는 자동 감쇠
""",

# ══════════════════════════════════════════════════════════════════════
"백테스트 탭": """
백테스트 탭 — Walk-Forward 성과 검증

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

【Walk-Forward 백테스트란?】
  미래를 모른다는 전제로 과거를 시뮬레이션합니다.
  "실제 투자와 동일한 환경"에서 검증하는 기관급 방법론입니다.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

【백테스트 설정】

  초기 자본: 기본 1억원 (변경 가능)
  학습 윈도우: 504일 (2년)
  테스트 윈도우: 126일 (6개월)
  슬라이딩 스텝: 63일 (3개월)

  거래 비용 (현실적):
    거래 수수료: 0.15% 편도
    슬리피지:    0.05%
    체결 지연:   1영업일 (신호 → 익일 체결)
    총 왕복: 약 0.4%

  리밸런싱 주기:
    daily / weekly / monthly 선택

【결과 해석 — 성과 지표】

  CAGR (연환산 수익률):
    우수: > 15%  |  양호: 8~15%  |  부족: < 8%
    (KOSPI 장기 평균 ≈ 8%)

  Sharpe Ratio (위험조정 수익률):
    우수: > 1.0  |  양호: 0.5~1.0  |  부족: < 0.5
    (헤지펀드 평균 ≈ 0.7)

  Max Drawdown (최대 낙폭):
    우수: < 10%  |  양호: 10~20%  |  주의: > 20%

  Sortino Ratio:
    하락 리스크만 고려한 Sharpe
    우수: > 1.5

  Calmar Ratio:
    CAGR / MDD  |  우수: > 0.5

  승률 (Win Rate):
    리밸런싱 기간 중 KOSPI 초과 수익 비율

【차트 화면 구성】

  ① 누적 수익 곡선 (파란선: 전략, 회색선: KOSPI 벤치마크)
  ② 드로다운 차트 (낙폭 추이)
  ③ 월별 수익률 히트맵
  ④ 롤링 Sharpe 변화

【세션 저장/불러오기】
  "세션 저장": 결과를 outputs/backtest_sessions/ 에 JSON 저장
  "세션 불러오기": 이전 결과 재열람

【과적합 경고 기준】
  OOS Sharpe / IS Sharpe < 0.5 → 과적합 의심
  조치:
    1. 피처 수 감소
    2. 모델 단순화 (레이어 축소)
    3. Dropout / Weight Decay 강화
    4. 학습 데이터 기간 확대

【뉴스 기반 백테스트 (신규)】
  뉴스 피처 ON으로 학습된 모델:
    → 백테스트 기간 중 각 날짜의 뉴스 피처 자동 로드
    → "뉴스 없음" vs "뉴스 있음" 성과 비교 가능
    → Ablation 탭에서 정량 분석 가능
""",

# ══════════════════════════════════════════════════════════════════════
"포트폴리오 탭": """
포트폴리오 탭 — 최적 포트폴리오 구성 및 관리

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

【기능 개요】
  학습된 모델의 예측 신호를 바탕으로
  최적 포트폴리오 비중을 계산하고 리밸런싱 권고를 제공합니다.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

【포트폴리오 구성 방법】

  Risk Parity (기본 권장):
    모든 종목의 리스크 기여분을 동등하게 배분
    w_i ∝ 1/σ_i  (역변동성 가중)
    장점: 안정적, 분산 효과 최대화
    단점: 모델 신호를 약하게 반영

  Mean-Variance (MVO):
    Sharpe 비율 최대화
    max Sharpe = (μᵀw - Rf) / √(wᵀΣw)
    장점: 이론적으로 최적
    단점: 공분산 추정 오류에 민감

  Volatility Scaling:
    종목별 목표 변동성으로 비중 결정
    w_i = σ_target / σ_i
    장점: 직관적, 변동성 관리 용이

  Equal Weight (벤치마크):
    1/N 단순 균등 배분
    비교용으로 사용

【제약 조건 (항상 적용)】
  종목 최대 비중: 35% (집중 리스크 방지)
  종목 최소 비중: 0% (숏 포지션 없음)
  회전율 한도:   30% (1회 리밸런싱 기준)
  상관관계 상한: 70% (유사 종목 집중 방지)

【리밸런싱 주기】
  daily:   매일 (높은 거래 비용)
  weekly:  매주 금요일 (권장)
  monthly: 매월 말

【리스크 조정 후 최종 비중】
  기본 비중 → 레짐 조정 → 변동성 타겟팅 → 최종 비중

  예시:
    기본: 삼성전자 25%, SK하이닉스 20%, NAVER 15%...
    약세장 감지 → 전체 × 0.5
    최종: 삼성전자 12.5%, SK하이닉스 10%, NAVER 7.5%...

【포트폴리오 화면 구성】
  ① 파이 차트: 현재 권고 비중
  ② 막대 차트: 종목별 비중 + 모델 신호 방향
  ③ 리밸런싱 테이블:
     종목 | 현재비중 | 목표비중 | 변화 | 권고 행동

【현금 포지션】
  리스크 관리 후 남은 비중 = 현금 보유
  예: 레짐 조정 후 총 비중 합 = 75% → 현금 25%
""",

# ══════════════════════════════════════════════════════════════════════
"히스토리 탭 / 실패분석 탭": """
히스토리 탭 / 실패분석 탭 — 예측 이력 및 성과 분석

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

【히스토리 탭】

  과거 예측 이력 조회:
    날짜 | 종목 | 예측방향 | 예측수익률 | 실제수익률 | 결과(성공/실패)

  정렬/필터:
    종목별 필터
    날짜 범위 필터
    성공/실패 필터

  집계 통계:
    전체 예측 정확도 (방향 기준)
    종목별 정확도
    기간별 정확도 추이

  자동 검증:
    예측 후 lookahead 일수가 지나면
    실제 수익률과 자동 비교 → 성공/실패 기록

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

【실패분석 탭】

  틀린 예측만 모아서 원인 분석:

  분석 항목:
    ① 실패 빈도가 높은 시장 환경
       예: 급등락 당일, 고변동성 구간, 특정 요일
    ② 실패 빈도가 높은 종목
       예: 소형주, 특정 섹터 종목
    ③ 실패 시 뉴스 환경
       예: 충격 이벤트 발생 당일 실패율 증가
    ④ 실패 시 지표 패턴
       예: σ > 5% 일 때 방향 예측 실패 많음

  권고 사항 자동 생성:
    "삼성전자 실패율 35% → 학습 데이터 확충 권고"
    "충격 이벤트 당일 신뢰도 낮음 → 신호 강도 임계값 상향 권고"

  뉴스 연관 분석:
    실패한 예측 날짜의 뉴스 환경 vs 성공한 날의 뉴스 환경 비교
    → 어떤 뉴스 카테고리에서 모델이 약한지 파악
""",

# ══════════════════════════════════════════════════════════════════════
"GPU 탭": """
GPU 탭 — 하드웨어 실시간 모니터

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

【표시 항목】

  GPU 사용률 (%):
    0~30%:   대기 중 또는 비학습 상태
    30~80%:  정상 학습 중
    80~100%: GPU 풀 활용 (최적)

  VRAM 사용량 (MB):
    학습 중 점진적 증가
    최대 VRAM의 80% 이하 유지 권장

  온도 (°C):
    < 70°C:  정상
    70~85°C: 고부하 (쿨링 상태 확인)
    > 85°C:  열 제한 가능 → 성능 저하 가능

  전력 (W):
    최대 TDP의 70~90% = 효율적 사용

  팬 속도 (%):
    자동 제어 또는 수동 설정에 따라 변동

【학습 중 OOM 발생 시 (Out Of Memory)】

  즉각 조치:
    1. 배치 크기 절반으로 감소 (32 → 16)
    2. 이미지 크기 감소 (64 → 32)
    3. d_model 감소 (256 → 128)
    4. Transformer 레이어 수 감소 (4 → 2)

  VRAM별 권장 설정:
    4GB:  배치=16, 이미지=32, d_model=128
    6GB:  배치=24, 이미지=64, d_model=128
    8GB:  배치=32, 이미지=64, d_model=256
    12GB: 배치=64, 이미지=64, d_model=256

【GPU가 없는 경우 (CPU 모드)】
  자동으로 CPU 모드로 전환됩니다.
  학습 시간: GPU 대비 5~10배 증가
  예: GPU 3분 → CPU 15~30분

  ✅ CPU 모드에서도 모든 기능 정상 작동
  ✅ 학습 결과는 동일 (속도만 차이)

【pynvml 없는 경우】
  GPU 탭에 "pynvml 미설치" 표시
  설치: pip install pynvml
  (GPU 모니터링만 안 될 뿐 학습에는 영향 없음)
""",

# ══════════════════════════════════════════════════════════════════════
"설정 탭": """
설정 탭 — 전체 파라미터 조정

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

【데이터 설정】

  데이터 기간: 1y / 2y / 5y / max
    길수록: 학습 데이터 풍부 → 학습 시간 증가
    권장: 5y

  데이터 소스: yfinance / pykrx
    yfinance: 글로벌, 안정적 (기본)
    pykrx:    한국 전용, 더 정확

  캐시 TTL: 기본 23시간
    짧게: 더 자주 갱신 (네트워크 트래픽 증가)
    길게: 오래된 데이터 사용 가능

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

【ROI 감지 설정】

  세그먼트 길이 (20~60):
    30: 기본 (약 1.5개월)
    작을수록: 빠른 패턴 학습
    클수록: 장기 패턴 학습

  Lookahead (1~20):
    5: 기본 (5일 후 수익률 예측)
    단기: 1~3일 (노이즈 많음)
    중기: 5~10일 (권장)
    장기: 10~20일 (트렌드 추종)

  변동성 임계값 (Z-score):
    낮을수록 더 많은 ROI 감지 → 학습 데이터 증가
    높을수록 더 선별적 → 중요 패턴만 학습

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

【모델 설정】

  CNN 출력 차원 (64/128/256):
    클수록: 표현력 향상, VRAM 사용량 증가

  Transformer d_model (128/256/512):
    128: 소형 모델 (빠른 학습)
    256: 기본 (권장)
    512: 대형 모델 (충분한 데이터 필요)

  Transformer 헤드 수 (4/8/16):
    d_model / 헤드수 = 각 헤드 차원 (최소 32 권장)
    d_model=256, nhead=8: 헤드당 32차원

  Transformer 레이어 수 (2~6):
    2: 빠른 학습, 단순 패턴
    4: 기본 (권장)
    6: 복잡한 패턴, 과적합 주의

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

【뉴스 AI 설정 (신규)】

  뉴스 모듈 활성화:
    ON:  뉴스 수집 + 피처 생성 활성화
    OFF: 뉴스 기능 전체 비활성화

  수집 간격 (분):
    15분: 기본 (권장)
    짧을수록: 최신 뉴스 반영 빠름, 트래픽 증가
    길수록: 뉴스 피처 신선도 감소

  시간 감쇠 계수 (λ):
    0.1: 기본 (하루 후 약 9% 영향 유지)
    클수록: 빠르게 뉴스 영향 감쇠
    작을수록: 오래된 뉴스도 영향 유지

  충격 이벤트 임계값 (0.0~1.0):
    0.75: 기본 (75% 이상 강도 = 충격)
    낮추면: 더 많은 이벤트를 충격으로 분류

  클러스터링 유사도 임계값:
    0.75: 기본 (75% 이상 유사 = 같은 이슈)
    높이면: 중복 제거 엄격 → 대표 이슈만 남음

  뉴스 피처 사용:
    ON:  학습/예측 시 뉴스 40D 피처 자동 포함
    OFF: 기존 가격/기술지표만 사용

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

【포트폴리오 설정】

  방법: risk_parity / mean_variance / vol_scaling / equal_weight
  종목 최대 비중: 0.35 (35%)
  회전율 한도:   0.30 (30%)
  목표 변동성:   0.15 (15%)
  리밸런싱 주기: daily / weekly / monthly

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

【리스크 설정】

  목표 변동성: 0.15 (15% 연환산)
  최대 낙폭 한도: 0.20 (20%)
  킬스위치 Sharpe: -0.5 (60일 롤링 Sharpe < -0.5 시 청산)
  레짐 감지: ON/OFF

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

【설정 저장/불러오기】
  모든 설정은 settings.json 에 자동 저장
  "기본값 복원" 버튼: 공장 초기화
  수동 편집: settings.json 직접 수정 가능
""",

# ══════════════════════════════════════════════════════════════════════
"모델 아키텍처": """
CNN + Transformer + NewsEncoder 하이브리드 모델

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

【전체 구조 (뉴스 포함)】

  ROI 이미지 (B, 1, 64, 64)
      ↓
  CNN Encoder → 128D
      │
  OHLCV 시계열 (B, 30, 5)
      ↓
  Transformer Encoder → 256D
      │
  뉴스 피처 (B, 40) ← [선택적]
      ↓
  NewsFeatureEncoder
    Branch A: 카테고리점수[0-11] → 32D
    Branch B: 윈도우점수[12-29] → 32D
    Branch C: 파생지표[30-39]   → 16D
    Merge:    80D → 64D
      │
  concat: 128 + 256 + 64 = 448D
      ↓
  Fusion MLP (448D → 224D → LayerNorm → GELU → Dropout)
      ↓
  Prediction Head
    μ_head: 224D → 1 (예상 수익률)
    σ_head: 224D → 1 (불확실성, Softplus)
      ↓
  (μ, σ) — 확률적 출력

【GAF (Gramian Angular Field) 이미지】

  종가 시계열 → [-1, 1] 정규화
  → arccos 변환 → 각도 시퀀스 φ
  → GASF = cos(φ_i + φ_j)  [공간 패턴 인코딩]
  → 64×64 이미지 → CNN 입력

【손실 함수: Gaussian NLL】

  L = log(σ) + (y - μ)² / (2σ²)
  σ를 함께 학습 → 불확실성 자동 추정
  고변동성 구간: σ 자동 증가 → 포지션 크기 감소

【학습 안정화 기법】

  σ 클리핑: [1e-3, 2.0] 범위 강제 (그래디언트 폭발 방지)
  손실 클리핑: 최대 100 (이상치 샘플이 학습 지배 방지)
  Gradient Clipping: 1.0
  Early Stopping: patience=15 epoch
  Pre-Norm Transformer: 깊은 레이어도 안정적 학습

【파라미터 수 기준 (기본 설정)】

  뉴스 OFF: 약 270만 파라미터
  뉴스 ON:  약 275만 파라미터 (+4만)
""",

# ══════════════════════════════════════════════════════════════════════
"ROI 기반 피처 설계": """
ROI 기반 학습 시스템

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

【ROI란?】
전체 시계열에서 학습에 의미 있는 구간만 추출하는 기법.
주식 데이터는 대부분 노이즈이며,
패턴이 명확한 구간(ROI)에서 학습하면 성능이 향상됩니다.

【ROI 감지 조건 (OR 결합)】

  1. 변동성 급증 (Rolling Volatility Z-score)
     롤링 변동성의 z-score > 임계값 (기본: 1.5)
     시장 충격, 뉴스, 이벤트로 인한 변동성 폭발 감지

  2. 가격 돌파 (Price Breakout Z-score)
     롤링 평균 대비 가격 z-score > 임계값 (기본: 2.0)
     지지/저항선 돌파, 추세 전환 감지

  3. 거래량 급증 (Volume Spike Z-score)
     롤링 log-거래량의 z-score > 임계값 (기본: 2.0)
     기관 매매, 뉴스 반응 감지

【미래 누수 방지 (CRITICAL)】
  ✅ ROI 감지: 현재 시점까지의 과거 롤링 통계만 사용
  ✅ 레이블: ROI 감지 시점 이후 N일 수익률
  ✅ 정규화: 세그먼트 내부 데이터만으로 계산
  ✅ 뉴스 피처: reference_time 이후 뉴스 완전 차단
  ❌ 금지: 미래 종가 참조, 미래 변동성 사용

【세그먼트 추출】
  고정 길이 T=30 캔들 (설정 가능)
  [t-29, t] 구간 → (30, 5) OHLCV 배열
  가격 정규화: 첫 종가 대비 상대 수익률 스케일
  거래량 정규화: log 변환 + z-score

【레이블 생성】
  연속 레이블: log(Price[t+lookahead] / Price[t])
  학습 손실: Gaussian NLL (기댓값 + 불확실성 동시 학습)
""",

# ══════════════════════════════════════════════════════════════════════
"리스크 관리": """
리스크 관리 엔진

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

【변동성 타겟팅】
  목표: 포트폴리오 연간 변동성 = 15%
  방법: Exposure = σ_target / σ_realized
  효과: 고변동성 구간 자동 포지션 축소

【최대낙폭 제한】
  현재 낙폭 > 20% → 익스포저 비례 축소
  낙폭 회복 시 자동으로 익스포저 복원

【시장 레짐 감지】
  강세장:      최근 20일 평균 수익률 > 5% → 익스포저 ×1.0
  중립:        그 외 → 익스포저 ×0.9
  고변동성:    최근 20일 변동성 > 35% → 익스포저 ×0.7
  약세장:      최근 20일 평균 수익률 < -5% → 익스포저 ×0.5

  뉴스 기반 레짐 조정 (신규):
    충격 이벤트 감지 시 → 익스포저 추가 축소
    외부환경 스코어 < -0.6 → 방어적 포지션

【킬스위치】
  트리거: 60일 롤링 Sharpe < -0.5
  동작:   전체 포지션 즉시 현금화
  해제:   수동 (GUI 버튼)

【상관관계 제어】
  상관관계 > 70% 쌍 발견 시
  더 작은 비중의 종목 비중 50% 축소
  목적: 한 섹터 집중 리스크 방지
""",

# ══════════════════════════════════════════════════════════════════════
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
    scikit-learn>=1.3.0     ← 뉴스 클러스터링용 TF-IDF
    matplotlib>=3.7.0
    yfinance>=0.2.36
    pyarrow>=12.0.0         ← Parquet 캐시용

  딥러닝 (CPU):
    torch>=2.1.0

  딥러닝 (GPU):
    pip install torch torchvision \\
      --index-url https://download.pytorch.org/whl/cu121

  GPU 모니터:
    pynvml>=11.0.0

【뉴스 AI 관련 패키지 (선택)】
  scikit-learn 설치 시:
    TF-IDF 기반 고품질 클러스터링 (20% 이상 성능 향상)

  설치 안 해도:
    자체 구현 클러스터링 자동 fallback (기능 정상)

【시스템 요구사항】
  RAM:   최소 8GB (16GB 권장)
  GPU:   VRAM 4GB 이상 (없어도 CPU로 작동)
  디스크: 3GB 이상 (데이터 캐시 + 뉴스 DB 포함)
  OS:    Windows 10/11, Linux, macOS

【주요 경로】
  outputs/news.db      ← 뉴스 SQLite DB (자동 생성)
  outputs/models/      ← 학습 모델 저장
  cache/               ← 가격 데이터 캐시
  settings.json        ← 전체 설정 파일

【실행 방법】
  python main.py
  또는
  python -m quant_trading_system

【빠른 시작 순서】
  1. pip install -r requirements.txt
  2. python main.py
  3. 데이터 탭 → 종목 추가 → 전체 다운로드
  4. 데이터 탭 → 뉴스 수집 시작 (선택, 권장)
  5. 학습 탭 → 종목 선택 → 학습 시작
  6. 예측 탭 → 전체 예측 실행
  7. 포트폴리오 탭 → 권고 비중 확인
""",

}  # end HELP_CONTENT


class HelpDialog(tk.Toplevel):
    """상세 도움말 다이얼로그"""

    def __init__(self, parent):
        super().__init__(parent)
        self.title("도움말 — 퀀트 트레이딩 시스템 완전 가이드")
        self.geometry("1100x760")
        self.configure(bg="#1e1e2e")
        self.resizable(True, True)
        self._build()

    def _build(self):
        # 제목
        tk.Label(
            self,
            text="  도움말  — 탭별/기능별 완전 가이드",
            font=("맑은 고딕", 13, "bold"),
            bg="#11111b", fg="#cba6f7",
            anchor="w", padx=12, pady=8,
        ).pack(fill="x")

        # 검색 바
        search_fr = tk.Frame(self, bg="#181825")
        search_fr.pack(fill="x", padx=8, pady=(4, 0))
        tk.Label(search_fr, text="검색:", font=("맑은 고딕", 10),
                 bg="#181825", fg="#a6adc8").pack(side="left", padx=4)
        self._search_var = tk.StringVar()
        search_entry = tk.Entry(search_fr, textvariable=self._search_var,
                                font=("맑은 고딕", 10),
                                bg="#313244", fg="#cdd6f4",
                                insertbackground="#cdd6f4", relief="flat",
                                width=30)
        search_entry.pack(side="left", padx=4, ipady=3)
        tk.Button(search_fr, text="찾기",
                  font=("맑은 고딕", 9),
                  bg="#89b4fa", fg="#1e1e2e",
                  activebackground="#74c7ec",
                  relief="flat", padx=8,
                  command=self._search).pack(side="left", padx=2)
        tk.Button(search_fr, text="다음",
                  font=("맑은 고딕", 9),
                  bg="#45475a", fg="#cdd6f4",
                  activebackground="#585b70",
                  relief="flat", padx=8,
                  command=self._search_next).pack(side="left", padx=2)
        self._match_label = tk.Label(search_fr, text="",
                                     font=("맑은 고딕", 9),
                                     bg="#181825", fg="#a6adc8")
        self._match_label.pack(side="left", padx=8)
        self._search_positions = []
        self._search_idx = 0

        # 메인 분할
        main_fr = tk.Frame(self, bg="#1e1e2e")
        main_fr.pack(fill="both", expand=True, padx=8, pady=6)

        # 좌측: 목차
        toc_fr = tk.Frame(main_fr, bg="#181825", width=220)
        toc_fr.pack(side="left", fill="y", padx=(0, 8))
        toc_fr.pack_propagate(False)

        tk.Label(toc_fr, text="목차", font=("맑은 고딕", 11, "bold"),
                 bg="#181825", fg="#89b4fa").pack(pady=(10, 4))
        ttk.Separator(toc_fr).pack(fill="x", padx=8, pady=2)

        # 목차 스크롤 가능
        toc_canvas = tk.Canvas(toc_fr, bg="#181825",
                               highlightthickness=0, width=210)
        toc_scroll = ttk.Scrollbar(toc_fr, orient="vertical",
                                    command=toc_canvas.yview)
        toc_canvas.configure(yscrollcommand=toc_scroll.set)
        toc_scroll.pack(side="right", fill="y")
        toc_canvas.pack(fill="both", expand=True)
        toc_inner = tk.Frame(toc_canvas, bg="#181825")
        toc_canvas.create_window((0, 0), window=toc_inner, anchor="nw")

        self._toc_buttons = []
        for topic in HELP_CONTENT.keys():
            btn = tk.Button(
                toc_inner, text=f"  {topic}",
                font=("맑은 고딕", 9),
                bg="#181825", fg="#cdd6f4",
                activebackground="#313244", activeforeground="#cba6f7",
                relief="flat", anchor="w",
                wraplength=190, justify="left",
                command=lambda t=topic: self._show_topic(t),
            )
            btn.pack(fill="x", padx=4, pady=1, ipady=3)
            self._toc_buttons.append((topic, btn))

        toc_inner.update_idletasks()
        toc_canvas.config(scrollregion=toc_canvas.bbox("all"))

        # 우측: 내용
        content_fr = tk.Frame(main_fr, bg="#1e1e2e")
        content_fr.pack(side="left", fill="both", expand=True)

        self.topic_label = tk.Label(
            content_fr, text="",
            font=("맑은 고딕", 12, "bold"),
            bg="#11111b", fg="#cba6f7", anchor="w",
            padx=10, pady=6,
        )
        self.topic_label.pack(fill="x")

        text_fr = tk.Frame(content_fr, bg="#1e1e2e")
        text_fr.pack(fill="both", expand=True)

        self.text = tk.Text(
            text_fr,
            font=("Consolas", 10),
            bg="#181825", fg="#cdd6f4",
            insertbackground="#cdd6f4",
            relief="flat", wrap="word",
            padx=14, pady=10,
            state="disabled",
            spacing1=1, spacing3=1,
        )
        vsb = ttk.Scrollbar(text_fr, orient="vertical",
                             command=self.text.yview)
        self.text.configure(yscrollcommand=vsb.set)
        vsb.pack(side="right", fill="y")
        self.text.pack(fill="both", expand=True)

        # 색상 태그
        self.text.tag_configure("header",
                                foreground="#89b4fa",
                                font=("Consolas", 10, "bold"))
        self.text.tag_configure("subheader",
                                foreground="#74c7ec",
                                font=("Consolas", 10, "bold"))
        self.text.tag_configure("check",
                                foreground="#a6e3a1")
        self.text.tag_configure("cross",
                                foreground="#f38ba8")
        self.text.tag_configure("note",
                                foreground="#f9e2af")
        self.text.tag_configure("highlight",
                                background="#f9e2af",
                                foreground="#1e1e2e")
        self.text.tag_configure("news",
                                foreground="#cba6f7",
                                font=("Consolas", 10, "bold"))

        # 닫기 버튼
        tk.Button(
            self, text="닫기",
            command=self.destroy,
            font=("맑은 고딕", 10),
            bg="#45475a", fg="#cdd6f4",
            activebackground="#f38ba8",
            relief="flat", pady=6,
        ).pack(fill="x", padx=8, pady=6)

        # 첫 항목 표시
        self._show_topic(list(HELP_CONTENT.keys())[0])

    def _show_topic(self, topic: str):
        self.topic_label.config(text=f"  {topic}")
        content = HELP_CONTENT.get(topic, "")
        self._render(content)

        # 활성 버튼 하이라이트
        for t, btn in self._toc_buttons:
            if t == topic:
                btn.config(bg="#313244", fg="#cba6f7",
                           font=("맑은 고딕", 9, "bold"))
            else:
                btn.config(bg="#181825", fg="#cdd6f4",
                           font=("맑은 고딕", 9))

    def _render(self, content: str):
        self.text.config(state="normal")
        self.text.delete("1.0", "end")
        self._search_positions = []
        self._match_label.config(text="")

        for line in content.split("\n"):
            stripped = line.strip()
            if "━" in line:
                self.text.insert("end", line + "\n", "header")
            elif stripped.startswith("【") and stripped.endswith("】"):
                self.text.insert("end", line + "\n", "header")
            elif stripped.startswith("【") and "】" in stripped:
                self.text.insert("end", line + "\n", "subheader")
            elif stripped.startswith("STEP ") or stripped.startswith("▶"):
                self.text.insert("end", line + "\n", "subheader")
            elif "뉴스" in stripped and ("★" in stripped or "신규" in stripped
                                         or "별도" in stripped or "자동" in stripped):
                self.text.insert("end", line + "\n", "news")
            elif stripped.startswith("✅") or "✅" in stripped[:4]:
                self.text.insert("end", line + "\n", "check")
            elif stripped.startswith("❌") or "❌" in stripped[:4]:
                self.text.insert("end", line + "\n", "cross")
            elif stripped.startswith("⚠") or stripped.startswith("  ⚠"):
                self.text.insert("end", line + "\n", "note")
            else:
                self.text.insert("end", line + "\n")

        self.text.config(state="disabled")
        self.text.see("1.0")

    def _search(self):
        query = self._search_var.get().strip()
        if not query:
            return

        self.text.config(state="normal")
        self.text.tag_remove("highlight", "1.0", "end")
        self._search_positions = []
        self._search_idx = 0

        start = "1.0"
        while True:
            pos = self.text.search(query, start, stopindex="end",
                                   nocase=True)
            if not pos:
                break
            end = f"{pos}+{len(query)}c"
            self.text.tag_add("highlight", pos, end)
            self._search_positions.append(pos)
            start = end

        self.text.config(state="disabled")

        n = len(self._search_positions)
        if n > 0:
            self._match_label.config(text=f"{n}건 발견")
            self._search_next()
        else:
            # 전체 탭에서 찾기
            for topic, _ in self._toc_buttons:
                if query.lower() in HELP_CONTENT.get(topic, "").lower():
                    self._show_topic(topic)
                    self._search()
                    return
            self._match_label.config(text="없음")

    def _search_next(self):
        if not self._search_positions:
            return
        pos = self._search_positions[self._search_idx % len(self._search_positions)]
        self.text.see(pos)
        self._search_idx += 1
        n = len(self._search_positions)
        self._match_label.config(
            text=f"{self._search_idx % n + 1}/{n}건")
