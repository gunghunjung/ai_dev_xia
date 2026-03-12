"""
HelpDialog — AI 예측 메커니즘 및 UI 컨트롤 상세 설명 다이얼로그
"""
from __future__ import annotations
from PyQt6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QTabWidget, QWidget,
    QTextBrowser, QPushButton, QLabel, QScrollArea, QFrame,
    QSizePolicy,
)
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QFont


# ──────────────────────────────────────────────────────────────────────────────
#  각 탭별 HTML 도움말 내용
# ──────────────────────────────────────────────────────────────────────────────

_STYLE = """
<style>
  body  { font-family: 'Malgun Gothic', sans-serif; font-size: 13px;
          background:#0d1117; color:#c9d1d9; margin:12px; }
  h2    { color:#58a6ff; border-bottom:1px solid #30363d;
          padding-bottom:6px; margin-top:18px; }
  h3    { color:#79c0ff; margin-top:14px; margin-bottom:4px; }
  table { border-collapse:collapse; width:100%; margin:8px 0; }
  th    { background:#161b22; color:#8b949e; padding:6px 10px;
          border:1px solid #30363d; text-align:left; font-weight:bold; }
  td    { padding:5px 10px; border:1px solid #21262d; vertical-align:top; }
  tr:nth-child(even) td { background:#161b22; }
  .tag  { display:inline-block; background:#1f6feb; color:#fff;
          border-radius:3px; padding:1px 6px; font-size:11px; margin:0 2px; }
  .warn { color:#f0883e; font-weight:bold; }
  .good { color:#3fb950; font-weight:bold; }
  .code { font-family:Consolas,monospace; background:#161b22;
          color:#79c0ff; padding:1px 5px; border-radius:3px; }
  .tip  { background:#1c2128; border-left:3px solid #388bfd;
          padding:8px 12px; margin:10px 0; border-radius:0 4px 4px 0; }
  .flow { background:#1c2128; border:1px solid #30363d; border-radius:6px;
          padding:10px 16px; margin:10px 0; font-size:12px; }
  .flow span { color:#58a6ff; font-weight:bold; }
</style>
"""

# ── 전체 흐름 탭 ─────────────────────────────────────────────────────────────
_HTML_OVERVIEW = _STYLE + """
<h2>🤖 AI 예측 전체 흐름</h2>

<div class="flow">
<span>① 데이터 로드</span> → <span>② 피처 생성</span> → <span>③ 정규화</span>
→ <span>④ 시퀀스 데이터셋</span> → <span>⑤ 딥러닝 학습</span>
→ <span>⑥ 슬라이딩 윈도우 예측</span> → <span>⑦ 신뢰구간 계산</span>
→ <span>⑧ 백테스트</span>
</div>

<h3>① 데이터 로드</h3>
<p>yfinance를 통해 종목의 OHLCV(시가·고가·저가·종가·거래량) 데이터를 가져옵니다.
캐시 파일이 있으면 즉시 로드합니다.</p>

<h3>② 피처 생성 (Feature Engineering)</h3>
<p>원시 OHLCV 데이터에서 <b>29개의 기술적 지표</b>를 자동 계산합니다.</p>
<table>
<tr><th>그룹</th><th>피처</th><th>의미</th></tr>
<tr><td><span class="tag">price</span></td>
    <td>ret, log_ret, ret5, ret20, mom10, mom20, rv5, rv20, z5, z20</td>
    <td>수익률·모멘텀·변동성·Z-score (정규화된 값)</td></tr>
<tr><td><span class="tag">ta</span></td>
    <td>sma5/20/60, ema20, bb_width, bb_pct, rsi14/7, macd/signal/hist, atr14, obv, cci20, adx14, stoch_k/d</td>
    <td>이동평균·볼린저밴드·RSI·MACD·ATR·OBV 등</td></tr>
<tr><td><span class="tag">market</span></td>
    <td>bm_return, rel_strength</td>
    <td>벤치마크 대비 상대 강도</td></tr>
</table>

<h3>③ 정규화 (RobustScaler)</h3>
<p>SMA·ATR·OBV 같은 <b>가격 스케일 피처</b>(수만 원 단위)는 그대로 신경망에 넣으면
<b>Transformer의 Softmax가 수치 오버플로우</b>를 일으켜 NaN 손실이 발생합니다.<br>
따라서 <b>중앙값(Median)과 IQR</b>을 기준으로 정규화합니다:</p>
<p class="code">&nbsp;&nbsp;정규화값 = (X − 중앙값) / IQR</p>
<div class="tip">⚠ 학습 데이터에서만 중앙값·IQR을 계산하고 (fit),
예측 시에는 동일한 값으로 변환(transform)합니다 — 미래 데이터 누수 없음.</div>

<h3>④ 시퀀스 데이터셋 (Sliding Window)</h3>
<p>과거 <b>Sequence Length</b>개 봉을 하나의 입력 윈도우로 만들어,
다음 <b>Horizon</b>일 후의 수익률(또는 종가·방향)을 예측 목표로 삼습니다.</p>
<p class="code">&nbsp;&nbsp;입력: X[i-seq : i] 형태 (seq × 피처수)<br>
&nbsp;&nbsp;목표: y[i] (horizon일 후 값)</p>

<h3>⑤ 딥러닝 학습</h3>
<p>Train / Val 분리(시간 순 유지) → 미니배치 학습 → EarlyStopping → 최적 가중치 복원</p>

<h3>⑥ 슬라이딩 윈도우 예측</h3>
<p>학습이 끝난 모델로 전체 기간을 한 칸씩 이동하며 예측값을 생성합니다.
MC Dropout이 활성화된 경우, 드롭아웃을 켜둔 채 여러 번 추론하여 불확실성을 추정합니다.</p>

<h3>⑦ 신뢰구간</h3>
<p>예측값의 95% 신뢰구간 = <span class="code">pred ± 1.96 × std</span>
(std는 MC Dropout 반복 또는 Bootstrap에서 계산)</p>

<h3>⑧ 백테스트</h3>
<p>예측 방향에 따라 매수/매도 신호를 생성하고, 실제 가격으로 수익률을 계산합니다.
수수료(0.015%)·슬리피지가 반영됩니다.</p>

<div class="tip"><b>현재 로그 분석 결과:</b><br>
• DA(방향 정확도) = <span class="good">69.9%</span> — 랜덤(50%) 대비 우수<br>
• R² = <span class="good">0.304</span> — 예측이 실제 변동의 30% 설명<br>
• MAPE = <span class="warn">164%</span> — 수익률이 0에 가까운 날이 많아 상대오차 과장됨 (무시 가능)</div>
"""

# ── 데이터 탭 ─────────────────────────────────────────────────────────────────
_HTML_DATA = _STYLE + """
<h2>📊 데이터 패널</h2>
<p>주가 데이터를 불러오는 설정입니다. <b>데이터 로드</b> 버튼을 누르면 yfinance에서
다운로드하거나 로컬 캐시에서 읽습니다.</p>

<table>
<tr><th>컨트롤</th><th>내용</th><th>예시 / 권장값</th></tr>

<tr><td><b>종목 코드</b></td>
    <td>조회할 주식의 티커 심볼.<br>
    한국 주식은 <span class="code">.KS</span>(유가증권) 또는
    <span class="code">.KQ</span>(코스닥) 접미사를 붙입니다.</td>
    <td><span class="code">064350.KS</span> (한화에어로스페이스)<br>
        <span class="code">005930.KS</span> (삼성전자)<br>
        <span class="code">AAPL</span> (애플)</td></tr>

<tr><td><b>벤치마크</b></td>
    <td>비교 지수 심볼. <span class="tag">market</span> 피처(상대 강도)를
    계산할 때 사용됩니다. 없으면 0으로 처리됩니다.</td>
    <td><span class="code">^GSPC</span> (S&amp;P500)<br>
        <span class="code">^KS11</span> (KOSPI)<br>
        <span class="code">^KQ11</span> (KOSDAQ)</td></tr>

<tr><td><b>기간 (Period)</b></td>
    <td>다운로드할 과거 데이터의 총 기간.<br>
    길수록 학습 데이터가 많아지나 오래된 패턴이 포함됩니다.</td>
    <td><span class="code">1y</span> · <span class="code">2y</span> ·
        <span class="code">5y</span> (권장) · <span class="code">10y</span></td></tr>

<tr><td><b>간격 (Interval)</b></td>
    <td>봉의 단위. 일봉이 기본값이며, 분봉은 단기간만 제공됩니다.</td>
    <td><span class="code">1d</span> (일봉, 권장)<br>
        <span class="code">1wk</span> (주봉)</td></tr>

<tr><td><b>강제 새로고침</b></td>
    <td>체크하면 캐시를 무시하고 yfinance에서 새로 다운로드합니다.<br>
    최신 데이터가 필요할 때 사용합니다.</td>
    <td>체크 해제(기본) → 캐시 우선<br>
        체크 → 항상 최신</td></tr>
</table>

<div class="tip">💡 캐시 파일은 <span class="code">data/cache/</span> 폴더에
저장됩니다. 같은 종목을 자주 분석하면 캐시로 빠르게 로드됩니다.</div>
"""

# ── 피처 탭 ──────────────────────────────────────────────────────────────────
_HTML_FEATURE = _STYLE + """
<h2>🔧 피처 패널</h2>
<p>원시 데이터를 AI 모델 입력으로 변환하는 설정입니다.
여기서의 선택이 <b>모델이 무엇을 학습하는지</b>를 결정합니다.</p>

<table>
<tr><th>컨트롤</th><th>내용</th><th>권장값 / 주의사항</th></tr>

<tr><td><b>피처 그룹</b></td>
    <td>어떤 종류의 지표를 사용할지 선택합니다.<br>
    <span class="tag">price</span> 수익률·모멘텀·변동성<br>
    <span class="tag">ta</span> RSI·MACD·볼린저밴드 등 기술적 지표<br>
    <span class="tag">market</span> 벤치마크 대비 상대 강도</td>
    <td>세 그룹 모두 선택 (기본)<br>
    피처 수: 29개</td></tr>

<tr><td><b>예측 대상 (Target Type)</b></td>
    <td>모델이 예측할 값의 종류를 지정합니다.<br>
    <b>return</b>: 수익률(%) — 가장 일반적, 정규화 불필요<br>
    <b>close</b>: 절대 종가 — 스케일이 커서 손실값도 큼<br>
    <b>direction</b>: 상승(1)/하락(0) — 분류 문제</td>
    <td><span class="good">return</span> 강력 권장<br>
    close는 Loss가 수만 단위로 커짐<br>
    direction은 Loss를 <span class="code">bce</span>로 변경 필요</td></tr>

<tr><td><b>예측 기간 (Horizon)</b></td>
    <td>몇 봉 후를 예측 목표로 삼을지 설정합니다.<br>
    horizon=5이면 5일 후 수익률을 예측합니다.</td>
    <td>1 ~ 20 사이 권장<br>
    길수록 불확실성 증가</td></tr>

<tr><td><b>Sequence Length</b></td>
    <td>모델이 한 번에 참조하는 과거 봉의 수입니다.<br>
    이 값이 곧 모델 입력의 시간 축 길이입니다.</td>
    <td>30 ~ 60 권장<br>
    너무 길면 학습 샘플 수 감소<br>
    너무 짧으면 패턴 학습 어려움</td></tr>
</table>

<div class="tip">💡 <b>Sequence Length = 60, Horizon = 5</b>의 의미:<br>
과거 60일의 데이터를 보고 → 5일 후 수익률을 예측</div>

<h3>피처 자동 생성 과정</h3>
<table>
<tr><th>지표</th><th>계산 방식</th><th>의미</th></tr>
<tr><td>ret</td><td>close.pct_change()</td><td>1일 수익률</td></tr>
<tr><td>z20</td><td>(close − SMA20) / std20</td><td>20일 평균 대비 Z-score</td></tr>
<tr><td>rsi14</td><td>RSI(14일)</td><td>과매수/과매도 0~100</td></tr>
<tr><td>bb_pct</td><td>(close−하단) / (상단−하단)</td><td>볼린저밴드 내 위치 0~1</td></tr>
<tr><td>macd</td><td>EMA12 − EMA26</td><td>추세 강도</td></tr>
<tr><td>atr14</td><td>평균 True Range(14일)</td><td>변동성 (원 단위)</td></tr>
<tr><td>obv</td><td>거래량 누적 방향합</td><td>거래량 추세</td></tr>
<tr><td>rel_strength</td><td>종목 수익률 − 벤치마크 수익률</td><td>상대 강도</td></tr>
</table>
"""

# ── 학습 탭 ──────────────────────────────────────────────────────────────────
_HTML_TRAINING = _STYLE + """
<h2>🧠 학습 패널</h2>
<p>딥러닝 모델의 구조와 학습 방법을 설정합니다.
설정을 마친 후 <b>🚀 학습 시작</b>을 누르면 백그라운드에서 학습이 진행됩니다.</p>

<h3>모델 선택</h3>
<table>
<tr><th>모델</th><th>구조</th><th>특징</th><th>권장 상황</th></tr>
<tr><td><b>lstm</b></td>
    <td>LSTM (Long Short-Term Memory)</td>
    <td>시계열의 장기 의존성 학습<br>안정적·빠름</td>
    <td>기본값, 처음 시작할 때</td></tr>
<tr><td><b>gru</b></td>
    <td>GRU (Gated Recurrent Unit)</td>
    <td>LSTM보다 파라미터 적음<br>빠른 학습</td>
    <td>데이터 적을 때</td></tr>
<tr><td><b>transformer</b></td>
    <td>Transformer Encoder<br>(Multi-Head Attention)</td>
    <td>병렬 처리 · 장거리 패턴 포착<br>파라미터 많음</td>
    <td>GPU 있을 때, 데이터 많을 때</td></tr>
<tr><td><b>temporal_cnn</b></td>
    <td>1D Dilated CNN (TCN)</td>
    <td>지역 패턴 포착 · 빠름</td>
    <td>단기 패턴 중심 분석</td></tr>
<tr><td><b>tft_like</b></td>
    <td>Temporal Fusion Transformer</td>
    <td>변수별 중요도 학습<br>설명 가능성 높음</td>
    <td>피처 중요도가 궁금할 때</td></tr>
<tr><td><b>nbeats_like</b></td>
    <td>N-BEATS 기반</td>
    <td>추세·계절성 분리<br>통계 해석 가능</td>
    <td>추세 분석 중심</td></tr>
<tr><td><b>cnn_lstm</b></td>
    <td>CNN + LSTM 하이브리드</td>
    <td>지역·순차 패턴 동시 포착</td>
    <td>단·장기 패턴 혼합</td></tr>
</table>

<h3>학습 하이퍼파라미터</h3>
<table>
<tr><th>컨트롤</th><th>의미</th><th>내부 동작</th><th>권장값</th></tr>

<tr><td><b>Epoch</b></td>
    <td>전체 학습 데이터를 몇 번 반복할지</td>
    <td>EarlyStopping이 활성화되어 있어 실제로는 더 일찍 멈출 수 있습니다.
    Patience(10) 동안 Val Loss가 개선 없으면 자동 종료</td>
    <td>50 ~ 200<br>EarlyStopping 덕분에 크게 설정해도 무방</td></tr>

<tr><td><b>Batch Size</b></td>
    <td>한 번의 가중치 업데이트에 사용할 샘플 수</td>
    <td>크면 → 안정적·느린 수렴, 많은 메모리 필요<br>
    작으면 → 잡음 많지만 빠른 수렴</td>
    <td>32 ~ 64 권장<br>GPU 메모리가 작으면 16</td></tr>

<tr><td><b>Learning Rate</b></td>
    <td>한 번의 업데이트에서 가중치를 얼마나 바꿀지</td>
    <td>OneCycle 스케줄러 사용 시: 이 값의 3배까지 올랐다가 다시 감소
    (max_lr = lr × 3)</td>
    <td>0.0005 (5e-4) 권장<br>수렴 안 되면 낮추세요</td></tr>

<tr><td><b>Hidden Dim</b></td>
    <td>모델 내부 벡터(임베딩)의 크기</td>
    <td>크면 → 표현력 증가, 과적합 위험<br>
    작으면 → 빠르지만 표현력 제한</td>
    <td>64 ~ 128 권장<br>GPU 있으면 256도 가능</td></tr>

<tr><td><b>Num Layers</b></td>
    <td>LSTM/Transformer 등의 레이어(층) 수</td>
    <td>깊을수록 복잡한 패턴 학습 가능하나 학습이 어려워짐</td>
    <td>2 ~ 3 권장</td></tr>

<tr><td><b>Dropout</b></td>
    <td>학습 중 뉴런을 무작위로 비활성화하는 비율</td>
    <td>과적합 방지 정규화 기법.<br>
    예측 시에는 자동으로 비활성화됨 (MC Dropout 제외)</td>
    <td>0.1 ~ 0.3 권장<br>데이터 적으면 높이세요</td></tr>

<tr><td><b>Loss (손실 함수)</b></td>
    <td>모델이 최소화할 오차 측정 방식</td>
    <td><b>huber</b>: 이상값에 강함 (권장)<br>
    <b>mse</b>: 큰 오차에 민감<br>
    <b>mae</b>: 중간값 예측 경향<br>
    <b>direction</b>: 방향 패널티 추가 MSE</td>
    <td><span class="good">huber</span> 강력 권장</td></tr>

<tr><td><b>Optimizer</b></td>
    <td>가중치 업데이트 알고리즘</td>
    <td><b>adamw</b>: Adam + Weight Decay (권장)<br>
    <b>adam</b>: 기본 Adam<br>
    <b>rmsprop</b>: 과거 기울기 기반 적응형</td>
    <td><span class="good">adamw</span> 권장</td></tr>

<tr><td><b>Scheduler</b></td>
    <td>학습 중 Learning Rate를 어떻게 변화시킬지</td>
    <td><b>onecycle</b>: 상승 후 하강, 빠른 수렴 (권장)<br>
    <b>cosine</b>: 코사인 곡선으로 서서히 감소<br>
    <b>plateau</b>: Val Loss 정체 시 자동 감소<br>
    <b>none</b>: 고정 LR</td>
    <td><span class="good">onecycle</span> 또는
    <span class="good">cosine</span></td></tr>

<tr><td><b>Device</b></td>
    <td>연산을 수행할 하드웨어</td>
    <td><b>auto</b>: CUDA GPU가 있으면 자동 사용<br>
    <b>cpu</b>: CPU 강제 사용<br>
    <b>cuda</b>: GPU 강제 사용 (없으면 오류)</td>
    <td><span class="good">auto</span> 권장<br>
    현재 RTX 2070 감지됨 ✅</td></tr>

<tr><td><b>Data Augmentation</b></td>
    <td>학습 데이터에 인위적 변형을 가해 다양성을 높임</td>
    <td>학습 배치에 적용되는 3가지 증강:<br>
    · 가우시안 노이즈(±0.5%) 50% 확률<br>
    · 랜덤 스케일(±5%) 30% 확률<br>
    · 피처 드롭아웃(5%) 20% 확률</td>
    <td>데이터 적을 때 체크 권장<br>
    과적합 방지 효과</td></tr>
</table>

<h3>학습 진행 상황</h3>
<table>
<tr><th>표시</th><th>의미</th></tr>
<tr><td><b>Progress Bar</b></td>
    <td>전체 Epoch 대비 진행률 (0~100%)</td></tr>
<tr><td><b>학습 로그</b></td>
    <td><span class="code">Epoch N/M train=X val=Y Zs</span><br>
    • train: 학습 데이터 손실 (작을수록 좋음)<br>
    • val: 검증 데이터 손실 (작을수록 좋음, 과적합 감지용)<br>
    • Z초: 에포크당 소요 시간</td></tr>
<tr><td><b>Early stopping at epoch N</b></td>
    <td>Val Loss가 10 에포크 연속 개선되지 않아 자동 조기 종료.<br>
    최적 가중치(가장 낮은 Val Loss)가 자동 복원됩니다.</td></tr>
</table>

<div class="tip">💡 <b>좋은 학습의 신호:</b><br>
train과 val이 함께 감소 → 정상 학습 ✅<br>
train만 감소, val 증가 → 과적합 (Dropout 올리거나 Epoch 줄이세요)<br>
둘 다 감소 안 함 → Learning Rate가 너무 작거나 모델이 너무 단순</div>
"""

# ── 예측 탭 ──────────────────────────────────────────────────────────────────
_HTML_PREDICTION = _STYLE + """
<h2>🔮 예측 패널</h2>
<p>학습된 모델로 전체 기간에 걸쳐 예측을 실행하고 결과를 확인합니다.
<b>학습 완료 후</b> 🔮 예측 실행 버튼을 누르세요.</p>

<h3>예측 요약 박스</h3>
<table>
<tr><th>항목</th><th>의미</th><th>해석 방법</th></tr>

<tr><td><b>다음 예측값 (수익률)</b></td>
    <td>가장 최근 슬라이딩 윈도우의 예측 수익률<br>
    Target Type = return일 때 해당</td>
    <td><span class="code">+0.0234</span> → 약 +2.34% 상승 예측<br>
    <span class="code">-0.0150</span> → 약 -1.50% 하락 예측</td></tr>

<tr><td><b>방향</b></td>
    <td>예측값의 부호에 따른 방향성 판단</td>
    <td>📈 상승 예측 / 📉 하락 예측</td></tr>

<tr><td><b>신뢰도 (std)</b></td>
    <td>예측의 불확실성. 표준편차(σ) 값.<br>
    MC Dropout이 꺼져 있으면 95% CI로부터 역산</td>
    <td>작을수록 예측이 안정적<br>
    <span class="good">0.01 이하</span>: 높은 신뢰<br>
    <span class="warn">0.05 이상</span>: 불확실성 높음</td></tr>

<tr><td><b>하한 / 상한 (95% CI)</b></td>
    <td>예측값의 95% 신뢰구간<br>
    실제 값이 이 범위 안에 들어올 확률 95%</td>
    <td>하한: <span class="code">pred − 1.96 × std</span><br>
    상한: <span class="code">pred + 1.96 × std</span></td></tr>
</table>

<h3>전체 예측 결과 테이블</h3>
<table>
<tr><th>열</th><th>의미</th></tr>
<tr><td><b>Date</b></td><td>해당 예측의 기준 날짜</td></tr>
<tr><td><b>Actual</b></td><td>실제 수익률 (과거 데이터는 정답 있음)</td></tr>
<tr><td><b>Pred</b></td><td>모델 예측 수익률</td></tr>
<tr><td><b>Lower</b></td><td>95% 신뢰구간 하한</td></tr>
<tr><td><b>Upper</b></td><td>95% 신뢰구간 상한</td></tr>
</table>

<h3>성능 지표 (상태바 표시)</h3>
<table>
<tr><th>지표</th><th>의미</th><th>기준</th></tr>
<tr><td><b>DA (Direction Accuracy)</b></td>
    <td>방향(상승/하락) 예측 정확도</td>
    <td><span class="warn">50%</span>: 랜덤<br>
    <span class="good">60% 이상</span>: 유의미<br>
    현재: <span class="good">69.9%</span> ✅</td></tr>
<tr><td><b>RMSE</b></td>
    <td>예측 오차의 제곱평균 제곱근<br>수익률 단위</td>
    <td>작을수록 좋음</td></tr>
<tr><td><b>R²</b></td>
    <td>예측이 실제 분산의 몇 %를 설명하는지</td>
    <td><span class="warn">0 이하</span>: 평균만 못함<br>
    <span class="good">0.3 이상</span>: 양호<br>
    현재: <span class="good">0.304</span> ✅</td></tr>
<tr><td><b>MAE</b></td>
    <td>평균 절대 오차 (수익률 단위)</td>
    <td>작을수록 좋음</td></tr>
<tr><td><b>MAPE</b></td>
    <td>평균 절대 백분율 오차.<br>
    수익률이 0에 가까운 날이 많으면 과장됨</td>
    <td>수익률 예측에서는 신뢰도 낮음<br>DA / R²를 우선 참고</td></tr>
</table>

<h3>💾 CSV 저장</h3>
<p>예측 결과를 CSV 파일로 저장합니다. 컬럼: date, actual, pred, lower, upper</p>
"""

# ── 백테스트 탭 ───────────────────────────────────────────────────────────────
_HTML_BACKTEST = _STYLE + """
<h2>📈 백테스트 패널</h2>
<p>예측 결과를 바탕으로 실제 매매를 시뮬레이션하여 전략의 수익성을 검증합니다.
<b>예측 실행 후</b> 백테스트를 실행하세요.</p>

<table>
<tr><th>컨트롤</th><th>의미</th><th>권장값</th></tr>

<tr><td><b>초기 자본</b></td>
    <td>백테스트 시작 시 보유 현금 (원)</td>
    <td>10,000,000원 (1천만 원)</td></tr>

<tr><td><b>수수료 (Fee)</b></td>
    <td>매수/매도 시 발생하는 거래 수수료 비율</td>
    <td>0.00015 (0.015%, 실제 증권사 수준)</td></tr>

<tr><td><b>슬리피지 (Slippage)</b></td>
    <td>주문 시 원하는 가격과 실제 체결 가격의 차이</td>
    <td>0.0005 (0.05%)</td></tr>

<tr><td><b>매수 임계값</b></td>
    <td>예측 수익률이 이 값 이상이면 매수 신호</td>
    <td>0.01 (1% 이상 상승 예측 시 매수)</td></tr>

<tr><td><b>매도 임계값</b></td>
    <td>예측 수익률이 이 값 이하이면 매도 신호</td>
    <td>-0.01 (-1% 이하 하락 예측 시 매도)</td></tr>

<tr><td><b>포지션 크기</b></td>
    <td>신호 발생 시 보유 현금의 몇 %를 투자할지</td>
    <td>1.0 (100%, 전량 투자)</td></tr>
</table>

<h3>백테스트 성능 지표</h3>
<table>
<tr><th>지표</th><th>의미</th><th>기준</th></tr>
<tr><td><b>CAGR</b></td>
    <td>연환산 복리 수익률</td>
    <td><span class="good">연 10% 이상</span>: 우수</td></tr>
<tr><td><b>Sharpe Ratio</b></td>
    <td>위험 대비 수익률 (수익률 / 변동성)</td>
    <td><span class="warn">1.0 미만</span>: 보통<br>
    <span class="good">2.0 이상</span>: 우수</td></tr>
<tr><td><b>MDD (최대 낙폭)</b></td>
    <td>최고점 대비 최대 손실 비율</td>
    <td><span class="good">-20% 이내</span>: 안정적</td></tr>
<tr><td><b>승률</b></td>
    <td>수익 거래 수 / 전체 거래 수</td>
    <td><span class="good">55% 이상</span>: 양호</td></tr>
</table>
"""

# ── 설정 탭 ──────────────────────────────────────────────────────────────────
_HTML_SETTINGS = _STYLE + """
<h2>⚙️ 설정 패널</h2>
<p>애플리케이션 전체에 적용되는 환경 설정입니다.</p>

<table>
<tr><th>컨트롤</th><th>의미</th><th>비고</th></tr>

<tr><td><b>다크 모드</b></td>
    <td>UI 색상 테마. 체크 시 어두운 배경으로 전환됩니다.</td>
    <td>재시작 없이 즉시 적용</td></tr>

<tr><td><b>로그 레벨</b></td>
    <td>출력할 로그의 최소 중요도를 설정합니다.<br>
    DEBUG → INFO → WARNING → ERROR 순으로 상세도 감소</td>
    <td>평소: <span class="code">INFO</span><br>
    문제 발생 시: <span class="code">DEBUG</span></td></tr>

<tr><td><b>출력 디렉토리</b></td>
    <td>모델 파일(.pt), 예측 결과, 백테스트 결과,
    실험 로그가 저장되는 폴더 경로</td>
    <td>기본: <span class="code">outputs/</span><br>
    📁 Browse 버튼으로 변경 가능</td></tr>

<tr><td><b>설정 초기화</b></td>
    <td>모든 설정을 기본값으로 되돌립니다.<br>
    <span class="warn">⚠ 되돌릴 수 없습니다!</span></td>
    <td>재시작 후 완전 적용</td></tr>
</table>

<h3>자동 저장 구조</h3>
<table>
<tr><th>파일/폴더</th><th>내용</th></tr>
<tr><td><span class="code">outputs/predictions/</span></td>
    <td>학습된 모델 가중치 파일 (.pt)<br>
    예: <span class="code">064350.KS_transformer.pt</span></td></tr>
<tr><td><span class="code">outputs/experiments/</span></td>
    <td>학습 이력(손실 곡선), 성능 지표 JSON</td></tr>
<tr><td><span class="code">outputs/backtests/</span></td>
    <td>백테스트 결과 JSON</td></tr>
<tr><td><span class="code">data/cache/</span></td>
    <td>다운로드한 주가 데이터 CSV 캐시</td></tr>
<tr><td><span class="code">logs/app.log</span></td>
    <td>전체 실행 로그</td></tr>
<tr><td><span class="code">config.json</span></td>
    <td>UI 설정값 자동 저장 (종료 시)</td></tr>
</table>
"""

# ── 트러블슈팅 탭 ─────────────────────────────────────────────────────────────
_HTML_TROUBLE = _STYLE + """
<h2>🔧 트러블슈팅 &amp; 팁</h2>

<h3>자주 발생하는 문제</h3>
<table>
<tr><th>증상</th><th>원인</th><th>해결책</th></tr>

<tr><td><span class="warn">train=nan</span></td>
    <td>Inf 피처가 LayerNorm에서 NaN 생성</td>
    <td>자동으로 해당 배치 skip됨 ✅<br>
    지속되면 데이터를 확인하세요</td></tr>

<tr><td><span class="warn">val 손실이 전혀 변하지 않음</span></td>
    <td>입력 데이터 미정규화로 모델 미학습</td>
    <td>RobustScaler 정규화가 자동 적용됨 ✅</td></tr>

<tr><td>학습이 너무 빨리 멈춤<br>(Early stopping)</td>
    <td>Patience(10) 내에 Val Loss 개선 없음</td>
    <td>Epoch를 늘리거나 LR을 낮추세요.<br>
    Scheduler를 cosine으로 바꿔보세요</td></tr>

<tr><td>에포크 23~24에서 스파이크</td>
    <td>OneCycleLR 피크 구간에서 과도한 LR</td>
    <td>이미 max_lr=lr×3으로 수정됨 ✅<br>
    그래도 발생하면 Scheduler를 cosine으로 변경</td></tr>

<tr><td>GPU인데 CPU처럼 느림</td>
    <td>Device가 cpu로 설정됨</td>
    <td>Device를 <span class="code">auto</span>로 변경<br>
    RTX 2070 감지 확인: 로그에 "GPU 감지" 출력</td></tr>

<tr><td>데이터 로드 실패</td>
    <td>네트워크 또는 잘못된 티커</td>
    <td>티커 형식 확인 (한국: .KS/.KQ 필요)<br>
    강제 새로고침 체크 후 재시도</td></tr>

<tr><td>MAPE가 100% 이상</td>
    <td>수익률이 0에 가까운 날 분모가 매우 작아짐</td>
    <td>정상입니다. DA와 R²를 보세요 ✅</td></tr>
</table>

<h3>성능 개선 팁</h3>
<div class="tip">
<b>1. 기본 설정 (빠른 실험)</b><br>
모델: lstm · Epoch: 100 · LR: 0.0005 · Hidden: 64 · Layers: 2<br>
Scheduler: cosine · Loss: huber · Optimizer: adamw
</div>
<div class="tip">
<b>2. 고성능 설정 (GPU 권장)</b><br>
모델: transformer · Epoch: 200 · LR: 0.0003 · Hidden: 128 · Layers: 3<br>
Scheduler: onecycle · Loss: huber · Data Augmentation: 켜기
</div>
<div class="tip">
<b>3. 수렴이 안 될 때</b><br>
LR을 10배 낮추세요 (0.0005 → 0.00005)<br>
Batch Size를 낮추세요 (64 → 32)<br>
Scheduler를 plateau로 변경 (자동 LR 감소)
</div>
<div class="tip">
<b>4. 과적합 시</b><br>
Dropout 올리기 (0.2 → 0.4)<br>
Data Augmentation 켜기<br>
Hidden Dim 줄이기 (128 → 64)
</div>
"""


# ──────────────────────────────────────────────────────────────────────────────
#  HelpDialog 클래스
# ──────────────────────────────────────────────────────────────────────────────

class HelpDialog(QDialog):
    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setWindowTitle("📖 도움말 — AI 예측 메커니즘 가이드")
        self.setMinimumSize(820, 640)
        self.resize(900, 700)
        self._build()

    def _build(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(6)

        # 탭 위젯
        tabs = QTabWidget()
        tabs.setDocumentMode(True)

        tab_data = [
            ("🤖 전체 흐름",   _HTML_OVERVIEW),
            ("📊 데이터",       _HTML_DATA),
            ("🔧 피처",         _HTML_FEATURE),
            ("🧠 학습",         _HTML_TRAINING),
            ("🔮 예측",         _HTML_PREDICTION),
            ("📈 백테스트",     _HTML_BACKTEST),
            ("⚙️ 설정",         _HTML_SETTINGS),
            ("🔧 트러블슈팅",   _HTML_TROUBLE),
        ]

        for title, html in tab_data:
            browser = QTextBrowser()
            browser.setOpenExternalLinks(False)
            browser.setHtml(html)
            browser.setStyleSheet(
                "QTextBrowser { background:#0d1117; color:#c9d1d9;"
                " border:none; font-size:13px; }"
                "QScrollBar:vertical { background:#0d1117; width:8px; }"
                "QScrollBar::handle:vertical { background:#30363d; border-radius:4px; }"
            )
            tabs.addTab(browser, title)

        layout.addWidget(tabs)

        # 닫기 버튼
        btn_row = QHBoxLayout()
        btn_row.addStretch()
        close_btn = QPushButton("✕  닫기")
        close_btn.setFixedWidth(90)
        close_btn.clicked.connect(self.close)
        btn_row.addWidget(close_btn)
        layout.addLayout(btn_row)

        # 다크 스타일
        self.setStyleSheet("""
            QDialog { background:#0d1117; color:#c9d1d9; }
            QTabBar::tab { background:#161b22; color:#8b949e;
                           padding:6px 14px; border:1px solid #30363d;
                           border-bottom:none; }
            QTabBar::tab:selected { background:#21262d; color:#c9d1d9; }
            QTabWidget::pane { border:1px solid #30363d; background:#0d1117; }
            QPushButton { background:#21262d; border:1px solid #30363d;
                          border-radius:4px; padding:4px 12px; color:#c9d1d9; }
            QPushButton:hover { background:#30363d; }
        """)
