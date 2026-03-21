"""
gui/help_dialog.py — 상세 사용법 도움말 다이얼로그 (한글)

탭 구성:
  빠른 시작    — 처음 사용자를 위한 3단계 가이드
  소스 설정    — 비디오/카메라/이미지 시퀀스 상세 설명
  ROI 관리     — ROI 개념 및 그리기 방법
  AI 분석      — 모델 종류 및 판정 로직 설명
  이상 감지    — 6가지 이상 패턴 및 판정 규칙
  로그 & 내보내기 — 저장 위치 및 CSV/JSON 활용법
  단축키 & 팁  — 키보드 단축키 및 트러블슈팅
"""

from __future__ import annotations

import tkinter as tk
from tkinter import ttk


# ---------------------------------------------------------------------------
# 도움말 내용 정의
# ---------------------------------------------------------------------------

HELP_CONTENT: dict[str, list[tuple[str, str]]] = {
    "⚡ 빠른 시작": [
        ("STEP 1 — 소스 선택", """\
[설정] 탭 → [비디오 소스] 섹션에서 입력 방식을 선택합니다.

  • 파일     : MP4, AVI, MOV, MKV 등 동영상 파일
  • 카메라   : USB 웹캠 또는 캡처 카드 (번호 0=기본 카메라)
  • 이미지 폴더 : JPG/PNG 이미지가 들어 있는 디렉터리

파일/폴더 경로는 [찾기…] 버튼으로 탐색할 수 있습니다."""),

        ("STEP 2 — ROI 지정", """\
[ROI 관리] 탭에서 분석할 관심 영역을 지정합니다.

방법 A — 마우스 드래그:
  1. [✎ ROI 그리기] 클릭
  2. 비디오 화면에서 원하는 영역을 드래그
  3. 마우스를 놓으면 ROI가 자동 등록됩니다.

방법 B — 좌표 직접 입력:
  하단 [좌표 직접 입력] 섹션에서 X, Y, 너비, 높이 값을 입력 후
  [+ 추가] 클릭.

여러 개의 ROI를 동시에 등록하여 복수 영역을 동시 분석할 수 있습니다."""),

        ("STEP 3 — 분석 시작", """\
[설정] 탭 → [▶ 시작] 버튼을 누르면 분석이 시작됩니다.

  • 분석 결과는 비디오 화면에 실시간으로 오버레이됩니다.
  • [◉ 상태] 탭에서 시스템 상태와 ROI별 수치를 확인하세요.
  • 이상이 감지되면 자동으로 [⚠ 이벤트] 탭에 기록됩니다.
  • [■ 정지] 버튼을 누르면 분석이 중단되고 로그가 저장됩니다."""),
    ],

    "📁 소스 설정": [
        ("파일 소스 (권장)", """\
지원 형식: MP4, AVI, MOV, MKV, WMV, MPEG 등 OpenCV가 지원하는 모든 형식

옵션:
  반복 재생  — 영상 끝에 도달하면 처음부터 다시 재생
  목표 FPS   — 분석 속도를 제한 (실제 영상 FPS보다 낮게 설정 가능)
  최대 프레임 — 지정한 프레임 수 이후 자동 정지 (비워두면 무제한)

팁: 고해상도(4K 이상) 파일은 [화면 표시] → 소스 FPS를 낮춰 사용하면
    CPU 부하를 줄일 수 있습니다."""),

        ("카메라 소스", """\
카메라 번호:
  0 = 노트북 내장 카메라 또는 첫 번째 USB 카메라
  1, 2, … = 추가로 연결된 카메라 장치

목표 FPS: 카메라가 지원하는 최대 FPS를 초과하면 자동으로 낮춰집니다.

IP 카메라(RTSP) 사용 시:
  [파일] 모드를 선택하고 경로란에 rtsp:// URL을 직접 입력하면 됩니다.
  예) rtsp://admin:1234@192.168.1.100:554/stream"""),

        ("이미지 시퀀스 소스", """\
폴더 안의 이미지 파일을 파일명 순서로 읽어 동영상처럼 처리합니다.

지원 형식: JPG, JPEG, PNG, BMP, TIFF, TIF
정렬 기준: 파일명 알파벳 순서 (001.jpg, 002.jpg … 권장)

목표 FPS: 이미지 간 간격을 제어합니다.
  예) FPS=10 → 각 이미지를 0.1초 간격으로 처리

적합한 경우:
  • 열화상 카메라 원시 데이터 처리
  • 타임랩스 데이터 분석
  • 배치 이미지 검사 자동화"""),
    ],

    "🔲 ROI 관리": [
        ("ROI란?", """\
ROI(Region of Interest, 관심 영역)는 영상 내에서 집중적으로 분석할
직사각형 영역입니다.

ROI를 사용하면:
  • 불필요한 배경 영역을 제외해 분석 정확도 향상
  • 여러 영역을 독립적으로 동시 감시 가능
  • 각 ROI마다 별도의 상태(정상/경고/이상)가 부여됨

권장: ROI는 1~6개 이내로 설정하는 것이 성능상 유리합니다."""),

        ("ROI 그리기 (마우스)", """\
1. [ROI 관리] 탭 클릭
2. [✎ ROI 그리기] 버튼 클릭 → 커서가 십자선으로 변경
3. 비디오 화면에서 마우스 왼쪽 버튼을 누른 채 드래그
4. 마우스를 놓으면 ROI가 등록되고 목록에 나타납니다.

주의:
  • 너무 작은 영역(8×8픽셀 미만)은 무시됩니다.
  • [✎ ROI 그리기] 클릭 후 캔버스를 드래그해야 합니다."""),

        ("ROI 이름 변경 및 삭제", """\
이름 변경:
  목록에서 ROI 클릭 → 하단 [표시 이름] 입력 → [적용] 클릭

개별 삭제:
  목록에서 ROI 선택 → [✕ 삭제] 클릭

전체 삭제:
  [✕ 전체 삭제] 클릭 → 확인 대화상자에서 [예]

좌표 수정:
  삭제 후 [좌표 직접 입력]으로 새로 추가하는 방식을 사용하세요."""),

        ("ROI 색상 코드", """\
각 ROI는 상태에 따라 색상으로 구분됩니다:

  🟢 초록   — 정상 (NORMAL)
  🟡 노란색 — 경고 (WARNING)
  🔴 빨강   — 이상 감지 (ABNORMAL)
  🟠 주황   — 고착 (STUCK)
  🔵 파랑   — 드리프트 (DRIFTING)
  🟣 보라   — 진동 (OSCILLATING)
  ⭕ 진빨강 — 급격한 변화 (SUDDEN_CHANGE)"""),
    ],

    "🤖 AI 분석": [
        ("AI 모델 종류", """\
1. 기본(테스트용) — Placeholder Model
   실제 모델 없이 랜덤 예측 결과를 반환합니다.
   처음 시스템 구성 및 UI 확인 시 사용하세요.
   별도 설치 없이 즉시 사용 가능합니다.

2. ONNX 모델
   .onnx 확장자 파일을 사용합니다.
   설치: pip install onnxruntime
   장점: CPU에서 빠른 추론, 프레임워크 독립적

3. PyTorch 모델
   .pt 또는 .pth 파일을 사용합니다.
   설치: pip install torch torchvision
   장점: 커스텀 아키텍처 지원, GPU 가속 가능"""),

        ("모델 출력 규약", """\
이 시스템은 모든 모델이 다음 형식의 출력을 반환한다고 가정합니다:

  입력: BGR 이미지 (ROI 크롭 또는 전체 프레임)
  출력: [정상 점수, 이상 점수] 형태의 1D 배열
        (합계가 1이 되도록 softmax 처리됨)

클래스 인덱스:
  0 = 정상 (normal)
  1 = 이상 (abnormal)

AI 판정 임계값 (기본값):
  이상 판정: 이상 점수 ≥ 0.70
  경고 판정: 이상 점수 ≥ 0.40"""),

        ("AI 판정 + 규칙 기반 검증", """\
AI 판정 결과는 단독으로 최종 결정에 사용되지 않습니다.
다음 6가지 규칙과 함께 복합 판정을 수행합니다:

  1. AI 점수 규칙     — AI 이상 확률이 임계값 초과
  2. 급변 감지 규칙   — 프레임 간 강도 차이가 임계값 초과
  3. 고착 감지 규칙   — 분산이 너무 낮아 화면이 멈춘 상태
  4. 드리프트 규칙    — 슬라이딩 윈도우에서 점진적 평균 이탈
  5. 진동 감지 규칙   — 주기적인 강도 변동 패턴 검출
  6. 경고 신뢰도 규칙 — AI 점수가 경계 영역에 있을 때 경고

히스테리시스 적용:
  N 프레임 연속으로 이상 판정 시에만 ABNORMAL 상태로 전환됩니다.
  (기본값: 5프레임 연속 이상 → ABNORMAL 확정)"""),
    ],

    "🚨 이상 감지": [
        ("6가지 이상 패턴", """\
이 시스템은 다음 6가지 시계열 이상 패턴을 자동으로 감지합니다:

① 고착 (STUCK)
   증상: ROI 영상이 일정 기간 거의 변하지 않음
   원인: 카메라 가림, 장비 잠김, 센서 고장
   판정: 슬라이딩 윈도우 내 강도 표준편차 < 임계값

② 드리프트 (DRIFTING)
   증상: ROI 강도 평균이 시간에 따라 점진적으로 이탈
   원인: 조명 변화, 렌즈 오염, 온도 드리프트
   판정: 윈도우 초반 평균 vs 후반 평균 차이 > 임계값

③ 진동 (OSCILLATING)
   증상: ROI 강도가 주기적으로 오르내리는 패턴
   원인: 기계 진동, 전기 노이즈, 불안정한 공정
   판정: 슬라이딩 윈도우 내 부호 교차 횟수 ≥ 임계값

④ 급격한 변화 (SUDDEN_CHANGE)
   증상: 연속된 두 프레임 간 강도 차이가 급격히 증가
   원인: 갑작스러운 조명 변화, 외부 충격, 물체 침입
   판정: 프레임 간 평균 절대 차이 > 임계값

⑤ AI 이상 감지 (AI_DETECTION)
   증상: 학습된 모델이 이상 패턴을 직접 분류
   원인: 모델이 학습한 시각적 이상 패턴

⑥ 낮은 신뢰도 경고 (WARNING)
   증상: AI 점수가 정상과 이상의 경계 영역
   용도: 운영자에게 주의를 요청하는 소프트 알림"""),

        ("판정 흐름 (히스테리시스)", """\
단일 프레임 이상 감지 → 즉시 ABNORMAL 전환 방지를 위해
히스테리시스(Hysteresis) 로직을 사용합니다.

ABNORMAL 전환 조건:
  연속 5프레임(기본값) 이상 이상 점수 ≥ 0.70 이어야 ABNORMAL로 전환

NORMAL 복구 조건:
  연속 10프레임(기본값) 이상 정상 점수여야 NORMAL로 복구

WARNING 조건:
  현재 프레임의 이상 점수가 0.40~0.70 사이

INITIALIZING 상태:
  시작 후 5프레임 동안은 초기화 상태를 유지합니다.

슬라이딩 윈도우 크기: 기본 60프레임 (약 2초 @ 30fps)"""),

        ("이벤트 저장", """\
NORMAL → ABNORMAL로 상태가 전환될 때 자동으로 이벤트가 기록됩니다.

저장 내용:
  • 이벤트 스냅샷 이미지 (.jpg)
  • 전후 프레임 클립 영상 (.avi)
    - 사전 저장: 이벤트 발생 30프레임 전 (기본값)
    - 사후 저장: 이벤트 발생 30프레임 후 (기본값)
  • 이벤트 메타데이터 (events.json)
  • 프레임별 로그 (frame_log_*.csv / *.json)

저장 위치: 실행 디렉터리의 logs/events/ 폴더"""),
    ],

    "💾 로그 & 내보내기": [
        ("저장 파일 구조", """\
프로그램 종료(■ 정지) 시 자동으로 다음 파일이 생성됩니다:

logs/
├── system.log              ← 시스템 실행 로그 (텍스트)
├── frame_log_YYYYMMDD_HHmmss.csv   ← 프레임별 분석 결과
├── frame_log_YYYYMMDD_HHmmss.json  ← 위와 동일 (JSON 형식)
└── events/
    ├── events.json                 ← 이벤트 목록 (전체)
    ├── evt_XXXXXXXX_XXXX_snapshot.jpg  ← 이벤트 스냅샷
    └── evt_XXXXXXXX_XXXX_clip.avi      ← 이벤트 클립"""),

        ("CSV 파일 활용법", """\
frame_log CSV 파일에는 프레임당 다음 컬럼이 포함됩니다:

  frame_index   — 프레임 번호
  timestamp     — 유닉스 타임스탬프
  system_state  — 시스템 상태 문자열
  state_conf    — 상태 신뢰도 (0~1)
  inference_ms  — AI 추론 소요 시간 (ms)
  is_event      — 이벤트 발생 여부 (True/False)
  roi_*_state   — 각 ROI 상태
  roi_*_mean_intensity — 각 ROI 평균 강도
  triggered_rules — 발동된 규칙 이름 목록

Excel, Python pandas, R 등으로 바로 분석 가능합니다.

예) pandas 로드:
  import pandas as pd
  df = pd.read_csv('logs/frame_log_*.csv')
  abnormal = df[df['system_state'] == 'ABNORMAL']"""),

        ("events.json 활용법", """\
events.json 형식:
[
  {
    "event_id": "evt_00001234_0001",
    "frame_index": 1234,
    "timestamp": 1710000000.123,
    "event_type": "state_change_abnormal",
    "severity": 0.85,
    "system_state": "ABNORMAL",
    "roi_id": "roi_0",
    "abnormality_type": "ai_detection|sudden_change",
    "message": "ROI roi_0: 이상 신뢰도=0.85",
    "snapshot_path": "logs/events/evt_....jpg"
  },
  ...
]

이 파일을 기반으로 이벤트 리뷰 도구나
자동 보고서 생성 스크립트를 구성할 수 있습니다."""),
    ],

    "⌨ 단축키 & 팁": [
        ("키보드 단축키", """\
[헤드리스 CLI 모드에서 — python main.py --no-gui]
  Q          — 분석 종료 (OpenCV 창에서)

[GUI 모드]
  현재 버전에서는 GUI 버튼/메뉴로 모든 조작이 가능합니다.
  추후 업데이트에서 단축키가 추가될 예정입니다."""),

        ("성능 최적화 팁", """\
분석이 느릴 때:
  • [목표 FPS]를 낮게 설정 (예: 15fps)
  • ROI 개수를 줄임 (1~2개 권장)
  • [디버그 오버레이]를 비활성화
  • 소스 해상도를 낮춤 (전처리 리사이즈 설정 추가)

GPU 가속 (PyTorch 모델):
  config.py의 AIConfig.device = "cuda" 로 변경
  (NVIDIA GPU + CUDA 환경 필요)

ONNX 모델 변환 (권장):
  PyTorch 모델을 ONNX로 변환하면 torch 없이도 빠르게 실행됩니다.
  torch.onnx.export(model, dummy_input, "model.onnx")"""),

        ("트러블슈팅", """\
증상: 소스를 열 수 없습니다 오류
  → 파일 경로에 한글/특수문자가 포함된 경우 영문 경로로 이동
  → 카메라 번호를 0, 1, 2 순으로 시도

증상: 화면이 표시되지 않음 / 프레임 없음
  → OpenCV 코덱 설치 확인: pip install opencv-python
  → 비디오 파일이 손상됐는지 미디어 플레이어로 확인

증상: AI 추론 시간이 매우 느림
  → 모델 타입을 "기본(테스트용)"으로 변경하여 테스트
  → ONNX 모델 사용 권장 (CPU 최적화됨)

증상: ROI가 화면에 표시되지 않음
  → 소스 시작 전에 ROI를 등록해야 합니다.

증상: 이벤트가 너무 자주 발생
  → [ai/decision_engine.py] DecisionConfig의
    consecutive_abnormal_frames 값을 높이거나
    abnormal_score_threshold 를 높여 민감도를 낮추세요."""),

        ("실제 AI 모델 연결 방법", """\
1. ONNX 모델 사용:
   a) pip install onnxruntime
   b) [AI 모델] → [ONNX] 선택
   c) [모델 경로] → .onnx 파일 선택
   d) ▶ 시작

2. PyTorch 모델 사용:
   a) pip install torch
   b) [AI 모델] → [PyTorch] 선택
   c) [모델 경로] → .pt / .pth 파일 선택
   d) ▶ 시작

모델 출력 형식:
  모델이 [정상 확률, 이상 확률] 형태의 2-class 분류 출력을
  반환해야 합니다. 다른 형식의 모델은 ai/inference_engine.py
  _scores_to_result() 메서드를 수정하여 적용할 수 있습니다."""),
    ],
}


# ---------------------------------------------------------------------------
# HelpDialog
# ---------------------------------------------------------------------------

class HelpDialog(tk.Toplevel):
    """
    탭 구성 상세 도움말 다이얼로그.
    App 인스턴스 또는 임의 Tk 위젯을 master로 받아 생성합니다.
    """

    def __init__(self, master):
        super().__init__(master)
        self.title("사용법 도움말 — 비디오 분석 시스템")
        self.geometry("820x620")
        self.resizable(True, True)
        self.minsize(640, 480)

        # 다크 테마 배경
        self.configure(bg="#1e1e2e")
        self.grab_set()   # 모달

        self._build_ui()
        self.focus_set()

    def _build_ui(self) -> None:
        # 헤더
        hdr = tk.Frame(self, bg="#181825", pady=10)
        hdr.pack(fill="x")
        tk.Label(hdr, text="📖  비디오 분석 시스템 — 상세 사용법",
                 bg="#181825", fg="#cba6f7",
                 font=("맑은 고딕", 13, "bold")).pack(side="left", padx=16)
        ttk.Button(hdr, text="✕  닫기", command=self.destroy).pack(side="right", padx=12)

        # 탭 노트북
        nb = ttk.Notebook(self)
        nb.pack(fill="both", expand=True, padx=10, pady=(6, 4))

        for tab_name, sections in HELP_CONTENT.items():
            tab_frame = ttk.Frame(nb, padding=0)
            nb.add(tab_frame, text=tab_name)
            self._build_tab(tab_frame, sections)

        # 하단 버튼
        foot = tk.Frame(self, bg="#1e1e2e", pady=6)
        foot.pack(fill="x")
        ttk.Button(foot, text="닫기", command=self.destroy, width=12).pack(side="right", padx=12)
        tk.Label(foot, text="문의: GitHub Issues 페이지를 이용해 주세요.",
                 bg="#1e1e2e", fg="#585b70", font=("맑은 고딕", 8)).pack(side="left", padx=12)

    def _build_tab(self, parent: ttk.Frame, sections: list) -> None:
        """각 탭 안에 섹션별 제목 + 텍스트 박스를 세로로 쌓습니다."""
        canvas = tk.Canvas(parent, bg="#1e1e2e", highlightthickness=0)
        vsb = ttk.Scrollbar(parent, orient="vertical", command=canvas.yview)
        canvas.configure(yscrollcommand=vsb.set)
        vsb.pack(side="right", fill="y")
        canvas.pack(side="left", fill="both", expand=True)

        inner = tk.Frame(canvas, bg="#1e1e2e")
        inner_id = canvas.create_window((0, 0), window=inner, anchor="nw")

        def _on_frame_configure(e):
            canvas.configure(scrollregion=canvas.bbox("all"))

        def _on_canvas_configure(e):
            canvas.itemconfig(inner_id, width=e.width)

        inner.bind("<Configure>", _on_frame_configure)
        canvas.bind("<Configure>", _on_canvas_configure)
        canvas.bind_all("<MouseWheel>",
                        lambda e: canvas.yview_scroll(int(-1 * (e.delta / 120)), "units"))

        for title, body in sections:
            # 섹션 제목
            tk.Label(inner, text=title,
                     bg="#313244", fg="#cba6f7",
                     font=("맑은 고딕", 10, "bold"),
                     anchor="w", padx=10, pady=4).pack(fill="x", pady=(10, 0))

            # 내용 텍스트 (읽기 전용)
            txt = tk.Text(inner, bg="#1e1e2e", fg="#cdd6f4",
                          font=("맑은 고딕", 9),
                          relief="flat", wrap="word",
                          padx=14, pady=8,
                          cursor="arrow",
                          state="normal")
            txt.insert("1.0", body.strip())
            txt.config(state="disabled")

            # 높이 자동 계산
            line_count = body.strip().count("\n") + 2
            txt.config(height=min(line_count, 22))
            txt.pack(fill="x", pady=(0, 4))


# ---------------------------------------------------------------------------
# 독립 실행 테스트
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    root = tk.Tk()
    root.withdraw()
    dlg = HelpDialog(root)
    root.mainloop()
