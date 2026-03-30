"""
app_gui.py ─ 오프라인 문서 변경 분석 AI  (단일 GUI 프로그램)

실행:
    python app_gui.py

구성 탭:
    1. 학습   ─ 단일/다중 파일 → 컬럼 매핑 → 학습 파라미터 → 신규/증분 학습
    2. 예측   ─ 모델 로드 → 파일 선택 → 배치 예측 → 결과 미리보기
    3. 모델   ─ 저장 모델 목록 / 상세 정보
    4. 로그   ─ 실시간 로그 + 내보내기

특징:
    ·  tkinter + ttk (아나콘다 기본 포함, 추가 설치 불필요)
    ·  학습/예측은 별도 Thread → UI 비멈춤
    ·  로그 핸들러 → 로그 탭 실시간 반영
    ·  오프라인 환경 강제 설정 (TRANSFORMERS_OFFLINE=1)
"""

import os
import sys
import queue
import logging
import warnings
import threading
import traceback
import datetime
import tkinter as tk
from tkinter import ttk, filedialog, messagebox, scrolledtext

# ── pynvml/torch deprecated 경고 억제 ────────────────────────
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)

# ── 오프라인 강제 설정 ────────────────────────────────────────
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_DATASETS_OFFLINE"]  = "1"
os.environ["HF_HUB_OFFLINE"]       = "1"


# ═══════════════════════════════════════════════════════════════
# 로그 → 큐 핸들러 (Thread-safe GUI 로그)
# ═══════════════════════════════════════════════════════════════

class QueueLogHandler(logging.Handler):
    def __init__(self, log_queue: queue.Queue):
        super().__init__()
        self.log_queue = log_queue
        self.setFormatter(logging.Formatter(
            "%(asctime)s [%(levelname)s] %(name)s - %(message)s",
            datefmt="%H:%M:%S",
        ))

    def emit(self, record):
        msg = self.format(record)
        self.log_queue.put(msg)


# ═══════════════════════════════════════════════════════════════
# 색상 / 스타일 상수
# ═══════════════════════════════════════════════════════════════

CLR_BG       = "#1e1e2e"    # 배경
CLR_PANEL    = "#2a2a3e"    # 패널
CLR_ACCENT   = "#7c3aed"    # 강조 (보라)
CLR_BTN      = "#6d28d9"    # 버튼
CLR_BTN_HOV  = "#7c3aed"
CLR_SUCCESS  = "#10b981"    # 성공 (초록)
CLR_WARN     = "#f59e0b"    # 경고 (노랑)
CLR_ERR      = "#ef4444"    # 오류 (빨강)
CLR_FG       = "#e2e8f0"    # 텍스트
CLR_FG_DIM   = "#94a3b8"    # 흐린 텍스트
CLR_BORDER   = "#4a4a6a"    # 경계선
CLR_INPUT    = "#13131f"    # 입력창 배경
CLR_TRAIN    = "#0ea5e9"    # 학습 탭 강조
CLR_PRED     = "#10b981"    # 예측 탭 강조


def _hex(c): return c  # 색상 그대로 사용


# ═══════════════════════════════════════════════════════════════
# 공통 위젯 팩토리
# ═══════════════════════════════════════════════════════════════

def _label(parent, text, bold=False, color=CLR_FG, size=10, **kw):
    font = ("맑은 고딕", size, "bold" if bold else "normal")
    return tk.Label(parent, text=text, fg=color, bg=parent["bg"],
                    font=font, **kw)

def _entry(parent, textvariable, width=40, **kw):
    return tk.Entry(parent, textvariable=textvariable, width=width,
                    bg=CLR_INPUT, fg=CLR_FG, insertbackground=CLR_FG,
                    relief="flat", bd=4, font=("맑은 고딕", 10), **kw)

def _btn(parent, text, command, color=CLR_BTN, fg=CLR_FG, width=None, **kw):
    b = tk.Button(parent, text=text, command=command,
                  bg=color, fg=fg, activebackground=CLR_BTN_HOV,
                  activeforeground=fg, relief="flat", bd=0,
                  font=("맑은 고딕", 10, "bold"),
                  cursor="hand2", padx=12, pady=6, **kw)
    if width:
        b.config(width=width)
    return b

def _spinbox(parent, textvariable, from_, to, increment=1, width=8):
    return tk.Spinbox(parent, textvariable=textvariable,
                      from_=from_, to=to, increment=increment, width=width,
                      bg=CLR_INPUT, fg=CLR_FG, insertbackground=CLR_FG,
                      relief="flat", bd=4, font=("맑은 고딕", 10),
                      buttonbackground=CLR_PANEL)

def _frame(parent, bg=None, **kw):
    return tk.Frame(parent, bg=bg or parent["bg"], **kw)

def _lframe(parent, text, color=CLR_ACCENT):
    lf = tk.LabelFrame(parent, text=f"  {text}  ",
                        bg=parent["bg"], fg=color,
                        font=("맑은 고딕", 10, "bold"),
                        relief="solid", bd=1,
                        highlightbackground=CLR_BORDER)
    return lf

def _sep(parent, orient="horizontal"):
    return ttk.Separator(parent, orient=orient)


# ═══════════════════════════════════════════════════════════════
# 메인 애플리케이션
# ═══════════════════════════════════════════════════════════════

class DocChangeApp:

    def __init__(self, root: tk.Tk):
        self.root = root
        self._log_queue: queue.Queue = queue.Queue()
        self._worker_thread: threading.Thread = None
        self._running = False

        # ── 로깅 설정 ──
        self._setup_logging()

        # ── 윈도우 기본 설정 ──
        root.title("오프라인 문서 변경 분석 AI")
        root.configure(bg=CLR_BG)
        root.geometry("1100x780")
        root.minsize(900, 650)

        # ── ttk 스타일 ──
        self._apply_style()

        # ── UI 빌드 ──
        self._build_header()
        self._build_notebook()
        self._build_statusbar()

        # ── 로그 폴링 ──
        self._poll_log_queue()

    # ─────────────────────────────────────────────────────────
    # 로깅
    # ─────────────────────────────────────────────────────────

    def _setup_logging(self):
        handler = QueueLogHandler(self._log_queue)
        logging.basicConfig(
            level=logging.INFO,
            handlers=[handler, logging.StreamHandler(sys.stdout)],
            format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
            force=True,
        )
        self._logger = logging.getLogger("DocChangeApp")

    def _poll_log_queue(self):
        """100ms 마다 로그 큐 확인 후 Log 탭에 출력"""
        try:
            while True:
                msg = self._log_queue.get_nowait()
                if hasattr(self, "_log_text"):
                    self._log_text.configure(state="normal")
                    self._log_text.insert("end", msg + "\n")
                    self._log_text.see("end")
                    self._log_text.configure(state="disabled")
        except queue.Empty:
            pass
        self.root.after(100, self._poll_log_queue)

    # ─────────────────────────────────────────────────────────
    # ttk 스타일
    # ─────────────────────────────────────────────────────────

    def _apply_style(self):
        style = ttk.Style()
        style.theme_use("clam")

        style.configure("TNotebook", background=CLR_BG, borderwidth=0)
        style.configure("TNotebook.Tab",
                        background=CLR_PANEL, foreground=CLR_FG_DIM,
                        font=("맑은 고딕", 11, "bold"),
                        padding=[18, 8])
        style.map("TNotebook.Tab",
                  background=[("selected", CLR_ACCENT)],
                  foreground=[("selected", "#ffffff")])

        style.configure("TProgressbar",
                        troughcolor=CLR_INPUT,
                        background=CLR_ACCENT,
                        thickness=8)

        style.configure("Treeview",
                        background=CLR_INPUT, foreground=CLR_FG,
                        fieldbackground=CLR_INPUT,
                        rowheight=28,
                        font=("맑은 고딕", 10))
        style.configure("Treeview.Heading",
                        background=CLR_PANEL, foreground=CLR_ACCENT,
                        font=("맑은 고딕", 10, "bold"))
        style.map("Treeview",
                  background=[("selected", CLR_ACCENT)],
                  foreground=[("selected", "#ffffff")])

        style.configure("TScrollbar",
                        background=CLR_PANEL,
                        troughcolor=CLR_BG,
                        arrowcolor=CLR_FG_DIM)

    # ─────────────────────────────────────────────────────────
    # 헤더
    # ─────────────────────────────────────────────────────────

    def _build_header(self):
        hdr = _frame(self.root, bg=CLR_PANEL)
        hdr.pack(fill="x", side="top")

        _label(hdr, "  문서 변경 분석 AI", bold=True,
               color=CLR_FG, size=14).pack(side="left", pady=12)
        _label(hdr, "오프라인 전용  |  HuggingFace 로컬 모델  |  CPU / GPU 자동",
               color=CLR_FG_DIM, size=9).pack(side="left", padx=16, pady=12)

        # GPU/CPU 뱃지 - 초기엔 "확인 중..." 표시 후 백그라운드에서 감지
        self._device_badge = _label(hdr, "  장치 확인 중...  ", bold=True,
                                    color=CLR_FG_DIM, size=9)
        self._device_badge.pack(side="right", padx=16)

        # 백그라운드 스레드에서 torch 로딩 (UI 블로킹 방지)
        threading.Thread(target=self._detect_device, daemon=True).start()

    def _detect_device(self):
        """백그라운드에서 torch/GPU 감지 후 뱃지 업데이트"""
        try:
            import torch
            is_gpu = torch.cuda.is_available()
            device = "GPU ✓" if is_gpu else "CPU 모드"
            dc = CLR_SUCCESS if is_gpu else CLR_WARN
        except ImportError:
            device = "torch 없음"
            dc = CLR_ERR
        # UI 업데이트는 메인 스레드에서
        self.root.after(0, self._device_badge.config,
                        {"text": f"  {device}  ", "fg": dc})

    # ─────────────────────────────────────────────────────────
    # 탭 노트북
    # ─────────────────────────────────────────────────────────

    def _build_notebook(self):
        nb = ttk.Notebook(self.root)
        nb.pack(fill="both", expand=True, padx=8, pady=(6, 2))

        tab1 = _frame(nb, bg=CLR_BG)
        tab2 = _frame(nb, bg=CLR_BG)
        tab3 = _frame(nb, bg=CLR_BG)
        tab4 = _frame(nb, bg=CLR_BG)

        nb.add(tab1, text="  학습  ")
        nb.add(tab2, text="  예측  ")
        nb.add(tab3, text="  모델 관리  ")
        nb.add(tab4, text="  로그  ")

        self._build_train_tab(tab1)
        self._build_predict_tab(tab2)
        self._build_model_tab(tab3)
        self._build_log_tab(tab4)

    # ─────────────────────────────────────────────────────────
    # 상태바
    # ─────────────────────────────────────────────────────────

    def _build_statusbar(self):
        self._status_var = tk.StringVar(value="대기 중")
        bar = _frame(self.root, bg=CLR_PANEL)
        bar.pack(fill="x", side="bottom")
        tk.Label(bar, textvariable=self._status_var,
                 bg=CLR_PANEL, fg=CLR_FG_DIM,
                 font=("맑은 고딕", 9), anchor="w").pack(
                     side="left", padx=10, pady=4)

    def _set_status(self, msg: str, color=CLR_FG_DIM):
        self._status_var.set(msg)

    # ═══════════════════════════════════════════════════════
    # TAB 1 — 학습
    # ═══════════════════════════════════════════════════════

    def _build_train_tab(self, parent):
        parent.configure(bg=CLR_BG)
        pane = _frame(parent, bg=CLR_BG)
        pane.pack(fill="both", expand=True, padx=16, pady=12)

        # 좌: 설정  /  우: 진행상태
        left  = _frame(pane, bg=CLR_BG)
        right = _frame(pane, bg=CLR_BG)
        left.pack(side="left", fill="y", padx=(0, 10))
        right.pack(side="left", fill="both", expand=True)

        # ── 입력 소스 ──────────────────────────────────────
        src_lf = _lframe(left, "① 입력 데이터", CLR_TRAIN)
        src_lf.pack(fill="x", pady=(0, 8))
        src_lf.configure(bg=CLR_BG)

        self._t_src_mode = tk.StringVar(value="file")
        f_rb = tk.Frame(src_lf, bg=CLR_BG)
        f_rb.pack(fill="x", padx=8, pady=4)
        for val, lbl in [("file", "단일 파일"), ("dir", "디렉터리 (다중)")]:
            tk.Radiobutton(f_rb, text=lbl, variable=self._t_src_mode, value=val,
                           bg=CLR_BG, fg=CLR_FG, selectcolor=CLR_PANEL,
                           activebackground=CLR_BG, activeforeground=CLR_FG,
                           font=("맑은 고딕", 10),
                           command=self._on_train_src_toggle).pack(
                               side="left", padx=6)

        self._t_input_var = tk.StringVar()
        f_in = _frame(src_lf, bg=CLR_BG)
        f_in.pack(fill="x", padx=8, pady=(0, 8))
        _entry(f_in, self._t_input_var, width=32).pack(side="left")
        self._t_browse_btn = _btn(f_in, "파일 선택",
                                   self._browse_train_input,
                                   color=CLR_PANEL)
        self._t_browse_btn.pack(side="left", padx=4)

        self._t_save_merged = tk.BooleanVar(value=True)
        tk.Checkbutton(src_lf, text="통합 데이터셋 저장 (merged_dataset.xlsx)",
                       variable=self._t_save_merged,
                       bg=CLR_BG, fg=CLR_FG_DIM,
                       selectcolor=CLR_PANEL,
                       activebackground=CLR_BG,
                       font=("맑은 고딕", 9)).pack(
                           anchor="w", padx=8, pady=(0, 6))

        # ── 컬럼 매핑 ──────────────────────────────────────
        col_lf = _lframe(left, "② 컬럼 매핑", CLR_TRAIN)
        col_lf.pack(fill="x", pady=(0, 8))
        col_lf.configure(bg=CLR_BG)

        # 건너뛸 행 수 (상단 양식/제목 행)
        skip_row = _frame(col_lf, bg=CLR_BG)
        skip_row.pack(fill="x", padx=8, pady=(4, 1))
        _label(skip_row, "상단 건너뛸 행:", color=CLR_FG_DIM).pack(side="left")
        _label(skip_row, "(양식/제목 행 수)", color=CLR_FG_DIM, size=8).pack(side="left", padx=2)
        self._t_skiprows = tk.IntVar(value=0)
        _spinbox(skip_row, self._t_skiprows, 0, 50, 1).pack(side="right")

        # 헤더 없음 체크박스
        hdr_chk_row = _frame(col_lf, bg=CLR_BG)
        hdr_chk_row.pack(fill="x", padx=8, pady=(1, 2))
        self._t_no_header = tk.BooleanVar(value=True)
        tk.Checkbutton(
            hdr_chk_row,
            text="헤더 없음 (열 위치로 선택: A열, B열, G열...)",
            variable=self._t_no_header,
            bg=CLR_BG, fg=CLR_WARN, selectcolor=CLR_PANEL,
            activebackground=CLR_BG,
            font=("맑은 고딕", 9, "bold"),
        ).pack(anchor="w")

        # 자동 감지 버튼
        detect_row = _frame(col_lf, bg=CLR_BG)
        detect_row.pack(fill="x", padx=8, pady=(1, 2))
        _btn(detect_row, "파일 미리보기 & 컬럼 감지", self._detect_columns,
             color=CLR_PANEL).pack(fill="x")
        self._detect_hint = _label(detect_row, "← 파일 선택 후 클릭하세요",
                                   color=CLR_FG_DIM, size=8)
        self._detect_hint.pack(anchor="w", pady=1)

        # 컬럼 드롭다운
        self._col_vars = {}
        self._col_combos = {}
        col_defs = [
            ("before_col",  "변경 전 (필수)"),
            ("after_col",   "변경 후 (필수)"),
            ("summary_col", "요약     (선택)"),
            ("reason_col",  "사유     (선택)"),
            ("code_col",    "코드     (필수)"),
        ]
        for key, lbl in col_defs:
            row = _frame(col_lf, bg=CLR_BG)
            row.pack(fill="x", padx=8, pady=2)
            _label(row, f"{lbl}:", color=CLR_FG_DIM, size=9).pack(side="left")
            var = tk.StringVar(value="")
            self._col_vars[key] = var
            cb = ttk.Combobox(row, textvariable=var, width=22,
                              font=("맑은 고딕", 9))
            cb.pack(side="right")
            self._col_combos[key] = cb

        # ── 모델 경로 ──────────────────────────────────────
        mdl_lf = _lframe(left, "③ 모델 경로", CLR_TRAIN)
        mdl_lf.pack(fill="x", pady=(0, 8))
        mdl_lf.configure(bg=CLR_BG)

        self._t_base_model = tk.StringVar(value="./local_model/kobart")
        self._t_save_path  = tk.StringVar(value="./saved_model/model_v1")

        for var, lbl, browse_fn in [
            (self._t_base_model, "로컬 모델:",   self._browse_base_model),
            (self._t_save_path,  "저장 경로:",   self._browse_save_path),
        ]:
            row = _frame(mdl_lf, bg=CLR_BG)
            row.pack(fill="x", padx=8, pady=3)
            _label(row, lbl, color=CLR_FG_DIM).pack(side="left")
            _entry(row, var, width=22).pack(side="left", padx=4)
            _btn(row, "선택", browse_fn, color=CLR_PANEL).pack(side="left")

        # ── 학습 파라미터 ──────────────────────────────────
        prm_lf = _lframe(left, "④ 학습 파라미터", CLR_TRAIN)
        prm_lf.pack(fill="x", pady=(0, 8))
        prm_lf.configure(bg=CLR_BG)

        self._t_epochs    = tk.IntVar(value=10)
        self._t_batch     = tk.IntVar(value=4)
        self._t_lr_idx    = tk.StringVar(value="3e-5")
        self._t_max_in    = tk.IntVar(value=256)
        self._t_max_tgt   = tk.IntVar(value=128)
        self._t_patience  = tk.IntVar(value=3)
        self._t_incremental = tk.BooleanVar(value=False)

        params = [
            ("에폭 수",        self._t_epochs,   1,   50, 1),
            ("배치 크기",      self._t_batch,    1,   32, 1),
            ("입력 토큰 수",   self._t_max_in,  64,  512, 32),
            ("생성 토큰 수",   self._t_max_tgt, 32,  256, 16),
            ("Early Stop",     self._t_patience, 1,   10, 1),
        ]
        for lbl, var, fr, to, inc in params:
            row = _frame(prm_lf, bg=CLR_BG)
            row.pack(fill="x", padx=8, pady=2)
            _label(row, f"{lbl}:", color=CLR_FG_DIM).pack(side="left")
            _spinbox(row, var, fr, to, inc).pack(side="right")

        # LR 선택
        lr_row = _frame(prm_lf, bg=CLR_BG)
        lr_row.pack(fill="x", padx=8, pady=2)
        _label(lr_row, "학습률:", color=CLR_FG_DIM).pack(side="left")
        lr_combo = ttk.Combobox(lr_row, textvariable=self._t_lr_idx,
                                values=["1e-5", "2e-5", "3e-5", "5e-5", "1e-4"],
                                width=9, state="readonly")
        lr_combo.pack(side="right")

        tk.Checkbutton(prm_lf, text="증분 학습 (기존 모델에 추가 데이터 fine-tuning)",
                       variable=self._t_incremental,
                       bg=CLR_BG, fg=CLR_FG_DIM, selectcolor=CLR_PANEL,
                       activebackground=CLR_BG,
                       font=("맑은 고딕", 9)).pack(anchor="w", padx=8, pady=(0, 6))

        # ── 우측: 진행 상태 + 결과 ────────────────────────
        ctrl_lf = _lframe(right, "⑤ 실행 제어", CLR_TRAIN)
        ctrl_lf.pack(fill="x", pady=(0, 8))
        ctrl_lf.configure(bg=CLR_BG)

        self._train_prog = ttk.Progressbar(ctrl_lf, mode="indeterminate",
                                            style="TProgressbar")
        self._train_prog.pack(fill="x", padx=8, pady=6)
        self._train_status = tk.StringVar(value="대기 중")
        _label(ctrl_lf, "", color=CLR_FG_DIM,
               textvariable=self._train_status).pack(pady=2)

        btn_row = _frame(ctrl_lf, bg=CLR_BG)
        btn_row.pack(fill="x", padx=8, pady=6)
        self._train_btn = _btn(btn_row, "▶  학습 시작",
                                self._start_train, color=CLR_ACCENT)
        self._train_btn.pack(side="left", padx=4)
        self._stop_btn = _btn(btn_row, "■  중지",
                               self._stop_worker, color=CLR_ERR)
        self._stop_btn.pack(side="left", padx=4)
        self._stop_btn.configure(state="disabled")

        # ── 학습 이력 표 ──
        hist_lf = _lframe(right, "학습 이력", CLR_TRAIN)
        hist_lf.pack(fill="both", expand=True)
        hist_lf.configure(bg=CLR_BG)

        cols = ("epoch", "train_loss", "train_acc", "val_loss", "val_acc")
        hdrs = ("에폭", "Train Loss", "Train Acc", "Val Loss", "Val Acc")
        self._hist_tree = ttk.Treeview(hist_lf, columns=cols,
                                        show="headings", height=8)
        for c, h in zip(cols, hdrs):
            self._hist_tree.heading(c, text=h)
            self._hist_tree.column(c, width=100, anchor="center")
        vsb = ttk.Scrollbar(hist_lf, orient="vertical",
                             command=self._hist_tree.yview)
        self._hist_tree.configure(yscrollcommand=vsb.set)
        self._hist_tree.pack(side="left", fill="both", expand=True, padx=4, pady=4)
        vsb.pack(side="right", fill="y", pady=4)

    # ═══════════════════════════════════════════════════════
    # TAB 2 — 예측
    # ═══════════════════════════════════════════════════════

    def _build_predict_tab(self, parent):
        parent.configure(bg=CLR_BG)
        pane = _frame(parent, bg=CLR_BG)
        pane.pack(fill="both", expand=True, padx=16, pady=12)

        left  = _frame(pane, bg=CLR_BG)
        right = _frame(pane, bg=CLR_BG)
        left.pack(side="left", fill="y", padx=(0, 10))
        right.pack(side="left", fill="both", expand=True)

        # ── 모델 경로 ──────────────────────────────────────
        mdl_lf = _lframe(left, "① 저장 모델 경로", CLR_PRED)
        mdl_lf.pack(fill="x", pady=(0, 8))
        mdl_lf.configure(bg=CLR_BG)

        self._p_model_var = tk.StringVar(value="./saved_model/model_v1")
        row = _frame(mdl_lf, bg=CLR_BG)
        row.pack(fill="x", padx=8, pady=6)
        _entry(row, self._p_model_var, width=28).pack(side="left")
        _btn(row, "선택", self._browse_pred_model, color=CLR_PANEL).pack(
            side="left", padx=4)

        # ── 입력 ──────────────────────────────────────────
        inp_lf = _lframe(left, "② 예측 입력", CLR_PRED)
        inp_lf.pack(fill="x", pady=(0, 8))
        inp_lf.configure(bg=CLR_BG)

        self._p_src_mode = tk.StringVar(value="file")
        rb_row = _frame(inp_lf, bg=CLR_BG)
        rb_row.pack(fill="x", padx=8, pady=4)
        for val, lbl in [("file", "단일 파일"), ("dir", "디렉터리 (다중)")]:
            tk.Radiobutton(rb_row, text=lbl, variable=self._p_src_mode, value=val,
                           bg=CLR_BG, fg=CLR_FG, selectcolor=CLR_PANEL,
                           activebackground=CLR_BG, activeforeground=CLR_FG,
                           font=("맑은 고딕", 10),
                           command=self._on_pred_src_toggle).pack(
                               side="left", padx=6)

        self._p_input_var = tk.StringVar()
        f_in = _frame(inp_lf, bg=CLR_BG)
        f_in.pack(fill="x", padx=8, pady=(0, 8))
        _entry(f_in, self._p_input_var, width=28).pack(side="left")
        self._p_browse_btn = _btn(f_in, "파일 선택",
                                   self._browse_pred_input, color=CLR_PANEL)
        self._p_browse_btn.pack(side="left", padx=4)

        # ── 출력 ──────────────────────────────────────────
        out_lf = _lframe(left, "③ 예측 출력", CLR_PRED)
        out_lf.pack(fill="x", pady=(0, 8))
        out_lf.configure(bg=CLR_BG)

        self._p_out_mode = tk.StringVar(value="separate")
        for val, lbl in [("separate", "파일별 저장"), ("combined", "통합 저장")]:
            tk.Radiobutton(out_lf, text=lbl,
                           variable=self._p_out_mode, value=val,
                           bg=CLR_BG, fg=CLR_FG, selectcolor=CLR_PANEL,
                           activebackground=CLR_BG, font=("맑은 고딕", 10),
                           command=self._on_pred_out_toggle).pack(
                               anchor="w", padx=8)

        self._p_output_var = tk.StringVar(value="./predictions/result.xlsx")
        out_row = _frame(out_lf, bg=CLR_BG)
        out_row.pack(fill="x", padx=8, pady=(4, 8))
        _label(out_row, "출력 경로:", color=CLR_FG_DIM).pack(side="left")
        _entry(out_row, self._p_output_var, width=22).pack(side="left", padx=4)
        _btn(out_row, "선택", self._browse_pred_output, color=CLR_PANEL).pack(
            side="left")

        # ── 컬럼 (선택) ──
        cmap_lf = _lframe(left, "④ 컬럼 설정 (비우면 자동)", CLR_PRED)
        cmap_lf.pack(fill="x", pady=(0, 8))
        cmap_lf.configure(bg=CLR_BG)

        self._p_before_col = tk.StringVar()
        self._p_after_col  = tk.StringVar()
        for var, lbl in [(self._p_before_col, "변경 전 컬럼:"),
                          (self._p_after_col,  "변경 후 컬럼:")]:
            row = _frame(cmap_lf, bg=CLR_BG)
            row.pack(fill="x", padx=8, pady=2)
            _label(row, lbl, color=CLR_FG_DIM).pack(side="left")
            _entry(row, var, width=16).pack(side="right")

        # ── 예측 파라미터 ──────────────────────────────────
        prm_lf = _lframe(left, "⑤ 예측 파라미터", CLR_PRED)
        prm_lf.pack(fill="x", pady=(0, 8))
        prm_lf.configure(bg=CLR_BG)

        self._p_batch   = tk.IntVar(value=4)
        self._p_beams   = tk.IntVar(value=2)
        self._p_max_in  = tk.IntVar(value=256)
        self._p_max_tok = tk.IntVar(value=128)

        for lbl, var, fr, to, inc in [
            ("배치 크기", self._p_batch,   1, 32,  1),
            ("Beam 수",   self._p_beams,   1,  8,  1),
            ("입력 토큰", self._p_max_in, 64, 512, 32),
            ("생성 토큰", self._p_max_tok,32, 256, 16),
        ]:
            row = _frame(prm_lf, bg=CLR_BG)
            row.pack(fill="x", padx=8, pady=2)
            _label(row, f"{lbl}:", color=CLR_FG_DIM).pack(side="left")
            _spinbox(row, var, fr, to, inc).pack(side="right")

        # ── 실행 버튼 ──────────────────────────────────────
        ctrl_lf = _lframe(left, "⑥ 실행", CLR_PRED)
        ctrl_lf.pack(fill="x", pady=(0, 8))
        ctrl_lf.configure(bg=CLR_BG)

        self._pred_prog = ttk.Progressbar(ctrl_lf, mode="indeterminate")
        self._pred_prog.pack(fill="x", padx=8, pady=4)
        self._pred_status = tk.StringVar(value="대기 중")
        _label(ctrl_lf, "", color=CLR_FG_DIM,
               textvariable=self._pred_status).pack(pady=2)

        btn_row = _frame(ctrl_lf, bg=CLR_BG)
        btn_row.pack(padx=8, pady=6)
        self._pred_btn = _btn(btn_row, "▶  예측 시작",
                               self._start_predict, color=CLR_PRED)
        self._pred_btn.pack(side="left", padx=4)

        # ── 우측: 결과 미리보기 ────────────────────────────
        prev_lf = _lframe(right, "예측 결과 미리보기", CLR_PRED)
        prev_lf.pack(fill="both", expand=True)
        prev_lf.configure(bg=CLR_BG)

        res_cols = ("pred_summary", "pred_reason", "pred_code", "confidence_score")
        res_hdrs = ("예측 요약", "예측 사유", "예측 코드", "신뢰도")
        self._pred_tree = ttk.Treeview(prev_lf, columns=res_cols,
                                        show="headings", height=20)
        for c, h in zip(res_cols, res_hdrs):
            self._pred_tree.heading(c, text=h)
            w = 80 if c == "confidence_score" else 180
            if c == "pred_code":
                w = 100
            self._pred_tree.column(c, width=w, anchor="center" if c in ("pred_code", "confidence_score") else "w")
        vsb2 = ttk.Scrollbar(prev_lf, orient="vertical",
                              command=self._pred_tree.yview)
        hsb2 = ttk.Scrollbar(prev_lf, orient="horizontal",
                              command=self._pred_tree.xview)
        self._pred_tree.configure(yscrollcommand=vsb2.set,
                                   xscrollcommand=hsb2.set)
        vsb2.pack(side="right", fill="y")
        hsb2.pack(side="bottom", fill="x")
        self._pred_tree.pack(fill="both", expand=True, padx=4, pady=4)

        # 저장 버튼
        _btn(right, "결과 Excel 저장...", self._save_pred_result,
             color=CLR_PANEL).pack(anchor="e", padx=4, pady=4)
        self._pred_result_df = None

    # ═══════════════════════════════════════════════════════
    # TAB 3 — 모델 관리
    # ═══════════════════════════════════════════════════════

    def _build_model_tab(self, parent):
        parent.configure(bg=CLR_BG)
        pane = _frame(parent, bg=CLR_BG)
        pane.pack(fill="both", expand=True, padx=16, pady=12)

        # 상단 컨트롤
        top = _frame(pane, bg=CLR_BG)
        top.pack(fill="x", pady=(0, 8))
        _label(top, "모델 루트:", color=CLR_FG_DIM).pack(side="left")
        self._m_root_var = tk.StringVar(value="./saved_model")
        _entry(top, self._m_root_var, width=30).pack(side="left", padx=6)
        _btn(top, "선택", self._browse_model_root, color=CLR_PANEL).pack(side="left")
        _btn(top, "새로고침", self._refresh_model_list, color=CLR_ACCENT).pack(
            side="left", padx=8)

        # 모델 목록
        list_lf = _lframe(pane, "저장된 모델 목록", CLR_ACCENT)
        list_lf.pack(fill="both", expand=True, pady=(0, 8))
        list_lf.configure(bg=CLR_BG)

        m_cols = ("name", "model_type", "num_classes", "created_at")
        m_hdrs = ("모델명", "타입", "클래스 수", "생성일")
        self._model_tree = ttk.Treeview(list_lf, columns=m_cols,
                                         show="headings", height=6)
        for c, h in zip(m_cols, m_hdrs):
            self._model_tree.heading(c, text=h)
        self._model_tree.column("name",       width=180)
        self._model_tree.column("model_type", width=80, anchor="center")
        self._model_tree.column("num_classes",width=80, anchor="center")
        self._model_tree.column("created_at", width=180, anchor="center")
        vsb3 = ttk.Scrollbar(list_lf, orient="vertical",
                              command=self._model_tree.yview)
        self._model_tree.configure(yscrollcommand=vsb3.set)
        vsb3.pack(side="right", fill="y", pady=4)
        self._model_tree.pack(fill="both", expand=True, padx=4, pady=4)
        self._model_tree.bind("<<TreeviewSelect>>", self._on_model_select)

        # 모델 상세 정보
        info_lf = _lframe(pane, "모델 상세 정보", CLR_ACCENT)
        info_lf.pack(fill="both", expand=True)
        info_lf.configure(bg=CLR_BG)
        self._model_info_text = scrolledtext.ScrolledText(
            info_lf, bg=CLR_INPUT, fg=CLR_FG,
            font=("Consolas", 9), state="disabled",
            wrap="word", height=10)
        self._model_info_text.pack(fill="both", expand=True, padx=4, pady=4)

        # 초기 로드
        self.root.after(300, self._refresh_model_list)

    # ═══════════════════════════════════════════════════════
    # TAB 4 — 로그
    # ═══════════════════════════════════════════════════════

    def _build_log_tab(self, parent):
        parent.configure(bg=CLR_BG)
        ctrl = _frame(parent, bg=CLR_BG)
        ctrl.pack(fill="x", padx=16, pady=(8, 4))
        _label(ctrl, "실시간 시스템 로그", bold=True,
               color=CLR_FG, size=11).pack(side="left")
        _btn(ctrl, "로그 지우기", self._clear_log,
             color=CLR_PANEL).pack(side="right", padx=4)
        _btn(ctrl, "로그 저장", self._save_log,
             color=CLR_PANEL).pack(side="right", padx=4)

        self._log_text = scrolledtext.ScrolledText(
            parent, bg=CLR_INPUT, fg=CLR_FG,
            font=("Consolas", 9), state="disabled",
            wrap="word")
        self._log_text.pack(fill="both", expand=True, padx=16, pady=(0, 12))

    # ═══════════════════════════════════════════════════════
    # 파일 브라우저 헬퍼
    # ═══════════════════════════════════════════════════════

    def _browse_train_input(self):
        if self._t_src_mode.get() == "file":
            p = filedialog.askopenfilename(
                title="학습 엑셀 파일 선택",
                filetypes=[("Excel", "*.xlsx *.xls"), ("All", "*.*")])
        else:
            p = filedialog.askdirectory(title="학습 엑셀 디렉터리 선택")
        if p:
            self._t_input_var.set(p)

    def _detect_columns(self):
        """
        파일의 첫 몇 행을 읽어 드롭다운에 채운다.
        헤더 없음 모드: 'A열 (샘플: ...)' 형식으로 표시
        헤더 있음 모드: 실제 컬럼명 표시
        """
        import pandas as pd
        import glob as _glob

        inp = self._t_input_var.get().strip()
        if not inp:
            messagebox.showwarning("파일 없음", "먼저 파일 또는 폴더를 선택하세요.")
            return

        # 단일 파일 or 폴더에서 첫 번째 xlsx
        if os.path.isfile(inp):
            target = inp
        else:
            files = sorted(_glob.glob(os.path.join(inp, "*.xlsx")))
            if not files:
                messagebox.showwarning("파일 없음", f"폴더에 xlsx 파일이 없습니다:\n{inp}")
                return
            target = files[0]

        skiprows  = self._t_skiprows.get()
        no_header = self._t_no_header.get()

        try:
            if no_header:
                # 헤더 없음: 첫 3행 읽어서 샘플값 확인
                df_sample = pd.read_excel(
                    target, engine="openpyxl",
                    header=None, skiprows=skiprows, nrows=3
                )
                n_cols = len(df_sample.columns)
                # 열 문자 생성 (A,B,...,Z,AA,AB,...)
                def _idx_to_letter(idx):
                    result = ""
                    n = idx + 1
                    while n:
                        n, r = divmod(n - 1, 26)
                        result = chr(65 + r) + result
                    return result

                options = [""]
                for i in range(n_cols):
                    letter = _idx_to_letter(i)
                    sample = str(df_sample.iloc[0, i])[:15].replace("\n", " ")
                    options.append(f"{letter}열  (샘플: {sample})")

                # 내부적으로 저장할 실제 키: 열 문자 (A, B, G...)
                # _col_key_map: 드롭다운 표시값 → 실제 컬럼 인덱스(int)
                self._col_key_map = {}
                for opt in options[1:]:
                    letter = opt.split("열")[0].strip()
                    idx = sum((ord(c) - 64) * (26 ** (len(letter) - 1 - j))
                              for j, c in enumerate(letter)) - 1
                    self._col_key_map[opt] = idx

                for cb in self._col_combos.values():
                    cb["values"] = options

                hint = (f"헤더 없음 모드 | {os.path.basename(target)} | "
                        f"{n_cols}열 | 건너뜀={skiprows}행")
            else:
                # 헤더 있음: 첫 행을 헤더로
                df_sample = pd.read_excel(
                    target, engine="openpyxl",
                    header=0, skiprows=skiprows if skiprows > 0 else None,
                    nrows=1
                )
                cols = [str(c) for c in df_sample.columns]
                options = [""] + cols
                self._col_key_map = {}  # 헤더 모드는 컬럼명 직접 사용
                for cb in self._col_combos.values():
                    cb["values"] = options
                hint = (f"헤더 있음 모드 | {os.path.basename(target)} | "
                        f"{len(cols)}개 컬럼 | 건너뜀={skiprows}행")

        except Exception as e:
            messagebox.showerror("감지 실패", str(e))
            return

        self._detect_hint.configure(text=hint)
        self._logger.info(f"컬럼 감지: {hint}")
        self._logger.info(f"선택 가능 목록: {options[1:]}")

    def _on_train_src_toggle(self):
        lbl = "파일 선택" if self._t_src_mode.get() == "file" else "폴더 선택"
        self._t_browse_btn.configure(text=lbl)

    def _browse_base_model(self):
        p = filedialog.askdirectory(title="로컬 모델 폴더 선택")
        if p:
            self._t_base_model.set(p)

    def _browse_save_path(self):
        p = filedialog.askdirectory(title="모델 저장 폴더 선택")
        if p:
            self._t_save_path.set(p)

    def _browse_pred_model(self):
        p = filedialog.askdirectory(title="저장된 모델 폴더 선택")
        if p:
            self._p_model_var.set(p)

    def _browse_pred_input(self):
        if self._p_src_mode.get() == "file":
            p = filedialog.askopenfilename(
                title="예측 엑셀 파일 선택",
                filetypes=[("Excel", "*.xlsx *.xls"), ("All", "*.*")])
        else:
            p = filedialog.askdirectory(title="예측 엑셀 디렉터리 선택")
        if p:
            self._p_input_var.set(p)

    def _on_pred_src_toggle(self):
        lbl = "파일 선택" if self._p_src_mode.get() == "file" else "폴더 선택"
        self._p_browse_btn.configure(text=lbl)

    def _on_pred_out_toggle(self):
        mode = self._p_out_mode.get()
        default = ("./predictions/result.xlsx"
                   if mode == "combined" else "./predictions")
        self._p_output_var.set(default)

    def _browse_pred_output(self):
        if self._p_out_mode.get() == "combined":
            p = filedialog.asksaveasfilename(
                title="통합 결과 저장 경로",
                defaultextension=".xlsx",
                filetypes=[("Excel", "*.xlsx")])
        else:
            p = filedialog.askdirectory(title="파일별 저장 폴더 선택")
        if p:
            self._p_output_var.set(p)

    def _browse_model_root(self):
        p = filedialog.askdirectory(title="모델 루트 폴더 선택")
        if p:
            self._m_root_var.set(p)
            self._refresh_model_list()

    # ═══════════════════════════════════════════════════════
    # 워커 스레드 공통
    # ═══════════════════════════════════════════════════════

    def _stop_worker(self):
        self._running = False
        self._set_status("중지 요청됨 (현재 배치 완료 후 종료)")

    def _lock_ui(self, training=True):
        self._running = True
        if training:
            self._train_btn.configure(state="disabled")
            self._stop_btn.configure(state="normal")
            self._train_prog.start(12)
        else:
            self._pred_btn.configure(state="disabled")
            self._pred_prog.start(12)

    def _unlock_ui(self, training=True):
        self._running = False
        if training:
            self._train_btn.configure(state="normal")
            self._stop_btn.configure(state="disabled")
            self._train_prog.stop()
        else:
            self._pred_btn.configure(state="normal")
            self._pred_prog.stop()

    # ═══════════════════════════════════════════════════════
    # 학습 실행
    # ═══════════════════════════════════════════════════════

    def _start_train(self):
        inp = self._t_input_var.get().strip()
        if not inp:
            messagebox.showwarning("입력 없음", "학습 데이터 파일 또는 디렉터리를 선택하세요.")
            return

        col_map = {k: v.get().strip() for k, v in self._col_vars.items()}
        col_map = {k.replace("_col", ""): v for k, v in col_map.items()}

        skiprows  = self._t_skiprows.get()
        no_header = self._t_no_header.get()
        col_key_map = getattr(self, "_col_key_map", {})

        params = dict(
            src_mode    = self._t_src_mode.get(),
            input_path  = inp,
            col_map     = col_map,
            skiprows    = skiprows,
            no_header   = no_header,
            col_key_map = col_key_map,
            base_model  = self._t_base_model.get().strip(),
            save_path   = self._t_save_path.get().strip(),
            epochs      = self._t_epochs.get(),
            batch_size  = self._t_batch.get(),
            lr          = float(self._t_lr_idx.get()),
            max_input   = self._t_max_in.get(),
            max_target  = self._t_max_tgt.get(),
            patience    = self._t_patience.get(),
            incremental = self._t_incremental.get(),
            save_merged = self._t_save_merged.get(),
        )

        # 이력 초기화
        for item in self._hist_tree.get_children():
            self._hist_tree.delete(item)

        self._lock_ui(training=True)
        self._train_status.set("학습 준비 중...")
        self._set_status("학습 중...")

        t = threading.Thread(target=self._train_worker, args=(params,), daemon=True)
        t.start()
        self._worker_thread = t

    def _train_worker(self, params: dict):
        try:
            import pandas as pd
            from utils import (load_excel, load_excel_dir,
                               merge_files, build_merge_report, save_dataset)
            from train import train as do_train, incremental_train

            self._ui_call(self._train_status.set, "데이터 로드 중...")

            col_map     = params["col_map"]
            skiprows    = params.get("skiprows", 0)
            no_header   = params.get("no_header", False)
            col_key_map = params.get("col_key_map", {})
            pandas_header = None if no_header else 0
            pandas_skip   = skiprows if skiprows > 0 else None

            # 헤더 없음 모드: 드롭다운 표시값("G열  (샘플: ...)") → 정수 인덱스로 변환
            def _resolve_col_map(df, raw_col_map, col_key_map):
                resolved = {}
                for key, val in raw_col_map.items():
                    if not val:
                        resolved[key] = ""
                        continue
                    if val in col_key_map:
                        # 열 위치 → 실제 컬럼명
                        idx = col_key_map[val]
                        if 0 <= idx < len(df.columns):
                            resolved[key] = df.columns[idx]
                            self._log_info(f"  {key}: '{val}' → 열[{idx}] '{df.columns[idx]}'")
                        else:
                            resolved[key] = ""
                    else:
                        resolved[key] = val
                return resolved

            if params["src_mode"] == "file":
                df = load_excel(params["input_path"],
                                header=pandas_header, skiprows=pandas_skip)
                col_map = _resolve_col_map(df, col_map, col_key_map)
                self._log_info(f"단일 파일 로드: {params['input_path']} ({len(df)}행)")
                df, summary = merge_files([(df, col_map, os.path.basename(params["input_path"]))])
                self._log_info(build_merge_report(summary))
            else:
                loaded = load_excel_dir(params["input_path"], progress=False,
                                        header=pandas_header, skiprows=pandas_skip)
                ok = [(fp, d) for fp, d, e in loaded if d is not None]
                if not ok:
                    raise ValueError(f"로드 가능한 xlsx 없음: {params['input_path']}")
                triples = []
                for fp, d in ok:
                    resolved = _resolve_col_map(d, col_map, col_key_map)
                    triples.append((d, resolved, os.path.basename(fp)))
                df, summary = merge_files(triples)
                self._log_info(build_merge_report(summary))
                if params["save_merged"]:
                    p = os.path.join(params["save_path"], "merged_dataset.xlsx")
                    save_dataset(df, p)
                    self._log_info(f"통합 데이터셋 저장: {p}")

            self._log_info(f"최종 학습 데이터: {len(df)}행")
            self._ui_call(self._train_status.set,
                          f"{'증분' if params['incremental'] else '신규'} 학습 중...")

            def progress_fn(epoch, total, log_str):
                if not self._running:
                    raise InterruptedError("사용자 중지")
                self._ui_call(self._train_status.set,
                              f"Epoch {epoch}/{total}")
                self._ui_call(self._append_history_row, log_str)

            train_kwargs = dict(
                df           = df,
                col_map      = col_map,
                save_path    = params["save_path"],
                epochs       = params["epochs"],
                batch_size   = params["batch_size"],
                learning_rate= params["lr"],
                max_input_len= params["max_input"],
                max_target_len=params["max_target"],
                patience     = params["patience"],
                progress_fn  = progress_fn,
            )
            if params["incremental"]:
                result = incremental_train(
                    new_df=df, col_map=col_map,
                    model_dir=params["base_model"],
                    **{k: v for k, v in train_kwargs.items()
                       if k not in ("df", "col_map", "base_model_path")},
                )
            else:
                result = do_train(
                    base_model_path=params["base_model"],
                    **train_kwargs,
                )

            self._ui_call(self._on_train_done, result)

        except InterruptedError:
            self._ui_call(self._train_status.set, "사용자 중지")
            self._ui_call(messagebox.showinfo, "중지", "학습이 중지되었습니다.")
        except Exception as e:
            tb = traceback.format_exc()
            self._log_err(f"학습 오류: {e}\n{tb}")
            self._ui_call(messagebox.showerror, "학습 오류",
                          f"{type(e).__name__}: {e}\n\n{tb[:600]}")
            self._ui_call(self._train_status.set, "오류 발생")
        finally:
            self._ui_call(self._unlock_ui, True)
            self._ui_call(self._set_status, "대기 중")

    def _on_train_done(self, result: dict):
        best = result.get("best_val_loss", 0)
        self._train_status.set(
            f"완료! Best Val Loss: {best:.4f}  →  {result['save_path']}")
        messagebox.showinfo("학습 완료",
                            f"Best Val Loss: {best:.4f}\n"
                            f"저장 경로: {result['save_path']}\n"
                            f"클래스 수: {result.get('num_classes', '?')}")
        self._refresh_model_list()

    def _append_history_row(self, log_str: str):
        """'[Epoch 01/10] Train Loss=... Acc=... | Val ...' 파싱 → Treeview"""
        import re
        m = re.search(
            r"Epoch\s+(\d+)/\d+.*?Train Loss=([\d.]+)\s+Acc=([\d.]+).*?"
            r"Val Loss=([\d.]+)\s+Acc=([\d.]+)", log_str)
        if m:
            self._hist_tree.insert("", "end", values=(
                m.group(1),
                f"{float(m.group(2)):.4f}",
                f"{float(m.group(3)):.3f}",
                f"{float(m.group(4)):.4f}",
                f"{float(m.group(5)):.3f}",
            ))
            self._hist_tree.yview_moveto(1)

    # ═══════════════════════════════════════════════════════
    # 예측 실행
    # ═══════════════════════════════════════════════════════

    def _start_predict(self):
        model_path = self._p_model_var.get().strip()
        inp        = self._p_input_var.get().strip()

        if not model_path or not os.path.isdir(model_path):
            messagebox.showwarning("모델 없음", f"모델 경로를 확인하세요:\n{model_path}")
            return
        if not inp:
            messagebox.showwarning("입력 없음", "예측 파일 또는 디렉터리를 선택하세요.")
            return

        params = dict(
            model_path  = model_path,
            src_mode    = self._p_src_mode.get(),
            input_path  = inp,
            output_path = self._p_output_var.get().strip(),
            out_mode    = self._p_out_mode.get(),
            before_col  = self._p_before_col.get().strip() or None,
            after_col   = self._p_after_col.get().strip()  or None,
            batch_size  = self._p_batch.get(),
            num_beams   = self._p_beams.get(),
            max_input   = self._p_max_in.get(),
            max_tokens  = self._p_max_tok.get(),
        )

        # 결과 트리 초기화
        for item in self._pred_tree.get_children():
            self._pred_tree.delete(item)
        self._pred_result_df = None

        self._lock_ui(training=False)
        self._pred_status.set("예측 준비 중...")
        self._set_status("예측 중...")

        t = threading.Thread(target=self._predict_worker, args=(params,), daemon=True)
        t.start()

    def _predict_worker(self, params: dict):
        try:
            from predict import predict_single, predict_dir

            self._ui_call(self._pred_status.set, "모델 로드 중...")

            if params["src_mode"] == "file":
                result_df = predict_single(
                    model_path    = params["model_path"],
                    input_file    = params["input_path"],
                    output_file   = params["output_path"],
                    before_col    = params["before_col"],
                    after_col     = params["after_col"],
                    batch_size    = params["batch_size"],
                    max_input_len = params["max_input"],
                    max_new_tokens= params["max_tokens"],
                    num_beams     = params["num_beams"],
                )
                self._ui_call(self._load_pred_preview, result_df)
                self._pred_result_df = result_df

            else:
                results, combined_df = predict_dir(
                    model_path   = params["model_path"],
                    input_dir    = params["input_path"],
                    output_mode  = params["out_mode"],
                    output_dir   = (params["output_path"]
                                    if params["out_mode"] == "separate"
                                    else "./predictions"),
                    output_file  = (params["output_path"]
                                    if params["out_mode"] == "combined"
                                    else "./predictions/combined.xlsx"),
                    before_col   = params["before_col"],
                    after_col    = params["after_col"],
                    batch_size   = params["batch_size"],
                    max_input_len= params["max_input"],
                    max_new_tokens=params["max_tokens"],
                    num_beams    = params["num_beams"],
                )
                preview = combined_df if combined_df is not None else None
                if preview is None and results:
                    frames = [r["df"] for r in results if r["df"] is not None]
                    import pandas as pd
                    preview = pd.concat(frames, ignore_index=True) if frames else None
                if preview is not None:
                    self._ui_call(self._load_pred_preview, preview)
                    self._pred_result_df = preview

            self._ui_call(self._pred_status.set, "예측 완료!")
            self._ui_call(self._set_status, "예측 완료")
            rows = len(self._pred_result_df) if self._pred_result_df is not None else 0
            self._ui_call(messagebox.showinfo, "예측 완료",
                          f"{rows}건 예측 완료\n저장 경로: {params['output_path']}")

        except Exception as e:
            tb = traceback.format_exc()
            self._log_err(f"예측 오류: {e}\n{tb}")
            self._ui_call(messagebox.showerror, "예측 오류",
                          f"{type(e).__name__}: {e}\n\n{tb[:600]}")
            self._ui_call(self._pred_status.set, "오류 발생")
        finally:
            self._ui_call(self._unlock_ui, False)
            self._ui_call(self._set_status, "대기 중")

    def _load_pred_preview(self, df):
        for item in self._pred_tree.get_children():
            self._pred_tree.delete(item)
        cols = ["pred_summary", "pred_reason", "pred_code", "confidence_score"]
        available = [c for c in cols if c in df.columns]
        for _, row in df[available].head(200).iterrows():
            vals = []
            for c in available:
                v = row[c]
                if c == "confidence_score":
                    v = f"{float(v):.1%}"
                else:
                    v = str(v)[:60]
                vals.append(v)
            self._pred_tree.insert("", "end", values=vals)

    def _save_pred_result(self):
        if self._pred_result_df is None:
            messagebox.showwarning("저장 불가", "예측 결과가 없습니다.")
            return
        p = filedialog.asksaveasfilename(
            title="예측 결과 저장",
            defaultextension=".xlsx",
            filetypes=[("Excel", "*.xlsx")])
        if p:
            self._pred_result_df.to_excel(p, index=False, engine="openpyxl")
            messagebox.showinfo("저장 완료", f"저장: {p}")

    # ═══════════════════════════════════════════════════════
    # 모델 관리
    # ═══════════════════════════════════════════════════════

    def _refresh_model_list(self):
        from model_loader import list_models
        root_path = self._m_root_var.get().strip()
        models = list_models(root_path)
        for item in self._model_tree.get_children():
            self._model_tree.delete(item)
        for m in models:
            self._model_tree.insert("", "end", iid=m["path"], values=(
                m["name"],
                m.get("model_type", "?"),
                m.get("num_classes", "?"),
                m.get("created_at", "?")[:19],
            ))
        if not models:
            self._model_info_text.configure(state="normal")
            self._model_info_text.delete("1.0", "end")
            self._model_info_text.insert("end",
                f"저장된 모델 없음: {root_path}\n\n"
                "학습 탭에서 학습을 먼저 완료하세요.")
            self._model_info_text.configure(state="disabled")

    def _on_model_select(self, event):
        sel = self._model_tree.selection()
        if not sel:
            return
        model_path = sel[0]
        try:
            from model_loader import load_config
            config = load_config(model_path)
            lines = []
            for k, v in config.items():
                if isinstance(v, dict):
                    lines.append(f"\n[{k}]")
                    for kk, vv in v.items():
                        lines.append(f"  {kk:<22}: {vv}")
                elif isinstance(v, list):
                    lines.append(f"{k:<24}: {', '.join(str(x) for x in v)}")
                else:
                    lines.append(f"{k:<24}: {v}")
            text = "\n".join(lines)
        except Exception as e:
            text = f"config 로드 실패: {e}"

        self._model_info_text.configure(state="normal")
        self._model_info_text.delete("1.0", "end")
        self._model_info_text.insert("end", text)
        self._model_info_text.configure(state="disabled")

        # 예측 탭 모델 경로 자동 설정
        self._p_model_var.set(model_path)

    # ═══════════════════════════════════════════════════════
    # 로그 탭
    # ═══════════════════════════════════════════════════════

    def _clear_log(self):
        self._log_text.configure(state="normal")
        self._log_text.delete("1.0", "end")
        self._log_text.configure(state="disabled")

    def _save_log(self):
        ts  = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        p   = filedialog.asksaveasfilename(
            initialfile=f"log_{ts}.txt",
            defaultextension=".txt",
            filetypes=[("Text", "*.txt"), ("All", "*.*")])
        if p:
            content = self._log_text.get("1.0", "end")
            with open(p, "w", encoding="utf-8") as f:
                f.write(content)
            messagebox.showinfo("저장 완료", f"로그 저장: {p}")

    # ═══════════════════════════════════════════════════════
    # 유틸
    # ═══════════════════════════════════════════════════════

    def _ui_call(self, fn, *args):
        """스레드에서 안전하게 UI 함수 호출"""
        self.root.after(0, fn, *args)

    def _log_info(self, msg: str):
        logging.getLogger("Worker").info(msg)

    def _log_err(self, msg: str):
        logging.getLogger("Worker").error(msg)


# ═══════════════════════════════════════════════════════════════
# 진입점
# ═══════════════════════════════════════════════════════════════

def main():
    try:
        root = tk.Tk()
    except Exception as e:
        print(f"[치명적 오류] tkinter 초기화 실패: {e}")
        import traceback; traceback.print_exc()
        input("엔터를 눌러 종료...")
        return

    # DPI 대응 (Windows 고해상도)
    try:
        from ctypes import windll
        windll.shcore.SetProcessDpiAwareness(1)
    except Exception:
        pass
    try:
        root.tk.call("tk", "scaling", 1.25)
    except Exception:
        pass

    try:
        app = DocChangeApp(root)
    except Exception as e:
        import traceback
        err_msg = traceback.format_exc()
        print(f"[치명적 오류] GUI 초기화 실패:\n{err_msg}")
        try:
            messagebox.showerror("초기화 오류", f"{e}\n\n{err_msg}")
        except Exception:
            pass
        input("엔터를 눌러 종료...")
        return

    root.mainloop()


if __name__ == "__main__":
    main()
