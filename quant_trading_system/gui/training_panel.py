# gui/training_panel.py — 모델 학습 패널
from __future__ import annotations
import os, sys, threading, tkinter as tk
from tkinter import ttk, messagebox, scrolledtext
import logging
import numpy as np
import pandas as pd

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)

from config import AppSettings
from data import DataLoader
from data.korean_stocks import get_name
from features import ROIDetector, CVFeatureExtractor, TSFeatureExtractor
from models.hybrid import HybridModel
from models.trainer import ModelTrainer
from models.store import ModelStore
from gui.tooltip import add_tooltip
from gui.ui_meta import META, PRESETS, apply_preset

logger = logging.getLogger("quant.gui.train")


class TrainingPanel:
    """
    학습 패널
    - 데이터 탭에 등록된 종목 중 1개 선택 → 학습
    - GPU 가속 자동 적용 (CUDA 감지)
    - 학습 중 실시간 손실 곡선 표시
    - 모델 저장 / 불러오기 / 삭제
    """

    def __init__(self, parent, settings: AppSettings, on_change, on_complete=None):
        self.settings   = settings
        self.on_change  = on_change
        self._on_complete = on_complete   # 학습 완료 콜백
        self._stop_event = threading.Event()
        self._thread    = None
        self.trainer    = None
        self._store     = ModelStore(os.path.join(BASE_DIR, settings.model_dir))

        self.frame = ttk.Frame(parent)
        self._build()
        self.refresh_symbols()   # 초기 종목 목록 표시

    # ─────────────────────────────────────────────
    # 공개 API (main_window에서 호출)
    # ─────────────────────────────────────────────

    def refresh_symbols(self):
        """데이터 탭 종목 변경 시 호출 → 종목 Listbox 동기화

        성능: get_name() 결과를 _name_cache에 메모이제이션 → DB 반복 탐색 방지
        """
        if not hasattr(self, "_name_cache"):
            self._name_cache: dict[str, str] = {}

        symbols = self.settings.data.symbols
        self.sym_listbox.delete(0, "end")
        for t in symbols:
            if t not in self._name_cache:
                self._name_cache[t] = get_name(t)
            name = self._name_cache[t]
            icon = "📦" if self._store.has_model(t) else "  "
            self.sym_listbox.insert("end", f"{icon}  {t}   {name}")

        # 첫 번째 항목 자동 선택
        if symbols:
            self.sym_listbox.selection_set(0)
            self.sym_listbox.activate(0)
            self._on_sym_select()

        count = len(symbols)
        self.sym_count_var.set(
            f"총 {count}개 종목  (데이터 탭에서 추가하세요)" if count == 0
            else f"총 {count}개 종목"
        )

    # ─────────────────────────────────────────────
    # UI 구성
    # ─────────────────────────────────────────────

    def notify_data_updated(self):
        """데이터 탭에서 새 데이터 다운로드 완료 시 메인 윈도우가 호출"""
        if hasattr(self, "_data_update_bar"):
            self._data_update_bar.configure(height=30)

    def dismiss_data_update_notice(self):
        """탭 방문 or 학습 시작 시 배너 숨김"""
        if hasattr(self, "_data_update_bar"):
            self._data_update_bar.configure(height=0)

    def _build(self):
        # 데이터 업데이트 알림 배너 (초기 숨김 — notify_data_updated() 시 표시)
        self._data_update_bar = tk.Frame(self.frame, bg="#2a2a00", height=0)
        self._data_update_bar.pack(fill="x")
        self._data_update_bar.pack_propagate(False)
        _bar_inner = tk.Frame(self._data_update_bar, bg="#2a2a00")
        _bar_inner.pack(fill="both", expand=True, padx=10)
        tk.Label(
            _bar_inner,
            text="🔔  새 데이터가 다운로드되었습니다  —  최신 데이터로 모델을 재학습하면 예측 정확도가 향상됩니다",
            bg="#2a2a00", fg="#f9e2af",
            font=("맑은 고딕", 9, "bold"), anchor="w",
        ).pack(side="left", fill="y")
        ttk.Button(_bar_inner, text="✕ 닫기",
                   command=self.dismiss_data_update_notice).pack(side="right")

        # 안내 배너
        banner = tk.Frame(self.frame, bg="#1a1a2e", pady=7)
        banner.pack(fill="x", padx=6, pady=(6, 2))
        tk.Label(
            banner,
            text="🧠  2단계: AI 모델 학습  —  종목 선택 → 학습 시작 → 완료 후 백테스트 탭으로",
            bg="#1a1a2e", fg="#89b4fa",
            font=("맑은 고딕", 10, "bold"),
        ).pack(side="left", padx=12)
        tk.Label(
            banner,
            text="  (학습 없이도 백테스트 가능: 신호 소스 = 모멘텀 선택)",
            bg="#1a1a2e", fg="#9399b2",
            font=("맑은 고딕", 8),
        ).pack(side="left")

        pane = tk.PanedWindow(self.frame, orient="horizontal",
                              bg="#1e1e2e", sashwidth=4)
        pane.pack(fill="both", expand=True, padx=6, pady=2)

        # 왼쪽 패널: 스크롤 가능하게 감싸기
        left_outer = ttk.LabelFrame(pane, text="① 학습 설정", padding=4)
        right      = ttk.LabelFrame(pane, text="② 학습 진행 / 결과", padding=8)
        pane.add(left_outer, minsize=240)
        pane.add(right,      minsize=500)

        # Canvas + Scrollbar 스크롤 프레임
        _left_canvas = tk.Canvas(left_outer, bg="#1e1e2e",
                                 highlightthickness=0)
        _left_vsb    = ttk.Scrollbar(left_outer, orient="vertical",
                                     command=_left_canvas.yview)
        _left_canvas.configure(yscrollcommand=_left_vsb.set)
        _left_vsb.pack(side="right", fill="y")
        _left_canvas.pack(side="left", fill="both", expand=True)

        left = ttk.Frame(_left_canvas, padding=4)
        _win_id = _left_canvas.create_window((0, 0), window=left, anchor="nw")

        def _on_frame_resize(event):
            _left_canvas.configure(scrollregion=_left_canvas.bbox("all"))
            _left_canvas.itemconfig(_win_id, width=_left_canvas.winfo_width())

        left.bind("<Configure>", _on_frame_resize)
        _left_canvas.bind("<Configure>",
                          lambda e: _left_canvas.itemconfig(_win_id,
                                                            width=e.width))

        # 마우스 휠 스크롤
        def _on_mousewheel(event):
            _left_canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")
        _left_canvas.bind("<MouseWheel>", _on_mousewheel)

        self._build_left(left)
        self._build_right(right)

    def _build_left(self, parent):
        # ── 빠른 프리셋 선택 ───────────────────────
        preset_fr = ttk.LabelFrame(parent, text="⚡ 빠른 설정 프리셋", padding=4)
        preset_fr.pack(fill="x", pady=(0, 6))

        preset_row = ttk.Frame(preset_fr)
        preset_row.pack(fill="x")

        presets = [
            ("🟢 초보자 추천", "beginner",
             "빠른 학습, 안전한 기본값\nd_model=128, 레이어=2, epoch=50"),
            ("⚡ 단기 전략",   "short_term",
             "짧은 예측 기간, 작은 모델\nd_model=128, lookahead=3"),
            ("📈 중기 전략",   "mid_term",
             "균형 잡힌 설정\nd_model=256, 레이어=4, epoch=100"),
            ("🔬 고급",        "advanced",
             "고성능 모델\nd_model=512, 레이어=6, epoch=200\n⚠️ GPU 필요"),
        ]
        for name, key, tip in presets:
            btn = ttk.Button(
                preset_row, text=name,
                command=lambda k=key: self._apply_train_preset(k),
                width=12,
            )
            btn.pack(side="left", padx=2)
            add_tooltip(btn, tip)

        # ── ① 학습 종목 선택 (다중 선택 가능) ────────
        sym_fr = ttk.LabelFrame(
            parent,
            text="학습 종목 선택  (Ctrl·Shift+클릭 다중 선택 → 순차 학습)",
            padding=4,
        )
        sym_fr.pack(fill="x", pady=(0, 6))

        self.sym_count_var = tk.StringVar(value="")
        tk.Label(sym_fr, textvariable=self.sym_count_var,
                 bg="#1e1e2e", fg="#9399b2",
                 font=("맑은 고딕", 9)).pack(anchor="w")

        lb_fr = tk.Frame(sym_fr, bg="#1e1e2e")
        lb_fr.pack(fill="x")

        self.sym_listbox = tk.Listbox(
            lb_fr,
            selectmode="extended",          # ← 다중 선택
            bg="#181825", fg="#cdd6f4",
            selectbackground="#89b4fa", selectforeground="#1e1e2e",
            font=("맑은 고딕", 9),
            relief="flat", height=10,
            exportselection=False,          # 다른 위젯 클릭 시 선택 유지
        )
        vsb = ttk.Scrollbar(lb_fr, orient="vertical",
                            command=self.sym_listbox.yview)
        self.sym_listbox.configure(yscrollcommand=vsb.set)
        self.sym_listbox.pack(side="left", fill="both", expand=True)
        vsb.pack(side="right", fill="y")

        # 마우스 선택 이벤트
        self.sym_listbox.bind("<<ListboxSelect>>", lambda e: self._on_sym_select())

        # 키보드 ↑↓ 이동: active 위치를 selection으로 동기화 (modifier 없을 때만)
        def _lb_key_nav(event):
            def _sync(e=event):
                active = self.sym_listbox.index("active")
                if active < 0:
                    return
                # Shift/Ctrl 없으면 단일 선택으로 고정
                if e.state & 0x0001 == 0 and e.state & 0x0004 == 0:
                    self.sym_listbox.selection_clear(0, "end")
                    self.sym_listbox.selection_set(active)
                self._on_sym_select()
            self.frame.after(30, _sync)

        for _k in ("<Up>", "<Down>", "<Prior>", "<Next>", "<Home>", "<End>"):
            self.sym_listbox.bind(_k, _lb_key_nav)

        # 하단 버튼 행
        lb_btn = ttk.Frame(sym_fr)
        lb_btn.pack(fill="x", pady=(4, 0))
        ttk.Button(lb_btn, text="전체 선택",
                   command=lambda: self.sym_listbox.select_set(0, "end")
                   ).pack(side="left", padx=(0, 4))
        ttk.Button(lb_btn, text="선택 해제",
                   command=lambda: self.sym_listbox.selection_clear(0, "end")
                   ).pack(side="left", padx=(0, 8))
        ttk.Button(lb_btn, text="↺ 새로고침",
                   command=self.refresh_symbols).pack(side="left")

        # ── 뉴스 AI 상태 배지 ──────────────────────
        news_status_fr = tk.Frame(parent, bg="#1e1e2e")
        news_status_fr.pack(fill="x", pady=(0, 4))

        news_enabled = getattr(getattr(self.settings, "news", None),
                               "use_news_in_model", False)
        badge_text  = "뉴스 AI 피처: 활성 (40D)" if news_enabled else "뉴스 AI 피처: 비활성"
        badge_bg    = "#1a3020" if news_enabled else "#2a1a1a"
        badge_fg    = "#a6e3a1" if news_enabled else "#6c7086"
        badge_dot   = "●" if news_enabled else "○"
        self._news_badge = tk.Label(
            news_status_fr,
            text=f"  {badge_dot}  {badge_text}  ",
            bg=badge_bg, fg=badge_fg,
            font=("맑은 고딕", 8, "bold"),
            relief="flat", padx=4, pady=2,
        )
        self._news_badge.pack(side="left", padx=2)
        add_tooltip(self._news_badge,
                    "뉴스 AI 피처가 모델 학습에 포함되는지 표시합니다.\n"
                    "설정 탭 → 뉴스/외부환경 → '모델에 뉴스 피처 사용' 항목으로 전환하세요.")

        # ── ② 저장된 모델 정보 ─────────────────────
        mdl_fr = ttk.LabelFrame(parent, text="저장된 모델", padding=4)
        mdl_fr.pack(fill="x", pady=(0, 6))

        self.model_info_var = tk.StringVar(value="종목을 선택하세요")
        tk.Label(mdl_fr, textvariable=self.model_info_var,
                 bg="#1e1e2e", fg="#a6adc8",
                 font=("맑은 고딕", 9),
                 justify="left", wraplength=260).pack(anchor="w")

        mdl_btn_fr = ttk.Frame(mdl_fr)
        mdl_btn_fr.pack(fill="x", pady=(6, 0))
        self.load_btn = ttk.Button(
            mdl_btn_fr, text="📂 불러오기",
            command=self._load_model, state="disabled")
        self.load_btn.pack(side="left", padx=(0, 4))
        self.del_btn = ttk.Button(
            mdl_btn_fr, text="🗑 삭제",
            command=self._delete_model, state="disabled",
            style="Danger.TButton")
        self.del_btn.pack(side="left", padx=(0, 8))
        ttk.Button(
            mdl_btn_fr, text="📊 모델 정보",
            command=self._show_model_info_dialog,
        ).pack(side="left")
        add_tooltip(mdl_btn_fr.winfo_children()[-1],
                    "모든 종목의 AI 모델 학습 상태 (날짜·성능·버전)를\n"
                    "모달리스 창에서 한눈에 확인합니다.")

        # ── ③ 모델 파라미터 ────────────────────────
        param_fr = ttk.LabelFrame(parent, text="모델 파라미터  (🖱️ 항목에 마우스 올리면 설명)", padding=4)
        param_fr.pack(fill="x", pady=(0, 6))

        # (UI 라벨, param_vars 키, 기본값, META 키)
        params = [
            ("분석 구간 길이",        "segment_length", 30,    "roi.segment_length"),
            ("예측 기간",             "lookahead",      5,     "roi.lookahead"),
            ("모델 크기 (d_model)",   "d_model",        256,   "model.d_model"),
            ("Transformer 레이어",    "num_layers",     4,     "model.num_encoder_layers"),
            ("어텐션 헤드 수 (nhead)","nhead",          8,     "model.nhead"),
            ("드롭아웃",              "dropout",        0.1,   "model.dropout"),
            ("학습률",                "lr",             1e-4,  "model.learning_rate"),
            ("배치 크기",             "batch_size",     32,    "model.batch_size"),
            ("최대 에폭 수",          "epochs",         100,   "model.epochs"),
            ("Early Stopping 인내",   "patience",       15,    "model.patience"),
        ]

        self.param_vars: dict[str, tk.StringVar] = {}
        for i, (label, key, default, meta_key) in enumerate(params):
            meta_info    = META.get(meta_key, {})
            detail_text  = meta_info.get("detail", meta_info.get("help", ""))
            unit_text    = meta_info.get("unit", "")
            beginner_val = meta_info.get("beginner", default)

            # 레이블 (툴팁 포함)
            lbl = ttk.Label(param_fr, text=label + ":")
            lbl.grid(row=i, column=0, sticky="w", padx=(4, 2), pady=1)
            if detail_text:
                add_tooltip(lbl, detail_text)

            # 입력 필드 (툴팁 포함)
            var = tk.StringVar(value=str(default))
            self.param_vars[key] = var
            entry = ttk.Entry(param_fr, textvariable=var, width=9)
            entry.grid(row=i, column=1, sticky="w", padx=2)
            if detail_text:
                add_tooltip(entry, detail_text)

            # 초보자 추천값 배지 (클릭하면 적용)
            if str(beginner_val) != str(default):
                rec = tk.Label(
                    param_fr,
                    text=f"↩{beginner_val}",
                    bg="#1e1e2e", fg="#89b4fa",
                    font=("맑은 고딕", 8), cursor="hand2",
                )
                rec.grid(row=i, column=2, sticky="w", padx=(2, 0))
                rec.bind("<Button-1>",
                         lambda e, v=var, b=beginner_val: v.set(str(b)))
                add_tooltip(rec, f"클릭 → 초보 추천값 '{beginner_val}'")

        # ── ④ 하드웨어 ─────────────────────────────
        hw_fr = ttk.LabelFrame(parent, text="하드웨어", padding=4)
        hw_fr.pack(fill="x", pady=(0, 6))

        self.cuda_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(hw_fr, text="CUDA GPU 사용",
                        variable=self.cuda_var).pack(anchor="w")
        try:
            import torch
            if torch.cuda.is_available():
                name = torch.cuda.get_device_name(0)
                txt, fg = f"✅ GPU: {name}", "#a6e3a1"
            else:
                txt, fg = "⚠️  CUDA 없음 (CPU 학습)", "#f9e2af"
        except ImportError:
            txt, fg = "❌ PyTorch 미설치", "#f38ba8"
        tk.Label(hw_fr, text=txt, bg="#1e1e2e", fg=fg,
                 font=("맑은 고딕", 9)).pack(anchor="w")

        # ── ⑤ 실행 버튼 ────────────────────────────
        btn_fr = ttk.Frame(parent)
        btn_fr.pack(fill="x", pady=(4, 8))

        self.train_btn = ttk.Button(
            btn_fr, text="🚀 학습 시작",
            command=self._start_training,
            style="Accent.TButton")
        self.train_btn.pack(side="left", padx=(0, 4))

        self.stop_btn = ttk.Button(
            btn_fr, text="⏹ 중단",
            command=self._stop_training,
            style="Danger.TButton",
            state="disabled")
        self.stop_btn.pack(side="left")

    # ─────────────────────────────────────────────
    # 프리셋 적용
    # ─────────────────────────────────────────────

    def _apply_train_preset(self, preset_key: str):
        """
        프리셋 버튼 클릭 시 학습 파라미터 var에 해당 값을 일괄 적용.
        apply_preset()은 settings 객체에 쓰지만, 여기서는 UI 변수만 업데이트.
        """
        preset = PRESETS.get(preset_key)
        if not preset:
            return

        # PRESETS 점(.) 키 → param_vars 키 매핑
        _map = {
            "roi.segment_length":       "segment_length",
            "roi.lookahead":            "lookahead",
            "model.d_model":            "d_model",
            "model.num_encoder_layers": "num_layers",
            "model.nhead":              "nhead",
            "model.dropout":            "dropout",
            "model.learning_rate":      "lr",
            "model.batch_size":         "batch_size",
            "model.epochs":             "epochs",
            "model.patience":           "patience",
        }

        applied = []
        for dotted_key, value in preset["settings"].items():
            pkey = _map.get(dotted_key)
            if pkey and pkey in self.param_vars:
                self.param_vars[pkey].set(str(value))
                applied.append(f"{pkey}={value}")

        name = preset.get("name", preset_key)
        desc = preset.get("description", "")
        self._log(f"\n✅ 프리셋 적용: {name}")
        if desc:
            for line in desc.strip().split("\n"):
                self._log(f"   {line}")
        if applied:
            self._log(f"   변경된 파라미터: {', '.join(applied)}")

    def _build_right(self, parent):
        # ── 전체 배치 진행 현황 ─────────────────────
        batch_fr = tk.Frame(parent, bg="#181825", padx=6, pady=4)
        batch_fr.pack(fill="x", pady=(0, 4))
        self.batch_status_var = tk.StringVar(value="")
        tk.Label(batch_fr, textvariable=self.batch_status_var,
                 bg="#181825", fg="#f9e2af",
                 font=("맑은 고딕", 9, "bold")).pack(side="left")

        # ── 개별 에폭 진행바 ────────────────────────
        prog_fr = ttk.Frame(parent)
        prog_fr.pack(fill="x", pady=(0, 4))
        ttk.Label(prog_fr, text="Epoch:").pack(side="left", padx=4)
        self.progress = ttk.Progressbar(prog_fr, mode="determinate")
        self.progress.pack(side="left", padx=4, fill="x", expand=True)
        self.pct_var = tk.StringVar(value="0%")
        ttk.Label(prog_fr, textvariable=self.pct_var, width=6).pack(side="left")

        self.status_var = tk.StringVar(value="대기 중")
        ttk.Label(parent, textvariable=self.status_var,
                  foreground="#89b4fa",
                  font=("맑은 고딕", 10)).pack(anchor="w", pady=(0, 4))

        # 손실 그래프
        chart_fr = ttk.LabelFrame(parent, text="학습 손실 곡선", padding=4)
        chart_fr.pack(fill="x", pady=(0, 6))
        self.chart_canvas = tk.Canvas(chart_fr, height=180,
                                      bg="#181825", highlightthickness=0)
        self.chart_canvas.pack(fill="x")
        self._loss_history: dict = {"train": [], "val": []}

        # 결과 표
        res_fr = ttk.LabelFrame(parent, text="학습 결과", padding=6)
        res_fr.pack(fill="x", pady=(0, 6))
        cols = ("종목", "ROI수", "최적Epoch", "Val손실", "학습시간(s)", "파라미터수")
        self.result_tree = ttk.Treeview(res_fr, columns=cols,
                                        show="headings", height=4)
        for c in cols:
            self.result_tree.heading(c, text=c)
            self.result_tree.column(c, width=100, anchor="center")
        self.result_tree.column("종목", width=140)
        self.result_tree.pack(fill="x")
        # 키보드 이동 시 포커스/선택 유지
        for key in ("<Up>", "<Down>"):
            self.result_tree.bind(key, lambda e: self.frame.after(
                30, lambda t=e.widget: t.selection_set(t.focus()) if t.focus() else None))

        # 로그
        ttk.Label(parent, text="학습 로그:").pack(anchor="w")
        self.log_box = scrolledtext.ScrolledText(
            parent, height=10, bg="#181825", fg="#a6adc8",
            font=("Consolas", 9), relief="flat", state="disabled",
        )
        self.log_box.pack(fill="both", expand=True)

    # ─────────────────────────────────────────────
    # 종목 선택 이벤트
    # ─────────────────────────────────────────────

    def _on_sym_select(self):
        """Listbox 선택 변경 → 모델 정보 업데이트"""
        sym = self._get_selected_symbol()
        if not sym:
            self.model_info_var.set("종목을 선택하세요")
            self.load_btn.config(state="disabled")
            self.del_btn.config(state="disabled")
            return
        self._refresh_model_info(sym)

    def _get_selected_symbol(self) -> str:
        """
        현재 선택된 ticker 코드만 반환.

        Listbox 텍스트 형식: "📦  005930.KS   삼성전자"
        가장 안전한 방법은 settings.data.symbols 인덱스로 역참조하는 것.
        파싱 실패 시 인덱스 기반 fallback.
        """
        sel = self.sym_listbox.curselection()
        if not sel:
            return ""
        idx  = sel[0]
        syms = self.settings.data.symbols
        # ① 인덱스 기반 직접 참조 (가장 안전)
        if idx < len(syms):
            return syms[idx]
        # ② fallback: 텍스트에서 종목코드 토큰 추출
        text   = self.sym_listbox.get(idx)
        parts  = text.split()
        non_emoji = [p for p in parts if p.isprintable() and p not in ("📦", "✗")]
        for part in non_emoji:
            # KRX: "005930.KS" 또는 "005930"(6자리 숫자)
            stripped = part.strip()
            if stripped.isdigit() and len(stripped) == 6:
                return stripped
            if "." in part and any(part.upper().endswith(s)
                                   for s in (".KS", ".KQ", ".KX")):
                return part
            # 해외 주식: 알파벳만 2~6자 (AAPL, MSFT 등)
            if part.isascii() and part.isupper() and 2 <= len(part) <= 6:
                return part
        return non_emoji[0] if non_emoji else ""

    def _refresh_model_info(self, sym: str):
        """선택 종목의 저장된 모델 정보 표시"""
        # has_model / get_meta 는 내부적으로 _sym_dir() 사용 → safe name 변환 포함
        if not self._store.has_model(sym):
            self.model_info_var.set(f"[{sym}]\n저장된 모델 없음")
            self.load_btn.config(state="disabled")
            self.del_btn.config(state="disabled")
            return

        meta    = self._store.get_meta(sym)
        ts      = meta.get("timestamp", "")
        metrics = meta.get("metrics", {})

        if not ts and not metrics:
            self.model_info_var.set(f"[{sym}]\n메타 정보 없음 (모델 파일은 존재)")
            self.load_btn.config(state="normal")
            self.del_btn.config(state="normal")
            return

        ts_fmt  = (f"{ts[:4]}-{ts[4:6]}-{ts[6:8]} {ts[9:11]}:{ts[11:13]}"
                   if len(ts) >= 13 else ts or "—")
        ep      = metrics.get("best_epoch", "—")
        vl      = metrics.get("best_val_loss", metrics.get("val_loss", 0))
        np_     = metrics.get("n_params", "—")
        vl_str  = f"{vl:.6f}" if isinstance(vl, float) else str(vl)
        np_str  = f"{np_:,}" if isinstance(np_, int) else str(np_)

        info = (
            f"[{sym}]  latest.pt\n"
            f"최근 학습: {ts_fmt}\n"
            f"최적 Epoch: {ep}  |  Val Loss: {vl_str}\n"
            f"파라미터: {np_str}"
        )
        self.model_info_var.set(info)
        self.load_btn.config(state="normal")
        self.del_btn.config(state="normal")

    # ─────────────────────────────────────────────
    # 모델 불러오기 / 삭제
    # ─────────────────────────────────────────────

    def _load_model(self):
        """저장된 모델 불러오기 → 즉시 예측 가능 상태로 전환"""
        sym = self._get_selected_symbol()
        if not sym:
            return
        if not self._store.has_model(sym):
            messagebox.showwarning("알림", f"[{sym}] 저장된 모델이 없습니다.", parent=self.frame)
            return

        self._log(f"\n{'='*40}")
        self._log(f"[{sym}] 모델 불러오기 중...")
        self._set_status(f"[{sym}] 모델 로드 중...")

        def _do():
            try:
                params = self._get_params()
                cache_dir = os.path.join(BASE_DIR, self.settings.data.cache_dir)
                loader = DataLoader(cache_dir, self.settings.data.cache_ttl_hours)

                df = loader.load(sym, self.settings.data.period,
                                 self.settings.data.interval)
                if df is None or len(df) < 50:
                    self._log(f"[{sym}] 데이터 없음 — 데이터 탭에서 먼저 다운로드하세요.")
                    self._set_status("모델 로드 실패: 데이터 없음")
                    return

                roi_det = ROIDetector(
                    segment_length=params["segment_length"],
                    lookahead=params["lookahead"],
                )
                cv_ext = CVFeatureExtractor(
                    image_size=self.settings.model.image_size)
                ts_ext = TSFeatureExtractor(n_features=32)

                segs, labels, _ = roi_det.extract_segments(df)
                if len(segs) == 0:
                    self._log(f"[{sym}] ROI 구간 없음 — 데이터가 너무 짧습니다.")
                    self._set_status("모델 로드 실패: ROI 없음")
                    return

                images   = cv_ext.transform(segs)
                ts_feats = ts_ext.transform(segs)

                device = "cuda" if self.cuda_var.get() else "cpu"
                try:
                    import torch
                    if device == "cuda" and not torch.cuda.is_available():
                        device = "cpu"
                except ImportError:
                    device = "cpu"

                # ── 저장된 아키텍처 config 먼저 읽기 ──────────────────
                # checkpoint에 config가 없으면 현재 UI 파라미터로 fallback
                import torch as _torch
                latest_path = os.path.join(
                    self._store._sym_dir(sym), "latest.pt")
                saved_cfg = {}
                if os.path.exists(latest_path):
                    try:
                        raw = _torch.load(
                            latest_path, map_location="cpu",
                            weights_only=False)
                        saved_cfg = raw.get("config", {})
                    except Exception:
                        pass

                _ts_mode    = saved_cfg.get("ts_mode", "scalar")
                _seg_len    = int(segs.shape[1])
                _ts_indim   = (int(segs.shape[2])          # 5 (OHLCV) for segment mode
                               if _ts_mode == "segment"
                               else int(ts_feats.shape[1]))  # 32 for scalar mode

                arch_config = {
                    "img_in_channels":    saved_cfg.get("img_in_channels",    int(images.shape[1])),
                    "cnn_channels":       saved_cfg.get("cnn_channels",       self.settings.model.cnn_channels),
                    "cnn_out_dim":        saved_cfg.get("cnn_out_dim",        self.settings.model.cnn_out_dim),
                    "ts_input_dim":       saved_cfg.get("ts_input_dim",       _ts_indim),
                    "d_model":            saved_cfg.get("d_model",            params["d_model"]),
                    "nhead":              saved_cfg.get("nhead",              params["nhead"]),
                    "num_encoder_layers": saved_cfg.get("num_encoder_layers", params["num_layers"]),
                    "dim_feedforward":    saved_cfg.get("dim_feedforward",    params["d_model"] * 2),
                    "dropout":            saved_cfg.get("dropout",            params["dropout"]),
                    "ts_mode":            _ts_mode,
                    "segment_length":     saved_cfg.get("segment_length", _seg_len),
                }
                if saved_cfg:
                    self._log(f"[{sym}] 저장된 아키텍처 config 사용 (d_model={arch_config['d_model']}, ts_mode={_ts_mode})")
                else:
                    self._log(f"[{sym}] ⚠️ 저장된 config 없음 → UI 파라미터로 로드 시도 (ts_mode=scalar)")

                model = HybridModel(
                    img_in_channels    = arch_config["img_in_channels"],
                    cnn_channels       = arch_config["cnn_channels"],
                    cnn_out_dim        = arch_config["cnn_out_dim"],
                    ts_input_dim       = arch_config["ts_input_dim"],
                    d_model            = arch_config["d_model"],
                    nhead              = arch_config["nhead"],
                    num_encoder_layers = arch_config["num_encoder_layers"],
                    dim_feedforward    = arch_config["dim_feedforward"],
                    dropout            = arch_config["dropout"],
                    max_seq_len        = arch_config["segment_length"] + 10,
                )

                ok = self._store.load(model, sym, device=device)
                if not ok:
                    self._log(f"[{sym}] 모델 파일을 읽을 수 없습니다.")
                    self._set_status("모델 로드 실패")
                    return

                # 추론 (ts_mode에 따라 입력 결정)
                self.trainer = ModelTrainer(model, device=device,
                                            store=self._store, log_cb=self._log)
                n_test = min(20, len(segs))
                if _ts_mode == "segment":
                    mu_arr, sigma_arr = self.trainer.predict(
                        images[-n_test:], segs[-n_test:].astype(np.float32))
                else:
                    mu_arr, sigma_arr = self.trainer.predict(
                        images[-n_test:], ts_feats[-n_test:])

                mu_mean    = float(mu_arr.mean())
                sigma_mean = float(sigma_arr.mean())
                conf       = abs(mu_mean) / (sigma_mean + 1e-8)

                self._log(f"[{sym}] ✅ 모델 불러오기 완료!")
                self._log(f"  예측 수익률 (μ): {mu_mean:+.4f}  ({mu_mean*100:+.2f}%)")
                self._log(f"  불확실성  (σ): {sigma_mean:.4f}")
                self._log(f"  신뢰도    (μ/σ): {conf:.3f}")
                self._log(f"  시그널: {'📈 매수' if mu_mean > 0 else '📉 매도'} "
                          f"(신뢰도 {'높음' if conf > 1 else '낮음'})")
                self._set_status(f"[{sym}] 모델 로드 완료 — μ={mu_mean:+.4f}, σ={sigma_mean:.4f}")

                # 모델 정보 갱신
                self.frame.after(0, lambda: self._refresh_model_info(sym))

            except Exception as e:
                import traceback
                self._log(f"[{sym}] 오류:\n{traceback.format_exc()}")
                self._set_status("모델 로드 중 오류 발생")

        threading.Thread(target=_do, daemon=True).start()

    def _delete_model(self):
        sym = self._get_selected_symbol()
        if not sym:
            return
        if not messagebox.askyesno(
                "모델 삭제",
                f"[{sym}] 의 저장된 모델을 삭제하시겠습니까?\n이 작업은 되돌릴 수 없습니다.",
                parent=self.frame):
            return
        try:
            import shutil
            # _store._sym_dir() 로 실제 디렉터리 경로 (safe name) 사용
            model_dir = self._store._sym_dir(sym)
            if os.path.exists(model_dir):
                shutil.rmtree(model_dir)
                self._log(f"[{sym}] 모델 삭제 완료")
                self._refresh_model_info(sym)
                self.refresh_symbols()  # 아이콘 업데이트
            else:
                messagebox.showinfo("알림", "삭제할 모델이 없습니다.", parent=self.frame)
        except Exception as e:
            messagebox.showerror("오류", f"모델 삭제 실패:\n{e}", parent=self.frame)

    # ─────────────────────────────────────────────
    # 학습 실행
    # ─────────────────────────────────────────────

    def _get_selected_symbols(self) -> list[str]:
        """Listbox에서 선택된 모든 ticker 반환 (다중 선택 지원)"""
        sel = self.sym_listbox.curselection()
        if not sel:
            return []
        syms = self.settings.data.symbols
        result = []
        for i in sel:
            if i < len(syms):
                result.append(syms[i])
        return result

    def _start_training(self):
        if self._thread and self._thread.is_alive():
            messagebox.showinfo("알림", "이미 학습 중입니다.")
            return

        syms = self._get_selected_symbols()
        if not syms:
            messagebox.showwarning(
                "경고",
                "학습할 종목을 선택하세요.\n\n"
                "① 데이터 탭에서 종목 추가 후 다운로드\n"
                "② 목록에서 종목 클릭 (Ctrl·Shift+클릭으로 여러 개 선택 가능)")
            return

        # d_model % nhead 사전 검증
        params = self._get_params()
        if params["d_model"] % params["nhead"] != 0:
            messagebox.showerror(
                "설정 오류",
                f"d_model({params['d_model']})이 nhead({params['nhead']})로 나누어지지 않습니다.\n"
                f"d_model을 nhead의 배수로 설정하세요.\n"
                f"예) d_model=256, nhead=8  또는  d_model=128, nhead=4"
            )
            return

        self._stop_event.clear()
        self._loss_history = {"train": [], "val": []}
        self.result_tree.delete(*self.result_tree.get_children())
        self.progress["value"] = 0
        self.train_btn.config(state="disabled")
        self.stop_btn.config(state="normal")
        n = len(syms)
        names = [f"{s} ({get_name(s)})" for s in syms]
        self._log(f"\n{'='*44}")
        self._log(f"학습 대상  총 {n}개: {', '.join(names)}")
        self._log(f"{'='*44}")
        self.batch_status_var.set(f"총 {n}개 종목 학습 준비 중...")

        self._thread = threading.Thread(
            target=self._training_thread, args=(syms,), daemon=True)
        self._thread.start()

    def _stop_training(self):
        self._stop_event.set()
        self._log("학습 중단 요청...")
        self.stop_btn.config(state="disabled")

    def _training_thread(self, syms: list):
        """순차 배치 학습 — syms의 각 종목을 하나씩 학습"""
        params    = self._get_params()
        use_cuda  = self.cuda_var.get()
        cache_dir = os.path.join(BASE_DIR, self.settings.data.cache_dir)
        loader    = DataLoader(cache_dir, self.settings.data.cache_ttl_hours)
        total     = len(syms)
        done      = 0

        # 디바이스 결정 (공통)
        device = "cuda" if use_cuda else "cpu"
        try:
            import torch
            if device == "cuda" and not torch.cuda.is_available():
                device = "cpu"
                self._log("CUDA 없음 → CPU 사용")
        except ImportError:
            device = "cpu"

        for idx, sym in enumerate(syms):
            if self._stop_event.is_set():
                self._log("\n⏹ 학습 중단됨")
                break

            name = get_name(sym)
            # ── 배치 진행 상황 표시 ──────────────────
            self.frame.after(0, lambda i=idx, t=total, s=sym, n=name:
                self.batch_status_var.set(
                    f"▶  {t}개 중 {i+1}번째 학습 중:  {n} ({s.split('.')[0]})"
                    f"    완료 {i}개 / 남은 {t-i-1}개"
                )
            )

            self._log(f"\n{'─'*44}")
            self._log(f"[{idx+1}/{total}]  {sym}  ({name})")
            self._log(f"{'─'*44}")

            try:
                # ── 데이터 로드 ─────────────────────
                self._set_status(f"[{idx+1}/{total}] {name} — 데이터 로드 중...")
                df = loader.load(sym, self.settings.data.period,
                                 self.settings.data.interval)
                if df is None or len(df) < 100:
                    n_rows = len(df) if df is not None else 0
                    self._log(f"  ⚠️ 데이터 부족 ({n_rows}행 < 100) — 건너뜀")
                    if total == 1:                      # 단일 선택 시에는 다이얼로그
                        msg = (f"[{sym}] 데이터가 없거나 부족합니다 ({n_rows}행).\n\n"
                               "해결 방법:\n① 데이터 탭 → 해당 종목 선택\n"
                               "② [💾 데이터 다운로드] 클릭 후 다시 시도")
                        self.frame.after(0, lambda m=msg: messagebox.showwarning(
                            "데이터 없음", m, parent=self.frame))
                    continue

                self._log(f"  데이터: {len(df)}행  "
                          f"({df.index[0].date()} ~ {df.index[-1].date()})")

                # ── ROI 추출 ─────────────────────────
                roi_det = ROIDetector(
                    segment_length=params["segment_length"],
                    lookahead=params["lookahead"],
                    vol_z_threshold=1.5,
                    breakout_threshold=2.0,
                )
                self._set_status(f"[{idx+1}/{total}] {name} — ROI 감지 중...")
                segs, labels, dates = roi_det.extract_segments(df)
                n_roi = len(labels)
                self._log(f"  ROI 구간: {n_roi}개")

                if n_roi < 20:
                    self._log(f"  ⚠️ ROI 구간 부족 ({n_roi}개 < 20) — 건너뜀")
                    continue

                # ── 피처 추출 ────────────────────────
                self._set_status(f"[{idx+1}/{total}] {name} — 피처 추출 중...")
                cv_ext   = CVFeatureExtractor(
                    image_size=self.settings.model.image_size, method="gasf")
                images   = cv_ext.transform(segs)
                ts_ext   = TSFeatureExtractor(n_features=32)
                ts_feats = ts_ext.transform(segs)
                self._log(f"  이미지: {images.shape}  TS: {ts_feats.shape}")

                # ── 뉴스 AI 피처 생성 (설정에서 활성화 시) ─────────────────────
                # settings.news.use_news_in_model = True 이면 각 ROI 날짜에
                # 대응하는 40D 뉴스 특징 벡터를 생성하여 모델에 주입한다.
                # DB가 없거나 뉴스가 부족하면 None (모델은 자동으로 fallback).
                news_feats_arr = None
                _use_news = getattr(
                    getattr(self.settings, "news", None),
                    "use_news_in_model", False)
                if _use_news:
                    try:
                        from features.news_features import NewsFeatureGenerator
                        from data.news_db import NewsDB
                        db_path = os.path.join(
                            BASE_DIR, self.settings.news.db_path)
                        news_db = NewsDB(db_path)
                        nfg     = NewsFeatureGenerator(
                            decay_lambda=self.settings.news.decay_lambda)
                        # ROI 날짜 목록으로 각 시점의 뉴스 벡터 생성
                        news_list = []
                        for roi_date in dates:
                            ref = pd.Timestamp(roi_date) if not isinstance(
                                roi_date, pd.Timestamp) else roi_date
                            events = news_db.query_events_before(
                                symbol=sym,
                                before_dt=ref.to_pydatetime(),
                                max_hours=24 * 20,   # 최근 20일치 이벤트
                            )
                            vec = nfg.generate(
                                events=events,
                                reference_time=ref.to_pydatetime(),
                                symbol=sym,
                            )
                            news_list.append(vec)
                        if news_list:
                            news_feats_arr = np.stack(news_list).astype(np.float32)
                            self._log(
                                f"  뉴스 AI 피처: {news_feats_arr.shape}  "
                                f"(비영 비율: "
                                f"{(news_feats_arr != 0).any(axis=1).mean():.0%})"
                            )
                        else:
                            self._log("  뉴스 AI 피처: DB에 이벤트 없음 — 미주입")
                    except Exception as _ne:
                        self._log(f"  뉴스 AI 피처 생성 실패 (무시됨): {_ne}")
                        news_feats_arr = None
                else:
                    self._log("  뉴스 AI 피처: 비활성 (설정 탭에서 활성화 가능)")

                if self._stop_event.is_set():
                    self._log("⏹ 학습 중단됨")
                    break

                # ── 모델 ────────────────────────────
                # ts_input_dim = 5 (OHLCV 채널) — 세그먼트를 Transformer에 직접 전달
                # 이전 방식(ts_feats 스칼라, T=1)보다 자기-어텐션이 실제 의미 있게 동작
                _ts_input_dim  = int(segs.shape[2])   # 5 (OHLCV)
                _seg_len       = int(segs.shape[1])   # segment_length
                _news_feat_dim = (int(news_feats_arr.shape[1])
                                  if news_feats_arr is not None else 0)

                arch_config = {
                    "img_in_channels":    int(images.shape[1]),
                    "cnn_channels":       self.settings.model.cnn_channels,
                    "cnn_out_dim":        self.settings.model.cnn_out_dim,
                    "ts_input_dim":       _ts_input_dim,
                    "d_model":            params["d_model"],
                    "nhead":              params["nhead"],
                    "num_encoder_layers": params["num_layers"],
                    "dim_feedforward":    params["d_model"] * 2,
                    "dropout":            params["dropout"],
                    # ── 피처 메타 (추론 파이프라인 복원용) ──
                    "segment_length":     _seg_len,
                    "lookahead":          params["lookahead"],
                    "image_size":         self.settings.model.image_size,
                    "ts_mode":            "segment",  # "segment" | "scalar"
                    "news_feat_dim":      _news_feat_dim,
                    "use_news":           news_feats_arr is not None,
                }
                model    = HybridModel(
                    img_in_channels    = arch_config["img_in_channels"],
                    cnn_channels       = arch_config["cnn_channels"],
                    cnn_out_dim        = arch_config["cnn_out_dim"],
                    ts_input_dim       = arch_config["ts_input_dim"],
                    d_model            = arch_config["d_model"],
                    nhead              = arch_config["nhead"],
                    num_encoder_layers = arch_config["num_encoder_layers"],
                    dim_feedforward    = arch_config["dim_feedforward"],
                    dropout            = arch_config["dropout"],
                    max_seq_len        = _seg_len + 10,  # 여유 길이
                )
                n_params = model.count_parameters()
                self._log(f"  파라미터: {n_params:,}개  |  디바이스: {device}")
                self._log(f"  TS모드: segment ({_seg_len}T × {_ts_input_dim}F)"
                          + (f"  | 뉴스: {_news_feat_dim}D"
                             if news_feats_arr is not None else ""))

                # ── 학습 ────────────────────────────
                self.trainer = ModelTrainer(
                    model, device=device,
                    store=self._store,
                    log_cb=self._log,
                )

                def _prog_cb(pct, msg, _i=idx, _t=total, _n=name):
                    self.frame.after(0, lambda p=pct, m=msg:
                                     self._update_progress(p, m))

                metrics = self.trainer.train(
                    images=images, ts_feats=ts_feats, labels=labels,
                    segments=segs,               # ← 전체 OHLCV 세그먼트 (Transformer용)
                    news_feats=news_feats_arr,   # ← 뉴스 AI 피처 (None이면 미주입)
                    symbol=sym, model_config=arch_config,
                    lr=params["lr"], batch_size=params["batch_size"],
                    epochs=params["epochs"], patience=params["patience"],
                    stop_event=self._stop_event, progress_cb=_prog_cb,
                )

                # ── 결과 행 추가 ─────────────────────
                if metrics:
                    row = (
                        f"{sym} ({name})",
                        str(n_roi),
                        str(metrics.get("best_epoch", "—")),
                        f"{metrics.get('best_val_loss', 0):.6f}",
                        f"{metrics.get('train_time_s', 0):.1f}",
                        f"{n_params:,}",
                    )
                    self.frame.after(0, lambda r=row:
                                     self.result_tree.insert("", "end", values=r))

                # 손실 곡선 (마지막 종목 기준)
                if self.trainer.history.get("train_loss"):
                    self._loss_history["train"] = self.trainer.history["train_loss"]
                    self._loss_history["val"]   = self.trainer.history["val_loss"]
                    self.frame.after(0, self._draw_loss_chart)

                done += 1
                self._log(f"  ✅ 완료  [{done}/{total}]")

                # 아이콘 갱신 (완료한 종목 즉시 반영)
                self.frame.after(0, self.refresh_symbols)

            except Exception:
                import traceback
                self._log(f"  ❌ 오류:\n{traceback.format_exc()}")

        # ── 배치 완료 ────────────────────────────────
        self.frame.after(0, lambda d=done, t=total:
            self.batch_status_var.set(
                f"✅ 학습 완료  {t}개 중 {d}개 성공"
                + ("  (중단됨)" if self._stop_event.is_set() else "")
            )
        )
        # 마지막으로 선택 종목 모델 정보 갱신
        last_sym = syms[-1] if syms else None
        if last_sym:
            def _post_all(s=last_sym):
                self.refresh_symbols()
                self._refresh_model_info(s)
            self.frame.after(0, _post_all)
        self.frame.after(0, self._on_train_complete)

    def _on_train_complete(self):
        self.train_btn.config(state="normal")
        self.stop_btn.config(state="disabled")
        self.progress["value"] = 100
        self.pct_var.set("100%")
        self._set_status("완료")
        if self._on_complete:
            self._on_complete()

    def _update_progress(self, pct: float, msg: str):
        self.progress["value"] = min(pct, 100)
        self.pct_var.set(f"{pct:.0f}%")
        self.status_var.set(msg)

    # ─────────────────────────────────────────────
    # 손실 곡선
    # ─────────────────────────────────────────────

    def _draw_loss_chart(self):
        canvas = self.chart_canvas
        canvas.delete("all")
        W = canvas.winfo_width()
        H = canvas.winfo_height()
        if W < 10 or H < 10:
            return

        train_loss = self._loss_history["train"]
        val_loss   = self._loss_history["val"]
        if not train_loss:
            return

        pad = 40
        all_v = train_loss + val_loss
        vmin, vmax = min(all_v), max(all_v)
        vrange = max(vmax - vmin, 1e-10)
        n = len(train_loss)

        def xc(i): return pad + (W - 2*pad) * i / max(n-1, 1)
        def yc(v): return H - pad - (H - 2*pad) * (v - vmin) / vrange

        # 격자
        for frac in [0, 0.5, 1.0]:
            v  = vmin + vrange * frac
            y  = yc(v)
            canvas.create_line(pad, y, W-pad, y, fill="#313244", dash=(2, 4))
            canvas.create_text(pad-4, y, text=f"{v:.4f}",
                               fill="#9399b2", font=("Consolas", 7), anchor="e")

        # 훈련 (파란)
        pts_t = []
        for i in range(n):
            pts_t.extend([xc(i), yc(train_loss[i])])
        if len(pts_t) >= 4:
            canvas.create_line(*pts_t, fill="#89b4fa", width=1.8, smooth=True)

        # 검증 (오렌지)
        pts_v = []
        for i in range(min(len(val_loss), n)):
            pts_v.extend([xc(i), yc(val_loss[i])])
        if len(pts_v) >= 4:
            canvas.create_line(*pts_v, fill="#fab387", width=1.8, smooth=True)

        # 범례
        canvas.create_line(W-130, 14, W-108, 14, fill="#89b4fa", width=2)
        canvas.create_text(W-104, 14, text="훈련손실",
                           fill="#89b4fa", font=("맑은 고딕", 8), anchor="w")
        canvas.create_line(W-55, 14, W-33, 14, fill="#fab387", width=2)
        canvas.create_text(W-29, 14, text="검증손실",
                           fill="#fab387", font=("맑은 고딕", 8), anchor="w")

    # ─────────────────────────────────────────────
    # 유틸
    # ─────────────────────────────────────────────

    def _get_params(self) -> dict:
        defaults = {
            "segment_length": 30, "lookahead": 5,
            "d_model": 256, "num_layers": 4, "nhead": 8,
            "dropout": 0.1, "lr": 1e-4,
            "batch_size": 32, "epochs": 100, "patience": 15,
        }
        p = {}
        for k, d in defaults.items():
            try:
                p[k] = type(d)(self.param_vars[k].get())
            except Exception:
                p[k] = d
        return p

    # ─────────────────────────────────────────────
    # 모델 정보 모달리스 창
    # ─────────────────────────────────────────────

    def _show_model_info_dialog(self):
        """모든 종목의 AI 모델 학습 상태를 모달리스 Toplevel 창에 표시."""
        if not hasattr(self, "_info_win"):
            self._info_win = None
        if self._info_win and self._info_win.winfo_exists():
            self._info_win.lift()
            self._info_win.focus_force()
            self._refresh_info_dialog()
            return

        win = tk.Toplevel(self.frame)
        win.title("📊 모델 정보 — AI 학습 상태")
        win.geometry("820x480")
        win.configure(bg="#1e1e2e")
        win.resizable(True, True)
        self._info_win = win

        # ── 상단 버튼 ───────────────────────────
        top = tk.Frame(win, bg="#1e1e2e")
        top.pack(fill="x", padx=8, pady=(8, 4))
        ttk.Button(top, text="🔄 새로고침",
                   command=self._refresh_info_dialog).pack(side="left", padx=(0, 8))
        tk.Label(
            top,
            text="전 종목 AI 모델 상태  |  📦 = 학습 완료  ✗ = 미학습  ⚠️ = 30일 이상 오래됨",
            bg="#1e1e2e", fg="#9399b2", font=("맑은 고딕", 9),
        ).pack(side="left")

        # ── Treeview ────────────────────────────
        cols = ("종목명", "코드", "상태", "최근 학습", "Best Epoch",
                "Val Loss", "파라미터수", "버전")
        tree = ttk.Treeview(win, columns=cols, show="headings")
        tree.heading("종목명",   text="종목명")
        tree.heading("코드",     text="코드")
        tree.heading("상태",     text="상태")
        tree.heading("최근 학습",text="최근 학습")
        tree.heading("Best Epoch", text="Best Epoch")
        tree.heading("Val Loss", text="Val Loss")
        tree.heading("파라미터수", text="파라미터수")
        tree.heading("버전",     text="버전")

        tree.column("종목명",    width=150, anchor="w")
        tree.column("코드",      width=80,  anchor="center")
        tree.column("상태",      width=70,  anchor="center")
        tree.column("최근 학습", width=135, anchor="center")
        tree.column("Best Epoch",width=80,  anchor="center")
        tree.column("Val Loss",  width=90,  anchor="center")
        tree.column("파라미터수",width=90,  anchor="center")
        tree.column("버전",      width=50,  anchor="center")

        tree.tag_configure("has_model",   foreground="#a6e3a1")
        tree.tag_configure("no_model",    foreground="#585b70")
        tree.tag_configure("stale_model", foreground="#f9e2af")

        vsb = ttk.Scrollbar(win, orient="vertical", command=tree.yview)
        tree.configure(yscrollcommand=vsb.set)
        tree.pack(side="left", fill="both", expand=True, padx=(8, 0), pady=(0, 8))
        vsb.pack(side="right", fill="y", padx=(0, 8), pady=(0, 8))

        self._info_tree = tree
        self._refresh_info_dialog()

        win.protocol("WM_DELETE_WINDOW",
                     lambda: (win.destroy(),
                              setattr(self, "_info_win", None)))

    def _refresh_info_dialog(self):
        """모델 정보 Treeview 내용 전체 갱신."""
        if not hasattr(self, "_info_tree"):
            return
        import datetime
        tree = self._info_tree
        tree.delete(*tree.get_children())
        now = datetime.datetime.now()

        if not hasattr(self, "_name_cache"):
            self._name_cache: dict[str, str] = {}

        for sym in self.settings.data.symbols:
            if sym not in self._name_cache:
                self._name_cache[sym] = get_name(sym)
            name = self._name_cache[sym]
            code = sym.split(".")[0]

            if not self._store.has_model(sym):
                tree.insert("", "end", tags=("no_model",),
                            values=(name, code, "✗ 미학습", "—", "—", "—", "—", "0"))
                continue

            meta   = self._store.get_meta(sym)
            ts_raw = meta.get("timestamp", "")
            m      = meta.get("metrics", {})

            if not ts_raw and not m:
                tree.insert("", "end", tags=("has_model",),
                            values=(name, code, "📦 있음", "메타 없음", "—", "—", "—", "—"))
                continue

            if len(ts_raw) >= 13:
                ts_fmt = (f"{ts_raw[:4]}-{ts_raw[4:6]}-{ts_raw[6:8]} "
                          f"{ts_raw[9:11]}:{ts_raw[11:13]}")
                try:
                    dt    = datetime.datetime.strptime(ts_fmt, "%Y-%m-%d %H:%M")
                    stale = (now - dt).days > 30
                except Exception:
                    stale = False
            else:
                ts_fmt = ts_raw or "—"
                stale  = False

            ep  = m.get("best_epoch", "—")
            vl  = m.get("best_val_loss", m.get("val_loss", None))
            np_ = m.get("n_params", "—")
            vl_str = f"{vl:.6f}" if isinstance(vl, float) else "—"
            np_str = f"{np_:,}"  if isinstance(np_, int)   else str(np_)

            tag    = "stale_model" if stale else "has_model"
            status = "⚠️ 오래됨" if stale else "📦 학습됨"
            tree.insert("", "end", tags=(tag,),
                        values=(name, code, status, ts_fmt,
                                ep, vl_str, np_str, "1"))

    def _set_status(self, msg: str):
        self.frame.after(0, lambda: self.status_var.set(msg))

    def _log(self, msg: str):
        def _do():
            self.log_box.config(state="normal")
            self.log_box.insert("end", msg + "\n")
            self.log_box.see("end")
            self.log_box.config(state="disabled")
        self.frame.after(0, _do)
