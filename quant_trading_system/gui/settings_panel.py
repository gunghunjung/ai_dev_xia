# gui/settings_panel.py — 설정 패널 (초보자 친화적 재설계)
"""
초보자와 전문가 모두를 위한 설정 패널.

구성:
  ┌─────────────────────────────────────────────────────┐
  │ [프리셋 선택] 초보자추천 / 단기전략 / 중기전략 / ...  │
  ├──────────────────────────────────────────────────────┤
  │ [기본 설정] [고급 설정] ← 탭                         │
  │  · 기본 설정: 초보자가 자주 바꾸는 항목만            │
  │  · 고급 설정: 전문가용 세부 파라미터                 │
  ├──────────────────────────────────────────────────────┤
  │ [💾 적용 및 저장]  [↩ 기본값 복원]                   │
  └──────────────────────────────────────────────────────┘
"""
from __future__ import annotations
import tkinter as tk
from tkinter import ttk, messagebox
import logging
from dataclasses import asdict

from config import AppSettings, save_settings, load_settings
from .tooltip import add_tooltip, add_help_label
from .ui_meta import META, PRESETS, get_detail, get_help, apply_preset

logger = logging.getLogger("quant.gui.settings")


# ─── 색상 상수 ───────────────────────────────────────────────────────────────
C_BG     = "#1e1e2e"
C_PANEL  = "#181825"
C_FG     = "#cdd6f4"
C_DIM    = "#9399b2"
C_ACC    = "#89b4fa"
C_GREEN  = "#a6e3a1"
C_YELL   = "#f9e2af"
C_SEL    = "#313244"


class SettingsPanel:
    """
    설정 패널 — 초보자/전문가 이중 레이어 구조.
    """

    def __init__(self, parent, settings: AppSettings, on_change):
        self.settings  = settings
        self.on_change = on_change
        self._vars: dict[str, tk.StringVar] = {}
        self._is_beginner = True   # 기본: 초보자 모드

        self.frame = ttk.Frame(parent)
        self._build()

    # ─────────────────────────────────────────────────────────────────────────
    # 외부 API
    # ─────────────────────────────────────────────────────────────────────────

    def set_mode(self, is_beginner: bool):
        """메인 윈도우에서 모드 전환 시 호출"""
        self._is_beginner = is_beginner
        # 고급 탭 표시/숨김
        try:
            if is_beginner:
                self._nb.tab(1, state="hidden")
            else:
                self._nb.tab(1, state="normal")
        except Exception:
            pass

    # ─────────────────────────────────────────────────────────────────────────
    # UI 빌드
    # ─────────────────────────────────────────────────────────────────────────

    def _build(self):
        root = self.frame

        # ── 프리셋 배너 ──────────────────────────────────────────────────────
        self._build_preset_bar(root)

        # ── 탭 (기본 / 고급) ─────────────────────────────────────────────────
        self._nb = ttk.Notebook(root)
        self._nb.pack(fill="both", expand=True, padx=6, pady=(4, 0))

        basic_scroll = self._make_scroll_frame(self._nb)
        adv_scroll   = self._make_scroll_frame(self._nb)

        self._nb.add(basic_scroll["outer"], text="  📋 기본 설정  ")
        self._nb.add(adv_scroll["outer"],   text="  🔬 고급 설정  ")

        self._build_basic_tab(basic_scroll["inner"])
        self._build_advanced_tab(adv_scroll["inner"])

        # 초보자 모드: 고급 탭 숨김
        # (전문가 모드로 전환하면 set_mode()에서 표시)
        # self._nb.tab(1, state="hidden")  # 일단 항상 표시

        # ── 하단 버튼 ─────────────────────────────────────────────────────────
        self._build_buttons(root)

    def _make_scroll_frame(self, parent) -> dict:
        """스크롤 가능한 컨테이너 생성, {'outer': Canvas+Scrollbar frame, 'inner': inner Frame}"""
        outer = ttk.Frame(parent)

        canvas = tk.Canvas(outer, bg=C_BG, highlightthickness=0)
        vsb    = ttk.Scrollbar(outer, orient="vertical", command=canvas.yview)
        inner  = ttk.Frame(canvas)

        inner.bind("<Configure>",
                   lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
        canvas.create_window((0, 0), window=inner, anchor="nw")
        canvas.configure(yscrollcommand=vsb.set)

        canvas.pack(side="left",  fill="both", expand=True)
        vsb.pack(side="right", fill="y")

        canvas.bind_all("<MouseWheel>",
                        lambda e: canvas.yview_scroll(-1 * (e.delta // 120), "units"))

        return {"outer": outer, "inner": inner}

    # ─────────────────────────────────────────────────────────────────────────
    # 프리셋 선택 바
    # ─────────────────────────────────────────────────────────────────────────

    def _build_preset_bar(self, parent):
        bar = tk.Frame(parent, bg=C_PANEL, pady=8)
        bar.pack(fill="x", padx=6, pady=(6, 0))

        tk.Label(bar, text="⚡ 빠른 설정 프리셋:", bg=C_PANEL, fg=C_DIM,
                 font=("맑은 고딕", 9, "bold")).pack(side="left", padx=(12, 8))

        preset_styles = {
            "beginner":   ("#a6e3a1", "#1e3a2f"),
            "short_term": ("#89b4fa", "#1e2a3e"),
            "mid_term":   ("#89b4fa", "#1e2a3e"),
            "stable":     ("#f9e2af", "#3d3115"),
            "advanced":   ("#cba6f7", "#2a1e3e"),
        }

        for key, info in PRESETS.items():
            fg, abg = preset_styles.get(key, (C_FG, C_SEL))
            btn = tk.Button(
                bar,
                text=info["name"],
                bg=abg, fg=fg,
                relief="flat", bd=0,
                font=("맑은 고딕", 9, "bold"),
                padx=10, pady=3,
                cursor="hand2",
                command=lambda k=key: self._apply_preset(k),
            )
            btn.pack(side="left", padx=3)
            add_tooltip(btn, info["description"])

        # 프리셋 설명 라벨
        self._preset_desc_var = tk.StringVar(value="원하는 프리셋을 선택하면 관련 설정이 자동으로 채워집니다")
        tk.Label(bar, textvariable=self._preset_desc_var,
                 bg=C_PANEL, fg=C_DIM,
                 font=("맑은 고딕", 8),
                 anchor="w").pack(side="left", padx=12, fill="x", expand=True)

    def _apply_preset(self, preset_key: str):
        info = PRESETS.get(preset_key, {})
        if not info:
            return
        if not messagebox.askyesno(
            "프리셋 적용",
            f"'{info['name']}' 프리셋을 적용할까요?\n\n"
            f"{info['description']}\n\n"
            f"현재 설정이 일부 변경됩니다.",
            icon="question",
        ):
            return

        apply_preset(preset_key, self.settings)
        self._refresh_vars_from_settings(self.settings)
        self.on_change(self.settings)
        self._preset_desc_var.set(f"✅ '{info['name']}' 적용됨")
        messagebox.showinfo(
            "프리셋 적용 완료",
            f"'{info['name']}' 설정이 적용되었습니다.\n\n"
            f"'💾 적용 및 저장' 버튼을 눌러 저장하세요.",
        )

    # ─────────────────────────────────────────────────────────────────────────
    # 기본 설정 탭 — 초보자가 자주 바꾸는 핵심 항목만
    # ─────────────────────────────────────────────────────────────────────────

    def _build_basic_tab(self, parent):
        col_fr = tk.Frame(parent, bg=C_BG)
        col_fr.pack(fill="both", expand=True, padx=8, pady=8)

        # 왼쪽 열
        left = ttk.Frame(col_fr)
        left.grid(row=0, column=0, sticky="nw", padx=(0, 16))

        # 오른쪽 열
        right = ttk.Frame(col_fr)
        right.grid(row=0, column=1, sticky="nw")

        col_fr.columnconfigure(0, weight=1)
        col_fr.columnconfigure(1, weight=1)

        # ── 왼쪽: 데이터 + 투자 기간 ─────────────────────────────────────────
        self._section(left, "📥 데이터 수집",
                      "분석에 사용할 주가 데이터 기간과 방식을 설정합니다")

        self._combo_field(left, "data.period", {
            "1년": "1y", "2년": "2y", "3년": "3y",
            "5년 (추천)": "5y", "10년": "10y", "전체": "max",
        })
        self._combo_field(left, "data.interval", {
            "일봉 — 매일 (추천)": "1d",
            "주봉 — 매주":        "1wk",
        })
        self._text_field(left, "data.benchmark")

        tk.Frame(left, bg=C_BG, height=12).pack()

        # ── 왼쪽: 예측 설정 ───────────────────────────────────────────────────
        self._section(left, "🎯 AI 예측 기간",
                      "AI가 몇 거래일 뒤 주가를 예측할지 설정합니다")

        self._text_field(left, "roi.lookahead")
        self._text_field(left, "roi.segment_length")

        tk.Frame(left, bg=C_BG, height=12).pack()

        # ── 왼쪽: 학습 속도 ───────────────────────────────────────────────────
        self._section(left, "🚀 학습 속도 / 정확도",
                      "빠른 실험 vs 정밀 학습 사이의 균형을 설정합니다")

        self._text_field(left, "model.epochs")
        self._text_field(left, "model.patience")
        self._text_field(left, "model.learning_rate")

        # ── 오른쪽: 포트폴리오 ────────────────────────────────────────────────
        self._section(right, "💼 포트폴리오 구성",
                      "여러 종목에 어떻게 투자 비중을 나눌지 설정합니다")

        self._combo_field(right, "portfolio.method", {
            "균등 분배 (초보자 추천)": "equal_weight",
            "위험 균등 (Risk Parity)": "risk_parity",
            "기대수익 최적화 (고급)":  "mean_variance",
            "변동성 비례 조정 (고급)": "vol_scaling",
        })
        self._combo_field(right, "portfolio.rebalance_freq", {
            "매일":      "daily",
            "매주 (추천)": "weekly",
            "매월":      "monthly",
        })
        self._text_field(right, "portfolio.max_weight")

        tk.Frame(right, bg=C_BG, height=12).pack()

        # ── 오른쪽: 백테스트 ──────────────────────────────────────────────────
        self._section(right, "📈 백테스트",
                      "과거 데이터로 전략을 검증할 때 사용하는 조건입니다")

        self._text_field(right, "backtest.initial_capital")
        self._text_field(right, "backtest.transaction_cost")
        self._text_field(right, "backtest.slippage")

        tk.Frame(right, bg=C_BG, height=12).pack()

        # ── 오른쪽: 리스크 ────────────────────────────────────────────────────
        self._section(right, "🛡️ 위험 관리",
                      "손실이 일정 수준을 넘었을 때 자동으로 보호하는 설정입니다")

        self._text_field(right, "risk.max_drawdown_limit")
        self._text_field(right, "risk.vol_target")

    # ─────────────────────────────────────────────────────────────────────────
    # 고급 설정 탭 — 전문가 세부 파라미터
    # ─────────────────────────────────────────────────────────────────────────

    def _build_advanced_tab(self, parent):
        note = tk.Label(
            parent,
            text=(
                "⚠️  고급 설정입니다. 각 파라미터의 의미를 정확히 이해하고 변경하세요.\n"
                "   잘못된 값은 모델 오류나 부정확한 백테스트 결과를 유발할 수 있습니다."
            ),
            bg="#2a1e1e", fg=C_YELL,
            font=("맑은 고딕", 9),
            justify="left", anchor="w",
            padx=12, pady=6,
        )
        note.pack(fill="x", pady=(8, 4), padx=6)

        col_fr = tk.Frame(parent, bg=C_BG)
        col_fr.pack(fill="both", expand=True, padx=8, pady=4)

        col0 = ttk.Frame(col_fr)
        col1 = ttk.Frame(col_fr)
        col2 = ttk.Frame(col_fr)
        col0.grid(row=0, column=0, sticky="nw", padx=(0, 12))
        col1.grid(row=0, column=1, sticky="nw", padx=(0, 12))
        col2.grid(row=0, column=2, sticky="nw")
        col_fr.columnconfigure(0, weight=1)
        col_fr.columnconfigure(1, weight=1)
        col_fr.columnconfigure(2, weight=1)

        # ── 모델 아키텍처 ────────────────────────────────────────────────────
        self._section(col0, "🧠 모델 아키텍처",
                      "Transformer 신경망 구조 파라미터")

        self._text_field(col0, "model.d_model")
        self._text_field(col0, "model.nhead")
        self._text_field(col0, "model.num_encoder_layers")
        self._text_field(col0, "model.dim_feedforward",
                         fallback_label="FFN 내부 차원",
                         fallback_help="Transformer 내부 피드포워드 레이어 크기")
        self._text_field(col0, "model.dropout")
        self._text_field(col0, "model.batch_size")
        self._text_field(col0, "model.image_size")

        # ── ROI 감지 ──────────────────────────────────────────────────────────
        self._section(col1, "🔍 ROI 감지 (피처 추출)",
                      "주가에서 중요 구간을 찾아내는 파라미터")

        self._text_field(col1, "roi.vol_z_threshold")
        self._text_field(col1, "roi.breakout_threshold",
                         fallback_label="돌파 민감도 (Z)",
                         fallback_help="가격 돌파를 감지하는 임계값")
        self._text_field(col1, "roi.volume_spike_threshold",
                         fallback_label="거래량 급증 감지 (Z)",
                         fallback_help="거래량 급증을 감지하는 임계값")
        self._text_field(col1, "roi.min_roi_spacing",
                         fallback_label="ROI 간격 최소치",
                         fallback_help="중요 구간 사이 최소 간격 (거래일)")
        self._text_field(col1, "roi.rolling_window",
                         fallback_label="롤링 윈도우",
                         fallback_help="통계 계산에 사용하는 이동 윈도우 크기")

        # ── Walk-Forward 백테스트 ─────────────────────────────────────────────
        self._section(col2, "🔄 Walk-Forward 검증",
                      "시간 순서를 지키며 전략을 검증하는 방법의 세부 설정")

        self._text_field(col2, "backtest.wf_train_days")
        self._text_field(col2, "backtest.wf_test_days")
        self._text_field(col2, "backtest.wf_step_days")
        self._text_field(col2, "backtest.execution_delay",
                         fallback_label="체결 지연 (일)",
                         fallback_help="주문 후 실제 체결까지의 지연 시간")

        # ── 리스크 상세 ───────────────────────────────────────────────────────
        self._section(col2, "🛡️ 리스크 상세",
                      "고급 리스크 관리 파라미터")

        self._text_field(col2, "risk.vol_lookback",
                         fallback_label="변동성 계산 기간",
                         fallback_help="변동성 추정에 사용하는 과거 기간")
        self._text_field(col2, "risk.correlation_cap",
                         fallback_label="상관관계 상한",
                         fallback_help="종목 간 상관관계 최대 허용값 (분산 투자 강제)")
        self._text_field(col2, "risk.kill_switch_sharpe",
                         fallback_label="킬스위치 Sharpe 기준",
                         fallback_help="Sharpe가 이 값 미만이면 포지션 전량 청산")

        # 레짐 감지 체크박스
        tk.Frame(col2, bg=C_BG, height=8).pack()
        var = tk.StringVar(value=str(self.settings.risk.regime_detection))
        self._vars["risk.regime_detection"] = var
        cb = ttk.Checkbutton(
            col2, text="시장 국면 감지 사용",
            variable=var, onvalue="True", offvalue="False",
        )
        cb.pack(anchor="w", padx=4)
        add_tooltip(cb,
                    "시장이 상승장/하락장/횡보장인지 감지하여\n"
                    "포트폴리오 전략을 동적으로 조정합니다.\n\n"
                    "추천: 켜기 (True)")

    # ─────────────────────────────────────────────────────────────────────────
    # 하단 버튼
    # ─────────────────────────────────────────────────────────────────────────

    def _build_buttons(self, parent):
        btn_fr = tk.Frame(parent, bg=C_PANEL, pady=8)
        btn_fr.pack(fill="x", padx=6, pady=(4, 6))

        save_btn = ttk.Button(
            btn_fr, text="💾 적용 및 저장",
            command=self._apply_and_save,
            style="Accent.TButton",
        )
        save_btn.pack(side="left", padx=(12, 8))
        add_tooltip(save_btn,
                    "현재 화면에 입력된 값을 적용하고 파일에 저장합니다.\n"
                    "프로그램을 다시 시작해도 이 설정이 유지됩니다.")

        reset_btn = ttk.Button(
            btn_fr, text="↩ 기본값 복원",
            command=self._restore_defaults,
        )
        reset_btn.pack(side="left")
        add_tooltip(reset_btn,
                    "모든 설정을 프로그램 기본값으로 되돌립니다.\n"
                    "이 작업은 취소할 수 없습니다.")

        # 저장 안내
        tk.Label(
            btn_fr,
            text="※ 설정을 바꾼 후 반드시 '적용 및 저장'을 눌러주세요",
            bg=C_PANEL, fg=C_DIM,
            font=("맑은 고딕", 8),
        ).pack(side="left", padx=16)

    # ─────────────────────────────────────────────────────────────────────────
    # 위젯 빌더 헬퍼
    # ─────────────────────────────────────────────────────────────────────────

    def _section(self, parent, title: str, description: str = ""):
        """섹션 헤더 (타이틀 + 짧은 설명)"""
        fr = tk.Frame(parent, bg=C_BG)
        fr.pack(fill="x", pady=(8, 2), padx=4)
        tk.Label(fr, text=title, bg=C_BG, fg=C_ACC,
                 font=("맑은 고딕", 10, "bold")).pack(anchor="w")
        if description:
            tk.Label(fr, text=description, bg=C_BG, fg=C_DIM,
                     font=("맑은 고딕", 8)).pack(anchor="w")
        tk.Frame(parent, bg="#313244", height=1).pack(fill="x", padx=4, pady=(0, 6))

    def _text_field(
        self,
        parent,
        key: str,
        fallback_label: str = "",
        fallback_help: str = "",
    ):
        """
        입력 필드 (Label + Entry + 도움말 텍스트 + 툴팁).
        key = ui_meta 의 META 키 (예: "model.d_model")
        """
        meta  = META.get(key, {})
        label = meta.get("label", fallback_label or key.split(".")[-1])
        help_ = meta.get("help",  fallback_help)
        detail = meta.get("detail", help_)
        default = self._get_settings_value(key)
        unit  = meta.get("unit", "")
        beginner = meta.get("beginner", "")

        # 현재 값 또는 기본값
        current = default if default is not None else meta.get("default", "")

        fr = tk.Frame(parent, bg=C_BG)
        fr.pack(fill="x", padx=4, pady=2)

        # 레이블
        lbl_text = label + ":"
        if unit:
            lbl_text += f"  ({unit})"
        lbl = tk.Label(fr, text=lbl_text, bg=C_BG, fg=C_FG,
                       font=("맑은 고딕", 9), anchor="w")
        lbl.pack(anchor="w")

        # 입력 필드 행
        entry_fr = tk.Frame(fr, bg=C_BG)
        entry_fr.pack(fill="x")

        var = tk.StringVar(value=str(current) if current is not None else "")
        self._vars[key] = var
        entry = ttk.Entry(entry_fr, textvariable=var, width=16)
        entry.pack(side="left")

        # 초보자 추천값 뱃지
        if beginner and str(beginner) != str(current):
            rec_lbl = tk.Label(
                entry_fr,
                text=f"추천: {beginner}",
                bg="#1e3a2f", fg=C_GREEN,
                font=("맑은 고딕", 7),
                padx=4, pady=1, cursor="hand2",
            )
            rec_lbl.pack(side="left", padx=4)
            rec_lbl.bind("<Button-1>",
                         lambda e, v=var, b=beginner: v.set(str(b)))
            add_tooltip(rec_lbl,
                        f"클릭하면 초보자 추천값 '{beginner}'으로 자동 입력됩니다")

        # 툴팁 (상세 설명)
        if detail:
            add_tooltip(lbl,   detail)
            add_tooltip(entry, detail)

        # 도움말 텍스트 (한 줄 요약)
        if help_ and help_ != detail:
            add_help_label(fr, f"  {help_}", padx=0)
        elif help_:
            # 짧은 도움말만
            add_help_label(fr, f"  {help_[:70]}{'…' if len(help_) > 70 else ''}", padx=0)

    def _combo_field(
        self,
        parent,
        key: str,
        display_to_value: dict,  # {"표시이름": "내부값", ...}
    ):
        """
        콤보박스 필드 (Label + Combobox + 도움말 + 툴팁).
        display_to_value: 화면에 보여줄 이름 → 실제 저장값 매핑
        """
        meta   = META.get(key, {})
        label  = meta.get("label", key.split(".")[-1])
        help_  = meta.get("help", "")
        detail = meta.get("detail", help_)

        current_val = self._get_settings_value(key)

        # 현재값 → 표시이름 역매핑
        value_to_display = {v: k for k, v in display_to_value.items()}
        current_display  = value_to_display.get(str(current_val), str(current_val))

        fr = tk.Frame(parent, bg=C_BG)
        fr.pack(fill="x", padx=4, pady=2)

        lbl = tk.Label(fr, text=label + ":", bg=C_BG, fg=C_FG,
                       font=("맑은 고딕", 9), anchor="w")
        lbl.pack(anchor="w")

        # StringVar: 표시 이름을 저장, 적용 시 역매핑으로 실제 값 추출
        display_var = tk.StringVar(value=current_display)

        # 내부 값을 위한 별도 var (기존 self._vars 호환 유지)
        inner_var = tk.StringVar(value=str(current_val))
        self._vars[key] = inner_var

        # 표시 이름이 변경되면 내부 값도 동기화
        def _on_display_change(*_):
            selected = display_var.get()
            inner_var.set(display_to_value.get(selected, selected))

        display_var.trace_add("write", _on_display_change)

        cb = ttk.Combobox(
            fr,
            textvariable=display_var,
            values=list(display_to_value.keys()),
            width=28,
            state="readonly",
        )
        cb.pack(anchor="w")

        if detail:
            add_tooltip(lbl, detail)
            add_tooltip(cb,  detail)
        if help_:
            add_help_label(fr, f"  {help_[:70]}{'…' if len(help_) > 70 else ''}", padx=0)

    def _get_settings_value(self, key: str):
        """dotted key → settings 객체에서 현재 값 추출 (data.period → settings.data.period)"""
        parts = key.split(".", 1)
        if len(parts) != 2:
            return None
        section = getattr(self.settings, parts[0], None)
        if section is None:
            return None
        return getattr(section, parts[1], None)

    # ─────────────────────────────────────────────────────────────────────────
    # 저장 / 복원
    # ─────────────────────────────────────────────────────────────────────────

    def _apply_and_save(self):
        """설정 적용 및 저장"""
        try:
            s = self.settings

            def _bool(v):  return str(v).lower() not in ("false", "0", "no")
            def _f(k, d=0.0):
                try:    return float(self._vars[k].get())
                except: return d
            def _i(k, d=0):
                try:    return int(float(self._vars[k].get()))
                except: return d
            def _s(k, d=""):
                return self._vars.get(k, tk.StringVar(value=d)).get()

            # 검증
            d_model = _i("model.d_model", s.model.d_model)
            nhead   = _i("model.nhead",   s.model.nhead)
            if d_model % nhead != 0:
                messagebox.showerror(
                    "설정 오류",
                    f"모델 크기(d_model={d_model})가 어텐션 헤드 수(nhead={nhead})로\n"
                    f"나누어지지 않습니다.\n\n"
                    f"d_model은 nhead의 배수여야 합니다.\n"
                    f"예) d_model=128, nhead=4  또는  d_model=256, nhead=8",
                )
                return

            # 데이터
            s.data.period          = _s("data.period",           "5y")
            s.data.interval        = _s("data.interval",         "1d")
            s.data.cache_ttl_hours = _i("data.cache_ttl_hours",  23)
            s.data.benchmark       = _s("data.benchmark",        "^KS11")

            # ROI
            s.roi.segment_length        = _i("roi.segment_length",        30)
            s.roi.lookahead             = _i("roi.lookahead",             5)
            s.roi.vol_z_threshold       = _f("roi.vol_z_threshold",       1.5)
            s.roi.breakout_threshold    = _f("roi.breakout_threshold",    2.0)
            s.roi.volume_spike_threshold= _f("roi.volume_spike_threshold",2.0)
            s.roi.min_roi_spacing       = _i("roi.min_roi_spacing",       5)
            s.roi.rolling_window        = _i("roi.rolling_window",        20)

            # 모델
            s.model.d_model              = d_model
            s.model.nhead                = nhead
            s.model.num_encoder_layers   = _i("model.num_encoder_layers", 4)
            s.model.dim_feedforward      = _i("model.dim_feedforward",    512)
            s.model.dropout              = _f("model.dropout",            0.1)
            s.model.learning_rate        = _f("model.learning_rate",      1e-4)
            s.model.weight_decay         = _f("model.weight_decay",       1e-5)
            s.model.batch_size           = _i("model.batch_size",         32)
            s.model.epochs               = _i("model.epochs",             100)
            s.model.patience             = _i("model.patience",           15)
            s.model.image_size           = _i("model.image_size",         64)

            # 포트폴리오
            s.portfolio.method           = _s("portfolio.method",         "equal_weight")
            s.portfolio.max_weight       = _f("portfolio.max_weight",     0.35)
            s.portfolio.min_weight       = _f("portfolio.min_weight",     0.0)
            s.portfolio.turnover_limit   = _f("portfolio.turnover_limit", 0.3)
            s.portfolio.target_volatility= _f("portfolio.target_volatility",0.15)
            s.portfolio.rebalance_freq   = _s("portfolio.rebalance_freq", "weekly")
            s.portfolio.lookback_vol     = _i("portfolio.lookback_vol",   20)

            # 백테스트
            s.backtest.initial_capital   = _f("backtest.initial_capital", 1e8)
            s.backtest.transaction_cost  = _f("backtest.transaction_cost",0.0015)
            s.backtest.slippage          = _f("backtest.slippage",        0.0005)
            s.backtest.execution_delay   = _i("backtest.execution_delay", 1)
            s.backtest.wf_train_days     = _i("backtest.wf_train_days",   504)
            s.backtest.wf_test_days      = _i("backtest.wf_test_days",    126)
            s.backtest.wf_step_days      = _i("backtest.wf_step_days",    63)

            # 리스크
            s.risk.vol_target            = _f("risk.vol_target",          0.15)
            s.risk.max_drawdown_limit    = _f("risk.max_drawdown_limit",  0.20)
            s.risk.kill_switch_sharpe    = _f("risk.kill_switch_sharpe",  -0.5)
            s.risk.vol_lookback          = _i("risk.vol_lookback",        20)
            s.risk.correlation_cap       = _f("risk.correlation_cap",     0.7)
            s.risk.regime_detection      = _bool(
                self._vars.get("risk.regime_detection",
                               tk.StringVar(value="True")).get())

            import os
            import sys
            BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            save_settings(s, os.path.join(BASE_DIR, "settings.json"))
            self.on_change(s)
            messagebox.showinfo(
                "설정 저장 완료",
                "설정이 저장되었습니다.\n\n"
                "새 설정은 다음 학습 / 백테스트 실행부터 적용됩니다.",
            )
        except Exception as e:
            messagebox.showerror("오류", f"설정 적용 실패:\n{e}")

    def _restore_defaults(self):
        if messagebox.askyesno(
            "기본값 복원",
            "모든 설정을 프로그램 기본값으로 되돌리시겠습니까?\n\n"
            "이 작업은 취소할 수 없습니다.",
            icon="warning",
        ):
            self.settings = AppSettings()
            self._refresh_vars_from_settings(self.settings)
            self.on_change(self.settings)
            messagebox.showinfo(
                "기본값 복원 완료",
                "모든 설정이 기본값으로 복원되었습니다.",
            )

    def _refresh_vars_from_settings(self, s: AppSettings) -> None:
        """self._vars 값을 settings 객체에서 갱신 (복원 / 프리셋 적용 시 UI 동기화)"""
        mapping = {
            "data.period":                  s.data.period,
            "data.interval":                s.data.interval,
            "data.cache_ttl_hours":         s.data.cache_ttl_hours,
            "data.benchmark":               s.data.benchmark,
            "roi.segment_length":           s.roi.segment_length,
            "roi.lookahead":                s.roi.lookahead,
            "roi.vol_z_threshold":          s.roi.vol_z_threshold,
            "roi.breakout_threshold":       s.roi.breakout_threshold,
            "roi.volume_spike_threshold":   s.roi.volume_spike_threshold,
            "roi.min_roi_spacing":          s.roi.min_roi_spacing,
            "roi.rolling_window":           s.roi.rolling_window,
            "model.d_model":                s.model.d_model,
            "model.nhead":                  s.model.nhead,
            "model.num_encoder_layers":     s.model.num_encoder_layers,
            "model.dim_feedforward":        s.model.dim_feedforward,
            "model.dropout":                s.model.dropout,
            "model.learning_rate":          s.model.learning_rate,
            "model.weight_decay":           s.model.weight_decay,
            "model.batch_size":             s.model.batch_size,
            "model.epochs":                 s.model.epochs,
            "model.patience":               s.model.patience,
            "model.image_size":             s.model.image_size,
            "portfolio.method":             s.portfolio.method,
            "portfolio.max_weight":         s.portfolio.max_weight,
            "portfolio.min_weight":         s.portfolio.min_weight,
            "portfolio.turnover_limit":     s.portfolio.turnover_limit,
            "portfolio.target_volatility":  s.portfolio.target_volatility,
            "portfolio.rebalance_freq":     s.portfolio.rebalance_freq,
            "portfolio.lookback_vol":       s.portfolio.lookback_vol,
            "backtest.initial_capital":     s.backtest.initial_capital,
            "backtest.transaction_cost":    s.backtest.transaction_cost,
            "backtest.slippage":            s.backtest.slippage,
            "backtest.execution_delay":     s.backtest.execution_delay,
            "backtest.wf_train_days":       s.backtest.wf_train_days,
            "backtest.wf_test_days":        s.backtest.wf_test_days,
            "backtest.wf_step_days":        s.backtest.wf_step_days,
            "risk.vol_target":              s.risk.vol_target,
            "risk.max_drawdown_limit":      s.risk.max_drawdown_limit,
            "risk.kill_switch_sharpe":      s.risk.kill_switch_sharpe,
            "risk.vol_lookback":            s.risk.vol_lookback,
            "risk.correlation_cap":         s.risk.correlation_cap,
            "risk.regime_detection":        s.risk.regime_detection,
        }
        for key, val in mapping.items():
            if key in self._vars:
                self._vars[key].set(str(val))
