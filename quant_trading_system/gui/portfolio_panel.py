# gui/portfolio_panel.py — 포트폴리오 모니터링 패널
from __future__ import annotations
import os
import sys
import threading
import tkinter as tk
from tkinter import ttk
import logging
import numpy as np
from datetime import datetime

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)

from config import AppSettings
from data import DataLoader
from data.korean_stocks import get_name
from features import ROIDetector, CVFeatureExtractor, TSFeatureExtractor
from models.hybrid import HybridModel
from models.trainer import ModelTrainer
from models.store import ModelStore
from signals import SignalGenerator
from portfolio import PortfolioConstructor
from risk import RiskManager
from gui.tooltip import add_tooltip

logger = logging.getLogger("quant.gui.portfolio")


class PortfolioPanel:
    def __init__(self, parent, settings: AppSettings):
        self.settings = settings
        self._stop_event = threading.Event()
        self._thread     = None
        self._name_cache: dict[str, str] = {}

        self._rt_after_id: str | None = None   # after() 스케줄러 ID
        self._rt_auto_on  = tk.BooleanVar(value=False)
        self._rt_interval = tk.StringVar(value="30초")
        self._rt_fetching = False

        self.frame = ttk.Frame(parent)
        self._build()
        self.refresh_symbols()

    # ─────────────────────────────────────────────
    # 공개 API
    # ─────────────────────────────────────────────

    def refresh_symbols(self):
        """데이터 탭 종목 변경 시 main_window에서 호출"""
        self._refresh_sym_bar()

    # ─────────────────────────────────────────────
    # 이름 헬퍼
    # ─────────────────────────────────────────────

    def _get_name(self, sym: str) -> str:
        if sym not in self._name_cache:
            self._name_cache[sym] = get_name(sym)
        return self._name_cache[sym]

    def _sym_label(self, sym: str) -> str:
        name = self._get_name(sym)
        code = sym.split(".")[0]
        return f"{name} ({code})" if name and name != sym else sym

    # ─────────────────────────────────────────────
    # UI 구성
    # ─────────────────────────────────────────────

    def _build(self):
        # ── 배너 ────────────────────────────────
        banner = tk.Frame(self.frame, bg="#1a1a2e", pady=7)
        banner.pack(fill="x", padx=6, pady=(6, 2))
        tk.Label(
            banner,
            text="💼  5단계: 포트폴리오  —  비중 계산 · 리스크 관리 · 실시간 신호 모니터링",
            bg="#1a1a2e", fg="#cba6f7",
            font=("맑은 고딕", 10, "bold"),
        ).pack(side="left", padx=12)
        tk.Label(
            banner,
            text="  (AI 모델 학습은 ② 학습 탭에서 하세요)",
            bg="#1a1a2e", fg="#9399b2",
            font=("맑은 고딕", 8),
        ).pack(side="left")

        # ── 대상 종목 띠 ─────────────────────────
        sym_bar = tk.Frame(self.frame, bg="#0f1a2a", height=26)
        sym_bar.pack(fill="x", padx=6, pady=(0, 2))
        sym_bar.pack_propagate(False)
        tk.Label(sym_bar, text="🎯 대상 종목:", bg="#0f1a2a", fg="#89b4fa",
                 font=("맑은 고딕", 9, "bold")).pack(side="left", padx=(10, 6))
        self._sym_bar_var = tk.StringVar(value="")
        tk.Label(sym_bar, textvariable=self._sym_bar_var,
                 bg="#0f1a2a", fg="#cdd6f4",
                 font=("맑은 고딕", 9), anchor="w").pack(side="left", fill="x", expand=True)

        # ── 상단 버튼 행 ─────────────────────────
        top_fr = tk.Frame(self.frame, bg="#1e1e2e", pady=4)
        top_fr.pack(fill="x", padx=6, pady=(0, 4))

        self.calc_btn = ttk.Button(top_fr, text="🔄 포트폴리오 계산",
                                   command=self._compute_portfolio,
                                   style="Accent.TButton")
        self.calc_btn.pack(side="left", padx=(0, 6))
        add_tooltip(self.calc_btn,
                    "등록된 모든 종목의 신호를 계산하여 최적 투자 비중을 산출합니다.\n\n"
                    "AI 모델이 학습된 종목은 모델 신호를, 그렇지 않으면 모멘텀 신호를 사용합니다.")

        self.status_var = tk.StringVar(value="대기 중  —  '포트폴리오 계산' 버튼을 누르세요")
        ttk.Label(top_fr, textvariable=self.status_var,
                  foreground="#89b4fa",
                  font=("맑은 고딕", 9)).pack(side="left", padx=4)

        # ── 실시간 시세 섹션 ─────────────────────
        rt_frame = ttk.LabelFrame(self.frame, text="📡 실시간 시세 (yfinance · ~15분 지연)",
                                  padding=(6, 4))
        rt_frame.pack(fill="x", padx=6, pady=(0, 4))
        self._build_realtime(rt_frame)

        # ── 메인 PanedWindow ─────────────────────
        pane = tk.PanedWindow(self.frame, orient="horizontal",
                              bg="#1e1e2e", sashwidth=4)
        pane.pack(fill="both", expand=True, padx=6, pady=(0, 6))

        left  = ttk.LabelFrame(pane, text="포트폴리오 비중", padding=8)
        right = ttk.LabelFrame(pane, text="신호 및 분석",   padding=8)
        pane.add(left,  minsize=350)
        pane.add(right, minsize=500)

        self._build_left(left)
        self._build_right(right)

    def _build_realtime(self, parent):
        # ── 컨트롤 행 ──────────────────────────
        ctrl = tk.Frame(parent, bg="#1e1e2e")
        ctrl.pack(fill="x", pady=(0, 4))

        self._rt_btn = ttk.Button(ctrl, text="🔄 시세 조회",
                                  command=self._fetch_realtime)
        self._rt_btn.pack(side="left", padx=(0, 6))
        add_tooltip(self._rt_btn, "등록된 모든 종목의 현재가를 yfinance로 조회합니다.\n(약 15분 지연)")

        ttk.Label(ctrl, text="자동새로고침:").pack(side="left", padx=(8, 2))
        interval_cb = ttk.Combobox(ctrl, textvariable=self._rt_interval,
                                   values=["15초", "30초", "1분", "5분"],
                                   width=5, state="readonly")
        interval_cb.pack(side="left", padx=(0, 4))

        self._rt_auto_chk = ttk.Checkbutton(
            ctrl, text="자동", variable=self._rt_auto_on,
            command=self._toggle_auto_refresh)
        self._rt_auto_chk.pack(side="left", padx=(0, 10))
        add_tooltip(self._rt_auto_chk, "켜면 선택한 간격으로 시세를 자동 갱신합니다.")

        self._rt_status_var = tk.StringVar(value="조회 전")
        ttk.Label(ctrl, textvariable=self._rt_status_var,
                  foreground="#9399b2",
                  font=("맑은 고딕", 8)).pack(side="left")

        # ── 시세 테이블 ────────────────────────
        cols = ("종목명", "현재가", "전일비", "등락률(%)", "고가", "저가", "거래량")
        self._rt_tree = ttk.Treeview(parent, columns=cols,
                                     show="headings", height=5)
        widths = [150, 90, 90, 90, 80, 80, 100]
        for col, w in zip(cols, widths):
            self._rt_tree.heading(col, text=col)
            self._rt_tree.column(col, width=w,
                                 anchor="e" if col != "종목명" else "w")

        vsb_rt = ttk.Scrollbar(parent, orient="vertical",
                               command=self._rt_tree.yview)
        self._rt_tree.configure(yscrollcommand=vsb_rt.set)
        self._rt_tree.pack(side="left", fill="x", expand=True)
        vsb_rt.pack(side="right", fill="y")

        self._rt_tree.tag_configure("up",   foreground="#a6e3a1")
        self._rt_tree.tag_configure("down", foreground="#f38ba8")
        self._rt_tree.tag_configure("flat", foreground="#9399b2")

    def _build_left(self, parent):
        # 파이 차트
        self.pie_canvas = tk.Canvas(parent, height=220,
                                    bg="#181825", highlightthickness=0)
        self.pie_canvas.pack(fill="x", pady=(0, 8))

        # 비중 테이블
        tk.Label(parent, text="비중 테이블",
                 bg="#1e1e2e", fg="#9399b2",
                 font=("맑은 고딕", 8)).pack(anchor="w")

        cols = ("종목명", "비중", "기댓값(μ)", "불확실성(σ)", "신호")
        self.weight_tree = ttk.Treeview(parent, columns=cols,
                                        show="headings", height=12)
        self.weight_tree.heading("종목명",    text="종목명")
        self.weight_tree.heading("비중",      text="비중")
        self.weight_tree.heading("기댓값(μ)", text="기댓값(μ)")
        self.weight_tree.heading("불확실성(σ)",text="불확실성(σ)")
        self.weight_tree.heading("신호",      text="신호")

        self.weight_tree.column("종목명",     width=160, anchor="w")
        self.weight_tree.column("비중",       width=55,  anchor="center")
        self.weight_tree.column("기댓값(μ)",  width=75,  anchor="center")
        self.weight_tree.column("불확실성(σ)",width=75,  anchor="center")
        self.weight_tree.column("신호",       width=70,  anchor="center")

        vsb = ttk.Scrollbar(parent, orient="vertical",
                             command=self.weight_tree.yview)
        self.weight_tree.configure(yscrollcommand=vsb.set)
        self.weight_tree.pack(side="left", fill="both", expand=True)
        vsb.pack(side="right", fill="y")

        self.weight_tree.tag_configure("long",    foreground="#a6e3a1")
        self.weight_tree.tag_configure("short",   foreground="#f38ba8")
        self.weight_tree.tag_configure("neutral", foreground="#9399b2")

        # 키보드 이동 시 선택 유지
        for key in ("<Up>", "<Down>", "<Prior>", "<Next>"):
            self.weight_tree.bind(key, self._on_tree_keypress)

    def _build_right(self, parent):
        # 포트폴리오 통계
        stat_fr = ttk.LabelFrame(parent, text="기대 성과", padding=8)
        stat_fr.pack(fill="x", pady=(0, 8))

        self.stat_vars = {}
        stats = [("기대 연수익률", "exp_ret"), ("기대 변동성", "exp_vol"),
                 ("기대 Sharpe",   "exp_sharpe"), ("레짐",     "regime"),
                 ("킬스위치",      "killswitch")]
        for i, (label, key) in enumerate(stats):
            col = i % 2
            row = i // 2
            ttk.Label(stat_fr, text=label + ":").grid(
                row=row, column=col*2, sticky="e", padx=4, pady=2)
            var = tk.StringVar(value="—")
            self.stat_vars[key] = var
            ttk.Label(stat_fr, textvariable=var,
                      foreground="#89b4fa",
                      font=("맑은 고딕", 10, "bold")).grid(
                row=row, column=col*2+1, sticky="w", padx=4)

        # 리스크 게이지
        risk_fr = ttk.LabelFrame(parent, text="리스크 게이지", padding=8)
        risk_fr.pack(fill="x", pady=(0, 8))
        self.risk_canvas = tk.Canvas(risk_fr, height=60,
                                     bg="#181825", highlightthickness=0)
        self.risk_canvas.pack(fill="x")

        # 신호 세부 정보
        sig_fr = ttk.LabelFrame(parent, text="종목별 신호 상세", padding=4)
        sig_fr.pack(fill="both", expand=True)

        cols2 = ("종목명", "기댓값(μ)", "불확실성(σ)", "신뢰도(μ/σ)", "신호", "비중")
        self.sig_tree = ttk.Treeview(sig_fr, columns=cols2,
                                     show="headings", height=12)
        self.sig_tree.heading("종목명",      text="종목명")
        self.sig_tree.heading("기댓값(μ)",   text="기댓값(μ)")
        self.sig_tree.heading("불확실성(σ)", text="불확실성(σ)")
        self.sig_tree.heading("신뢰도(μ/σ)", text="신뢰도(μ/σ)")
        self.sig_tree.heading("신호",        text="신호")
        self.sig_tree.heading("비중",        text="비중")

        self.sig_tree.column("종목명",       width=160, anchor="w")
        self.sig_tree.column("기댓값(μ)",    width=80,  anchor="center")
        self.sig_tree.column("불확실성(σ)",  width=80,  anchor="center")
        self.sig_tree.column("신뢰도(μ/σ)",  width=80,  anchor="center")
        self.sig_tree.column("신호",         width=70,  anchor="center")
        self.sig_tree.column("비중",         width=60,  anchor="center")

        vsb2 = ttk.Scrollbar(sig_fr, orient="vertical",
                              command=self.sig_tree.yview)
        self.sig_tree.configure(yscrollcommand=vsb2.set)
        self.sig_tree.pack(side="left", fill="both", expand=True)
        vsb2.pack(side="right", fill="y")

        for key in ("<Up>", "<Down>", "<Prior>", "<Next>"):
            self.sig_tree.bind(key, self._on_tree_keypress)

    # ─────────────────────────────────────────────
    # 종목 띠 갱신
    # ─────────────────────────────────────────────

    def _refresh_sym_bar(self):
        syms = self.settings.data.symbols
        if not syms:
            self._sym_bar_var.set("(등록된 종목 없음 — 데이터 탭에서 종목을 추가하세요)")
            return
        parts = [self._sym_label(s) for s in syms]
        MAX = 6
        shown = parts[:MAX]
        rest  = len(parts) - MAX
        text  = "  |  ".join(shown) + (f"  +{rest}개" if rest > 0 else "")
        self._sym_bar_var.set(f"총 {len(syms)}개:  " + text)

    # ─────────────────────────────────────────────
    # 키보드 네비게이션
    # ─────────────────────────────────────────────

    def _on_tree_keypress(self, event):
        tree = event.widget
        def _sync():
            focused = tree.focus()
            if focused:
                tree.selection_set(focused)
        self.frame.after(30, _sync)

    # ─────────────────────────────────────────────
    # 포트폴리오 계산
    # ─────────────────────────────────────────────

    def _compute_portfolio(self):
        if self._thread and self._thread.is_alive():
            return
        self.weight_tree.delete(*self.weight_tree.get_children())
        self.sig_tree.delete(*self.sig_tree.get_children())
        self._set_status("계산 중...")
        self.calc_btn.config(state="disabled")
        self._thread = threading.Thread(target=self._compute_thread, daemon=True)
        self._thread.start()

    def _compute_thread(self):
        try:
            symbols = self.settings.data.symbols
            if not symbols:
                self._set_status("종목 없음. 데이터 탭에서 다운로드하세요.")
                return

            cache_dir = os.path.join(BASE_DIR, self.settings.data.cache_dir)
            loader    = DataLoader(cache_dir)
            store     = ModelStore(os.path.join(BASE_DIR, self.settings.model_dir))

            roi_det      = ROIDetector(segment_length=30, lookahead=5)
            cv_ext       = CVFeatureExtractor(image_size=64, method="gasf")
            ts_ext       = TSFeatureExtractor(n_features=32)
            signal_gen   = SignalGenerator(method="ranking")
            portfolio_ctor = PortfolioConstructor(
                method=self.settings.portfolio.method,
                max_weight=self.settings.portfolio.max_weight,
                target_vol=self.settings.portfolio.target_volatility,
            )
            risk_mgr = RiskManager(
                vol_target=self.settings.risk.vol_target,
                max_drawdown_limit=self.settings.risk.max_drawdown_limit,
            )

            preds       = {}
            all_returns = {}

            for sym in symbols:
                df = loader.load(sym, self.settings.data.period)
                if df is None or len(df) < 60:
                    continue

                ret = df["Close"].pct_change().dropna()
                all_returns[sym] = ret

                segs, _, _ = roi_det.extract_segments(df)
                if len(segs) == 0:
                    preds[sym] = (ret.mean() * 252, ret.std() * np.sqrt(252))
                    continue

                use_segs    = segs[-min(5, len(segs)):]
                model_loaded = False

                if store.has_model(sym):
                    try:
                        import torch as _pt
                        _pf_device = "cuda" if _pt.cuda.is_available() else "cpu"

                        # 저장된 config 먼저 읽어서 모델 파라미터 복원
                        _ckpt_path = os.path.join(
                            BASE_DIR, self.settings.model_dir,
                            sym.replace(".", "_"), "latest.pt"
                        )
                        _cfg = {}
                        if os.path.exists(_ckpt_path):
                            try:
                                _peeked = _pt.load(_ckpt_path, map_location="cpu",
                                                   weights_only=False)
                                _cfg = _peeked.get("config", {})
                            except Exception:
                                pass

                        imgs = cv_ext.transform(use_segs)
                        ts_f = ts_ext.transform(use_segs)

                        model = HybridModel(
                            img_in_channels=imgs.shape[1] if imgs.ndim >= 2 else 1,
                            ts_input_dim=_cfg.get("ts_input_dim", ts_f.shape[-1] if ts_f.ndim >= 2 else 32),
                            cnn_out_dim=_cfg.get("cnn_out_dim", 128),
                            d_model=_cfg.get("d_model", self.settings.model.d_model),
                            nhead=_cfg.get("nhead", self.settings.model.nhead),
                            num_encoder_layers=_cfg.get("num_encoder_layers",
                                                        self.settings.model.num_encoder_layers),
                            dropout=_cfg.get("dropout", self.settings.model.dropout),
                        )
                        ckpt = store.load(model, sym, device=_pf_device)
                        if ckpt:
                            trainer = ModelTrainer(model, device=_pf_device)
                            mu_arr, sigma_arr = trainer.predict(imgs, ts_f)
                            preds[sym]   = (float(mu_arr.mean()),
                                            float(sigma_arr.mean()))
                            model_loaded = True
                    except Exception as e:
                        logger.debug(f"{sym} 모델 로드 실패: {e}")

                if not model_loaded:
                    preds[sym] = (ret.mean() * 252, ret.std() * np.sqrt(252))

            if not preds:
                self._set_status("예측 데이터 없음")
                return

            import pandas as pd
            signal_df   = signal_gen.generate_from_dict(preds)
            returns_df  = pd.DataFrame(all_returns).dropna()
            weights     = portfolio_ctor.construct(signal_df, returns_df)

            if all_returns:
                combined_ret = (returns_df.mean(axis=1)
                                if not returns_df.empty
                                else pd.Series(dtype=float))
                weights = risk_mgr.adjust_weights(weights, combined_ret, 1.0)

            stats = portfolio_ctor.compute_portfolio_stats(weights, returns_df)
            self.frame.after(0, lambda: self._update_display(
                signal_df, weights, stats, risk_mgr))

        except Exception as e:
            import traceback
            logger.error(traceback.format_exc())
            self._set_status(f"오류: {e}")
        finally:
            self.frame.after(0, lambda: self.calc_btn.config(state="normal"))

    def _update_display(self, signal_df, weights, stats, risk_mgr):
        # 비중 테이블
        self.weight_tree.delete(*self.weight_tree.get_children())
        for row in signal_df.itertuples():
            w   = weights.get(row.symbol, 0)
            tag = ("long"    if row.signal == 1
                   else "short"   if row.signal == -1
                   else "neutral")
            self.weight_tree.insert("", "end", tags=(tag,), values=(
                self._sym_label(row.symbol),
                f"{w:.1%}",
                f"{row.mu:+.4f}",
                f"{row.sigma:.4f}",
                "🔼 롱" if row.signal == 1 else (
                    "🔽 숏" if row.signal == -1 else "—"),
            ))

        # 신호 상세
        self.sig_tree.delete(*self.sig_tree.get_children())
        for row in signal_df.sort_values("confidence", ascending=False).itertuples():
            self.sig_tree.insert("", "end", values=(
                self._sym_label(row.symbol),
                f"{row.mu:+.4f}",
                f"{row.sigma:.4f}",
                f"{row.confidence:+.3f}",
                "롱" if row.signal == 1 else (
                    "숏" if row.signal == -1 else "중립"),
                f"{weights.get(row.symbol, 0):.1%}",
            ))

        # 통계
        self.stat_vars["exp_ret"].set(
            f"{stats.get('expected_annual_return', 0):.2%}")
        self.stat_vars["exp_vol"].set(
            f"{stats.get('expected_annual_vol', 0):.2%}")
        self.stat_vars["exp_sharpe"].set(
            f"{stats.get('expected_sharpe', 0):.3f}")
        self.stat_vars["regime"].set(risk_mgr.get_regime())
        self.stat_vars["killswitch"].set(
            "🚨 활성" if risk_mgr.is_kill_switch_active() else "✅ 비활성")

        self._draw_pie(weights)
        self._set_status(f"포트폴리오 계산 완료: {len(weights)}개 종목")

    def _draw_pie(self, weights):
        canvas = self.pie_canvas
        canvas.delete("all")
        W = canvas.winfo_width()
        H = canvas.winfo_height()
        if W < 50 or H < 50 or not weights:
            return

        import math
        cx, cy, r = W//2, H//2, min(W, H)//2 - 20
        total  = sum(weights.values()) or 1
        colors = ["#89b4fa", "#a6e3a1", "#fab387", "#f9e2af",
                  "#cba6f7", "#89dceb", "#f38ba8", "#74c7ec"]

        start = -90
        for i, (sym, w) in enumerate(weights.items()):
            extent = w / total * 360
            color  = colors[i % len(colors)]
            canvas.create_arc(cx-r, cy-r, cx+r, cy+r,
                              start=start, extent=extent,
                              fill=color, outline="#1e1e2e", width=2)
            mid_angle = math.radians(start + extent/2)
            lx = cx + (r * 0.65) * math.cos(mid_angle)
            ly = cy + (r * 0.65) * math.sin(mid_angle)
            name = self._get_name(sym)
            disp = name[:4] if (name and name != sym) else sym.split(".")[0]
            canvas.create_text(lx, ly,
                               text=f"{disp}\n{w:.0%}",
                               fill="#1e1e2e",
                               font=("맑은 고딕", 8, "bold"),
                               justify="center")
            start += extent

    def _set_status(self, msg: str):
        self.frame.after(0, lambda: self.status_var.set(msg))

    # ─────────────────────────────────────────────
    # 실시간 시세 조회
    # ─────────────────────────────────────────────

    _INTERVAL_MAP = {"15초": 15_000, "30초": 30_000, "1분": 60_000, "5분": 300_000}

    def _fetch_realtime(self):
        """백그라운드 스레드에서 실시간 시세 조회"""
        if self._rt_fetching:
            return
        symbols = self.settings.data.symbols
        if not symbols:
            self._rt_status_var.set("종목 없음")
            return
        self._rt_fetching = True
        self._rt_btn.config(state="disabled")
        self._rt_status_var.set("조회 중…")
        threading.Thread(target=self._fetch_thread, daemon=True).start()

    def _fetch_thread(self):
        try:
            cache_dir = os.path.join(BASE_DIR, self.settings.data.cache_dir)
            loader    = DataLoader(cache_dir)
            symbols   = self.settings.data.symbols
            prices    = loader.get_realtime_prices(symbols)
            self.frame.after(0, lambda: self._update_rt_table(prices))
        except Exception as e:
            logger.error(f"실시간 시세 조회 오류: {e}")
            self.frame.after(0, lambda: self._rt_status_var.set(f"오류: {e}"))
        finally:
            self._rt_fetching = False
            self.frame.after(0, lambda: self._rt_btn.config(state="normal"))

    def _update_rt_table(self, prices: dict):
        self._rt_tree.delete(*self._rt_tree.get_children())

        def _fmt_price(v, currency=""):
            if v is None:
                return "—"
            if currency in ("KRW", ""):
                return f"{v:,.0f}"
            return f"{v:,.2f}"

        def _fmt_vol(v):
            if v is None:
                return "—"
            v = int(v)
            if v >= 1_000_000:
                return f"{v/1_000_000:.1f}M"
            if v >= 1_000:
                return f"{v/1_000:.0f}K"
            return str(v)

        for sym in self.settings.data.symbols:
            d = prices.get(sym)
            if d is None:
                self._rt_tree.insert("", "end", tags=("flat",), values=(
                    self._sym_label(sym), "—", "—", "—", "—", "—", "—"))
                continue

            cur   = d.get("currency", "")
            chg   = d.get("change", 0) or 0
            chgp  = d.get("change_pct", 0) or 0
            tag   = "up" if chg > 0 else ("down" if chg < 0 else "flat")
            sign  = "+" if chg >= 0 else ""

            self._rt_tree.insert("", "end", tags=(tag,), values=(
                self._sym_label(sym),
                _fmt_price(d.get("price"),      cur),
                f"{sign}{_fmt_price(chg, cur)}",
                f"{sign}{chgp:.2f}%",
                _fmt_price(d.get("high"),       cur),
                _fmt_price(d.get("low"),        cur),
                _fmt_vol(d.get("volume")),
            ))

        now = datetime.now().strftime("%H:%M:%S")
        total = len(prices)
        self._rt_status_var.set(f"최종 갱신: {now}  ({total}/{len(self.settings.data.symbols)}개 성공)")

        # 자동갱신 재스케줄
        if self._rt_auto_on.get():
            self._schedule_next()

    def _toggle_auto_refresh(self):
        if self._rt_auto_on.get():
            self._fetch_realtime()
        else:
            self._cancel_schedule()

    def _schedule_next(self):
        self._cancel_schedule()
        ms = self._INTERVAL_MAP.get(self._rt_interval.get(), 30_000)
        self._rt_after_id = self.frame.after(ms, self._auto_refresh_tick)

    def _auto_refresh_tick(self):
        if self._rt_auto_on.get():
            self._fetch_realtime()

    def _cancel_schedule(self):
        if self._rt_after_id:
            self.frame.after_cancel(self._rt_after_id)
            self._rt_after_id = None
