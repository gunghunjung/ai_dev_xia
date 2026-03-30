# gui/data_panel.py — 데이터 패널 (종목 관리 + 다운로드)
from __future__ import annotations
import os, sys, threading, tkinter as tk
from tkinter import ttk, messagebox, scrolledtext
import logging
from datetime import datetime

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)

from config import AppSettings
from data import DataLoader
from data.realtime import RealtimeFetcher
from data.korean_stocks import get_info, get_name
from gui.stock_search_dialog import StockSearchDialog
from gui.tooltip import add_tooltip

logger = logging.getLogger("quant.gui.data")

PERIODS   = {"1년": "1y", "2년": "2y", "5년": "5y", "최대": "max"}
INTERVALS = {"일봉": "1d", "주봉": "1wk"}

# 실시간 갱신 주기 (표시명 → 초)
RT_INTERVALS = {"3초": 3, "5초": 5, "10초": 10, "15초": 15,
                "30초": 30, "1분": 60, "3분": 180, "5분": 300}


class DataPanel:
    """
    데이터 패널
    - 종목 검색 다이얼로그로 등록 (유효성 검사 포함)
    - 잘못된 종목 추가 원천 차단
    - OHLCV 다운로드 및 캐시 관리
    - 미리보기 테이블
    """

    def __init__(self, parent, settings: AppSettings, on_change,
                 on_data_downloaded=None):
        self.settings = settings
        self.on_change = on_change
        self._on_data_downloaded = on_data_downloaded   # 다운로드 완료 전용 콜백
        self._thread: threading.Thread | None = None
        self._stop_event = threading.Event()
        self._rt_thread: threading.Thread | None = None
        self._rt_fetching = False
        self._rt_after_id: str | None = None
        self._rt_auto_on  = tk.BooleanVar(value=False)
        self._rt_interval = tk.StringVar(value="30초")

        # 현재 등록된 종목 목록: [(ticker, name), ...]
        self._portfolio: list[tuple[str, str]] = [
            (t, get_name(t)) for t in settings.data.symbols
        ]

        # 실시간 시세 관련
        self._rt_fetcher   = RealtimeFetcher(cache_sec=2)
        self._rt_thread: threading.Thread | None = None
        self._rt_stop      = threading.Event()
        self._rt_running   = False

        self.frame = ttk.Frame(parent)
        self._build()

        # 프로그램 시작 시 자동으로 실시간 갱신 시작
        self.frame.after(500, self._start_realtime)

    # ─────────────────────────────────────────────
    # UI 구성
    # ─────────────────────────────────────────────

    def _build(self):
        # 안내 배너 (1단계 가이드)
        banner = tk.Frame(self.frame, bg="#1a2a1a", pady=7)
        banner.pack(fill="x", padx=6, pady=(6, 2))
        tk.Label(
            banner,
            text="📊  1단계: 분석할 종목을 추가하고, 데이터를 다운로드하세요",
            bg="#1a2a1a", fg="#a6e3a1",
            font=("맑은 고딕", 10, "bold"),
        ).pack(side="left", padx=12)
        tk.Label(
            banner,
            text="  종목 추가 → 데이터 다운로드 → 학습 탭으로 이동",
            bg="#1a2a1a", fg="#9399b2",
            font=("맑은 고딕", 9),
        ).pack(side="left")

        pane = tk.PanedWindow(self.frame, orient="horizontal",
                              bg="#1e1e2e", sashwidth=4)
        pane.pack(fill="both", expand=True, padx=6, pady=4)

        left  = ttk.LabelFrame(pane, text="① 종목 목록 관리", padding=8)
        right = ttk.LabelFrame(pane, text="② 다운로드 결과 미리보기", padding=8)
        pane.add(left,  minsize=240)
        pane.add(right, minsize=480)

        self._build_left(left)
        self._build_right(right)

    def _build_left(self, parent):
        # 기간 / 주기 설정
        cfg_fr = ttk.LabelFrame(parent, text="데이터 수집 설정", padding=6)
        cfg_fr.pack(fill="x", pady=(0, 8))

        cfg = ttk.Frame(cfg_fr)
        cfg.pack(fill="x")

        ttk.Label(cfg, text="기간:").grid(row=0, column=0, sticky="w", padx=4)
        self.period_var = tk.StringVar(value="5년")
        period_cb = ttk.Combobox(cfg, textvariable=self.period_var,
                     values=list(PERIODS), width=9, state="readonly")
        period_cb.grid(row=0, column=1, padx=4)
        add_tooltip(period_cb,
                    "몇 년치 과거 주가 데이터를 가져올지 설정합니다.\n\n"
                    "초보자 추천: 5년\n"
                    "기간이 길수록 AI 학습에 유리하지만 다운로드 시간이 길어집니다.")

        ttk.Label(cfg, text="주기:").grid(row=0, column=2, sticky="w", padx=(12, 4))
        self.interval_var = tk.StringVar(value="일봉")
        interval_cb = ttk.Combobox(cfg, textvariable=self.interval_var,
                     values=list(INTERVALS), width=9, state="readonly")
        interval_cb.grid(row=0, column=3, padx=4)
        add_tooltip(interval_cb,
                    "일봉: 매일의 시가/고가/저가/종가 데이터 (추천)\n"
                    "주봉: 매주 한 번의 데이터 (장기 분석에 적합)")

        tk.Label(cfg_fr,
                 text="  💡 초보자 추천: 기간=5년, 주기=일봉",
                 bg="#1e1e2e", fg="#9399b2",
                 font=("맑은 고딕", 8)).pack(anchor="w", pady=(4, 0))

        # 종목 리스트
        tk.Label(parent,
                 text="등록 종목  (분석할 주식을 추가하세요):",
                 bg="#1e1e2e", fg="#cdd6f4",
                 font=("맑은 고딕", 9, "bold")).pack(anchor="w", pady=(0, 2))

        list_fr = ttk.Frame(parent)
        list_fr.pack(fill="both", expand=True)

        cols = ("ticker", "name", "status")
        self.port_tree = ttk.Treeview(list_fr, columns=cols,
                                      show="headings", height=6,
                                      selectmode="extended")
        self.port_tree.heading("ticker", text="종목코드")
        self.port_tree.heading("name",   text="종목명")
        self.port_tree.heading("status", text="상태")
        self.port_tree.column("ticker", width=110, anchor="center")
        self.port_tree.column("name",   width=120, anchor="w")
        self.port_tree.column("status", width=60,  anchor="center")
        self.port_tree.tag_configure("ok",      foreground="#a6e3a1")
        self.port_tree.tag_configure("error",   foreground="#f38ba8")
        self.port_tree.tag_configure("pending", foreground="#f9e2af")

        vsb = ttk.Scrollbar(list_fr, orient="vertical",
                            command=self.port_tree.yview)
        self.port_tree.configure(yscrollcommand=vsb.set)
        self.port_tree.pack(side="left", fill="both", expand=True)
        vsb.pack(side="right", fill="y")

        self._refresh_portfolio_tree()

        # 관리 버튼
        btn_fr = ttk.Frame(parent)
        btn_fr.pack(fill="x", pady=(8, 0))

        add_btn = ttk.Button(btn_fr, text="🔍 종목 검색 및 추가",
                   command=self._open_search,
                   style="Accent.TButton")
        add_btn.pack(side="left", padx=(0, 4))
        add_tooltip(add_btn,
                    "종목명이나 종목코드로 검색하여 분석 목록에 추가합니다.\n\n"
                    "예) '삼성전자' 또는 '005930' 으로 검색")

        rem_btn = ttk.Button(btn_fr, text="✕ 제거",
                   command=self._remove_selected)
        rem_btn.pack(side="left", padx=(0, 4))
        add_tooltip(rem_btn, "선택한 종목을 목록에서 제거합니다")

        up_btn = ttk.Button(btn_fr, text="▲", command=self._move_up, width=3)
        up_btn.pack(side="left", padx=2)
        add_tooltip(up_btn, "선택한 종목을 위로 이동합니다")

        dn_btn = ttk.Button(btn_fr, text="▼", command=self._move_down, width=3)
        dn_btn.pack(side="left", padx=2)
        add_tooltip(dn_btn, "선택한 종목을 아래로 이동합니다")

        ttk.Separator(parent, orient="horizontal").pack(fill="x", pady=8)

        # 다운로드 버튼
        dl_fr = ttk.Frame(parent)
        dl_fr.pack(fill="x")

        dl_btn = ttk.Button(dl_fr, text="💾 데이터 다운로드",
                   command=self._download)
        dl_btn.pack(side="left", padx=(0, 4))
        add_tooltip(dl_btn,
                    "목록에 있는 모든 종목의 주가 데이터를 다운로드합니다.\n\n"
                    "한 번 다운로드한 데이터는 23시간 동안 캐시에 저장됩니다.\n"
                    "데이터 다운로드 후 → 학습 탭으로 이동하세요.")

        cache_btn = ttk.Button(dl_fr, text="🗑️ 캐시 삭제",
                   command=self._clear_cache)
        cache_btn.pack(side="left")
        add_tooltip(cache_btn,
                    "저장된 캐시 데이터를 삭제합니다.\n"
                    "다음 다운로드 시 최신 데이터를 새로 받아옵니다.")

        # 실시간 현재가 버튼 행
        rt_fr = ttk.Frame(parent)
        rt_fr.pack(fill="x", pady=(6, 0))

        self._rt_btn = ttk.Button(rt_fr, text="📡 현재가 조회",
                                  command=self._fetch_realtime,
                                  style="Accent.TButton")
        self._rt_btn.pack(side="left", padx=(0, 4))
        add_tooltip(self._rt_btn,
                    "등록된 종목의 현재가를 yfinance로 즉시 조회합니다.\n"
                    "(약 15분 지연 · 다운로드 없이 빠르게 확인)")

        ttk.Label(rt_fr, text="자동:").pack(side="left", padx=(6, 2))
        rt_interval_cb = ttk.Combobox(rt_fr, textvariable=self._rt_interval,
                                      values=["15초", "30초", "1분", "5분"],
                                      width=5, state="readonly")
        rt_interval_cb.pack(side="left", padx=(0, 3))
        self._rt_auto_chk = ttk.Checkbutton(
            rt_fr, text="자동갱신", variable=self._rt_auto_on,
            command=self._toggle_rt_auto)
        self._rt_auto_chk.pack(side="left")
        add_tooltip(self._rt_auto_chk, "켜면 선택 간격으로 현재가를 자동 갱신합니다.")

        self._rt_status_var = tk.StringVar(value="")
        ttk.Label(rt_fr, textvariable=self._rt_status_var,
                  foreground="#9399b2",
                  font=("맑은 고딕", 8)).pack(side="left", padx=(6, 0))

        # 진행 상태
        self.progress = ttk.Progressbar(parent, mode="determinate",
                                        maximum=100, value=0)
        self.progress.pack(fill="x", pady=(8, 2))
        self.status_var = tk.StringVar(value="준비 — 종목을 추가하고 '데이터 다운로드'를 누르세요")
        ttk.Label(parent, textvariable=self.status_var,
                  font=("맑은 고딕", 9)).pack(anchor="w")

        # 다운로드 완료 후 "다음 단계" 안내 배너 (초기에는 숨김)
        self._next_step_bar = tk.Frame(parent, bg="#1a3a1a", height=0)
        self._next_step_bar.pack(fill="x", pady=(4, 0))
        self._next_step_bar.pack_propagate(False)
        tk.Label(
            self._next_step_bar,
            text="✅  데이터 업데이트 완료!  →  🧠 학습 탭에서 모델을 재학습하고  →  🔮 예측 탭에서 예측을 갱신하세요",
            bg="#1a3a1a", fg="#a6e3a1",
            font=("맑은 고딕", 9, "bold"), anchor="w",
        ).pack(fill="both", expand=True, padx=10)

    def _build_right(self, parent):
        # 요약 카드
        sum_fr = ttk.Frame(parent)
        sum_fr.pack(fill="x", pady=(0, 6))

        self.summary_vars = {}
        cols_info = [("종목 수", "n"), ("총 데이터", "rows"),
                     ("시작일", "start"), ("종료일", "end")]
        for i, (lbl, key) in enumerate(cols_info):
            ttk.Label(sum_fr, text=lbl + ":").grid(
                row=0, column=i*2, sticky="e", padx=(8, 2))
            var = tk.StringVar(value="—")
            self.summary_vars[key] = var
            ttk.Label(sum_fr, textvariable=var,
                      foreground="#89b4fa",
                      font=("맑은 고딕", 10, "bold")).grid(
                row=0, column=i*2+1, sticky="w", padx=(0, 12))

        # 데이터 미리보기 테이블
        tree_fr = ttk.Frame(parent)
        tree_fr.pack(fill="x")

        cols = ("종목코드", "종목명", "행수", "시작일", "종료일",
                "현재가", "전일비", "등락률", "시장", "상태")
        self.data_tree = ttk.Treeview(tree_fr, columns=cols,
<<<<<<< HEAD
                                      show="headings", height=5)
        col_widths = [110, 120, 70, 90, 90, 90, 70, 70, 70]
=======
                                      show="headings", height=8)
        col_widths = [110, 120, 60, 90, 90, 90, 80, 72, 60, 60]
>>>>>>> fc3701554ef854ff18ab2cb7f0bca37a9183375d
        for c, w in zip(cols, col_widths):
            self.data_tree.heading(c, text=c)
            self.data_tree.column(c, width=w, anchor="center")
        self.data_tree.column("종목명", anchor="w")
        self.data_tree.tag_configure("ok",    foreground="#a6e3a1")
        self.data_tree.tag_configure("fail",  foreground="#f38ba8")
        self.data_tree.tag_configure("rt_up",   foreground="#a6e3a1")
        self.data_tree.tag_configure("rt_down", foreground="#f38ba8")

        vsb = ttk.Scrollbar(tree_fr, orient="vertical",
                            command=self.data_tree.yview)
        self.data_tree.configure(yscrollcommand=vsb.set)
        self.data_tree.pack(side="left", fill="x", expand=True)
        vsb.pack(side="right", fill="y")

        # 로그
        tk.Label(parent,
                 text="다운로드 로그  (각 종목의 다운로드 상태가 여기에 표시됩니다):",
                 bg="#1e1e2e", fg="#9399b2",
                 font=("맑은 고딕", 9)).pack(anchor="w", pady=(6, 2))
        self.log_box = scrolledtext.ScrolledText(
            parent, height=3, bg="#181825", fg="#a6adc8",
            font=("Consolas", 9), relief="flat", state="disabled",
        )
        self.log_box.pack(fill="x")

        # ── 실시간 시세 모니터 ─────────────────────────
        self._build_realtime(parent)

    # ─────────────────────────────────────────────
    # 종목 관리
    # ─────────────────────────────────────────────

    def _open_search(self):
        """종목 검색 다이얼로그 열기"""
        StockSearchDialog(
            self.frame.winfo_toplevel(),
            on_add=self._on_tickers_added,
        )

    def _on_tickers_added(self, tickers: list[str]):
        """검색 다이얼로그에서 종목 추가 콜백"""
        added = 0
        for ticker in tickers:
            if not any(t == ticker for t, _ in self._portfolio):
                name = get_name(ticker)
                self._portfolio.append((ticker, name))
                added += 1
        if added:
            self._refresh_portfolio_tree()
            self._sync_settings()
            self.status_var.set(f"{added}개 종목 추가됨")
            self._log(f"종목 추가: {[t for t in tickers if t not in [p[0] for p in self._portfolio[:-added]]]}")

    def _remove_selected(self):
        sel = self.port_tree.selection()
        if not sel:
            return
        to_remove = set(self.port_tree.item(iid, "values")[0] for iid in sel)
        self._portfolio = [(t, n) for t, n in self._portfolio
                           if t not in to_remove]
        self._refresh_portfolio_tree()
        self._sync_settings()

    def _move_up(self):
        sel = self.port_tree.selection()
        if not sel:
            return
        iid = sel[0]
        idx = self.port_tree.index(iid)
        if idx == 0:
            return
        self._portfolio[idx-1], self._portfolio[idx] = \
            self._portfolio[idx], self._portfolio[idx-1]
        self._refresh_portfolio_tree()
        # 선택 복원
        children = self.port_tree.get_children()
        if idx-1 < len(children):
            self.port_tree.selection_set(children[idx-1])

    def _move_down(self):
        sel = self.port_tree.selection()
        if not sel:
            return
        iid = sel[0]
        idx = self.port_tree.index(iid)
        if idx >= len(self._portfolio) - 1:
            return
        self._portfolio[idx], self._portfolio[idx+1] = \
            self._portfolio[idx+1], self._portfolio[idx]
        self._refresh_portfolio_tree()
        children = self.port_tree.get_children()
        if idx+1 < len(children):
            self.port_tree.selection_set(children[idx+1])

    def _refresh_portfolio_tree(self):
        """포트폴리오 트리뷰 새로고침"""
        self.port_tree.delete(*self.port_tree.get_children())
        for ticker, name in self._portfolio:
            self.port_tree.insert("", "end", iid=ticker,
                                  values=(ticker, name, "—"),
                                  tags=("pending",))

    def _sync_settings(self):
        """포트폴리오를 settings에 동기화"""
        self.settings.data.symbols = [t for t, _ in self._portfolio]
        self.on_change(self.settings)
        # 실시간 테이블 종목 목록도 갱신
        if hasattr(self, "_rt_rows"):
            self._init_rt_table()

    # ─────────────────────────────────────────────
    # 데이터 다운로드
    # ─────────────────────────────────────────────

    def _download(self):
        if self._thread and self._thread.is_alive():
            messagebox.showinfo("알림", "이미 다운로드 중입니다.")
            return
        if not self._portfolio:
            messagebox.showwarning("경고", "다운로드할 종목이 없습니다.\n먼저 '🔍 종목 추가'로 종목을 등록하세요.")
            return

        period   = PERIODS[self.period_var.get()]
        interval = INTERVALS[self.interval_var.get()]
        self.settings.data.period   = period
        self.settings.data.interval = interval
        self._sync_settings()

        self._stop_event.clear()
        self.progress["value"] = 0
        self._thread = threading.Thread(
            target=self._download_thread,
            args=(list(self._portfolio), period, interval),
            daemon=True,
        )
        self._thread.start()

    def _download_thread(self, portfolio: list, period: str, interval: str):
        cache_dir = os.path.join(BASE_DIR, self.settings.data.cache_dir)
        loader    = DataLoader(cache_dir, self.settings.data.cache_ttl_hours)

        self.frame.after(0, lambda: self.data_tree.delete(*self.data_tree.get_children()))

        total_rows, starts, ends = 0, [], []
        n_total = len(portfolio)

        for i, (ticker, name) in enumerate(portfolio):
            if self._stop_event.is_set():
                break

            pct = int(i / n_total * 100)
            self.frame.after(0, lambda p=pct: self.progress.configure(value=p))
            self._set_status(f"({i+1}/{n_total}) 다운로드: {ticker} ({name})")
            self._log(f"[{ticker}] {name} 로드 시작...")

            try:
                df = loader.load(ticker, period, interval)
            except Exception as e:
                self._log(f"[{ticker}] 오류: {e}")
                self._update_port_status(ticker, "❌")
                self.frame.after(0, lambda t=ticker, n=name: self.data_tree.insert(
                    "", "end", iid=t,
                    values=(t, n, "0", "—", "—", "—", "—", "—", "—", "❌ 실패"),
                    tags=("fail",)
                ))
                continue

            if df is None or df.empty:
                self._log(f"[{ticker}] 데이터 없음 — 종목코드 확인 필요")
                self._update_port_status(ticker, "❌")
                self.frame.after(0, lambda t=ticker, n=name: self.data_tree.insert(
                    "", "end", iid=t,
                    values=(t, n, "0", "—", "—", "—", "—", "—", "—", "❌ 없음"),
                    tags=("fail",)
                ))
                continue

            # 정상 — 히스토리컬 마지막 종가
            info     = get_info(ticker) or {}
            market   = info.get("market", "—")
            hist_cls = df["Close"].iloc[-1] if "Close" in df.columns else 0
            hist_end = str(df.index[-1].date())

            total_rows += len(df)
            starts.append(str(df.index[0].date()))
            ends.append(hist_end)

            # 실시간 현재가 조회 (실패해도 히스토리 값으로 대체)
            rt = loader.get_realtime_price(ticker)
            if rt and rt.get("price"):
                cur    = rt["currency"] or ""
                price  = rt["price"]
                chg    = rt.get("change", 0) or 0
                chgp   = rt.get("change_pct", 0) or 0
                sign   = "+" if chg >= 0 else ""
                price_str = f"{price:,.0f}" if cur in ("KRW", "") else f"{price:,.2f}"
                chg_str   = f"{sign}{chg:,.0f}" if cur in ("KRW", "") else f"{sign}{chg:,.2f}"
                chgp_str  = f"{sign}{chgp:.2f}%"
                rt_tag    = "rt_up" if chg > 0 else ("rt_down" if chg < 0 else "ok")
                self._log(f"[{ticker}] 현재가: {price_str}  ({chgp_str})")
            else:
                # 실시간 조회 실패 시 히스토리 마지막 값 사용
                price_str = f"{hist_cls:,.0f}"
                chg_str   = "—"
                chgp_str  = "—"
                rt_tag    = "ok"

            row = (ticker, name, f"{len(df):,}",
                   str(df.index[0].date()), hist_end,
                   price_str, chg_str, chgp_str,
                   market, "✅ 완료")
            self._update_port_status(ticker, "✅")
            self._log(f"[{ticker}] {name}: {len(df):,}행 완료")
            self.frame.after(0, lambda r=row, tg=rt_tag: self.data_tree.insert(
                "", "end", iid=r[0], values=r, tags=(tg,)
            ))

        # 요약 업데이트
        n = n_total
        def _finish():
            self.summary_vars["n"].set(str(n))
            self.summary_vars["rows"].set(f"{total_rows:,}")
            self.summary_vars["start"].set(min(starts) if starts else "—")
            self.summary_vars["end"].set(max(ends) if ends else "—")
            self.progress.configure(value=100)
        self.frame.after(0, _finish)
        self._set_status(f"다운로드 완료: {n}개 종목 ✅  (현재가 포함)")
        now = datetime.now().strftime("%H:%M:%S")
        self.frame.after(0, lambda: self._rt_status_var.set(f"최종: {now}"))

        # 다음 단계 안내 배너 표시
        self.frame.after(0, lambda: self._next_step_bar.configure(height=32))

        # 다운로드 완료 → 메인 윈도우에 알림 (신선도 알람 즉시 재검사용)
        self.frame.after(500, lambda: self.on_change(self.settings))

        # 학습/예측 탭 업데이트 알림 전용 콜백
        if self._on_data_downloaded:
            self.frame.after(600, self._on_data_downloaded)

    def _update_port_status(self, ticker: str, status: str):
        tag = "ok" if "✅" in status else "error"
        def _do():
            if self.port_tree.exists(ticker):
                vals = list(self.port_tree.item(ticker, "values"))
                vals[2] = status
                self.port_tree.item(ticker, values=vals, tags=(tag,))
        self.frame.after(0, _do)

    # ─────────────────────────────────────────────
    # 실시간 시세 모니터 UI
    # ─────────────────────────────────────────────

    def _build_realtime(self, parent):
        """실시간 시세 모니터 섹션 구성"""
        rt_fr = ttk.LabelFrame(
            parent,
            text="📡  실시간 시세 모니터  (장중 자동 갱신)",
            padding=6,
        )
        rt_fr.pack(fill="both", expand=True, pady=(8, 0))

        # 상단 컨트롤 행
        ctrl = ttk.Frame(rt_fr)
        ctrl.pack(fill="x", pady=(0, 4))

        ttk.Label(ctrl, text="갱신 주기:").pack(side="left", padx=(0, 4))
        self._rt_interval_var = tk.StringVar(value="5초")
        rt_cb = ttk.Combobox(
            ctrl, textvariable=self._rt_interval_var,
            values=list(RT_INTERVALS), width=6, state="readonly",
        )
        rt_cb.pack(side="left", padx=(0, 8))
        add_tooltip(rt_cb, "실시간 시세를 몇 초/분마다 갱신할지 설정합니다.\n\n"
                    "3초 / 5초: 빠른 갱신 (yfinance API 요청 빈번)\n"
                    "10초 / 15초: 균형 잡힌 갱신 속도 (추천)\n"
                    "30초 이상: 부하 최소화")

        self._rt_btn = ttk.Button(
            ctrl, text="▶ 시작",
            command=self._toggle_realtime,
            style="Accent.TButton",
            width=8,
        )
        self._rt_btn.pack(side="left", padx=(0, 8))
        add_tooltip(self._rt_btn, "실시간 시세 자동 갱신을 시작/중지합니다.")

        self._rt_status_var = tk.StringVar(value="중지됨")
        self._rt_status_lbl = tk.Label(
            ctrl, textvariable=self._rt_status_var,
            bg="#1e1e2e", fg="#9399b2",
            font=("맑은 고딕", 9),
        )
        self._rt_status_lbl.pack(side="left")

        # 실시간 시세 테이블
        rt_tree_fr = ttk.Frame(rt_fr)
        rt_tree_fr.pack(fill="both", expand=True)

        cols = ("종목코드", "종목명", "현재가", "등락", "등락률", "업데이트")
        self._rt_tree = ttk.Treeview(
            rt_tree_fr, columns=cols, show="headings", height=12,
        )
        widths = [110, 120, 100, 90, 80, 80]
        anchors = ["center", "w", "e", "e", "e", "center"]
        for c, w, a in zip(cols, widths, anchors):
            self._rt_tree.heading(c, text=c)
            self._rt_tree.column(c, width=w, anchor=a)

        self._rt_tree.tag_configure("up",      foreground="#f38ba8")  # 상승: 빨강
        self._rt_tree.tag_configure("down",    foreground="#89b4fa")  # 하락: 파랑
        self._rt_tree.tag_configure("flat",    foreground="#a6adc8")
        self._rt_tree.tag_configure("stale",   foreground="#6c7086")  # 오래된 데이터

        rt_vsb = ttk.Scrollbar(rt_tree_fr, orient="vertical",
                               command=self._rt_tree.yview)
        self._rt_tree.configure(yscrollcommand=rt_vsb.set)
        self._rt_tree.pack(side="left", fill="both", expand=True)
        rt_vsb.pack(side="right", fill="y")

        # 종목 목록 변경 시 실시간 테이블도 초기화
        self._rt_rows: dict[str, str] = {}   # symbol → iid
        self._init_rt_table()

    def _init_rt_table(self):
        """포트폴리오 종목으로 실시간 테이블 행 초기화"""
        self._rt_tree.delete(*self._rt_tree.get_children())
        self._rt_rows = {}
        for ticker, name in self._portfolio:
            iid = self._rt_tree.insert(
                "", "end",
                values=(ticker, name, "—", "—", "—", "—"),
                tags=("flat",),
            )
            self._rt_rows[ticker] = iid

    def _toggle_realtime(self):
        """실시간 갱신 시작/중지 토글"""
        if self._rt_running:
            self._stop_realtime()
        else:
            self._start_realtime()

    def _start_realtime(self):
        if self._rt_running:
            return
        if not self._portfolio:
            messagebox.showwarning("알림", "등록된 종목이 없습니다.\n먼저 종목을 추가하세요.",
                                   parent=self.frame)
            return

        self._init_rt_table()   # 현재 포트폴리오로 행 재구성
        self._rt_stop.clear()
        self._rt_running = True
        self._rt_btn.config(text="■ 중지")
        self._rt_status_lbl.config(fg="#a6e3a1")
        self._rt_status_var.set("갱신 중...")

        self._rt_thread = threading.Thread(
            target=self._realtime_loop, daemon=True,
        )
        self._rt_thread.start()

    def _stop_realtime(self):
        self._rt_stop.set()
        self._rt_running = False
        self._rt_btn.config(text="▶ 시작")
        self._rt_status_lbl.config(fg="#9399b2")
        self._rt_status_var.set("중지됨")

    def _realtime_loop(self):
        """백그라운드 스레드: 주기적으로 시세 조회 후 UI 업데이트"""
        while not self._rt_stop.is_set():
            symbols = [t for t, _ in self._portfolio]
            if not symbols:
                self._rt_stop.wait(timeout=5)
                continue

            results = self._rt_fetcher.fetch_all(symbols)
            self.frame.after(0, lambda r=results: self._update_rt_table(r))

            interval_sec = RT_INTERVALS.get(self._rt_interval_var.get(), 5)
            # 짧은 단위로 쪼개서 wait → 주기 변경 즉시 반영, 빠른 중지 가능
            elapsed = 0.0
            while elapsed < interval_sec and not self._rt_stop.is_set():
                self._rt_stop.wait(timeout=1.0)
                elapsed += 1.0

        # 스레드 종료 시 상태 초기화
        self.frame.after(0, lambda: self._rt_status_var.set("중지됨"))

    def _update_rt_table(self, results: dict):
        """시세 조회 결과로 실시간 테이블 갱신"""
        from datetime import datetime
        now_str = datetime.now().strftime("%H:%M:%S")

        success = 0
        for sym, snap in results.items():
            iid = self._rt_rows.get(sym)
            if iid is None:
                continue
            name = next((n for t, n in self._portfolio if t == sym), sym)

            if snap is None:
                self._rt_tree.item(iid, values=(
                    sym, name, "오류", "—", "—", now_str,
                ), tags=("stale",))
                continue

            price_str  = f"{snap.price:,.0f}"
            change_str = f"{snap.change:+,.0f}"
            pct_str    = f"{snap.change_pct:+.2%}"
            ts_str     = snap.fetched_str

            if snap.stale:
                tag = "stale"
            elif snap.change > 0:
                tag = "up"
            elif snap.change < 0:
                tag = "down"
            else:
                tag = "flat"

            self._rt_tree.item(iid, values=(
                sym, name, price_str, change_str, pct_str, ts_str,
            ), tags=(tag,))
            success += 1

        interval_label = self._rt_interval_var.get()
        self._rt_status_var.set(
            f"마지막 갱신: {now_str}  ({success}/{len(results)}개)  "
            f"— {interval_label}마다 갱신"
        )

    def _clear_cache(self):
        from data.cache import CacheManager
        cache_dir = os.path.join(BASE_DIR, self.settings.data.cache_dir)
        n = CacheManager(cache_dir).clear_all()
        messagebox.showinfo("캐시 삭제", f"{n}개 캐시 파일 삭제 완료")
        self._log(f"캐시 {n}개 삭제됨")

    # ─────────────────────────────────────────────
    # 유틸
    # ─────────────────────────────────────────────

    def _set_status(self, msg: str):
        self.frame.after(0, lambda: self.status_var.set(msg))

    def _log(self, msg: str):
        def _do():
            self.log_box.config(state="normal")
            self.log_box.insert("end", msg + "\n")
            self.log_box.see("end")
            self.log_box.config(state="disabled")
        self.frame.after(0, _do)

    # ─────────────────────────────────────────────
    # 실시간 현재가 조회 (독립 버튼 / 자동갱신)
    # ─────────────────────────────────────────────

    _RT_INTERVAL_MAP = {"15초": 15_000, "30초": 30_000, "1분": 60_000, "5분": 300_000}

    def _fetch_realtime(self):
        """현재가만 빠르게 조회 (OHLCV 다운로드 없음)"""
        if self._rt_fetching:
            return
        if not self._portfolio:
            self._rt_status_var.set("종목 없음")
            return
        self._rt_fetching = True
        self._rt_btn.config(state="disabled")
        self._rt_status_var.set("조회 중…")
        self._rt_thread = threading.Thread(target=self._realtime_thread, daemon=True)
        self._rt_thread.start()

    def _realtime_thread(self):
        try:
            cache_dir = os.path.join(BASE_DIR, self.settings.data.cache_dir)
            loader    = DataLoader(cache_dir)
            symbols   = [t for t, _ in self._portfolio]
            prices    = loader.get_realtime_prices(symbols)
            self.frame.after(0, lambda: self._apply_realtime(prices))
        except Exception as e:
            logger.error(f"실시간 현재가 조회 오류: {e}")
            self.frame.after(0, lambda: self._rt_status_var.set(f"오류: {e}"))
        finally:
            self._rt_fetching = False
            self.frame.after(0, lambda: self._rt_btn.config(state="normal"))

    def _apply_realtime(self, prices: dict):
        """조회 결과를 data_tree 행에 반영, 없는 종목은 신규 삽입"""
        for ticker, name in self._portfolio:
            rt = prices.get(ticker)
            if rt is None:
                continue
            cur   = rt.get("currency", "") or ""
            price = rt.get("price")
            chg   = rt.get("change", 0) or 0
            chgp  = rt.get("change_pct", 0) or 0
            sign  = "+" if chg >= 0 else ""

            if price is None:
                continue

            price_str = f"{price:,.0f}" if cur in ("KRW", "") else f"{price:,.2f}"
            chg_str   = (f"{sign}{chg:,.0f}" if cur in ("KRW", "")
                         else f"{sign}{chg:,.2f}")
            chgp_str  = f"{sign}{chgp:.2f}%"
            rt_tag    = "rt_up" if chg > 0 else ("rt_down" if chg < 0 else "ok")

            if self.data_tree.exists(ticker):
                # 기존 행: 현재가·전일비·등락률만 갱신
                vals = list(self.data_tree.item(ticker, "values"))
                vals[5] = price_str   # 현재가
                vals[6] = chg_str     # 전일비
                vals[7] = chgp_str    # 등락률
                self.data_tree.item(ticker, values=vals, tags=(rt_tag,))
            else:
                # 다운로드 전에 현재가만 먼저 조회한 경우 → 새 행 삽입
                info   = get_info(ticker) or {}
                market = info.get("market", "—")
                self.data_tree.insert("", "end", iid=ticker, tags=(rt_tag,),
                    values=(ticker, name, "—", "—", "—",
                            price_str, chg_str, chgp_str, market, "📡 실시간"))

        ok_cnt = len(prices)
        total  = len(self._portfolio)
        now    = datetime.now().strftime("%H:%M:%S")
        self._rt_status_var.set(f"최종: {now}  ({ok_cnt}/{total})")
        self._log(f"[현재가 조회] {ok_cnt}/{total}개 성공  ({now})")

        # 자동갱신 재스케줄
        if self._rt_auto_on.get():
            self._schedule_rt_next()

    def _toggle_rt_auto(self):
        if self._rt_auto_on.get():
            self._fetch_realtime()
        else:
            self._cancel_rt_schedule()

    def _schedule_rt_next(self):
        self._cancel_rt_schedule()
        ms = self._RT_INTERVAL_MAP.get(self._rt_interval.get(), 30_000)
        self._rt_after_id = self.frame.after(ms, self._rt_auto_tick)

    def _rt_auto_tick(self):
        if self._rt_auto_on.get():
            self._fetch_realtime()

    def _cancel_rt_schedule(self):
        if self._rt_after_id:
            self.frame.after_cancel(self._rt_after_id)
            self._rt_after_id = None
