# gui/stock_search_dialog.py — 상용 수준 종목 검색 다이얼로그
from __future__ import annotations
import os, sys, threading, tkinter as tk
from tkinter import ttk, messagebox
from typing import List, Dict, Optional, Callable

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)

from data.korean_stocks import (
    search, validate_ticker_yfinance, MARKETS, get_info,
    load_krx_async, is_krx_loaded, get_krx_count, refresh_krx,
)


class StockSearchDialog(tk.Toplevel):
    """
    상용 수준 종목 검색 다이얼로그
    - KRX 전체 종목 (KOSPI/KOSDAQ 2,800개+) 검색
    - 실시간 텍스트 필터 / 시장 필터
    - yfinance 실시간 유효성 검사 (선택 시)
    - 직접 입력 → 자동 .KS/.KQ 시도
    - on_add(tickers: list) 콜백
    """

    COL_DEF = {
        "ticker":     ("종목코드",   110, "center"),
        "name":       ("종목명",     180, "w"),
        "market":     ("시장",        72, "center"),
        "sector":     ("섹터",        80, "center"),
        "price":      ("현재가",      88, "e"),
        "change_pct": ("등락률",      70, "e"),
        "market_cap": ("시가총액",    88, "e"),
        "status":     ("상태",        52, "center"),
    }

    def __init__(self, parent, on_add: Callable[[List[str]], None],
                 title: str = "종목 검색"):
        super().__init__(parent)
        self.on_add    = on_add
        self._alive    = True          # 위젯 생존 플래그
        self._search_job = None

        self.title(f"  {title}")
        self.geometry("1080x720")
        self.configure(bg="#1e1e2e")
        self.resizable(True, True)
        self.transient(parent)
        self.grab_set()
        self.protocol("WM_DELETE_WINDOW", self._on_close)

        self._selected_tickers: List[str] = []
        self._validate_cache: Dict[str, Dict] = {}
        self._validate_lock = threading.Lock()

        self._build()
        self._refresh_db_status()
        self._run_search("")

        if not is_krx_loaded():
            self._set_db_status("KRX 종목 로딩 중... (백그라운드)")
            load_krx_async(self._on_krx_loaded)

        self.after(100, lambda: self.search_entry.focus_set())

    # ─────────────────────────────────────────────
    # 종료
    # ─────────────────────────────────────────────

    def _on_close(self):
        self._alive = False
        self.grab_release()
        self.destroy()

    # ─────────────────────────────────────────────
    # KRX 로딩 콜백
    # ─────────────────────────────────────────────

    def _on_krx_loaded(self, count: int):
        if not self._alive:
            return
        try:
            self.after(0, lambda c=count: self._after_krx_loaded(c))
        except Exception:
            pass

    def _after_krx_loaded(self, count: int):
        if not self._alive:
            return
        try:
            self._refresh_db_status()
            q = self.search_var.get().strip()
            m = self.market_var.get()
            self._run_search(q, m)
        except Exception:
            pass

    def _set_db_status(self, msg: str):
        try:
            if self._alive:
                self.db_status_var.set(msg)
        except Exception:
            pass

    def _refresh_db_status(self):
        if not self._alive:
            return
        try:
            n = get_krx_count()
            if n > 0:
                self.db_status_var.set(f"  KRX {n:,}개 종목 로드됨  (KOSPI / KOSDAQ / KONEX 전체 포함)")
            elif is_krx_loaded():
                self.db_status_var.set("  pykrx 미설치 — 정적 목록 사용 중  (pip install pykrx)")
            else:
                self.db_status_var.set("  KRX 종목 로딩 중...")
        except Exception:
            pass

    # ─────────────────────────────────────────────
    # UI 구성
    # Tkinter 패킹 규칙:
    #   side="bottom" 위젯을 먼저 pack → 항상 하단 표시
    #   side="top"    위젯을 나중 pack → 남은 공간 사용
    # ─────────────────────────────────────────────

    def _build(self):
        # ══ ① 하단 버튼바 (side="bottom", 가장 먼저 pack → 항상 표시) ══
        self._build_bottom()

        # ══ ② 상단 검색 바 (side="top") ══
        top = tk.Frame(self, bg="#11111b", pady=8)
        top.pack(fill="x", side="top")

        tk.Label(top, text="  종목 검색:", font=("맑은 고딕", 11, "bold"),
                 bg="#11111b", fg="#cba6f7").pack(side="left")

        self.search_var = tk.StringVar()
        self.search_var.trace_add("write", lambda *_: self._on_search_change())
        self.search_entry = ttk.Entry(top, textvariable=self.search_var,
                                      font=("맑은 고딕", 12), width=28)
        self.search_entry.pack(side="left", padx=6)

        tk.Label(top, text="시장:", bg="#11111b", fg="#a6adc8").pack(
            side="left", padx=(12, 4))
        self.market_var = tk.StringVar(value="전체")
        self.market_var.trace_add("write", lambda *_: self._on_search_change())
        ttk.Combobox(top, textvariable=self.market_var,
                     values=MARKETS, width=10, state="readonly",
                     font=("맑은 고딕", 10)).pack(side="left")

        ttk.Button(top, text="  새로고침",
                   command=self._on_refresh).pack(side="left", padx=(10, 0))

        self.count_var = tk.StringVar(value="")
        tk.Label(top, textvariable=self.count_var,
                 bg="#11111b", fg="#9399b2",
                 font=("맑은 고딕", 9)).pack(side="left", padx=12)

        # ══ ③ DB 상태바 (side="top") ══
        self.db_status_var = tk.StringVar(value="")
        sb = tk.Frame(self, bg="#181825")
        sb.pack(fill="x", side="top")
        tk.Label(sb, textvariable=self.db_status_var,
                 bg="#181825", fg="#89b4fa",
                 font=("맑은 고딕", 9), anchor="w").pack(
            fill="x", padx=0, pady=2)

        # ══ ④ 메인 영역 (side="top", expand=True → 남은 공간 채움) ══
        main = tk.PanedWindow(self, orient="horizontal",
                              bg="#1e1e2e", sashwidth=4)
        main.pack(fill="both", expand=True, padx=6, pady=4, side="top")

        left  = ttk.LabelFrame(main, text="검색 결과", padding=4)
        right = ttk.LabelFrame(main, text="추가할 종목", padding=4)
        main.add(left,  minsize=580)
        main.add(right, minsize=280)

        self._build_result_table(left)
        self._build_selected_panel(right)

    def _build_bottom(self):
        """
        하단 바 — side="bottom" 으로 pack → 항상 화면 최하단에 표시
        """
        # 구분선
        sep = tk.Frame(self, bg="#313244", height=1)
        sep.pack(fill="x", side="bottom")

        bot = tk.Frame(self, bg="#11111b", pady=8)
        bot.pack(fill="x", side="bottom")

        # ── 왼쪽: 직접 입력 ──────────────────────
        left_fr = tk.Frame(bot, bg="#11111b")
        left_fr.pack(side="left", padx=10)

        tk.Label(left_fr, text="직접 입력 (종목코드/티커):",
                 bg="#11111b", fg="#a6adc8",
                 font=("맑은 고딕", 9)).pack(anchor="w")

        ef = tk.Frame(left_fr, bg="#11111b")
        ef.pack()
        self.direct_var = tk.StringVar()
        de = ttk.Entry(ef, textvariable=self.direct_var,
                       font=("맑은 고딕", 11), width=16)
        de.pack(side="left", padx=(0, 4))
        de.bind("<Return>", lambda e: self._direct_add())
        ttk.Button(ef, text="검증 후 추가",
                   command=self._direct_add).pack(side="left")

        # ── 중앙: 상태 메시지 ────────────────────
        self.status_var = tk.StringVar(value="")
        tk.Label(bot, textvariable=self.status_var,
                 bg="#11111b", fg="#89b4fa",
                 font=("맑은 고딕", 9), wraplength=280).pack(
            side="left", padx=16)

        # ── 오른쪽: 버튼 그룹 ───────────────────
        right_fr = tk.Frame(bot, bg="#11111b")
        right_fr.pack(side="right", padx=10)

        ttk.Button(right_fr, text="✕ 취소",
                   command=self._on_close).pack(side="right", padx=(6, 0))

        self.confirm_btn = ttk.Button(
            right_fr,
            text="✅ 포트폴리오에 추가",
            command=self._confirm_add,
            style="Accent.TButton",
        )
        self.confirm_btn.pack(side="right", padx=4)

        # ── 핵심: 검색 결과 → 추가목록 이동 버튼 ─
        tk.Button(
            right_fr,
            text="  ➕ 선택 추가  →  ",
            command=self._add_selected,
            bg="#89b4fa", fg="#1e1e2e",
            font=("맑은 고딕", 10, "bold"),
            relief="flat", padx=8, pady=5,
            cursor="hand2", bd=0,
        ).pack(side="right", padx=4)

    def _build_result_table(self, parent):
        cols = list(self.COL_DEF.keys())
        self.tree = ttk.Treeview(parent, columns=cols,
                                 show="headings", selectmode="extended",
                                 height=24)
        for key, (label, width, anchor) in self.COL_DEF.items():
            self.tree.heading(key, text=label,
                              command=lambda k=key: self._sort_by(k))
            self.tree.column(key, width=width, anchor=anchor, minwidth=40)

        self.tree.tag_configure("pos",      foreground="#a6e3a1")
        self.tree.tag_configure("neg",      foreground="#f38ba8")
        self.tree.tag_configure("invalid",  foreground="#f38ba8")
        self.tree.tag_configure("checking", foreground="#f9e2af")
        self.tree.tag_configure("added",    background="#313244")

        vsb = ttk.Scrollbar(parent, orient="vertical",   command=self.tree.yview)
        hsb = ttk.Scrollbar(parent, orient="horizontal", command=self.tree.xview)
        self.tree.configure(yscrollcommand=vsb.set, xscrollcommand=hsb.set)
        self.tree.pack(side="left", fill="both", expand=True)
        vsb.pack(side="right",  fill="y")
        hsb.pack(side="bottom", fill="x")

        self.tree.bind("<Double-1>",         self._on_double_click)
        self.tree.bind("<<TreeviewSelect>>", self._on_select)
        self.tree.bind("<Return>",           lambda e: self._add_selected())

    def _build_selected_panel(self, parent):
        tk.Label(parent,
                 text="더블클릭 또는 [➕ 선택 추가 →] 버튼으로 담기",
                 bg="#1e1e2e", fg="#9399b2",
                 font=("맑은 고딕", 9)).pack(anchor="w")

        self.sel_listbox = tk.Listbox(
            parent, selectmode="extended",
            bg="#181825", fg="#cdd6f4",
            selectbackground="#89b4fa", selectforeground="#1e1e2e",
            font=("맑은 고딕", 10), relief="flat",
        )
        self.sel_listbox.pack(fill="both", expand=True)

        bf = ttk.Frame(parent)
        bf.pack(fill="x", pady=(4, 0))
        ttk.Button(bf, text="▲", width=3, command=self._move_up).pack(side="left", padx=2)
        ttk.Button(bf, text="▼", width=3, command=self._move_down).pack(side="left", padx=2)
        ttk.Button(bf, text="✕ 제거",
                   command=self._remove_selected_from_list,
                   style="Danger.TButton").pack(side="right", padx=2)

        self.val_var = tk.StringVar(value="")
        tk.Label(parent, textvariable=self.val_var,
                 bg="#1e1e2e", fg="#f9e2af",
                 font=("맑은 고딕", 9), wraplength=268, justify="left").pack(
            fill="x", pady=(4, 0))

    # ─────────────────────────────────────────────
    # 검색 / 필터
    # ─────────────────────────────────────────────

    def _on_search_change(self):
        if self._search_job:
            self.after_cancel(self._search_job)
        self._search_job = self.after(150, self._do_search)

    def _do_search(self):
        self._search_job = None
        if not self._alive:
            return
        self._run_search(self.search_var.get().strip(),
                         self.market_var.get())

    def _run_search(self, query: str = "", market: str = "전체"):
        if not self._alive:
            return
        results = search(query, market)

        MAX_DISPLAY = 500
        total = len(results)
        if total > MAX_DISPLAY and not query:
            display = results[:MAX_DISPLAY]
            self.count_var.set(
                f"검색 결과: {total:,}개  (상위 {MAX_DISPLAY}개 표시 — 검색어를 입력하세요)")
        else:
            display = results
            self.count_var.set(f"검색 결과: {total:,}개")

        self._populate_tree(display)

    def _populate_tree(self, rows: List[Dict]):
        if not self._alive:
            return
        try:
            children = self.tree.get_children()
            if children:
                self.tree.delete(*children)
        except Exception:
            return

        added_set = set(self._selected_tickers)
        for r in rows:
            ticker = r["ticker"]
            cached = self._validate_cache.get(ticker)

            price = f"{cached['price']:,.0f}" if cached and cached.get("valid") else "—"
            chg   = f"{cached['change_pct']:+.2f}%" if cached and cached.get("valid") else "—"
            mcap  = cached.get("market_cap", "—") if cached and cached.get("valid") else "—"
            st    = "✅" if (cached and cached.get("valid")) else \
                    "❌" if (cached and cached.get("valid") is False) else "—"

            tags = []
            if cached and cached.get("valid"):
                tags.append("pos" if cached.get("change_pct", 0) >= 0 else "neg")
            elif cached and cached.get("valid") is False:
                tags.append("invalid")
            if ticker in added_set:
                tags.append("added")

            try:
                self.tree.insert("", "end", iid=ticker, values=(
                    ticker, r["name"], r["market"], r.get("sector", ""),
                    price, chg, mcap, st,
                ), tags=tuple(tags))
            except Exception:
                pass

    def _on_refresh(self):
        self._set_db_status("KRX 종목 강제 갱신 중...")
        refresh_krx(self._on_krx_loaded)

    # ─────────────────────────────────────────────
    # 유효성 검사 (비동기)
    # ─────────────────────────────────────────────

    def _on_select(self, event=None):
        if not self._alive:
            return
        try:
            sel = self.tree.selection()
        except Exception:
            return
        if not sel:
            return
        ticker = sel[-1]
        if ticker not in self._validate_cache:
            self._async_validate(ticker)

    def _async_validate(self, ticker: str):
        with self._validate_lock:
            if ticker in self._validate_cache:
                return
            self._validate_cache[ticker] = {"valid": None, "checking": True}

        self._update_tree_row(ticker, checking=True)
        try:
            self.val_var.set(f"검증 중: {ticker} ...")
        except Exception:
            pass

        def _do():
            result = validate_ticker_yfinance(ticker, timeout=8.0)
            with self._validate_lock:
                self._validate_cache[ticker] = result
            if self._alive:
                try:
                    self.after(0, lambda: self._on_validate_done(ticker, result))
                except Exception:
                    pass

        threading.Thread(target=_do, daemon=True).start()

    def _on_validate_done(self, ticker: str, result: Dict):
        if not self._alive:
            return
        self._update_tree_row(ticker)
        try:
            if result.get("valid"):
                self.val_var.set(
                    f"✅ {ticker}: {result['name']} | "
                    f"{result['price']:,.0f} ({result['change_pct']:+.2f}%) | "
                    f"시총 {result['market_cap']}"
                )
            else:
                self.val_var.set(f"❌ {ticker}: {result.get('error', '유효하지 않은 종목')}")
        except Exception:
            pass

    def _update_tree_row(self, ticker: str, checking: bool = False):
        if not self._alive:
            return
        try:
            if not self.tree.exists(ticker):
                return
        except Exception:
            return

        cached = self._validate_cache.get(ticker, {})
        if checking:
            vals = list(self.tree.item(ticker, "values"))
            vals[7] = "⏳"
            self.tree.item(ticker, values=vals, tags=("checking",))
            return

        price = f"{cached['price']:,.0f}" if cached.get("valid") else "—"
        chg   = f"{cached['change_pct']:+.2f}%" if cached.get("valid") else "—"
        mcap  = cached.get("market_cap", "—") if cached.get("valid") else "—"
        st    = "✅" if cached.get("valid") else \
                ("❌" if cached.get("valid") is False else "—")

        tags = []
        if cached.get("valid"):
            tags.append("pos" if cached.get("change_pct", 0) >= 0 else "neg")
        elif cached.get("valid") is False:
            tags.append("invalid")
        if ticker in self._selected_tickers:
            tags.append("added")

        vals = list(self.tree.item(ticker, "values"))
        vals[4], vals[5], vals[6], vals[7] = price, chg, mcap, st
        self.tree.item(ticker, values=vals, tags=tuple(tags))

    # ─────────────────────────────────────────────
    # 종목 추가 / 제거 / 정렬
    # ─────────────────────────────────────────────

    def _add_selected(self):
        """검색 결과에서 선택 → 오른쪽 추가목록으로 이동"""
        if not self._alive:
            return
        try:
            sel = self.tree.selection()
        except Exception:
            return

        if not sel:
            self._set_status("먼저 검색 결과에서 종목을 클릭하여 선택하세요")
            return

        added = 0
        for ticker in sel:
            if ticker in self._selected_tickers:
                continue
            info = get_info(ticker)
            if info:
                self._selected_tickers.append(ticker)
                self.sel_listbox.insert("end", f"{ticker}  {info['name']}")
                added += 1
                try:
                    if self.tree.exists(ticker):
                        tags = list(self.tree.item(ticker, "tags"))
                        if "added" not in tags:
                            tags.append("added")
                        self.tree.item(ticker, tags=tuple(tags))
                except Exception:
                    pass
                if ticker not in self._validate_cache:
                    self._async_validate(ticker)
            else:
                self._async_validate(ticker)
                self._pending_add(ticker)

        if added:
            self._set_status(
                f"✅ {added}개가 추가 목록에 담겼습니다  →  [✅ 포트폴리오에 추가] 클릭으로 확정")

    def _pending_add(self, ticker: str):
        def _wait():
            for _ in range(20):
                with self._validate_lock:
                    c = self._validate_cache.get(ticker, {})
                if c.get("valid") is not None:
                    break
                import time; time.sleep(0.5)
            c = self._validate_cache.get(ticker, {})
            if self._alive:
                try:
                    self.after(0, lambda: self._finish_pending_add(ticker, c))
                except Exception:
                    pass
        threading.Thread(target=_wait, daemon=True).start()

    def _finish_pending_add(self, ticker: str, result: Dict):
        if not self._alive:
            return
        if result.get("valid"):
            if ticker not in self._selected_tickers:
                self._selected_tickers.append(ticker)
                name = result.get("name", ticker)
                self.sel_listbox.insert("end", f"{ticker}  {name}")
                self._set_status(f"✅ {ticker} ({name}) 추가됨")
        else:
            err = result.get("error", "유효하지 않은 종목")
            self._set_status(f"❌ {ticker}: {err}")
            messagebox.showerror(
                "종목 오류",
                f"'{ticker}'을(를) 추가할 수 없습니다.\n\n사유: {err}\n\n"
                "올바른 종목코드:\n"
                "• 064350.KS (현대로템),  365340.KQ (성일하이텍)\n"
                "• AAPL, MSFT, NVDA",
                parent=self,
            )

    def _direct_add(self):
        raw = self.direct_var.get().strip().upper()
        if not raw:
            return
        tickers = self._resolve_ticker(raw)
        self.direct_var.set("")
        self._set_status(f"'{raw}' 검증 중...")

        def _try(candidates):
            for t in candidates:
                result = validate_ticker_yfinance(t, timeout=8.0)
                with self._validate_lock:
                    self._validate_cache[t] = result
                if result.get("valid"):
                    if self._alive:
                        try:
                            self.after(0, lambda _t=t, r=result:
                                       self._finish_pending_add(_t, r))
                        except Exception:
                            pass
                    return
            if self._alive:
                try:
                    self.after(0, lambda: self._direct_add_failed(raw))
                except Exception:
                    pass

        threading.Thread(target=_try, args=(tickers,), daemon=True).start()

    def _direct_add_failed(self, raw: str):
        if not self._alive:
            return
        self._set_status(f"❌ '{raw}' 유효하지 않음")
        messagebox.showerror(
            "종목 오류",
            f"'{raw}'은(는) 유효하지 않은 종목코드입니다.\n\n"
            "올바른 입력 형식:\n"
            "• 국내 KOSPI  : 064350.KS  (현대로템)\n"
            "• 국내 KOSDAQ : 365340.KQ  (성일하이텍)\n"
            "• 숫자 6자리만 입력 시 .KS / .KQ 자동 시도\n"
            "• 미국 : AAPL, MSFT, NVDA",
            parent=self,
        )

    @staticmethod
    def _resolve_ticker(raw: str) -> List[str]:
        if any(raw.endswith(s) for s in (".KS", ".KQ", ".KP")):
            return [raw]
        if raw.isdigit() and len(raw) == 6:
            return [f"{raw}.KS", f"{raw}.KQ"]
        return [raw]

    def _remove_selected_from_list(self):
        sel = list(self.sel_listbox.curselection())
        for i in reversed(sel):
            ticker = self._selected_tickers[i]
            self._selected_tickers.pop(i)
            self.sel_listbox.delete(i)
            try:
                if self.tree.exists(ticker):
                    tags = [t for t in self.tree.item(ticker, "tags") if t != "added"]
                    self.tree.item(ticker, tags=tuple(tags))
            except Exception:
                pass

    def _move_up(self):
        sel = self.sel_listbox.curselection()
        if not sel or sel[0] == 0:
            return
        i = sel[0]
        self._selected_tickers[i-1], self._selected_tickers[i] = \
            self._selected_tickers[i], self._selected_tickers[i-1]
        item = self.sel_listbox.get(i)
        self.sel_listbox.delete(i)
        self.sel_listbox.insert(i-1, item)
        self.sel_listbox.selection_set(i-1)

    def _move_down(self):
        sel = self.sel_listbox.curselection()
        if not sel or sel[0] >= self.sel_listbox.size() - 1:
            return
        i = sel[0]
        self._selected_tickers[i], self._selected_tickers[i+1] = \
            self._selected_tickers[i+1], self._selected_tickers[i]
        item = self.sel_listbox.get(i)
        self.sel_listbox.delete(i)
        self.sel_listbox.insert(i+1, item)
        self.sel_listbox.selection_set(i+1)

    def _on_double_click(self, event):
        self._add_selected()

    def _sort_by(self, col: str):
        try:
            rows = [(self.tree.set(iid, col), iid)
                    for iid in self.tree.get_children("")]
            rows.sort(key=lambda x: x[0])
            for rank, (_, iid) in enumerate(rows):
                self.tree.move(iid, "", rank)
        except Exception:
            pass

    def _confirm_add(self):
        if not self._selected_tickers:
            messagebox.showwarning(
                "알림",
                "추가할 종목이 없습니다.\n\n"
                "사용 방법:\n"
                "① 왼쪽 검색 결과에서 종목 클릭 (선택)\n"
                "② 하단 [➕ 선택 추가 →] 버튼 클릭\n"
                "③ 오른쪽 목록에 담긴 것 확인\n"
                "④ [✅ 포트폴리오에 추가] 클릭으로 최종 확정",
                parent=self,
            )
            return
        self.on_add(list(self._selected_tickers))
        self._on_close()

    def _set_status(self, msg: str):
        try:
            if self._alive:
                self.status_var.set(msg)
        except Exception:
            pass
