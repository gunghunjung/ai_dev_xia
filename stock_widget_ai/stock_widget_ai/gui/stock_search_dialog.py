"""
StockSearchDialog — 종목 검색 다이얼로그
────────────────────────────────────────
검색 우선순위:
  1순위 : 로컬 KRX DB (한글 종목명 / 초성 / 티커코드) → 즉시 결과
  2순위 : Yahoo Finance API → 미국주식 / DB에 없는 종목 보충

사용 예시:
  "현대로템"  → 064350.KS  (로컬 DB 즉시 검색)
  "삼성"      → 삼성전자·삼성SDI·삼성화재 등
  "ㅎㄷㄹㅌ"  → 현대로템 (초성 검색)
  "AAPL"      → Apple Inc. (Yahoo Finance)
"""
from __future__ import annotations

import json
import urllib.request
import urllib.parse
from typing import Optional, List, Dict, Any

from PyQt6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout,
    QLineEdit, QPushButton,
    QTableWidget, QTableWidgetItem, QHeaderView,
    QLabel, QDialogButtonBox, QAbstractItemView,
    QTabWidget, QWidget, QScrollArea, QGridLayout,
)
from PyQt6.QtCore import QTimer, QThread, pyqtSignal

# ── 로컬 KRX DB ──────────────────────────────────────────────────────────
try:
    from ..data.korean_stocks_db import search_krx, KRX_STOCKS
    _HAS_KRX = True
except Exception:
    _HAS_KRX = False
    KRX_STOCKS = []
    def search_krx(query, top_n=30): return []

# ── 미국 인기 종목 ─────────────────────────────────────────────────────
_US_POPULAR = [
    ("AAPL","Apple"), ("MSFT","Microsoft"), ("NVDA","NVIDIA"),
    ("GOOGL","Alphabet"), ("AMZN","Amazon"), ("META","Meta"),
    ("TSLA","Tesla"), ("AVGO","Broadcom"), ("ORCL","Oracle"),
    ("AMD","AMD"), ("INTC","Intel"), ("QCOM","Qualcomm"),
    ("^GSPC","S&P 500"), ("^IXIC","NASDAQ"), ("^DJI","다우존스"),
]


# ── 백그라운드 검색 워커 ─────────────────────────────────────────────────
class _SearchWorker(QThread):
    results_ready = pyqtSignal(list)
    error         = pyqtSignal(str)

    def __init__(self, query: str) -> None:
        super().__init__()
        self._query     = query.strip()
        self._cancelled = False

    def run(self) -> None:
        combined: List[Dict] = []
        q = self._query

        # 1순위: 로컬 KRX DB (즉시, 네트워크 불필요)
        local = search_krx(q, top_n=30)
        combined.extend(local)

        # 2순위: Yahoo Finance API (미국주식 등 보충)
        existing = {r["symbol"] for r in combined}
        try:
            for r in self._yahoo(q):
                if r.get("symbol") not in existing:
                    combined.append(r)
                    existing.add(r.get("symbol", ""))
        except Exception:
            pass

        if not self._cancelled:
            self.results_ready.emit(combined)

    def _yahoo(self, q: str) -> List[Dict]:
        enc = urllib.parse.quote(q)
        hdr = {"User-Agent": "Mozilla/5.0"}
        for url in [
            f"https://query1.finance.yahoo.com/v1/finance/search?q={enc}&lang=ko-KR&region=KR&quotesCount=20&newsCount=0&enableFuzzyQuery=true",
            f"https://query1.finance.yahoo.com/v1/finance/search?q={enc}&lang=en-US&region=US&quotesCount=20&newsCount=0&enableFuzzyQuery=true",
        ]:
            if self._cancelled:
                break
            try:
                req = urllib.request.Request(url, headers=hdr)
                with urllib.request.urlopen(req, timeout=5) as r:
                    data = json.loads(r.read().decode())
                q_list = data.get("quotes", [])
                if q_list:
                    return q_list
            except Exception:
                continue
        return []

    def cancel(self) -> None:
        self._cancelled = True


# ════════════════════════════════════════════════════════════════════════
class StockSearchDialog(QDialog):
    """
    종목 검색 다이얼로그.
    dlg.selected_ticker 에 선택된 티커코드 저장.
    """

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setWindowTitle("종목 검색")
        self.setMinimumSize(740, 560)
        self.resize(780, 580)
        self.selected_ticker: Optional[str] = None
        self._worker:  Optional[_SearchWorker] = None
        self._debounce = QTimer(self)
        self._debounce.setSingleShot(True)
        self._debounce.timeout.connect(self._do_search)
        self._build()
        self._style()

    # ── UI ───────────────────────────────────────────────────────────────
    def _build(self) -> None:
        root = QVBoxLayout(self)
        root.setSpacing(8)

        hint = QLabel(
            "한글 종목명 · 초성 · 티커코드 모두 검색 가능\n"
            "예)  현대로템  /  삼성  /  ㅎㄷㄹㅌ  /  AAPL  /  064350"
        )
        hint.setStyleSheet("color:#8b949e; font-size:12px;")
        root.addWidget(hint)

        row = QHBoxLayout()
        self._input = QLineEdit()
        self._input.setPlaceholderText("종목명 · 초성 · 코드 입력 …")
        self._input.setMinimumHeight(38)
        self._input.textChanged.connect(self._on_text_changed)
        self._input.returnPressed.connect(self._do_search)
        btn = QPushButton("🔍 검색")
        btn.setFixedWidth(90)
        btn.setFixedHeight(38)
        btn.clicked.connect(self._do_search)
        row.addWidget(self._input)
        row.addWidget(btn)
        root.addLayout(row)

        tabs = QTabWidget()
        tabs.addTab(self._tab_search(), "🔍 검색 결과")
        tabs.addTab(self._tab_grid([{"s": s, "n": n} for s, n, _ in KRX_STOCKS[:60]], "kr"), "🇰🇷 국내 인기")
        tabs.addTab(self._tab_grid([{"s": s, "n": n} for s, n in _US_POPULAR], "us"),         "🇺🇸 미국 인기")
        self._tabs = tabs
        root.addWidget(tabs)

        self._status = QLabel("종목명을 입력하면 자동으로 검색됩니다")
        self._status.setStyleSheet("color:#8b949e; font-size:11px;")
        root.addWidget(self._status)

        box = QDialogButtonBox()
        ok  = box.addButton("✅ 선택", QDialogButtonBox.ButtonRole.AcceptRole)
        no  = box.addButton("취소",   QDialogButtonBox.ButtonRole.RejectRole)
        ok.clicked.connect(self._on_accept)
        no.clicked.connect(self.reject)
        root.addWidget(box)

    def _tab_search(self) -> QWidget:
        w = QWidget()
        v = QVBoxLayout(w)
        v.setContentsMargins(0, 4, 0, 0)
        self._table = QTableWidget(0, 4)
        self._table.setHorizontalHeaderLabels(["티커", "종목명", "시장", "유형"])
        h = self._table.horizontalHeader()
        h.setSectionResizeMode(0, QHeaderView.ResizeMode.ResizeToContents)
        h.setSectionResizeMode(1, QHeaderView.ResizeMode.Stretch)
        h.setSectionResizeMode(2, QHeaderView.ResizeMode.ResizeToContents)
        h.setSectionResizeMode(3, QHeaderView.ResizeMode.ResizeToContents)
        self._table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self._table.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self._table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self._table.setAlternatingRowColors(True)
        self._table.verticalHeader().setVisible(False)
        self._table.doubleClicked.connect(self._on_accept)
        v.addWidget(self._table)
        return w

    def _tab_grid(self, items: List[Dict], tag: str) -> QWidget:
        outer = QWidget()
        ov    = QVBoxLayout(outer)
        sc    = QScrollArea()
        sc.setWidgetResizable(True)
        inner = QWidget()
        g     = QGridLayout(inner)
        g.setSpacing(6)
        for i, item in enumerate(items):
            s = item.get("s") or item.get("symbol", "")
            n = item.get("n") or item.get("name",   "")
            b = QPushButton(f"{n}\n{s}")
            b.setFixedHeight(52)
            b.setToolTip(f"{n}  ({s})")
            b.clicked.connect(lambda _=False, sym=s: self._quick(sym))
            g.addWidget(b, i // 3, i % 3)
        sc.setWidget(inner)
        ov.addWidget(sc)
        return outer

    # ── 검색 ─────────────────────────────────────────────────────────────
    def _on_text_changed(self, text: str) -> None:
        if text.strip():
            self._debounce.start(350)

    def _do_search(self) -> None:
        q = self._input.text().strip()
        if not q:
            return
        if self._worker and self._worker.isRunning():
            self._worker.cancel()
            self._worker.wait(200)
        self._status.setText("🔍 검색 중…")
        self._table.setRowCount(0)
        self._tabs.setCurrentIndex(0)
        self._worker = _SearchWorker(q)
        self._worker.results_ready.connect(self._show_results)
        self._worker.error.connect(lambda e: self._status.setText(f"오류: {e}"))
        self._worker.start()

    _WANTED = {"EQUITY","ETF","MUTUALFUND","INDEX","CURRENCY","FUTURE"}
    _TKO    = {"EQUITY":"주식","ETF":"ETF","MUTUALFUND":"펀드",
               "INDEX":"지수","CURRENCY":"환율","FUTURE":"선물"}

    def _show_results(self, quotes: List[Dict]) -> None:
        rows = [q for q in quotes if q.get("quoteType","EQUITY") in self._WANTED] or quotes
        self._table.setRowCount(len(rows))
        for i, q in enumerate(rows):
            self._table.setItem(i, 0, QTableWidgetItem(q.get("symbol","")))
            self._table.setItem(i, 1, QTableWidgetItem(
                q.get("shortname") or q.get("name") or q.get("longname","")
            ))
            self._table.setItem(i, 2, QTableWidgetItem(q.get("exchange","")))
            self._table.setItem(i, 3, QTableWidgetItem(
                self._TKO.get(q.get("quoteType",""),"주식")
            ))
        if rows:
            self._table.selectRow(0)
        self._status.setText(
            f"✅ {len(rows)}개 종목  (더블클릭 또는 선택 버튼)" if rows
            else "❌ 결과 없음 — 국내 인기 탭을 이용하세요"
        )

    # ── 선택 ─────────────────────────────────────────────────────────────
    def _on_accept(self) -> None:
        if self._tabs.currentIndex() == 0:
            row = self._table.currentRow()
            if row >= 0:
                it = self._table.item(row, 0)
                if it and it.text():
                    self.selected_ticker = it.text()
                    self.accept()
                    return
        t = self._input.text().strip().upper()
        if t:
            self.selected_ticker = t
            self.accept()

    def _quick(self, sym: str) -> None:
        self.selected_ticker = sym
        self.accept()

    def closeEvent(self, event) -> None:
        if self._worker and self._worker.isRunning():
            self._worker.cancel()
            self._worker.wait(400)
        super().closeEvent(event)

    # ── 스타일 ───────────────────────────────────────────────────────────
    def _style(self) -> None:
        self.setStyleSheet("""
            QDialog  { background:#0d1117; color:#c9d1d9; }
            QLabel   { color:#c9d1d9; }
            QLineEdit {
                background:#161b22; border:1px solid #30363d;
                border-radius:4px; color:#c9d1d9; padding:4px 10px; font-size:13px;
            }
            QLineEdit:focus { border-color:#58a6ff; }
            QPushButton {
                background:#21262d; border:1px solid #30363d;
                border-radius:4px; color:#c9d1d9; padding:4px 8px;
            }
            QPushButton:hover   { background:#30363d; }
            QPushButton:pressed { background:#1f6feb; color:#fff; }
            QTableWidget {
                background:#161b22; alternate-background-color:#1c2128;
                color:#c9d1d9; gridline-color:#30363d;
                selection-background-color:#1f6feb; selection-color:#fff;
            }
            QHeaderView::section {
                background:#21262d; color:#8b949e; border:none; padding:4px; font-weight:bold;
            }
            QTabWidget::pane   { border:1px solid #30363d; }
            QTabBar::tab       { background:#21262d; color:#8b949e; padding:5px 14px; border:1px solid #30363d; }
            QTabBar::tab:selected { background:#161b22; color:#c9d1d9; }
            QScrollArea { border:none; }
        """)
