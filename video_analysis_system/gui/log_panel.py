"""
gui/log_panel.py — 로그 & 이벤트 패널 (독립형)

panels.py 의 EventPanel 을 확장하여:
  - 이벤트 로그 테이블 (타임스탬프/유형/심각도/ROI ID/메시지)
  - 실시간 텍스트 로그 스트림 (하단 섹션)
  - CSV / JSON 내보내기 버튼
  - 필터링 (심각도, 이벤트 유형)

외부 인터페이스:
  panel.add_event(event_dict)     — 이벤트 추가
  panel.append_log(text)          — 텍스트 로그 한 줄 추가
  panel.clear()                   — 전체 초기화
"""

from __future__ import annotations

import csv
import json
import tkinter as tk
from datetime import datetime
from pathlib import Path
from tkinter import filedialog, messagebox, ttk
from typing import Dict, List, Optional

from gui.scrollable_frame import ScrollableFrame


_KO = ("맑은 고딕", 9)
_MONO = ("Consolas", 9)


class LogPanel(ttk.Frame):
    """이벤트 로그 + 텍스트 스트림 통합 패널."""

    MAX_LOG_LINES = 500

    def __init__(self, master, **kwargs):
        super().__init__(master, padding=6, **kwargs)
        self._events: List[Dict] = []
        self._log_lines: List[str] = []

        self._build_ui()

    # ── UI 구성 ───────────────────────────────────────────────────────────

    def _build_ui(self) -> None:
        # 타이틀 + 버튼 행
        top = ttk.Frame(self)
        top.pack(fill="x", pady=(0, 4))
        ttk.Label(top, text="⚠  이벤트 & 로그",
                  font=("맑은 고딕", 10, "bold")).pack(side="left")
        ttk.Button(top, text="CSV 저장",  command=self._export_csv ).pack(side="right", padx=(2, 0))
        ttk.Button(top, text="JSON 저장", command=self._export_json).pack(side="right", padx=(2, 0))
        ttk.Button(top, text="초기화",    command=self.clear       ).pack(side="right")

        # 필터 행
        filter_row = ttk.Frame(self)
        filter_row.pack(fill="x", pady=(0, 4))
        ttk.Label(filter_row, text="심각도 ≥").pack(side="left")
        self._sev_filter = tk.DoubleVar(value=0.0)
        ttk.Spinbox(filter_row, from_=0.0, to=1.0, increment=0.1,
                    textvariable=self._sev_filter, width=5,
                    format="%.1f", command=self._apply_filter).pack(side="left", padx=(2, 8))
        ttk.Label(filter_row, text="유형:").pack(side="left")
        self._type_filter = tk.StringVar(value="전체")
        self._type_combo = ttk.Combobox(filter_row, textvariable=self._type_filter,
                                        width=14, state="readonly",
                                        values=["전체", "state_change_abnormal",
                                                "manual", "ai_detection"])
        self._type_combo.pack(side="left", padx=(2, 0))
        self._type_combo.bind("<<ComboboxSelected>>", lambda _: self._apply_filter())

        # 이벤트 트리뷰
        tree_frame = ttk.LabelFrame(self, text="이벤트 목록", padding=4)
        tree_frame.pack(fill="both", expand=True, pady=(0, 4))

        cols = ("time", "type", "sev", "roi", "message")
        self._tree = ttk.Treeview(tree_frame, columns=cols, show="headings",
                                  selectmode="browse", height=8)
        self._tree.heading("time",    text="시각")
        self._tree.heading("type",    text="유형")
        self._tree.heading("sev",     text="심각도")
        self._tree.heading("roi",     text="ROI")
        self._tree.heading("message", text="메시지")
        self._tree.column("time",    width=80,  anchor="center")
        self._tree.column("type",    width=90,  anchor="w")
        self._tree.column("sev",     width=50,  anchor="center")
        self._tree.column("roi",     width=60,  anchor="center")
        self._tree.column("message", width=160, anchor="w")

        vsb = ttk.Scrollbar(tree_frame, orient="vertical", command=self._tree.yview)
        self._tree.configure(yscrollcommand=vsb.set)
        self._tree.pack(side="left", fill="both", expand=True)
        vsb.pack(side="right", fill="y")

        # 텍스트 로그
        log_frame = ttk.LabelFrame(self, text="실시간 로그", padding=4)
        log_frame.pack(fill="x", pady=(0, 0))

        self._log_text = tk.Text(log_frame, height=6, state="disabled",
                                 bg="#181825", fg="#a6e3a1",
                                 font=_MONO, wrap="word",
                                 relief="flat")
        log_vsb = ttk.Scrollbar(log_frame, orient="vertical",
                                 command=self._log_text.yview)
        self._log_text.configure(yscrollcommand=log_vsb.set)
        self._log_text.pack(side="left", fill="both", expand=True)
        log_vsb.pack(side="right", fill="y")

    # ── 공용 API ──────────────────────────────────────────────────────────

    def add_event(self, event: Dict) -> None:
        """이벤트 딕셔너리를 추가한다."""
        ts = event.get("timestamp", 0.0)
        time_str = datetime.fromtimestamp(ts).strftime("%H:%M:%S") if ts else "—"
        entry = {
            "time":        time_str,
            "type":        event.get("event_type", ""),
            "sev":         f"{event.get('severity', 0.0):.2f}",
            "roi":         event.get("roi_id", ""),
            "message":     event.get("message", ""),
            "raw":         event,
        }
        self._events.append(entry)
        self._insert_tree_row(entry)
        self.append_log(f"[{time_str}] {entry['type']} | {entry['message']}")

    def append_log(self, text: str) -> None:
        """실시간 텍스트 로그에 한 줄 추가한다."""
        self._log_lines.append(text)
        if len(self._log_lines) > self.MAX_LOG_LINES:
            self._log_lines = self._log_lines[-self.MAX_LOG_LINES:]

        self._log_text.config(state="normal")
        self._log_text.insert("end", text + "\n")
        # 최대 줄 수 초과 시 위 줄 삭제
        lines = int(self._log_text.index("end-1c").split(".")[0])
        if lines > self.MAX_LOG_LINES:
            self._log_text.delete("1.0", "2.0")
        self._log_text.see("end")
        self._log_text.config(state="disabled")

    def clear(self) -> None:
        self._events.clear()
        self._log_lines.clear()
        for item in self._tree.get_children():
            self._tree.delete(item)
        self._log_text.config(state="normal")
        self._log_text.delete("1.0", "end")
        self._log_text.config(state="disabled")

    # ── 필터 ─────────────────────────────────────────────────────────────

    def _apply_filter(self) -> None:
        sev_min = self._sev_filter.get()
        type_filter = self._type_filter.get()

        for item in self._tree.get_children():
            self._tree.delete(item)

        for entry in self._events:
            sev = float(entry["sev"])
            t   = entry["type"]
            if sev < sev_min:
                continue
            if type_filter != "전체" and t != type_filter:
                continue
            self._insert_tree_row(entry)

    def _insert_tree_row(self, entry: Dict) -> None:
        sev = float(entry["sev"])
        tag = "high" if sev >= 0.8 else ("mid" if sev >= 0.5 else "low")
        self._tree.insert("", "end",
                          values=(entry["time"], entry["type"],
                                  entry["sev"], entry["roi"],
                                  entry["message"]),
                          tags=(tag,))
        self._tree.tag_configure("high", foreground="#f38ba8")
        self._tree.tag_configure("mid",  foreground="#fab387")
        self._tree.tag_configure("low",  foreground="#a6e3a1")
        self._tree.yview_moveto(1.0)

    # ── 내보내기 ─────────────────────────────────────────────────────────

    def _export_csv(self) -> None:
        if not self._events:
            messagebox.showinfo("내보내기", "저장할 이벤트가 없습니다.")
            return
        p = filedialog.asksaveasfilename(
            title="CSV 저장", defaultextension=".csv",
            filetypes=[("CSV", "*.csv"), ("전체", "*.*")],
        )
        if not p:
            return
        with open(p, "w", newline="", encoding="utf-8-sig") as f:
            writer = csv.DictWriter(f, fieldnames=["time", "type", "sev", "roi", "message"])
            writer.writeheader()
            for e in self._events:
                writer.writerow({k: e[k] for k in ["time", "type", "sev", "roi", "message"]})
        messagebox.showinfo("저장 완료", f"CSV 저장됨:\n{p}")

    def _export_json(self) -> None:
        if not self._events:
            messagebox.showinfo("내보내기", "저장할 이벤트가 없습니다.")
            return
        p = filedialog.asksaveasfilename(
            title="JSON 저장", defaultextension=".json",
            filetypes=[("JSON", "*.json"), ("전체", "*.*")],
        )
        if not p:
            return
        records = [e["raw"] for e in self._events if "raw" in e]
        Path(p).write_text(json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8")
        messagebox.showinfo("저장 완료", f"JSON 저장됨:\n{p}")
