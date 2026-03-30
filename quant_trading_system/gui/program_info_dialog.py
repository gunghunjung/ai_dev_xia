# gui/program_info_dialog.py — 프로그램 종합 정보 뷰어
"""
PROGRAM_INFO.txt 를 읽어 다이얼로그로 표시.
메인창 헤더 버튼 또는 메뉴에서 호출.
"""
from __future__ import annotations

import os
import tkinter as tk
from tkinter import ttk
import logging

logger = logging.getLogger("quant.gui.program_info")

_BASE_DIR   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_INFO_FILE  = os.path.join(_BASE_DIR, "outputs", "PROGRAM_INFO.txt")

# ─── 색상 ──────────────────────────────────────────────────────────────────
_C = {
    "bg":      "#1e1e2e",
    "bg2":     "#181825",
    "bg3":     "#11111b",
    "border":  "#313244",
    "text":    "#cdd6f4",
    "sub":     "#9399b2",
    "accent":  "#89b4fa",
    "green":   "#a6e3a1",
    "yellow":  "#f9e2af",
    "red":     "#f38ba8",
    "blue":    "#74c7ec",
    "purple":  "#cba6f7",
}


def open_program_info(parent: tk.Misc):
    """프로그램 종합 정보 다이얼로그 열기"""
    dlg = ProgramInfoDialog(parent)
    dlg.show()


class ProgramInfoDialog(tk.Toplevel):

    def __init__(self, parent):
        super().__init__(parent)
        self.title("프로그램 종합 정보 — AI 퀀트 트레이딩 시스템")
        self.geometry("920x700")
        self.minsize(760, 500)
        self.configure(bg=_C["bg"])
        self.resizable(True, True)

        self._content = self._load_content()
        self._build()

    # ──────────────────────────────────────────────────────────────────────
    # 로드
    # ──────────────────────────────────────────────────────────────────────

    def _load_content(self) -> str:
        try:
            with open(_INFO_FILE, encoding="utf-8") as f:
                return f.read()
        except FileNotFoundError:
            return "(파일 없음: outputs/PROGRAM_INFO.txt)"
        except Exception as e:
            return f"파일 로드 오류: {e}"

    # ──────────────────────────────────────────────────────────────────────
    # UI 구성
    # ──────────────────────────────────────────────────────────────────────

    def _build(self):
        # 헤더
        hdr = tk.Frame(self, bg="#12122a", pady=10)
        hdr.pack(fill="x")
        tk.Label(hdr,
                 text="📋  프로그램 종합 정보",
                 bg="#12122a", fg=_C["accent"],
                 font=("맑은 고딕", 12, "bold")).pack(side="left", padx=16)
        tk.Label(hdr,
                 text="요구사항 현황 · 구현 상태 · 파일 구조 · 데이터 플로우",
                 bg="#12122a", fg=_C["sub"],
                 font=("맑은 고딕", 8)).pack(side="left", padx=4)
        ttk.Button(hdr, text="✕ 닫기",
                   command=self.destroy).pack(side="right", padx=12)
        ttk.Button(hdr, text="↻ 새로고침",
                   command=self._reload).pack(side="right", padx=4)

        # 섹션 탭 (좌측 사이드바 + 우측 텍스트)
        body = tk.Frame(self, bg=_C["bg"])
        body.pack(fill="both", expand=True, padx=0, pady=0)

        # 좌측: 섹션 목차
        sidebar = tk.Frame(body, bg=_C["bg2"], width=180)
        sidebar.pack(side="left", fill="y", padx=0)
        sidebar.pack_propagate(False)

        tk.Label(sidebar, text="목차",
                 bg=_C["bg2"], fg=_C["sub"],
                 font=("맑은 고딕", 8, "bold")).pack(pady=(10, 4), padx=10, anchor="w")

        self._section_btns: list[tk.Button] = []
        sections = [
            ("1. 프로그램 개요",         "1. 프로그램 개요"),
            ("2. 전체 요청 사항",         "2. 전체 요청 사항"),
            ("3. 구현 현황",              "3. 구현 현황"),
            ("4. 미구현 / 잔여 과제",     "4. 미구현 / 잔여 과제"),
            ("5. 파일 구조",              "5. 전체 파일 구조"),
            ("6. 데이터 플로우",          "6. 주요 데이터 플로우"),
            ("7. 외부환경 상세",          "7. 외부환경 분석 상세"),
            ("8. 탭 기능 요약",           "8. 탭 구성 및 기능 요약"),
            ("9. 설치 의존성",            "9. 설치 의존성"),
        ]
        for label, anchor in sections:
            btn = tk.Button(
                sidebar, text=label,
                bg=_C["bg2"], fg=_C["sub"],
                font=("맑은 고딕", 8), relief="flat",
                anchor="w", padx=10, pady=3,
                activebackground=_C["bg3"], activeforeground=_C["accent"],
                cursor="hand2",
                command=lambda a=anchor: self._jump_to(a),
            )
            btn.pack(fill="x", padx=4, pady=1)
            self._section_btns.append(btn)

        # 구분선
        tk.Frame(body, bg=_C["border"], width=1).pack(side="left", fill="y")

        # 우측: 텍스트 뷰어
        right = tk.Frame(body, bg=_C["bg"])
        right.pack(side="left", fill="both", expand=True)

        # 검색 바
        search_bar = tk.Frame(right, bg=_C["bg3"], pady=4)
        search_bar.pack(fill="x", padx=6, pady=(6, 0))
        tk.Label(search_bar, text="검색:",
                 bg=_C["bg3"], fg=_C["sub"],
                 font=("맑은 고딕", 8)).pack(side="left", padx=6)
        self._search_var = tk.StringVar()
        search_entry = tk.Entry(search_bar, textvariable=self._search_var,
                                bg=_C["bg2"], fg=_C["text"],
                                insertbackground=_C["text"],
                                relief="flat", font=("맑은 고딕", 9), width=24)
        search_entry.pack(side="left", padx=4)
        search_entry.bind("<Return>",    lambda e: self._search())
        search_entry.bind("<KP_Enter>",  lambda e: self._search())
        ttk.Button(search_bar, text="찾기",
                   command=self._search, width=5).pack(side="left", padx=2)
        ttk.Button(search_bar, text="다음",
                   command=self._search_next, width=5).pack(side="left", padx=2)
        self._search_info = tk.Label(search_bar, text="",
                                      bg=_C["bg3"], fg=_C["sub"],
                                      font=("맑은 고딕", 7))
        self._search_info.pack(side="left", padx=8)

        # 텍스트 위젯
        txt_frame = tk.Frame(right, bg=_C["bg"])
        txt_frame.pack(fill="both", expand=True, padx=6, pady=6)

        self._txt = tk.Text(
            txt_frame,
            bg=_C["bg"], fg=_C["text"],
            font=("Consolas", 9),
            wrap="none",
            relief="flat",
            padx=12, pady=8,
            state="disabled",
            cursor="arrow",
            selectbackground=_C["border"],
        )
        vsb = ttk.Scrollbar(txt_frame, orient="vertical",
                             command=self._txt.yview)
        hsb = ttk.Scrollbar(txt_frame, orient="horizontal",
                             command=self._txt.xview)
        self._txt.configure(yscrollcommand=vsb.set,
                             xscrollcommand=hsb.set)
        vsb.pack(side="right",  fill="y")
        hsb.pack(side="bottom", fill="x")
        self._txt.pack(side="left", fill="both", expand=True)

        # 텍스트 색상 태그
        self._txt.tag_configure("head1",   foreground=_C["accent"],
                                  font=("Consolas", 10, "bold"))
        self._txt.tag_configure("head2",   foreground=_C["purple"],
                                  font=("Consolas", 9, "bold"))
        self._txt.tag_configure("done",    foreground=_C["green"])
        self._txt.tag_configure("partial", foreground=_C["yellow"])
        self._txt.tag_configure("todo",    foreground=_C["red"])
        self._txt.tag_configure("border",  foreground=_C["border"])
        self._txt.tag_configure("hilite",  background="#2a2a52",
                                  foreground=_C["yellow"])

        self._search_positions: list = []
        self._search_idx: int = 0

        self._render()

    # ──────────────────────────────────────────────────────────────────────
    # 렌더링
    # ──────────────────────────────────────────────────────────────────────

    def _render(self):
        self._txt.configure(state="normal")
        self._txt.delete("1.0", "end")

        for line in self._content.splitlines():
            tag = None
            stripped = line.strip()

            if stripped.startswith("╔") or stripped.startswith("╚") or \
               stripped.startswith("║") or stripped.startswith("━"):
                tag = "head1"
            elif stripped.startswith("  [") and "]" in stripped and \
                 stripped[stripped.index("]")-1].isdigit():
                # 섹션 제목 [A] [B] 등
                tag = "head2"
            elif stripped.startswith("━━"):
                tag = "head1"
            elif "  ✅" in line:
                tag = "done"
            elif "  ⚠" in line:
                tag = "partial"
            elif "  ❌" in line:
                tag = "todo"
            elif stripped.startswith("─────") or stripped.startswith("═════"):
                tag = "border"
            elif stripped.startswith("[1]") or stripped.startswith("[2]") or \
                 stripped.startswith("[3]") or stripped.startswith("[4]") or \
                 stripped.startswith("[5]") or stripped.startswith("[6]") or \
                 stripped.startswith("[7]") or stripped.startswith("[8]"):
                tag = "partial"

            if tag:
                self._txt.insert("end", line + "\n", tag)
            else:
                self._txt.insert("end", line + "\n")

        self._txt.configure(state="disabled")

    def _reload(self):
        self._content = self._load_content()
        self._render()
        self._search_positions.clear()
        self._search_info.configure(text="")

    # ──────────────────────────────────────────────────────────────────────
    # 섹션 점프
    # ──────────────────────────────────────────────────────────────────────

    def _jump_to(self, keyword: str):
        self._txt.configure(state="normal")
        # 기존 하이라이트 제거
        self._txt.tag_remove("hilite", "1.0", "end")
        pos = self._txt.search(keyword, "1.0", nocase=True)
        if pos:
            self._txt.see(pos)
            end_pos = f"{pos}+{len(keyword)}c"
            self._txt.tag_add("hilite", pos, end_pos)
        self._txt.configure(state="disabled")

        # 사이드바 버튼 하이라이트
        for btn in self._section_btns:
            btn.configure(bg=_C["bg2"], fg=_C["sub"])
        for btn in self._section_btns:
            if keyword[:6] in btn.cget("text"):
                btn.configure(bg=_C["bg3"], fg=_C["accent"])
                break

    # ──────────────────────────────────────────────────────────────────────
    # 검색
    # ──────────────────────────────────────────────────────────────────────

    def _search(self):
        kw = self._search_var.get().strip()
        if not kw:
            return
        self._txt.configure(state="normal")
        self._txt.tag_remove("hilite", "1.0", "end")

        positions = []
        start = "1.0"
        while True:
            pos = self._txt.search(kw, start, nocase=True, stopindex="end")
            if not pos:
                break
            end_pos = f"{pos}+{len(kw)}c"
            self._txt.tag_add("hilite", pos, end_pos)
            positions.append(pos)
            start = end_pos

        self._txt.configure(state="disabled")
        self._search_positions = positions
        self._search_idx = 0

        if positions:
            self._txt.see(positions[0])
            self._search_info.configure(
                text=f"{len(positions)}건 발견",
                fg=_C["green"])
        else:
            self._search_info.configure(text="없음", fg=_C["red"])

    def _search_next(self):
        if not self._search_positions:
            self._search()
            return
        self._search_idx = (self._search_idx + 1) % len(self._search_positions)
        self._txt.see(self._search_positions[self._search_idx])
        self._search_info.configure(
            text=f"{self._search_idx+1}/{len(self._search_positions)}",
            fg=_C["yellow"])

    # ──────────────────────────────────────────────────────────────────────

    def show(self):
        self.focus_force()
        # grab_set / wait_window 제거 → 최대화 버튼 활성화 (비모달)
