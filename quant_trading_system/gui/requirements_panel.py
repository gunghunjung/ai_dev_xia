# gui/requirements_panel.py — 요구사항 관리 시스템 (10단계)
"""
개발 방향을 추적하는 요구사항 관리 도구.

단순 메모장이 아니라 AI 시스템의 개발 로드맵을 관리하는 전용 패널.

기능:
  - 요구사항 추가 / 수정 / 삭제
  - 상태 관리: TODO / 진행중 / 완료 / 보류
  - 카테고리: AI모델 / UI / 데이터 / 버그 / 개선아이디어
  - 중요도: HIGH / MEDIUM / LOW
  - 생성일 / 수정일 자동 관리
  - JSON 영속 저장 (outputs/requirements.json)
  - 필터 / 정렬 / 검색
"""
from __future__ import annotations

import json
import logging
import os
import sys
import tkinter as tk
from tkinter import ttk, messagebox, scrolledtext
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)

logger = logging.getLogger("quant.gui.req")

# ─── 색상 상수 ──────────────────────────────────────────────────────────────
_BG       = "#1e1e2e"
_PANEL_BG = "#181825"
_FG       = "#cdd6f4"
_DIM      = "#9399b2"
_ACCENT   = "#89b4fa"
_GREEN    = "#a6e3a1"
_RED      = "#f38ba8"
_YELLOW   = "#f9e2af"
_PURPLE   = "#cba6f7"
_ORANGE   = "#fab387"

# ─── 상태 / 카테고리 / 중요도 정의 ─────────────────────────────────────────
_STATUSES = ["TODO", "진행중", "완료", "보류"]
_CATEGORIES = ["AI모델", "UI", "데이터", "버그", "개선아이디어", "기타"]
_PRIORITIES = ["HIGH", "MEDIUM", "LOW"]

_STATUS_COLORS = {
    "TODO":  _YELLOW,
    "진행중": _ACCENT,
    "완료":  _GREEN,
    "보류":  _DIM,
}

_PRIORITY_COLORS = {
    "HIGH":   _RED,
    "MEDIUM": _YELLOW,
    "LOW":    _DIM,
}

_CAT_ICONS = {
    "AI모델":    "🧠",
    "UI":        "🖥️",
    "데이터":    "📊",
    "버그":      "🐛",
    "개선아이디어": "💡",
    "기타":      "📋",
}


# ─── 데이터 모델 ────────────────────────────────────────────────────────────

@dataclass
class Requirement:
    """단일 요구사항 레코드."""
    id:          str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    title:       str = ""
    category:    str = "기타"
    status:      str = "TODO"
    priority:    str = "MEDIUM"
    created_at:  str = field(default_factory=lambda: datetime.now().strftime("%Y-%m-%d"))
    updated_at:  str = field(default_factory=lambda: datetime.now().strftime("%Y-%m-%d %H:%M"))
    notes:       str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "Requirement":
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


# ─── 저장소 ─────────────────────────────────────────────────────────────────

class RequirementsStore:
    """JSON 파일 기반 요구사항 영속 저장소."""

    def __init__(self, path: str):
        self._path = path
        self._items: List[Requirement] = []
        self._load()

    def _load(self):
        if os.path.exists(self._path):
            try:
                with open(self._path, "r", encoding="utf-8") as f:
                    raw = json.load(f)
                self._items = [Requirement.from_dict(r) for r in raw]
            except Exception as e:
                logger.warning(f"요구사항 파일 로드 실패: {e}")
                self._items = []
        else:
            self._items = []   # 빈 상태로 시작 — 사용자가 직접 입력

    def _save(self):
        os.makedirs(os.path.dirname(self._path), exist_ok=True)
        with open(self._path, "w", encoding="utf-8") as f:
            json.dump([r.to_dict() for r in self._items], f,
                      ensure_ascii=False, indent=2)
        # 버전 관리 파일에도 현재 요구사항 상태 동기 기록
        self._write_version_file()

    def _write_version_file(self):
        """requirements.json 변경 시마다 VERSION.txt에 현재 상태 덮어쓰기."""
        try:
            ver_path = os.path.join(os.path.dirname(self._path), "VERSION.txt")
            now      = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            items    = self._items

            # 상태별 집계
            by_status: Dict[str, List[Requirement]] = {}
            for r in items:
                by_status.setdefault(r.status, []).append(r)

            lines = [
                "=" * 60,
                "  AI 주식 분석 시스템 — 요구사항 현황",
                f"  최종 업데이트: {now}",
                "=" * 60,
                "",
            ]

            for status in ["진행중", "TODO", "보류", "완료"]:
                group = by_status.get(status, [])
                if not group:
                    continue
                icon = {"진행중": "🔄", "TODO": "📌", "보류": "⏸", "완료": "✅"}.get(status, "")
                lines.append(f"{icon} [{status}] ({len(group)}건)")
                lines.append("-" * 50)
                for r in sorted(group,
                                key=lambda x: {"HIGH":0,"MEDIUM":1,"LOW":2}.get(x.priority,9)):
                    prio_tag = {"HIGH":"🔴","MEDIUM":"🟡","LOW":"⚪"}.get(r.priority,"")
                    cat_icon = {"AI모델":"🧠","UI":"🖥","데이터":"📊",
                                "버그":"🐛","개선아이디어":"💡","기타":"📋"}.get(r.category,"")
                    lines.append(f"  {prio_tag} [{r.id}] {r.title}")
                    lines.append(f"       {cat_icon} {r.category}  |  생성: {r.created_at}  |  수정: {r.updated_at}")
                    if r.notes:
                        for note_line in r.notes.splitlines()[:3]:
                            lines.append(f"       → {note_line}")
                lines.append("")

            # 요약
            total  = len(items)
            done   = len(by_status.get("완료", []))
            active = len(by_status.get("진행중", [])) + len(by_status.get("TODO", []))
            lines += [
                "=" * 60,
                f"  총 {total}건  |  완료 {done}건  |  활성 {active}건  "
                f"|  보류 {len(by_status.get('보류',[]))}건",
                "=" * 60,
            ]

            with open(ver_path, "w", encoding="utf-8") as f:
                f.write("\n".join(lines))
        except Exception as e:
            logger.debug(f"VERSION.txt 쓰기 실패: {e}")

    def all(self) -> List[Requirement]:
        return list(self._items)

    def add(self, req: Requirement):
        self._items.insert(0, req)
        self._save()

    def update(self, req: Requirement):
        for i, r in enumerate(self._items):
            if r.id == req.id:
                req.updated_at = datetime.now().strftime("%Y-%m-%d %H:%M")
                self._items[i] = req
                self._save()
                return

    def delete(self, req_id: str):
        self._items = [r for r in self._items if r.id != req_id]
        self._save()

    def get(self, req_id: str) -> Optional[Requirement]:
        return next((r for r in self._items if r.id == req_id), None)


# ─── 편집 다이얼로그 ─────────────────────────────────────────────────────────

class RequirementDialog(tk.Toplevel):
    """요구사항 추가 / 수정 다이얼로그."""

    def __init__(self, parent, req: Optional[Requirement] = None):
        super().__init__(parent)
        self.result: Optional[Requirement] = None
        self._req = req or Requirement()

        title = "요구사항 수정" if req else "새 요구사항 추가"
        self.title(title)
        self.configure(bg=_BG)
        self.resizable(False, False)
        self.grab_set()

        self._build()
        self.wait_window()

    def _build(self):
        pad = {"padx": 10, "pady": 5}

        # 제목
        tk.Label(self, text="제목 *", bg=_BG, fg=_ACCENT,
                 font=("맑은 고딕", 9, "bold")).grid(row=0, column=0, sticky="w", **pad)
        self._title_var = tk.StringVar(value=self._req.title)
        title_ent = tk.Entry(self, textvariable=self._title_var,
                             bg=_PANEL_BG, fg=_FG, insertbackground=_FG,
                             font=("맑은 고딕", 10), width=50, relief="flat")
        title_ent.grid(row=0, column=1, columnspan=3, sticky="ew", **pad)
        title_ent.focus_set()

        # 카테고리
        tk.Label(self, text="카테고리", bg=_BG, fg=_FG,
                 font=("맑은 고딕", 9)).grid(row=1, column=0, sticky="w", **pad)
        self._cat_var = tk.StringVar(value=self._req.category)
        ttk.Combobox(self, textvariable=self._cat_var, values=_CATEGORIES,
                     state="readonly", width=16).grid(row=1, column=1, sticky="w", **pad)

        # 상태
        tk.Label(self, text="상태", bg=_BG, fg=_FG,
                 font=("맑은 고딕", 9)).grid(row=1, column=2, sticky="w", **pad)
        self._status_var = tk.StringVar(value=self._req.status)
        ttk.Combobox(self, textvariable=self._status_var, values=_STATUSES,
                     state="readonly", width=10).grid(row=1, column=3, sticky="w", **pad)

        # 중요도
        tk.Label(self, text="중요도", bg=_BG, fg=_FG,
                 font=("맑은 고딕", 9)).grid(row=2, column=0, sticky="w", **pad)
        self._prio_var = tk.StringVar(value=self._req.priority)
        for i, p in enumerate(_PRIORITIES):
            color = _PRIORITY_COLORS[p]
            rb = tk.Radiobutton(
                self, text=p, variable=self._prio_var, value=p,
                bg=_BG, fg=color, selectcolor=_PANEL_BG,
                activebackground=_BG, activeforeground=color,
                font=("맑은 고딕", 9, "bold"),
            )
            rb.grid(row=2, column=i + 1, sticky="w", **pad)

        # 메모
        tk.Label(self, text="메모", bg=_BG, fg=_FG,
                 font=("맑은 고딕", 9)).grid(row=3, column=0, sticky="nw", **pad)
        self._notes_text = scrolledtext.ScrolledText(
            self, height=6, width=60,
            bg=_PANEL_BG, fg=_FG, insertbackground=_FG,
            font=("맑은 고딕", 9), relief="flat", wrap="word",
        )
        self._notes_text.grid(row=3, column=1, columnspan=3, sticky="ew", **pad)
        self._notes_text.insert("1.0", self._req.notes)

        # 버튼
        btn_fr = tk.Frame(self, bg=_BG)
        btn_fr.grid(row=4, column=0, columnspan=4, pady=10)
        ttk.Button(btn_fr, text="💾 저장", command=self._save,
                   style="Accent.TButton").pack(side="left", padx=8)
        ttk.Button(btn_fr, text="취소", command=self.destroy).pack(side="left", padx=4)

        self.bind("<Return>", lambda e: self._save() if e.widget.__class__.__name__ != "ScrolledText" else None)
        self.bind("<Escape>", lambda e: self.destroy())

    def _save(self):
        title = self._title_var.get().strip()
        if not title:
            messagebox.showerror("오류", "제목을 입력하세요.", parent=self)
            return
        self._req.title    = title
        self._req.category = self._cat_var.get()
        self._req.status   = self._status_var.get()
        self._req.priority = self._prio_var.get()
        self._req.notes    = self._notes_text.get("1.0", "end-1c").strip()
        self.result = self._req
        self.destroy()


# ─── 메인 패널 ──────────────────────────────────────────────────────────────

class RequirementsPanel:
    """
    요구사항 관리 패널.

    AI 시스템의 개발 방향을 추적·관리하는 전용 패널.
    단순 메모장이 아니라 개발 로드맵 도구로 설계됨.
    """

    def __init__(self, parent: ttk.Notebook, output_dir: str):
        self._store = RequirementsStore(
            os.path.join(output_dir, "requirements.json")
        )
        self.frame = ttk.Frame(parent)
        self._selected_id: Optional[str] = None
        self._filter_status = tk.StringVar(value="전체")
        self._filter_cat    = tk.StringVar(value="전체")
        self._search_var    = tk.StringVar(value="")
        self._hide_done_var = tk.BooleanVar(value=True)   # 기본: 완료 숨김
        self._search_var.trace_add("write", self._on_search)

        self._build()
        self._refresh()

    # ══════════════════════════════════════════════════════════════════════════
    # UI 구성
    # ══════════════════════════════════════════════════════════════════════════

    def _build(self):
        # ── 헤더 ─────────────────────────────────────────────────────────
        hdr = tk.Frame(self.frame, bg="#11111b", pady=8)
        hdr.pack(fill="x")
        tk.Label(
            hdr,
            text="  📋 요구사항 관리  —  AI 시스템 개발 방향 로드맵",
            font=("맑은 고딕", 12, "bold"),
            bg="#11111b", fg=_PURPLE,
        ).pack(side="left", padx=12)

        # 오른쪽: 통계 배지
        self._stat_var = tk.StringVar()
        tk.Label(hdr, textvariable=self._stat_var,
                 bg="#11111b", fg=_DIM,
                 font=("맑은 고딕", 9)).pack(side="right", padx=16)

        # ── 툴바 ─────────────────────────────────────────────────────────
        toolbar = tk.Frame(self.frame, bg=_PANEL_BG, pady=6)
        toolbar.pack(fill="x", padx=4)

        ttk.Button(toolbar, text="➕ 추가",
                   command=self._add, style="Accent.TButton").pack(side="left", padx=(8, 4))
        ttk.Button(toolbar, text="✏️ 수정",
                   command=self._edit).pack(side="left", padx=4)
        ttk.Button(toolbar, text="🗑 삭제",
                   command=self._delete, style="Danger.TButton").pack(side="left", padx=4)

        # 상태 빠른 변경 (색상 구분)
        ttk.Separator(toolbar, orient="vertical").pack(side="left", fill="y", padx=8)
        tk.Label(toolbar, text="→ 상태:", bg=_PANEL_BG, fg=_DIM,
                 font=("맑은 고딕", 9)).pack(side="left")
        _STATUS_BTN_COLORS = {
            "TODO": ("#f9e2af", "#2a2000"),
            "진행중": ("#89b4fa", "#001a3d"),
            "완료": ("#a6e3a1", "#001a00"),
            "보류": ("#585b70", "#1e1e2e"),
        }
        for st in _STATUSES:
            fg_c, bg_c = _STATUS_BTN_COLORS.get(st, (_DIM, _PANEL_BG))
            btn = tk.Button(
                toolbar, text=st, width=5,
                bg=bg_c, fg=fg_c,
                activebackground=fg_c, activeforeground=bg_c,
                font=("맑은 고딕", 8, "bold"), relief="flat", padx=4,
                command=lambda s=st: self._quick_status(s),
            )
            btn.pack(side="left", padx=2)

        # 완료 항목 토글
        ttk.Separator(toolbar, orient="vertical").pack(side="left", fill="y", padx=8)
        hide_cb = ttk.Checkbutton(
            toolbar, text="완료 숨기기",
            variable=self._hide_done_var,
            command=self._refresh,
        )
        hide_cb.pack(side="left", padx=(2, 8))

        # 필터
        ttk.Separator(toolbar, orient="vertical").pack(side="left", fill="y", padx=4)
        tk.Label(toolbar, text="상태:", bg=_PANEL_BG, fg=_DIM,
                 font=("맑은 고딕", 9)).pack(side="left")
        ttk.Combobox(toolbar, textvariable=self._filter_status,
                     values=["전체"] + _STATUSES,
                     state="readonly", width=8).pack(side="left", padx=(2, 8))
        self._filter_status.trace_add("write", lambda *_: self._refresh())

        tk.Label(toolbar, text="분류:", bg=_PANEL_BG, fg=_DIM,
                 font=("맑은 고딕", 9)).pack(side="left")
        ttk.Combobox(toolbar, textvariable=self._filter_cat,
                     values=["전체"] + _CATEGORIES,
                     state="readonly", width=12).pack(side="left", padx=(2, 8))
        self._filter_cat.trace_add("write", lambda *_: self._refresh())

        # 검색
        tk.Label(toolbar, text="🔍", bg=_PANEL_BG, fg=_DIM).pack(side="left")
        tk.Entry(toolbar, textvariable=self._search_var,
                 bg=_PANEL_BG, fg=_FG, insertbackground=_FG,
                 font=("맑은 고딕", 9), width=20, relief="flat").pack(side="left", padx=4)

        # ── 메인 영역: 목록 + 상세 ───────────────────────────────────────
        pane = tk.PanedWindow(self.frame, orient="horizontal",
                              bg=_BG, sashwidth=4)
        pane.pack(fill="both", expand=True, padx=4, pady=4)

        # 왼쪽: Treeview
        list_fr = ttk.Frame(pane)
        pane.add(list_fr, minsize=480)
        self._build_tree(list_fr)

        # 오른쪽: 상세 / 통계
        right_fr = ttk.Frame(pane)
        pane.add(right_fr, minsize=260)
        self._build_detail(right_fr)

    def _build_tree(self, parent):
        cols = ("ID", "중요도", "카테고리", "제목", "상태", "생성일", "수정일")
        self._tree = ttk.Treeview(parent, columns=cols, show="headings", height=22)

        col_w = {
            "ID": 50, "중요도": 65, "카테고리": 80,
            "제목": 220, "상태": 65, "생성일": 85, "수정일": 95,
        }
        for c in cols:
            self._tree.heading(c, text=c, command=lambda col=c: self._sort(col))
            self._tree.column(c, width=col_w.get(c, 80), anchor="center")
        self._tree.column("제목", anchor="w")

        # 색상 태그
        self._tree.tag_configure("HIGH",   foreground=_RED)
        self._tree.tag_configure("MEDIUM", foreground=_YELLOW)
        self._tree.tag_configure("LOW",    foreground=_DIM)
        self._tree.tag_configure("완료",   foreground=_GREEN)
        self._tree.tag_configure("보류",   foreground="#585b70")

        vsb = ttk.Scrollbar(parent, orient="vertical", command=self._tree.yview)
        self._tree.configure(yscrollcommand=vsb.set)
        self._tree.pack(side="left", fill="both", expand=True)
        vsb.pack(side="right", fill="y")

        self._tree.bind("<<TreeviewSelect>>", self._on_select)
        self._tree.bind("<Double-1>",          lambda e: self._edit())

        # 정렬 상태
        self._sort_col = "중요도"
        self._sort_rev = False

    def _build_detail(self, parent):
        nb = ttk.Notebook(parent)
        nb.pack(fill="both", expand=True)

        # 상세 탭
        detail_fr = ttk.Frame(nb)
        nb.add(detail_fr, text="  📝 상세  ")

        self._detail_text = scrolledtext.ScrolledText(
            detail_fr, bg=_PANEL_BG, fg=_FG,
            font=("맑은 고딕", 9), relief="flat",
            wrap="word", state="disabled",
        )
        self._detail_text.pack(fill="both", expand=True)

        # 통계 탭
        stat_fr = ttk.Frame(nb)
        nb.add(stat_fr, text="  📊 통계  ")
        self._stat_canvas = tk.Canvas(
            stat_fr, bg=_PANEL_BG, highlightthickness=0,
        )
        self._stat_canvas.pack(fill="both", expand=True)

    # ══════════════════════════════════════════════════════════════════════════
    # 데이터 조작
    # ══════════════════════════════════════════════════════════════════════════

    def _add(self):
        dlg = RequirementDialog(self.frame.winfo_toplevel())
        if dlg.result:
            self._store.add(dlg.result)
            self._refresh()

    def _edit(self):
        if not self._selected_id:
            return
        req = self._store.get(self._selected_id)
        if not req:
            return
        dlg = RequirementDialog(self.frame.winfo_toplevel(), req)
        if dlg.result:
            self._store.update(dlg.result)
            self._refresh()

    def _delete(self):
        if not self._selected_id:
            return
        req = self._store.get(self._selected_id)
        if not req:
            return
        if messagebox.askyesno(
            "삭제 확인",
            f"'{req.title}'\n\n이 요구사항을 삭제하시겠습니까?",
            parent=self.frame.winfo_toplevel(),
        ):
            self._store.delete(self._selected_id)
            self._selected_id = None
            self._refresh()

    def _quick_status(self, status: str):
        """선택된 항목의 상태를 빠르게 변경."""
        if not self._selected_id:
            return
        req = self._store.get(self._selected_id)
        if not req:
            return
        req.status = status
        self._store.update(req)
        self._refresh()

    # ══════════════════════════════════════════════════════════════════════════
    # 표시 갱신
    # ══════════════════════════════════════════════════════════════════════════

    def _refresh(self):
        self._tree.delete(*self._tree.get_children())

        all_items = self._store.all()
        items     = list(all_items)

        # 완료 숨기기 토글
        hide_done = self._hide_done_var.get()
        if hide_done:
            active_items = [r for r in items if r.status != "완료"]
        else:
            active_items = items

        # 상태 필터
        status_f = self._filter_status.get()
        cat_f    = self._filter_cat.get()
        search_q = self._search_var.get().strip().lower()

        if status_f != "전체":
            active_items = [r for r in active_items if r.status == status_f]
        if cat_f != "전체":
            active_items = [r for r in active_items if r.category == cat_f]
        if search_q:
            active_items = [r for r in active_items
                            if search_q in r.title.lower() or search_q in r.notes.lower()]

        # 정렬: 진행중 > TODO > 보류 > 완료 / 그 안에서 중요도
        _prio_ord = {"HIGH": 0, "MEDIUM": 1, "LOW": 2}
        _stat_ord = {"진행중": 0, "TODO": 1, "보류": 2, "완료": 3}
        active_items.sort(key=lambda r: (
            _stat_ord.get(r.status, 9),
            _prio_ord.get(r.priority, 9),
            r.created_at,
        ))

        def _insert_row(req: Requirement):
            icon = _CAT_ICONS.get(req.category, "📋")
            # 완료: 취소선 느낌으로 회색, 진행중: 파랑, 보류: 회색, 그 외 중요도 색
            if req.status == "완료":
                tags = ("완료",)
            elif req.status == "진행중":
                tags = ("진행중_tag",)
            elif req.status == "보류":
                tags = ("보류",)
            else:
                tags = (req.priority,)

            # 완료 항목은 제목에 ✅ 접두사
            title_disp = (f"✅ {req.title}" if req.status == "완료"
                          else f"🔄 {req.title}" if req.status == "진행중"
                          else f"⏸ {req.title}" if req.status == "보류"
                          else req.title)
            self._tree.insert("", "end", iid=req.id, tags=tags, values=(
                req.id,
                req.priority,
                f"{icon} {req.category}",
                title_disp,
                req.status,
                req.created_at,
                req.updated_at,
            ))

        # 미완료 항목 먼저
        for req in active_items:
            _insert_row(req)

        # 완료 항목 숨김 모드에서도 구분선+완료 목록 하단에 별도 표시
        if hide_done and not search_q and status_f == "전체":
            done_items = sorted(
                [r for r in all_items if r.status == "완료"],
                key=lambda r: r.updated_at, reverse=True
            )
            if done_items:
                # 구분 헤더 행 삽입 (선택 불가용)
                sep_id = "__done_sep__"
                if not self._tree.exists(sep_id):
                    self._tree.insert("", "end", iid=sep_id, tags=("done_sep",),
                                      values=("", "", "",
                                              f"─── 완료됨 {len(done_items)}건 ───────────────",
                                              "", "", ""))
                for req in done_items:
                    _insert_row(req)

        # 완료-구분선 태그 색상
        self._tree.tag_configure("done_sep",
                                 foreground="#45475a",
                                 font=("맑은 고딕", 8))
        self._tree.tag_configure("진행중_tag", foreground=_ACCENT)

        # 통계 배지
        todo_n = sum(1 for r in all_items if r.status == "TODO")
        prog_n = sum(1 for r in all_items if r.status == "진행중")
        done_n = sum(1 for r in all_items if r.status == "완료")
        hold_n = sum(1 for r in all_items if r.status == "보류")
        high_n = sum(1 for r in all_items if r.priority == "HIGH" and r.status != "완료")
        self._stat_var.set(
            f"🔄 진행중 {prog_n}  📌 TODO {todo_n}  ⏸ 보류 {hold_n}  "
            f"✅ 완료 {done_n}  🔴 HIGH미완 {high_n}"
        )

        # 통계 차트 갱신
        self.frame.after(100, self._draw_stats)

        # 선택 복원
        if self._selected_id and self._tree.exists(self._selected_id):
            self._tree.selection_set(self._selected_id)

    def _on_search(self, *args):
        self._refresh()

    def _on_select(self, event):
        sel = self._tree.selection()
        if not sel:
            return
        self._selected_id = sel[0]
        req = self._store.get(self._selected_id)
        if not req:
            return
        self._show_detail(req)

    def _show_detail(self, req: Requirement):
        icon = _CAT_ICONS.get(req.category, "📋")
        prio_color = _PRIORITY_COLORS.get(req.priority, _DIM)
        stat_color = _STATUS_COLORS.get(req.status, _DIM)

        lines = [
            f"══════════════════════════════════════",
            f"[{req.id}] {req.title}",
            f"══════════════════════════════════════",
            f"",
            f"카테고리:  {icon} {req.category}",
            f"상태:      {req.status}",
            f"중요도:    {req.priority}",
            f"생성일:    {req.created_at}",
            f"수정일:    {req.updated_at}",
            f"",
            f"─── 메모 ───────────────────────────",
            req.notes if req.notes else "(메모 없음)",
            f"",
        ]
        self._detail_text.config(state="normal")
        self._detail_text.delete("1.0", "end")
        self._detail_text.insert("end", "\n".join(lines))
        self._detail_text.config(state="disabled")

    def _draw_stats(self):
        """통계 탭 — 도넛/막대 차트."""
        c = self._stat_canvas
        c.update_idletasks()
        W, H = c.winfo_width(), c.winfo_height()
        c.delete("all")
        if W < 50 or H < 50:
            return

        all_items = self._store.all()
        if not all_items:
            c.create_text(W // 2, H // 2, text="데이터 없음", fill=_DIM)
            return

        # ── 상태 분포 막대 ───────────────────────────────────────────
        status_counts = {s: sum(1 for r in all_items if r.status == s)
                         for s in _STATUSES}
        cat_counts    = {cat: sum(1 for r in all_items if r.category == cat)
                         for cat in _CATEGORIES}

        total    = max(len(all_items), 1)
        bar_x    = 20
        bar_w    = W - 40
        bar_h_u  = 18
        y_start  = 20

        c.create_text(bar_x, y_start - 4, text="상태별 분포",
                      fill=_ACCENT, font=("맑은 고딕", 8, "bold"), anchor="sw")

        y = y_start + 10
        for st, cnt in status_counts.items():
            pct  = cnt / total
            fill = _STATUS_COLORS.get(st, _DIM)
            c.create_rectangle(bar_x, y, bar_x + bar_w, y + bar_h_u,
                                fill="#2a2a3e", outline="")
            if pct > 0:
                c.create_rectangle(bar_x, y,
                                    bar_x + int(bar_w * pct), y + bar_h_u,
                                    fill=fill, outline="")
            c.create_text(bar_x + 4,    y + bar_h_u // 2, text=st,
                          fill="#1e1e2e" if pct > 0.15 else _FG,
                          font=("맑은 고딕", 8, "bold"), anchor="w")
            c.create_text(bar_x + bar_w + 4, y + bar_h_u // 2,
                          text=f"{cnt}", fill=_FG,
                          font=("맑은 고딕", 8), anchor="w")
            y += bar_h_u + 4

        # ── 카테고리 분포 ─────────────────────────────────────────────
        y += 14
        c.create_text(bar_x, y - 4, text="카테고리별 분포",
                      fill=_ACCENT, font=("맑은 고딕", 8, "bold"), anchor="sw")
        y += 10
        cat_colors = [_GREEN, _ACCENT, _YELLOW, _RED, _PURPLE, _DIM]
        for i, (cat, cnt) in enumerate(cat_counts.items()):
            pct  = cnt / total
            fill = cat_colors[i % len(cat_colors)]
            icon = _CAT_ICONS.get(cat, "📋")
            c.create_rectangle(bar_x, y, bar_x + bar_w, y + bar_h_u,
                                fill="#2a2a3e", outline="")
            if pct > 0:
                c.create_rectangle(bar_x, y,
                                    bar_x + int(bar_w * pct), y + bar_h_u,
                                    fill=fill, outline="")
            c.create_text(bar_x + 4, y + bar_h_u // 2,
                          text=f"{icon} {cat}",
                          fill="#1e1e2e" if pct > 0.25 else _FG,
                          font=("맑은 고딕", 8), anchor="w")
            c.create_text(bar_x + bar_w + 4, y + bar_h_u // 2,
                          text=f"{cnt}", fill=_FG,
                          font=("맑은 고딕", 8), anchor="w")
            y += bar_h_u + 4

        # ── HIGH 미완료 경고 ─────────────────────────────────────────
        high_open = [r for r in all_items
                     if r.priority == "HIGH" and r.status != "완료"]
        if high_open:
            y += 10
            c.create_text(
                W // 2, y,
                text=f"⚠️  HIGH 우선순위 미완료 {len(high_open)}개",
                fill=_RED, font=("맑은 고딕", 9, "bold"), anchor="n",
            )

    def _sort(self, col: str):
        if self._sort_col == col:
            self._sort_rev = not self._sort_rev
        else:
            self._sort_col = col
            self._sort_rev = False
        self._refresh()
