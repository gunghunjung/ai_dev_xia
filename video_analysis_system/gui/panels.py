"""
gui/panels.py — 우측 탭 패널 모음 (전체 한글 UI)

패널 목록:
  ControlPanel  — 소스 선택, 시작/정지, 모델 설정
  ROIPanel      — ROI 목록 관리 (추가/삭제/이름 변경)
  StatusPanel   — 실시간 시스템 상태, ROI 테이블, 발동된 규칙
  EventPanel    — 이벤트 로그 테이블
"""

from __future__ import annotations

import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from typing import Callable, Dict, List, Optional, Tuple

from core.states import ROI_STATE_COLORS, STATE_COLORS, ROIState, SystemState


def _bgr_to_hex(bgr: Tuple[int, int, int]) -> str:
    b, g, r = bgr
    return f"#{r:02x}{g:02x}{b:02x}"


STATE_HEX     = {s: _bgr_to_hex(c) for s, c in STATE_COLORS.items()}
ROI_STATE_HEX = {s: _bgr_to_hex(c) for s, c in ROI_STATE_COLORS.items()}

_KO = ("맑은 고딕", 9)
_KO_BOLD = ("맑은 고딕", 9, "bold")
_MONO = ("Consolas", 9)


# ---------------------------------------------------------------------------
# ControlPanel
# ---------------------------------------------------------------------------

class ControlPanel(ttk.Frame):
    """소스 선택, 시작/정지, 화면 옵션, AI 모델 설정 패널."""

    def __init__(self, master, on_start: Callable, on_stop: Callable, **kwargs):
        super().__init__(master, padding=8, **kwargs)
        self._on_start = on_start
        self._on_stop  = on_stop

        # ── 소스 유형 ─────────────────────────────────────────────────────
        src_frame = ttk.LabelFrame(self, text="📁  비디오 소스", padding=6)
        src_frame.pack(fill="x", pady=(0, 6))

        self._src_type = tk.StringVar(value="file")
        radio_row = ttk.Frame(src_frame)
        radio_row.pack(fill="x")
        for val, lbl in [("file", "파일"), ("camera", "카메라"), ("images", "이미지 폴더")]:
            ttk.Radiobutton(radio_row, text=lbl, variable=self._src_type,
                            value=val, command=self._on_src_change).pack(side="left", padx=4)

        path_row = ttk.Frame(src_frame)
        path_row.pack(fill="x", pady=(6, 0))
        self._path_var = tk.StringVar()
        self._path_entry = ttk.Entry(path_row, textvariable=self._path_var, width=26)
        self._path_entry.pack(side="left", fill="x", expand=True)
        self._browse_btn = ttk.Button(path_row, text="찾기…", width=5, command=self._browse)
        self._browse_btn.pack(side="left", padx=(3, 0))

        cam_row = ttk.Frame(src_frame)
        cam_row.pack(fill="x", pady=(5, 0))
        ttk.Label(cam_row, text="카메라 번호:").pack(side="left")
        self._cam_idx = tk.IntVar(value=0)
        ttk.Spinbox(cam_row, from_=0, to=8, textvariable=self._cam_idx, width=5).pack(side="left", padx=4)
        self._loop_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(cam_row, text="반복 재생", variable=self._loop_var).pack(side="left", padx=(8, 0))

        # ── 재생 제어 ──────────────────────────────────────────────────────
        ctrl_frame = ttk.LabelFrame(self, text="▶  재생 제어", padding=6)
        ctrl_frame.pack(fill="x", pady=(0, 6))

        btn_row = ttk.Frame(ctrl_frame)
        btn_row.pack(fill="x")
        self._start_btn = ttk.Button(btn_row, text="▶  시작", command=self._start,
                                     style="Accent.TButton")
        self._start_btn.pack(side="left", fill="x", expand=True, padx=(0, 3))
        self._stop_btn = ttk.Button(btn_row, text="■  정지", command=self._stop,
                                    state="disabled")
        self._stop_btn.pack(side="left", fill="x", expand=True)

        fps_row = ttk.Frame(ctrl_frame)
        fps_row.pack(fill="x", pady=(6, 0))
        ttk.Label(fps_row, text="목표 FPS:").pack(side="left")
        self._fps_var = tk.DoubleVar(value=30.0)
        ttk.Spinbox(fps_row, from_=1, to=120, increment=1,
                    textvariable=self._fps_var, width=7).pack(side="left", padx=4)

        mf_row = ttk.Frame(ctrl_frame)
        mf_row.pack(fill="x", pady=(3, 0))
        ttk.Label(mf_row, text="최대 프레임:").pack(side="left")
        self._max_frames = tk.StringVar(value="")
        ttk.Entry(mf_row, textvariable=self._max_frames, width=8).pack(side="left", padx=4)
        ttk.Label(mf_row, text="(비워두면 무제한)", foreground="#888888").pack(side="left")

        # ── 화면 옵션 ─────────────────────────────────────────────────────
        disp_frame = ttk.LabelFrame(self, text="🖥  화면 표시", padding=6)
        disp_frame.pack(fill="x", pady=(0, 6))

        self._show_debug = tk.BooleanVar(value=False)
        self._show_roi   = tk.BooleanVar(value=True)
        ttk.Checkbutton(disp_frame, text="디버그 오버레이 표시",
                        variable=self._show_debug).pack(anchor="w")
        ttk.Checkbutton(disp_frame, text="ROI 박스 표시",
                        variable=self._show_roi).pack(anchor="w")

        # ── AI 모델 ───────────────────────────────────────────────────────
        mdl_frame = ttk.LabelFrame(self, text="🤖  AI 모델", padding=6)
        mdl_frame.pack(fill="x")

        self._model_type = tk.StringVar(value="placeholder")
        radio_row2 = ttk.Frame(mdl_frame)
        radio_row2.pack(fill="x")
        for val, lbl in [("placeholder", "기본(테스트용)"), ("onnx", "ONNX"), ("pytorch", "PyTorch")]:
            ttk.Radiobutton(radio_row2, text=lbl,
                            variable=self._model_type, value=val).pack(side="left", padx=2)

        mp_row = ttk.Frame(mdl_frame)
        mp_row.pack(fill="x", pady=(5, 0))
        ttk.Label(mp_row, text="모델 경로:").pack(side="left")
        self._model_path = tk.StringVar()
        ttk.Entry(mp_row, textvariable=self._model_path, width=20).pack(side="left",
                                                                         fill="x", expand=True, padx=(4, 0))
        ttk.Button(mp_row, text="찾기", width=4,
                   command=lambda: self._model_path.set(
                       filedialog.askopenfilename(
                           title="모델 파일 선택",
                           filetypes=[("모델 파일", "*.onnx *.pt *.pth"), ("전체", "*.*")]
                       ) or self._model_path.get()
                   )).pack(side="left", padx=(3, 0))

        self._on_src_change()

    # ── Internal ──────────────────────────────────────────────────────────

    def _on_src_change(self) -> None:
        is_path = self._src_type.get() in ("file", "images")
        self._path_entry.config(state="normal" if is_path else "disabled")
        self._browse_btn.config(state="normal" if is_path else "disabled")

    def _browse(self) -> None:
        if self._src_type.get() == "file":
            p = filedialog.askopenfilename(
                title="비디오 파일 선택",
                filetypes=[("비디오 파일", "*.mp4 *.avi *.mov *.mkv *.wmv"), ("전체", "*.*")]
            )
        else:
            p = filedialog.askdirectory(title="이미지 폴더 선택")
        if p:
            self._path_var.set(p)

    def _start(self) -> None:
        max_f = None
        try:
            raw = self._max_frames.get().strip()
            if raw:
                max_f = int(raw)
        except ValueError:
            messagebox.showwarning("입력 오류", "최대 프레임은 정수로 입력하세요.")
            return

        overrides = {
            "source_type":  self._src_type.get(),
            "source_path":  self._path_var.get(),
            "camera_index": self._cam_idx.get(),
            "fps":          self._fps_var.get(),
            "loop":         self._loop_var.get(),
            "max_frames":   max_f,
            "model_type":   self._model_type.get(),
            "model_path":   self._model_path.get() or None,
            "show_debug":   self._show_debug.get(),
        }
        self._start_btn.config(state="disabled")
        self._stop_btn.config(state="normal")
        self._on_start(overrides)

    def _stop(self) -> None:
        self._stop_btn.config(state="disabled")
        self._start_btn.config(state="normal")
        self._on_stop()

    def set_stopped(self) -> None:
        self._stop_btn.config(state="disabled")
        self._start_btn.config(state="normal")


# ---------------------------------------------------------------------------
# ROIPanel
# ---------------------------------------------------------------------------

class ROIPanel(ttk.Frame):
    """ROI 목록 관리 (추가 / 삭제 / 이름 설정)."""

    def __init__(self, master, on_rois_changed: Callable, **kwargs):
        super().__init__(master, padding=8, **kwargs)
        self._on_rois_changed = on_rois_changed
        self._rois: List[dict] = []

        ttk.Label(self,
                  text="💡 비디오 화면에서 마우스를 드래그하여 관심 영역(ROI)을 지정하세요.",
                  foreground="#888888", wraplength=230, justify="left").pack(anchor="w", pady=(0, 6))

        btn_row = ttk.Frame(self)
        btn_row.pack(fill="x", pady=(0, 5))
        self._draw_btn = ttk.Button(btn_row, text="✎ ROI 그리기", command=self._toggle_draw)
        self._draw_btn.pack(side="left")
        ttk.Button(btn_row, text="✕ 삭제", command=self._remove_selected).pack(side="left", padx=(4, 0))
        ttk.Button(btn_row, text="✕ 전체 삭제", command=self._clear_all).pack(side="left", padx=(4, 0))

        # 목록
        list_frame = ttk.Frame(self)
        list_frame.pack(fill="both", expand=True)
        self._listbox = tk.Listbox(list_frame, height=10, selectmode="single",
                                   bg="#1e1e2e", fg="#cdd6f4",
                                   selectbackground="#313244",
                                   font=_MONO, relief="flat")
        sb = ttk.Scrollbar(list_frame, command=self._listbox.yview)
        self._listbox.config(yscrollcommand=sb.set)
        sb.pack(side="right", fill="y")
        self._listbox.pack(fill="both", expand=True)

        # 이름 편집
        edit_row = ttk.Frame(self)
        edit_row.pack(fill="x", pady=(6, 0))
        ttk.Label(edit_row, text="표시 이름:").pack(side="left")
        self._label_var = tk.StringVar()
        ttk.Entry(edit_row, textvariable=self._label_var, width=14).pack(side="left", padx=(4, 0))
        ttk.Button(edit_row, text="적용", command=self._set_label).pack(side="left", padx=(4, 0))

        # 좌표 직접 입력
        coord_frame = ttk.LabelFrame(self, text="좌표 직접 입력  (x, y, 너비, 높이)", padding=5)
        coord_frame.pack(fill="x", pady=(8, 0))
        coord_row = ttk.Frame(coord_frame)
        coord_row.pack(fill="x")
        self._cx = tk.StringVar(value="50")
        self._cy = tk.StringVar(value="50")
        self._cw = tk.StringVar(value="200")
        self._ch = tk.StringVar(value="150")
        for lbl, var in [("X:", self._cx), ("Y:", self._cy), ("W:", self._cw), ("H:", self._ch)]:
            ttk.Label(coord_row, text=lbl).pack(side="left")
            ttk.Entry(coord_row, textvariable=var, width=5).pack(side="left", padx=(1, 4))
        ttk.Button(coord_frame, text="+ 추가", command=self._add_from_coords).pack(pady=(4, 0))

        self._drawing_mode = False
        self.draw_callback: Optional[Callable] = None

    def _toggle_draw(self) -> None:
        self._drawing_mode = not self._drawing_mode
        self._draw_btn.config(text="✓ 그리는 중…" if self._drawing_mode else "✎ ROI 그리기")
        if self.draw_callback:
            self.draw_callback(self._drawing_mode)

    def add_roi(self, roi_id: str, bbox: tuple, label: str = "") -> None:
        entry = {"roi_id": roi_id, "bbox": list(bbox), "label": label or roi_id, "enabled": True}
        self._rois.append(entry)
        self._refresh_list()
        self._on_rois_changed(self._rois)
        self._drawing_mode = False
        self._draw_btn.config(text="✎ ROI 그리기")

    def _add_from_coords(self) -> None:
        try:
            x, y = int(self._cx.get()), int(self._cy.get())
            w, h = int(self._cw.get()), int(self._ch.get())
            if w <= 0 or h <= 0:
                raise ValueError
        except ValueError:
            messagebox.showwarning("입력 오류", "X, Y, 너비, 높이를 올바른 정수로 입력하세요.")
            return
        roi_id = f"roi_{len(self._rois)}"
        self.add_roi(roi_id, (x, y, w, h))

    def _remove_selected(self) -> None:
        sel = self._listbox.curselection()
        if not sel:
            return
        self._rois.pop(sel[0])
        self._refresh_list()
        self._on_rois_changed(self._rois)

    def _clear_all(self) -> None:
        if self._rois and not messagebox.askyesno("확인", "모든 ROI를 삭제하시겠습니까?"):
            return
        self._rois.clear()
        self._refresh_list()
        self._on_rois_changed(self._rois)

    def _set_label(self) -> None:
        sel = self._listbox.curselection()
        if not sel:
            messagebox.showinfo("알림", "목록에서 ROI를 먼저 선택하세요.")
            return
        self._rois[sel[0]]["label"] = self._label_var.get()
        self._refresh_list()
        self._on_rois_changed(self._rois)

    def _refresh_list(self) -> None:
        self._listbox.delete(0, "end")
        for r in self._rois:
            x, y, w, h = r["bbox"]
            self._listbox.insert("end", f"[{r['roi_id']}]  {r['label']}  ({x},{y},{w},{h})")

    def get_roi_configs(self) -> List[dict]:
        return list(self._rois)


# ---------------------------------------------------------------------------
# StatusPanel
# ---------------------------------------------------------------------------

class StatusPanel(ttk.Frame):
    """실시간 시스템 상태 / ROI 테이블 / 발동된 규칙 표시 패널."""

    _STATE_KO = {
        "INITIALIZING":  "초기화 중",
        "NORMAL":        "정상",
        "WARNING":       "경고",
        "ABNORMAL":      "이상 감지",
        "TRACKING_LOST": "추적 손실",
        "UNKNOWN":       "알 수 없음",
    }
    _ROI_STATE_KO = {
        "NORMAL":        "정상",
        "WARNING":       "경고",
        "ABNORMAL":      "이상",
        "STUCK":         "고착",
        "DRIFTING":      "드리프트",
        "OSCILLATING":   "진동",
        "SUDDEN_CHANGE": "급변",
        "UNKNOWN":       "미확인",
    }

    def __init__(self, master, **kwargs):
        super().__init__(master, padding=8, **kwargs)

        # 시스템 상태 배너
        state_frame = ttk.LabelFrame(self, text="🔵  시스템 상태", padding=6)
        state_frame.pack(fill="x", pady=(0, 6))
        self._state_var = tk.StringVar(value="—")
        self._conf_var  = tk.StringVar(value="")
        self._state_lbl = tk.Label(state_frame, textvariable=self._state_var,
                                   font=("맑은 고딕", 18, "bold"),
                                   bg="#1a1a2e", fg="#aaaaaa", width=12)
        self._state_lbl.pack()
        tk.Label(state_frame, textvariable=self._conf_var,
                 bg="#1a1a2e", fg="#888888", font=_MONO).pack()

        # 성능 지표
        perf_frame = ttk.LabelFrame(self, text="⏱  성능", padding=4)
        perf_frame.pack(fill="x", pady=(0, 6))
        self._fps_var  = tk.StringVar(value="FPS:       —")
        self._time_var = tk.StringVar(value="파이프라인:  — ms")
        self._inf_var  = tk.StringVar(value="AI 추론:    — ms")
        self._frame_var = tk.StringVar(value="처리 프레임: —")
        for v in (self._fps_var, self._time_var, self._inf_var, self._frame_var):
            tk.Label(perf_frame, textvariable=v, bg="#1a1a2e", fg="#aaaaaa",
                     font=_MONO, anchor="w").pack(fill="x")

        # ROI 상태 테이블
        roi_frame = ttk.LabelFrame(self, text="📦  ROI 상태", padding=4)
        roi_frame.pack(fill="x", pady=(0, 6))
        cols = ("roi", "state", "mean", "diff", "ai")
        self._roi_tree = ttk.Treeview(roi_frame, columns=cols, show="headings", height=5)
        for col, w, txt in [("roi",6,"ID"),("state",8,"상태"),("mean",6,"평균"),
                             ("diff",5,"변화"),("ai",10,"AI 판정")]:
            self._roi_tree.heading(col, text=txt)
            self._roi_tree.column(col, width=w*9, anchor="center")
        vsb = ttk.Scrollbar(roi_frame, command=self._roi_tree.yview)
        self._roi_tree.config(yscrollcommand=vsb.set)
        vsb.pack(side="right", fill="y")
        self._roi_tree.pack(fill="x")

        # 발동된 규칙
        rule_frame = ttk.LabelFrame(self, text="⚡  발동된 규칙", padding=4)
        rule_frame.pack(fill="both", expand=True)
        self._rule_text = tk.Text(rule_frame, height=6, bg="#1e1e2e", fg="#f38ba8",
                                  font=_MONO, relief="flat", state="disabled")
        rule_sb = ttk.Scrollbar(rule_frame, command=self._rule_text.yview)
        self._rule_text.config(yscrollcommand=rule_sb.set)
        rule_sb.pack(side="right", fill="y")
        self._rule_text.pack(fill="both", expand=True)

    def update_status(self, result) -> None:
        state = result.system_state
        hex_color = STATE_HEX.get(state, "#aaaaaa")
        ko_name = self._STATE_KO.get(state.value, state.value)
        self._state_var.set(ko_name)
        self._state_lbl.config(fg=hex_color)
        self._conf_var.set(f"신뢰도  {result.state_confidence:.3f}")

        self._fps_var.set(f"FPS:         {result.fps_actual:.1f}")
        self._time_var.set(f"파이프라인:  {result.pipeline_ms:.1f} ms")
        self._inf_var.set(f"AI 추론:     {result.inference_ms:.1f} ms")
        self._frame_var.set(f"처리 프레임: {result.frame_index}")

        self._roi_tree.delete(*self._roi_tree.get_children())
        for r in result.roi_summaries:
            flags = " ".join(
                f for f, v in [("고착", r["stuck"]), ("드리", r["drift"]), ("진동", r["oscillate"])] if v
            )
            ko_state = self._ROI_STATE_KO.get(r["state"], r["state"])
            ai_txt = f"{r['ai_label']} {r['ai_conf']:.2f}" if r["ai_conf"] > 0 else "—"
            row_id = self._roi_tree.insert("", "end", values=(
                r["roi_id"],
                ko_state + (" [" + flags + "]" if flags else ""),
                f"{r['mean_int']:.0f}",
                f"{r['diff']:.0f}",
                ai_txt,
            ))
            try:
                s = ROIState(r["state"])
                tag = s.value
                self._roi_tree.tag_configure(tag, foreground=ROI_STATE_HEX.get(s, "#cdd6f4"))
                self._roi_tree.item(row_id, tags=(tag,))
            except ValueError:
                pass

        self._rule_text.config(state="normal")
        self._rule_text.delete("1.0", "end")
        if result.triggered_rules:
            for rule in result.triggered_rules:
                self._rule_text.insert("end", f"▲ {rule.rule_name}\n   {rule.message}\n\n")
        else:
            self._rule_text.insert("end", "발동된 규칙 없음")
        self._rule_text.config(state="disabled")


# ---------------------------------------------------------------------------
# EventPanel
# ---------------------------------------------------------------------------

class EventPanel(ttk.Frame):
    """감지된 이상 이벤트 목록 및 상세 정보 패널."""

    _TYPE_KO = {
        "state_change_abnormal": "상태 → 이상 감지",
        "stuck":                 "고착 감지",
        "drift":                 "드리프트",
        "sudden_change":         "급격한 변화",
        "oscillation":           "진동 패턴",
        "ai_detection":          "AI 이상 감지",
    }

    def __init__(self, master, **kwargs):
        super().__init__(master, padding=8, **kwargs)

        hdr = ttk.Frame(self)
        hdr.pack(fill="x", pady=(0, 4))
        ttk.Label(hdr, text="⚠  감지된 이벤트", font=("맑은 고딕", 10, "bold")).pack(side="left")
        self._count_var = tk.StringVar(value="(0건)")
        ttk.Label(hdr, textvariable=self._count_var, foreground="#888888").pack(side="left", padx=8)
        ttk.Button(hdr, text="초기화", command=self._clear).pack(side="right")

        cols = ("time", "frame", "type", "sev", "roi", "reason")
        self._tree = ttk.Treeview(self, columns=cols, show="headings", height=13)
        hdrs_def = [
            ("time",  10, "시각"),
            ("frame",  7, "프레임"),
            ("type",  14, "유형"),
            ("sev",    6, "심각도"),
            ("roi",    6, "ROI"),
            ("reason", 20, "원인"),
        ]
        for col, w, txt in hdrs_def:
            self._tree.heading(col, text=txt)
            self._tree.column(col, width=w*8, anchor="w")

        vsb = ttk.Scrollbar(self, command=self._tree.yview)
        self._tree.config(yscrollcommand=vsb.set)
        vsb.pack(side="right", fill="y")
        self._tree.pack(fill="both", expand=True)

        detail_frame = ttk.LabelFrame(self, text="📋  상세 정보", padding=4)
        detail_frame.pack(fill="x", pady=(4, 0))
        self._detail = tk.Text(detail_frame, height=5, bg="#1e1e2e", fg="#cdd6f4",
                               font=_MONO, relief="flat", state="disabled")
        self._detail.pack(fill="x")
        self._tree.bind("<<TreeviewSelect>>", self._on_select)

        self._events: list = []

    def add_event(self, event) -> None:
        import time as _time
        ts  = _time.strftime("%H:%M:%S", _time.localtime(event.timestamp))
        sev = f"{event.severity:.2f}"
        ko_type = self._TYPE_KO.get(event.event_type, event.event_type)
        self._tree.insert("", "end", iid=event.event_id,
                          values=(ts, event.frame_index, ko_type,
                                  sev, event.roi_id, event.message[:55]),
                          tags=("event",))
        self._tree.tag_configure("event", foreground="#f38ba8")
        self._tree.see(event.event_id)
        self._events.append(event)
        self._count_var.set(f"({len(self._events)}건)")

    def _on_select(self, _) -> None:
        sel = self._tree.selection()
        if not sel:
            return
        evt = next((e for e in self._events if e.event_id == sel[0]), None)
        if evt is None:
            return
        ko_type = self._TYPE_KO.get(evt.event_type, evt.event_type)
        self._detail.config(state="normal")
        self._detail.delete("1.0", "end")
        self._detail.insert("end",
            f"이벤트 ID : {evt.event_id}\n"
            f"시스템 상태: {evt.system_state}\n"
            f"이벤트 유형: {ko_type}\n"
            f"이상 분류  : {evt.abnormality_type}\n"
            f"스냅샷    : {evt.snapshot_path or '(없음)'}\n"
            f"원인      : {evt.message}\n"
        )
        self._detail.config(state="disabled")

    def _clear(self) -> None:
        if self._events and not messagebox.askyesno("확인", "이벤트 목록을 초기화하시겠습니까?"):
            return
        self._tree.delete(*self._tree.get_children())
        self._events.clear()
        self._count_var.set("(0건)")
