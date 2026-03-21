"""
gui/ai_management_panel.py — AI 관리 통합 패널

ai_management/ 패키지의 8개 매니저를 하나의 탭 UI 로 통합한다.

탭 구성:
  📂 데이터셋   — DatasetManager (데이터셋 등록/조회/통계)
  🏋  학습      — TrainingManager (학습 작업 시작/중지/체크포인트)
  📊 평가       — EvaluationManager (모델 성능 평가/리포트)
  🧪 실험       — ExperimentManager (실험 이력 조회)
  🗂 레지스트리  — ModelRegistry (모델 등록/production 지정)
  🔍 예측 분석  — PredictionAnalysisManager (배치 예측/통계)
  🚀 배포 준비  — DeploymentPreparationManager (ONNX 내보내기/검증)
"""

from __future__ import annotations

import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from typing import Optional

from gui.scrollable_frame import ScrollableFrame


_KO = ("맑은 고딕", 9)
_KO_BOLD = ("맑은 고딕", 9, "bold")
_MONO = ("Consolas", 9)


def _err_label(parent, module: str) -> ttk.Label:
    return ttk.Label(
        parent,
        text=(
            f"⚠  {module} 모듈을 불러올 수 없습니다.\n\n"
            f"ai_management/{module.lower().replace(' ', '_')}.py 를 확인하세요."
        ),
        justify="left",
        wraplength=300,
    )


# ---------------------------------------------------------------------------
# 각 탭 구현
# ---------------------------------------------------------------------------

class _DatasetTab(ttk.Frame):
    def __init__(self, master):
        super().__init__(master, padding=8)
        try:
            from ai_management.dataset_manager import DatasetManager
            self._mgr = DatasetManager()
            self._build()
        except ImportError as e:
            _err_label(self, "DatasetManager").pack(padx=20, pady=20)

    def _build(self):
        ttk.Label(self, text="📂  데이터셋 관리", font=_KO_BOLD).pack(anchor="w", pady=(0, 6))

        btn_row = ttk.Frame(self)
        btn_row.pack(fill="x", pady=(0, 6))
        ttk.Button(btn_row, text="데이터셋 등록…", command=self._register).pack(side="left", padx=(0, 4))
        ttk.Button(btn_row, text="새로고침",        command=self._refresh ).pack(side="left")

        lf = ttk.LabelFrame(self, text="등록된 데이터셋", padding=4)
        lf.pack(fill="both", expand=True)

        cols = ("name", "path", "size", "labels")
        self._tree = ttk.Treeview(lf, columns=cols, show="headings", height=10)
        for col, w, lbl in [
            ("name", 100, "이름"), ("path", 180, "경로"),
            ("size", 60, "샘플 수"), ("labels", 80, "레이블"),
        ]:
            self._tree.heading(col, text=lbl)
            self._tree.column(col, width=w, anchor="w")
        vsb = ttk.Scrollbar(lf, orient="vertical", command=self._tree.yview)
        self._tree.configure(yscrollcommand=vsb.set)
        self._tree.pack(side="left", fill="both", expand=True)
        vsb.pack(side="right", fill="y")

        self._refresh()

    def _register(self):
        p = filedialog.askdirectory(title="데이터셋 폴더 선택")
        if p:
            try:
                info = self._mgr.register_dataset(path=p)
                self._refresh()
                messagebox.showinfo("등록 완료", f"데이터셋 등록됨:\n{info.get('name', p)}")
            except Exception as e:
                messagebox.showerror("등록 오류", str(e))

    def _refresh(self):
        if not hasattr(self, "_tree"):
            return
        for item in self._tree.get_children():
            self._tree.delete(item)
        try:
            for ds in self._mgr.list_datasets():
                self._tree.insert("", "end", values=(
                    ds.get("name", ""),
                    ds.get("path", ""),
                    ds.get("sample_count", "?"),
                    ", ".join(ds.get("labels", [])),
                ))
        except Exception:
            pass


class _TrainingTab(ttk.Frame):
    def __init__(self, master):
        super().__init__(master, padding=8)
        try:
            from ai_management.training_manager import TrainingManager
            from gui.training_panel import TrainingPanel
            TrainingPanel(self).pack(fill="both", expand=True)
        except ImportError:
            _err_label(self, "TrainingManager").pack(padx=20, pady=20)


class _EvaluationTab(ttk.Frame):
    def __init__(self, master):
        super().__init__(master, padding=8)
        try:
            from ai_management.evaluation_manager import EvaluationManager
            self._mgr = EvaluationManager()
            self._build()
        except ImportError:
            _err_label(self, "EvaluationManager").pack(padx=20, pady=20)

    def _build(self):
        ttk.Label(self, text="📊  모델 평가", font=_KO_BOLD).pack(anchor="w", pady=(0, 6))

        sf = ScrollableFrame(self)
        sf.pack(fill="both", expand=True)
        inner = sf.inner

        row = ttk.Frame(inner)
        row.pack(fill="x", pady=(0, 6))
        ttk.Label(row, text="모델 파일:").pack(side="left")
        self._model_path = tk.StringVar()
        ttk.Entry(row, textvariable=self._model_path, width=24).pack(side="left", fill="x", expand=True, padx=4)
        ttk.Button(row, text="찾기", command=self._browse_model).pack(side="left")

        row2 = ttk.Frame(inner)
        row2.pack(fill="x", pady=(0, 6))
        ttk.Label(row2, text="테스트 데이터:").pack(side="left")
        self._test_path = tk.StringVar()
        ttk.Entry(row2, textvariable=self._test_path, width=24).pack(side="left", fill="x", expand=True, padx=4)
        ttk.Button(row2, text="찾기", command=self._browse_data).pack(side="left")

        ttk.Button(inner, text="평가 실행", style="Accent.TButton",
                   command=self._run_eval).pack(anchor="w", pady=(0, 8))

        self._result_text = tk.Text(inner, height=10, state="disabled",
                                    bg="#181825", fg="#cdd6f4",
                                    font=_MONO, relief="flat")
        self._result_text.pack(fill="both", expand=True)

    def _browse_model(self):
        p = filedialog.askopenfilename(
            filetypes=[("모델 파일", "*.onnx *.pt *.pth"), ("전체", "*.*")])
        if p: self._model_path.set(p)

    def _browse_data(self):
        p = filedialog.askdirectory(title="테스트 데이터 폴더")
        if p: self._test_path.set(p)

    def _run_eval(self):
        try:
            result = self._mgr.evaluate(
                model_path=self._model_path.get(),
                dataset_path=self._test_path.get(),
            )
            self._result_text.config(state="normal")
            self._result_text.delete("1.0", "end")
            import json
            self._result_text.insert("end", json.dumps(result, indent=2, ensure_ascii=False))
            self._result_text.config(state="disabled")
        except Exception as e:
            messagebox.showerror("평가 오류", str(e))


class _ExperimentTab(ttk.Frame):
    def __init__(self, master):
        super().__init__(master, padding=8)
        try:
            from ai_management.experiment_manager import ExperimentManager
            self._mgr = ExperimentManager()
            self._build()
        except ImportError:
            _err_label(self, "ExperimentManager").pack(padx=20, pady=20)

    def _build(self):
        ttk.Label(self, text="🧪  실험 이력", font=_KO_BOLD).pack(anchor="w", pady=(0, 6))

        btn_row = ttk.Frame(self)
        btn_row.pack(fill="x", pady=(0, 6))
        ttk.Button(btn_row, text="새로고침", command=self._refresh).pack(side="left")
        ttk.Button(btn_row, text="선택 삭제", command=self._delete_selected).pack(side="left", padx=4)

        lf = ttk.LabelFrame(self, text="실험 목록", padding=4)
        lf.pack(fill="both", expand=True)

        cols = ("exp_id", "name", "status", "metrics", "start")
        self._tree = ttk.Treeview(lf, columns=cols, show="headings", height=12)
        for col, w, lbl in [
            ("exp_id", 80, "ID"), ("name", 100, "이름"),
            ("status", 70, "상태"), ("metrics", 120, "주요 지표"),
            ("start", 90, "시작 시각"),
        ]:
            self._tree.heading(col, text=lbl)
            self._tree.column(col, width=w, anchor="w")
        vsb = ttk.Scrollbar(lf, orient="vertical", command=self._tree.yview)
        self._tree.configure(yscrollcommand=vsb.set)
        self._tree.pack(side="left", fill="both", expand=True)
        vsb.pack(side="right", fill="y")

        self._refresh()

    def _refresh(self):
        for item in self._tree.get_children():
            self._tree.delete(item)
        try:
            for exp in self._mgr.list_experiments():
                metrics_str = ", ".join(
                    f"{k}={v:.3f}" for k, v in list(exp.get("metrics", {}).items())[:2])
                self._tree.insert("", "end", values=(
                    exp.get("experiment_id", "")[:8],
                    exp.get("name", ""),
                    exp.get("status", ""),
                    metrics_str,
                    exp.get("start_time_str", ""),
                ))
        except Exception:
            pass

    def _delete_selected(self):
        sel = self._tree.selection()
        if not sel:
            return
        if messagebox.askyesno("삭제 확인", "선택한 실험을 삭제하시겠습니까?"):
            for item in sel:
                exp_id = self._tree.item(item, "values")[0]
                try:
                    self._mgr.delete_experiment(exp_id)
                except Exception:
                    pass
            self._refresh()


class _RegistryTab(ttk.Frame):
    def __init__(self, master):
        super().__init__(master, padding=8)
        try:
            from ai_management.model_registry import ModelRegistry
            self._reg = ModelRegistry()
            self._build()
        except ImportError:
            _err_label(self, "ModelRegistry").pack(padx=20, pady=20)

    def _build(self):
        ttk.Label(self, text="🗂  모델 레지스트리", font=_KO_BOLD).pack(anchor="w", pady=(0, 6))

        btn_row = ttk.Frame(self)
        btn_row.pack(fill="x", pady=(0, 6))
        ttk.Button(btn_row, text="모델 등록…",      command=self._register).pack(side="left", padx=(0, 4))
        ttk.Button(btn_row, text="Production 지정",  command=self._set_prod ).pack(side="left", padx=(0, 4))
        ttk.Button(btn_row, text="새로고침",         command=self._refresh  ).pack(side="left")

        lf = ttk.LabelFrame(self, text="등록된 모델", padding=4)
        lf.pack(fill="both", expand=True)

        cols = ("name", "version", "framework", "status", "metrics")
        self._tree = ttk.Treeview(lf, columns=cols, show="headings", height=12)
        for col, w, lbl in [
            ("name", 100, "이름"), ("version", 60, "버전"),
            ("framework", 70, "프레임워크"), ("status", 80, "상태"),
            ("metrics", 140, "주요 지표"),
        ]:
            self._tree.heading(col, text=lbl)
            self._tree.column(col, width=w, anchor="w")
        vsb = ttk.Scrollbar(lf, orient="vertical", command=self._tree.yview)
        self._tree.configure(yscrollcommand=vsb.set)
        self._tree.pack(side="left", fill="both", expand=True)
        vsb.pack(side="right", fill="y")

        self._refresh()

    def _register(self):
        p = filedialog.askopenfilename(
            title="모델 파일 선택",
            filetypes=[("모델 파일", "*.onnx *.pt *.pth"), ("전체", "*.*")],
        )
        if p:
            try:
                self._reg.register(file_path=p)
                self._refresh()
                messagebox.showinfo("등록 완료", f"모델 등록됨:\n{p}")
            except Exception as e:
                messagebox.showerror("등록 오류", str(e))

    def _set_prod(self):
        sel = self._tree.selection()
        if not sel:
            messagebox.showinfo("선택 필요", "Production 으로 지정할 모델을 선택하세요.")
            return
        name = self._tree.item(sel[0], "values")[0]
        try:
            self._reg.set_production(name)
            self._refresh()
        except Exception as e:
            messagebox.showerror("오류", str(e))

    def _refresh(self):
        for item in self._tree.get_children():
            self._tree.delete(item)
        try:
            for m in self._reg.list_models():
                metrics_str = ", ".join(
                    f"{k}={v:.3f}" for k, v in list(m.get("metrics", {}).items())[:2])
                self._tree.insert("", "end", values=(
                    m.get("model_name", ""),
                    m.get("version", ""),
                    m.get("framework", ""),
                    m.get("status", ""),
                    metrics_str,
                ))
        except Exception:
            pass


class _PredictionTab(ttk.Frame):
    def __init__(self, master):
        super().__init__(master, padding=8)
        try:
            from ai_management.prediction_analysis_manager import PredictionAnalysisManager
            self._mgr = PredictionAnalysisManager()
            self._build()
        except ImportError:
            _err_label(self, "PredictionAnalysisManager").pack(padx=20, pady=20)

    def _build(self):
        ttk.Label(self, text="🔍  예측 분석", font=_KO_BOLD).pack(anchor="w", pady=(0, 6))

        sf = ScrollableFrame(self)
        sf.pack(fill="both", expand=True)
        inner = sf.inner

        row = ttk.Frame(inner)
        row.pack(fill="x", pady=(0, 4))
        ttk.Label(row, text="모델:").pack(side="left")
        self._model_path = tk.StringVar()
        ttk.Entry(row, textvariable=self._model_path, width=22).pack(side="left", fill="x", expand=True, padx=4)
        ttk.Button(row, text="찾기", command=lambda: self._browse(self._model_path)).pack(side="left")

        row2 = ttk.Frame(inner)
        row2.pack(fill="x", pady=(0, 4))
        ttk.Label(row2, text="입력 데이터:").pack(side="left")
        self._data_path = tk.StringVar()
        ttk.Entry(row2, textvariable=self._data_path, width=22).pack(side="left", fill="x", expand=True, padx=4)
        ttk.Button(row2, text="찾기", command=lambda: self._browse(self._data_path)).pack(side="left")

        ttk.Button(inner, text="배치 예측 실행", style="Accent.TButton",
                   command=self._run).pack(anchor="w", pady=(4, 8))

        self._text = tk.Text(inner, height=14, state="disabled",
                             bg="#181825", fg="#cdd6f4", font=_MONO, relief="flat")
        self._text.pack(fill="both", expand=True)

    def _browse(self, var: tk.StringVar):
        p = filedialog.askopenfilename()
        if p: var.set(p)

    def _run(self):
        try:
            result = self._mgr.run_batch(
                model_path=self._model_path.get(),
                input_path=self._data_path.get(),
            )
            import json
            self._text.config(state="normal")
            self._text.delete("1.0", "end")
            self._text.insert("end", json.dumps(result, indent=2, ensure_ascii=False))
            self._text.config(state="disabled")
        except Exception as e:
            messagebox.showerror("예측 오류", str(e))


class _DeployTab(ttk.Frame):
    def __init__(self, master):
        super().__init__(master, padding=8)
        try:
            from ai_management.deployment_preparation_manager import DeploymentPreparationManager
            self._mgr = DeploymentPreparationManager()
            self._build()
        except ImportError:
            _err_label(self, "DeploymentPreparationManager").pack(padx=20, pady=20)

    def _build(self):
        ttk.Label(self, text="🚀  배포 준비", font=_KO_BOLD).pack(anchor="w", pady=(0, 6))

        sf = ScrollableFrame(self)
        sf.pack(fill="both", expand=True)
        inner = sf.inner

        row = ttk.Frame(inner)
        row.pack(fill="x", pady=(0, 4))
        ttk.Label(row, text="모델 파일:").pack(side="left")
        self._model_path = tk.StringVar()
        ttk.Entry(row, textvariable=self._model_path, width=22).pack(side="left", fill="x", expand=True, padx=4)
        ttk.Button(row, text="찾기", command=self._browse_model).pack(side="left")

        row2 = ttk.Frame(inner)
        row2.pack(fill="x", pady=(0, 4))
        ttk.Label(row2, text="내보내기 형식:").pack(side="left")
        self._export_fmt = tk.StringVar(value="onnx")
        for v, l in [("onnx", "ONNX"), ("torchscript", "TorchScript")]:
            ttk.Radiobutton(row2, text=l, variable=self._export_fmt, value=v).pack(side="left", padx=4)

        row3 = ttk.Frame(inner)
        row3.pack(fill="x", pady=(0, 8))
        ttk.Label(row3, text="출력 경로:").pack(side="left")
        self._out_path = tk.StringVar(value="deploy_output")
        ttk.Entry(row3, textvariable=self._out_path, width=20).pack(side="left", padx=4)

        btn_row = ttk.Frame(inner)
        btn_row.pack(fill="x", pady=(0, 8))
        ttk.Button(btn_row, text="내보내기", style="Accent.TButton",
                   command=self._export).pack(side="left", padx=(0, 4))
        ttk.Button(btn_row, text="검증",     command=self._validate).pack(side="left")

        self._text = tk.Text(inner, height=10, state="disabled",
                             bg="#181825", fg="#cdd6f4", font=_MONO, relief="flat")
        self._text.pack(fill="both", expand=True)

    def _browse_model(self):
        p = filedialog.askopenfilename(
            filetypes=[("모델", "*.onnx *.pt *.pth"), ("전체", "*.*")])
        if p: self._model_path.set(p)

    def _export(self):
        try:
            result = self._mgr.export(
                model_path=self._model_path.get(),
                output_dir=self._out_path.get(),
                export_format=self._export_fmt.get(),
            )
            self._append(f"내보내기 완료:\n{result}\n")
        except Exception as e:
            messagebox.showerror("내보내기 오류", str(e))

    def _validate(self):
        try:
            ok, msg = self._mgr.validate(self._model_path.get())
            self._append(f"검증 {'✅ 통과' if ok else '❌ 실패'}:\n{msg}\n")
        except Exception as e:
            messagebox.showerror("검증 오류", str(e))

    def _append(self, text: str):
        self._text.config(state="normal")
        self._text.insert("end", text)
        self._text.see("end")
        self._text.config(state="disabled")


# ---------------------------------------------------------------------------
# AIManagementPanel — 통합 탭 컨테이너
# ---------------------------------------------------------------------------

class AIManagementPanel(ttk.Frame):
    """8개 AI 관리 서브시스템을 탭으로 통합한 메인 패널."""

    def __init__(self, master, **kwargs):
        super().__init__(master, **kwargs)
        nb = ttk.Notebook(self)
        nb.pack(fill="both", expand=True)

        tabs = [
            ("📂 데이터셋",   _DatasetTab),
            ("🏋 학습",       _TrainingTab),
            ("📊 평가",       _EvaluationTab),
            ("🧪 실험",       _ExperimentTab),
            ("🗂 레지스트리", _RegistryTab),
            ("🔍 예측 분석",  _PredictionTab),
            ("🚀 배포",       _DeployTab),
        ]
        for title, cls in tabs:
            frame = cls(nb)
            nb.add(frame, text=title)
