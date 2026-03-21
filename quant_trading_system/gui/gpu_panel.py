# gui/gpu_panel.py — GPU 실시간 모니터링 패널
from __future__ import annotations
import tkinter as tk
from tkinter import ttk
from typing import List, Dict
import threading


class GPUPanel:
    """
    GPU/CPU 실시간 사용량 모니터
    - GPU 사용률, VRAM 사용량, 온도, 전력
    - PyTorch CUDA 정보
    - 롤링 그래프 (최근 60초)
    """

    def __init__(self, parent):
        self.frame = ttk.Frame(parent)
        self._history: Dict[str, List[float]] = {
            "util_gpu": [], "util_mem": [], "temp": [], "power": [],
        }
        self._max_history = 60
        self._lock = threading.Lock()
        self._build()
        self._get_cuda_info()

    def _build(self):
        # 상단: CUDA 정보
        info_fr = ttk.LabelFrame(self.frame, text="CUDA / PyTorch 환경", padding=8)
        info_fr.pack(fill="x", padx=8, pady=(8, 4))

        self.cuda_vars = {}
        cuda_fields = [
            ("CUDA 사용 가능", "cuda_available"),
            ("CUDA 버전", "cuda_version"),
            ("PyTorch 버전", "torch_version"),
            ("GPU 개수", "device_count"),
            ("현재 GPU", "current_device"),
        ]
        for i, (label, key) in enumerate(cuda_fields):
            col = i % 3
            row = i // 3
            ttk.Label(info_fr, text=label + ":").grid(
                row=row, column=col*2, sticky="e", padx=4, pady=2)
            var = tk.StringVar(value="확인 중...")
            self.cuda_vars[key] = var
            ttk.Label(info_fr, textvariable=var,
                      foreground="#89b4fa",
                      font=("맑은 고딕", 10, "bold")).grid(
                row=row, column=col*2+1, sticky="w", padx=(0, 12))

        # GPU 상태 카드
        card_fr = ttk.Frame(self.frame)
        card_fr.pack(fill="x", padx=8, pady=(4, 4))

        self.gpu_cards = []
        for i in range(4):  # 최대 4개 GPU
            card = self._build_gpu_card(card_fr, i)
            card["frame"].grid(row=0, column=i, padx=4, sticky="n")
            self.gpu_cards.append(card)
            card["frame"].grid_remove()  # 기본 숨김

        # 실시간 그래프
        graph_fr = ttk.LabelFrame(self.frame, text="실시간 GPU 사용률 (최근 60초)",
                                   padding=4)
        graph_fr.pack(fill="both", expand=True, padx=8, pady=4)

        self.graph_canvas = tk.Canvas(graph_fr, bg="#181825",
                                      highlightthickness=0)
        self.graph_canvas.pack(fill="both", expand=True)
        self.graph_canvas.bind("<Configure>", lambda e: self._redraw_graph())

        # 범례
        legend_fr = ttk.Frame(graph_fr)
        legend_fr.pack(fill="x", pady=(4, 0))
        items = [("GPU 사용률", "#89b4fa"), ("VRAM 사용률", "#a6e3a1"),
                 ("온도", "#fab387"), ("전력", "#f9e2af")]
        for i, (label, color) in enumerate(items):
            tk.Label(legend_fr, text="━", fg=color, bg="#181825",
                     font=("Consolas", 12)).grid(row=0, column=i*2, padx=(8, 0))
            ttk.Label(legend_fr, text=label).grid(row=0, column=i*2+1, padx=(0, 8))

    def _build_gpu_card(self, parent, idx: int) -> Dict:
        """GPU 상태 카드"""
        fr = ttk.LabelFrame(parent, text=f"GPU {idx}", padding=8, width=200)

        gauges = {}
        gauge_items = [
            ("GPU 사용률", "util_gpu", "%", "#89b4fa"),
            ("VRAM 사용률", "util_mem", "%", "#a6e3a1"),
            ("온도", "temp", "°C", "#fab387"),
            ("전력", "power", "W", "#f9e2af"),
        ]

        for i, (label, key, unit, color) in enumerate(gauge_items):
            ttk.Label(fr, text=label + ":").grid(
                row=i, column=0, sticky="w", pady=2)
            var = tk.StringVar(value="0" + unit)
            gauges[key + "_var"] = var
            ttk.Label(fr, textvariable=var, foreground=color,
                      font=("Consolas", 11, "bold"), width=8).grid(
                row=i, column=1, sticky="e")

            pb = ttk.Progressbar(fr, length=120, maximum=100, mode="determinate")
            pb.grid(row=i, column=2, padx=(4, 0))
            gauges[key + "_pb"] = pb

        # VRAM 상세
        vram_var = tk.StringVar(value="—")
        gauges["vram_detail"] = vram_var
        ttk.Label(fr, textvariable=vram_var,
                  font=("Consolas", 8), foreground="#9399b2").grid(
            row=len(gauge_items), column=0, columnspan=3, sticky="w", pady=(4, 0))

        # GPU 이름
        name_var = tk.StringVar(value="—")
        gauges["name_var"] = name_var
        ttk.Label(fr, textvariable=name_var,
                  font=("맑은 고딕", 9), foreground="#cba6f7").grid(
            row=len(gauge_items)+1, column=0, columnspan=3, sticky="w")

        return {"frame": fr, **gauges}

    def update_stats(self, stats: List[Dict]) -> None:
        """GPU 통계 업데이트 (GPUMonitor 콜백에서 호출)"""
        if not stats:
            return

        with self._lock:
            s = stats[0]  # 첫 번째 GPU
            self._history["util_gpu"].append(s["util_gpu"])
            self._history["util_mem"].append(s["util_mem"])
            self._history["temp"].append(s["temperature"])
            pwr_pct = s["power_w"] / max(s.get("power_limit_w", 250) or 250, 1) * 100
            self._history["power"].append(min(pwr_pct, 100))

            # 최대 길이 유지
            for k in self._history:
                if len(self._history[k]) > self._max_history:
                    self._history[k] = self._history[k][-self._max_history:]

        try:
            self.frame.after(0, lambda: self._update_cards(stats))
            self.frame.after(0, self._redraw_graph)
        except Exception:
            pass

    def _update_cards(self, stats: List[Dict]) -> None:
        """GPU 카드 업데이트"""
        for i, s in enumerate(stats[:4]):
            if i >= len(self.gpu_cards):
                break
            card = self.gpu_cards[i]
            card["frame"].grid()

            card["util_gpu_var"].set(f"{s['util_gpu']:.0f}%")
            card["util_mem_var"].set(f"{s['util_mem']:.0f}%")
            card["temp_var"].set(f"{s['temperature']:.0f}°C")
            card["power_var"].set(f"{s['power_w']:.0f}W")

            card["util_gpu_pb"]["value"] = s["util_gpu"]
            card["util_mem_pb"]["value"] = s["util_mem"]

            # 온도 게이지 (최대 100°C)
            card["temp_pb"]["value"] = min(s["temperature"], 100)

            # 전력 게이지 (최대 300W)
            card["power_pb"]["value"] = min(s["power_w"] / 300 * 100, 100)

            card["vram_detail"].set(
                f"VRAM: {s['mem_used_mb']:.0f}/{s['mem_total_mb']:.0f} MB "
                f"({s['mem_used_mb']/max(s['mem_total_mb'],1)*100:.0f}%)"
            )
            card["name_var"].set(s["name"][:28])

    def _redraw_graph(self) -> None:
        """롤링 그래프 다시 그리기"""
        canvas = self.graph_canvas
        canvas.delete("all")
        W = canvas.winfo_width()
        H = canvas.winfo_height()
        if W < 50 or H < 50:
            return

        pad = 35
        graph_h = H - 2*pad
        graph_w = W - 2*pad

        # 격자
        for frac in [0, 25, 50, 75, 100]:
            y = pad + graph_h * (1 - frac/100)
            canvas.create_line(pad, y, W-pad, y,
                               fill="#313244", dash=(2, 4))
            canvas.create_text(pad-4, y, text=f"{frac}%",
                               fill="#9399b2", font=("Consolas", 8), anchor="e")

        series = [
            ("util_gpu", "#89b4fa"),
            ("util_mem", "#a6e3a1"),
            ("temp", "#fab387"),
            ("power", "#f9e2af"),
        ]

        with self._lock:
            hist_copy = {k: list(v) for k, v in self._history.items()}

        for key, color in series:
            data = hist_copy.get(key, [])
            n = len(data)
            if n < 2:
                continue

            pts = []
            for i, v in enumerate(data):
                x = pad + graph_w * i / (self._max_history - 1)
                y = pad + graph_h * (1 - min(v, 100)/100)
                pts.extend([x, y])

            if len(pts) >= 4:
                canvas.create_line(*pts, fill=color, width=2, smooth=True)

            # 현재 값 표시
            last = data[-1]
            canvas.create_text(W-pad+4,
                               pad + graph_h * (1 - min(last, 100)/100),
                               text=f"{last:.0f}",
                               fill=color, font=("Consolas", 8), anchor="w")

        # 시간 축
        canvas.create_text(pad, H-pad+12, text="60s 전",
                           fill="#9399b2", font=("Consolas", 8))
        canvas.create_text(W-pad, H-pad+12, text="현재",
                           fill="#9399b2", font=("Consolas", 8))

    def _get_cuda_info(self) -> None:
        """CUDA 정보 비동기 조회"""
        def _do():
            try:
                from utils.gpu_monitor import GPUMonitor
                monitor = GPUMonitor()
                info = monitor.get_cuda_info()
                self.frame.after(0, lambda: self._update_cuda_info(info))
            except Exception:
                pass
        threading.Thread(target=_do, daemon=True).start()

    def _update_cuda_info(self, info: Dict) -> None:
        self.cuda_vars["cuda_available"].set(
            "✅ 사용 가능" if info.get("cuda_available") else "❌ 사용 불가")
        self.cuda_vars["cuda_version"].set(info.get("cuda_version", "N/A"))
        self.cuda_vars["torch_version"].set(info.get("torch_version", "N/A"))
        self.cuda_vars["device_count"].set(str(info.get("device_count", 0)))
        self.cuda_vars["current_device"].set(info.get("current_device", "N/A")[:30])
