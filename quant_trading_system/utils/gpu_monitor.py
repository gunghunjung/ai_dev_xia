# utils/gpu_monitor.py — GPU 사용량 실시간 모니터링
from __future__ import annotations
import threading
import time
from typing import Optional, List, Dict, Callable


class GPUMonitor:
    """
    NVIDIA GPU 사용량 모니터
    - pynvml 기반 (CUDA 드라이버 필요)
    - 폴링 방식: update_interval 초마다 갱신
    - 콜백 등록으로 GUI 연동 가능
    """

    def __init__(self, update_interval: float = 1.0):
        self.update_interval = update_interval
        self._callbacks: List[Callable] = []
        self._thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self._nvml_ok = False
        self._gpu_count = 0
        self._last_stats: List[Dict] = []
        self._lock = threading.Lock()
        self._init_nvml()

    def _init_nvml(self) -> None:
        # pynvml 패키지를 경고 없이 임포트 (nvidia-ml-py도 pynvml로 노출됨)
        try:
            import warnings
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")   # FutureWarning 숨김
                import pynvml
            pynvml.nvmlInit()
            self._gpu_count = pynvml.nvmlDeviceGetCount()
            self._nvml_ok = True
            self._pynvml = pynvml
        except Exception:
            self._nvml_ok = False

    def get_stats(self) -> List[Dict]:
        """
        현재 GPU 통계 반환
        Returns: [{
            'index': int,
            'name': str,
            'util_gpu': float (0~100 %),
            'util_mem': float (0~100 %),
            'mem_used_mb': float,
            'mem_total_mb': float,
            'temperature': float (°C),
            'power_w': float (W),
            'cuda_allocated_mb': float,   # torch.cuda.memory_allocated()
            'cuda_reserved_mb': float,    # torch.cuda.memory_reserved()
        }]
        """
        if not self._nvml_ok:
            return self._dummy_stats()

        stats = []
        try:
            for i in range(self._gpu_count):
                h = self._pynvml.nvmlDeviceGetHandleByIndex(i)
                name = self._pynvml.nvmlDeviceGetName(h)
                if isinstance(name, bytes):
                    name = name.decode()

                util = self._pynvml.nvmlDeviceGetUtilizationRates(h)
                mem = self._pynvml.nvmlDeviceGetMemoryInfo(h)

                try:
                    temp = self._pynvml.nvmlDeviceGetTemperature(
                        h, self._pynvml.NVML_TEMPERATURE_GPU)
                except Exception:
                    temp = 0.0

                try:
                    power = self._pynvml.nvmlDeviceGetPowerUsage(h) / 1000.0
                except Exception:
                    power = 0.0

                # PyTorch CUDA 할당/예약 메모리 (GPU 인덱스 일치 시)
                try:
                    import torch
                    if torch.cuda.is_available() and i < torch.cuda.device_count():
                        cuda_alloc = torch.cuda.memory_allocated(i) / 1024 / 1024
                        cuda_reserv = torch.cuda.memory_reserved(i) / 1024 / 1024
                    else:
                        cuda_alloc = cuda_reserv = 0.0
                except Exception:
                    cuda_alloc = cuda_reserv = 0.0

                stats.append({
                    "index": i,
                    "name": name,
                    "util_gpu": float(util.gpu),
                    "util_mem": float(util.memory),
                    "mem_used_mb": mem.used / 1024 / 1024,
                    "mem_total_mb": mem.total / 1024 / 1024,
                    "temperature": float(temp),
                    "power_w": power,
                    "cuda_allocated_mb": cuda_alloc,
                    "cuda_reserved_mb": cuda_reserv,
                })
        except Exception as e:
            return self._dummy_stats(error=str(e))

        with self._lock:
            self._last_stats = stats
        return stats

    def _dummy_stats(self, error: str = "NVML 사용불가") -> List[Dict]:
        """NVML 없을 때 더미 데이터 반환 (PyTorch CUDA 정보는 여전히 시도)"""
        try:
            import torch
            if torch.cuda.is_available():
                cuda_alloc  = torch.cuda.memory_allocated(0) / 1024 / 1024
                cuda_reserv = torch.cuda.memory_reserved(0)  / 1024 / 1024
                total_mb    = torch.cuda.get_device_properties(0).total_memory / 1024 / 1024
                name        = torch.cuda.get_device_name(0)
            else:
                cuda_alloc = cuda_reserv = total_mb = 0.0
                name = f"GPU 없음 ({error})"
        except Exception:
            cuda_alloc = cuda_reserv = total_mb = 0.0
            name = f"GPU 없음 ({error})"
        return [{
            "index": 0,
            "name": name,
            "util_gpu": 0.0,
            "util_mem": cuda_alloc / max(total_mb, 1) * 100,
            "mem_used_mb": cuda_alloc,
            "mem_total_mb": total_mb,
            "temperature": 0.0,
            "power_w": 0.0,
            "cuda_allocated_mb": cuda_alloc,
            "cuda_reserved_mb": cuda_reserv,
        }]

    def register_callback(self, fn: Callable) -> None:
        """갱신 시 호출할 콜백 등록"""
        self._callbacks.append(fn)

    def start(self) -> None:
        """백그라운드 폴링 시작"""
        if self._thread and self._thread.is_alive():
            return
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._poll_loop, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        """폴링 중단"""
        self._stop_event.set()

    def _poll_loop(self) -> None:
        while not self._stop_event.wait(self.update_interval):
            stats = self.get_stats()
            for cb in self._callbacks:
                try:
                    cb(stats)
                except Exception:
                    pass

    @property
    def available(self) -> bool:
        return self._nvml_ok

    @property
    def gpu_count(self) -> int:
        return self._gpu_count

    def get_torch_device(self) -> str:
        """PyTorch에서 사용할 디바이스 문자열 반환"""
        try:
            import torch
            if torch.cuda.is_available():
                return "cuda"
        except ImportError:
            pass
        return "cpu"

    def get_cuda_info(self) -> Dict:
        """CUDA/PyTorch 정보 반환"""
        info = {
            "cuda_available": False,
            "cuda_version": "N/A",
            "torch_version": "N/A",
            "device_count": 0,
            "current_device": "N/A",
            "devices": [],
        }
        try:
            import torch
            info["cuda_available"] = torch.cuda.is_available()
            info["torch_version"] = torch.__version__
            if torch.cuda.is_available():
                info["cuda_version"] = torch.version.cuda or "N/A"
                info["device_count"] = torch.cuda.device_count()
                info["current_device"] = torch.cuda.get_device_name(0)
                info["devices"] = [
                    torch.cuda.get_device_name(i)
                    for i in range(torch.cuda.device_count())
                ]
        except ImportError:
            pass
        return info
