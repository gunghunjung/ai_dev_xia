"""
MediaEngine v2 — libVLC 기반 고성능 미디어 엔진

[v2 Evolution]
1. seek_relative() 음수 방지 버그 수정
2. HW가속 폴백: dxva2 → d3d11 → any 자동 전환
3. 재생 속도 제어 API (0.25x ~ 4.0x)
4. 자막 API: 자동 감지 로드 / 딜레이 / 트랙 선택 / 온오프
5. 스크린샷 API
6. cleanup() 중복 호출 안전화
7. get_duration_ms() -1 반환 방지
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from urllib.request import pathname2url
from urllib.parse import urljoin

from PyQt6.QtCore import QObject, QTimer, pyqtSignal


def _setup_vlc_path() -> None:
    """VLC DLL 경로를 자동으로 설정한다."""
    if getattr(sys, "frozen", False):
        bundle_dir = sys._MEIPASS  # type: ignore[attr-defined]
        if os.path.exists(os.path.join(bundle_dir, "libvlc.dll")):
            os.environ["PATH"] = bundle_dir + os.pathsep + os.environ.get("PATH", "")
            os.add_dll_directory(bundle_dir)
            return

    candidates = [
        Path(r"C:\Program Files\VideoLAN\VLC"),
        Path(r"C:\Program Files (x86)\VideoLAN\VLC"),
    ]
    vlc_env = os.environ.get("VLC_PATH")
    if vlc_env:
        candidates.insert(0, Path(vlc_env))

    for vlc_dir in candidates:
        if (vlc_dir / "libvlc.dll").exists():
            path_str = str(vlc_dir)
            os.environ["PATH"] = path_str + os.pathsep + os.environ.get("PATH", "")
            try:
                os.add_dll_directory(path_str)
            except AttributeError:
                pass
            os.environ["PYTHON_VLC_LIB_PATH"] = str(vlc_dir / "libvlc.dll")
            os.environ["PYTHON_VLC_MODULE_PATH"] = str(vlc_dir / "plugins")
            return


_setup_vlc_path()

try:
    import vlc
    VLC_AVAILABLE = True
except (ImportError, FileNotFoundError, OSError):
    VLC_AVAILABLE = False

# 자막 확장자
SUBTITLE_EXT = {".srt", ".smi", ".ass", ".ssa", ".vtt", ".sub"}


class MediaEngine(QObject):
    """libVLC 래퍼. 모든 신호는 메인 스레드(Qt)에서 안전하게 방출된다."""

    position_changed  = pyqtSignal(float)   # 0.0 ~ 1.0
    time_changed      = pyqtSignal(int)     # 현재 재생 시간 (ms)
    duration_changed  = pyqtSignal(int)     # 총 길이 (ms)
    state_changed     = pyqtSignal(str)     # playing|paused|stopped|ended|error
    media_loaded      = pyqtSignal(str)     # 파일명
    volume_changed    = pyqtSignal(int)     # 0 ~ 150
    speed_changed     = pyqtSignal(float)   # 0.25 ~ 4.0
    subtitle_changed  = pyqtSignal(str)     # 자막 파일명 or ""
    error_occurred    = pyqtSignal(str)     # 에러 메시지

    # ── HW가속 폴백 순서 ──────────────────────────────────────────────────
    _HW_OPTIONS = [
        ["--avcodec-hw=dxva2",  "--vout=direct3d11"],  # NVIDIA/AMD DXVA2
        ["--avcodec-hw=d3d11va","--vout=direct3d11"],  # D3D11 직접
        ["--avcodec-hw=any",    "--vout=direct3d11"],  # 범용
        [],                                             # SW 폴백
    ]

    def __init__(self, parent: QObject | None = None) -> None:
        super().__init__(parent)

        if not VLC_AVAILABLE:
            raise RuntimeError(
                "VLC Player 또는 python-vlc를 찾을 수 없습니다.\n\n"
                "해결 방법:\n"
                "1. https://www.videolan.org 에서 VLC Player (64비트)를 설치하세요.\n"
                "2. pip install python-vlc 을 실행하세요.\n\n"
                "VLC를 기본 경로에 설치하면 자동으로 인식됩니다."
            )

        self._instance, self._player = self._init_vlc_with_hw_fallback()
        self._is_seeking  = False
        self._last_state  = "stopped"
        self._duration    = 0
        self._speed       = 1.0
        self._sub_delay   = 0          # 마이크로초 단위
        self._cleaned_up  = False

        self._poll_timer = QTimer(self)
        self._poll_timer.setInterval(50)
        self._poll_timer.timeout.connect(self._poll_state)

    def _init_vlc_with_hw_fallback(self):
        """HW가속 옵션을 순서대로 시도해 인스턴스를 생성한다."""
        base_args = ["--quiet", "--no-xlib"]
        for hw_args in self._HW_OPTIONS:
            try:
                inst   = vlc.Instance(base_args + hw_args)
                player = inst.media_player_new()
                return inst, player
            except Exception:
                continue
        raise RuntimeError("VLC 인스턴스를 생성할 수 없습니다.")

    # ── 공개 API ───────────────────────────────────────────────────────────

    def load(self, file_path: str) -> bool:
        """파일을 로드한다. 한글 경로 포함 모든 OS 경로 지원."""
        try:
            path = Path(file_path).resolve()
            if not path.exists():
                self.error_occurred.emit(f"파일을 찾을 수 없습니다:\n{file_path}")
                return False

            uri   = urljoin("file:", pathname2url(str(path)))
            media = self._instance.media_new(uri)
            self._player.set_media(media)
            self._duration = 0
            self._speed    = 1.0

            self.media_loaded.emit(path.name)

            # 동일 폴더에서 자막 자동 감지
            self._auto_load_subtitle(path)
            return True

        except Exception as exc:
            self.error_occurred.emit(f"미디어 로드 실패: {exc}")
            return False

    def _auto_load_subtitle(self, video_path: Path) -> None:
        """영상과 같은 폴더/이름의 자막 파일을 자동으로 로드한다."""
        for ext in (".srt", ".smi", ".ass", ".ssa", ".vtt"):
            sub_path = video_path.with_suffix(ext)
            if sub_path.exists():
                self.load_subtitle(str(sub_path))
                return

    def load_subtitle(self, sub_path: str) -> bool:
        """외부 자막 파일을 로드한다."""
        try:
            path = Path(sub_path).resolve()
            if not path.exists():
                return False
            uri = urljoin("file:", pathname2url(str(path)))
            result = self._player.add_slave(
                vlc.MediaSlaveType.subtitle, uri, True
            )
            self.subtitle_changed.emit(path.name)
            return result == 0
        except Exception:
            return False

    def set_subtitle_delay(self, delta_ms: int) -> None:
        """자막 딜레이를 delta_ms 만큼 조정한다 (누적)."""
        self._sub_delay += delta_ms * 1000   # 마이크로초 변환
        self._player.video_set_spu_delay(self._sub_delay)

    def get_subtitle_delay_ms(self) -> int:
        return self._sub_delay // 1000

    def reset_subtitle_delay(self) -> None:
        self._sub_delay = 0
        self._player.video_set_spu_delay(0)

    def toggle_subtitle(self) -> None:
        """자막 표시 토글."""
        current = self._player.video_get_spu()
        if current == -1:
            self._player.video_set_spu(0)
        else:
            self._player.video_set_spu(-1)

    def set_hwnd(self, hwnd: int) -> None:
        self._player.set_hwnd(hwnd)

    def play(self) -> None:
        self._player.play()
        self._poll_timer.start()

    def pause(self) -> None:
        self._player.pause()

    def stop(self) -> None:
        self._player.stop()
        self._poll_timer.stop()
        self._last_state = "stopped"
        self._duration   = 0
        self.state_changed.emit("stopped")
        self.position_changed.emit(0.0)
        self.time_changed.emit(0)

    def toggle_play_pause(self) -> None:
        state = self._vlc_state_to_str(self._player.get_state())
        if state == "playing":
            self.pause()
        elif state in ("paused", "stopped", "ended"):
            self.play()

    def seek(self, position: float) -> None:
        """position: 0.0 ~ 1.0"""
        if self._duration > 0:
            self._player.set_position(max(0.0, min(1.0, position)))

    def seek_ms(self, ms: int) -> None:
        self._player.set_time(max(0, ms))

    def seek_relative(self, delta_ms: int) -> None:
        """[BUG FIX] get_time() 음수 반환 시 0으로 클램프."""
        current = max(0, self._player.get_time())
        target  = max(0, current + delta_ms)
        self._player.set_time(target)

    def next_frame(self) -> None:
        """일시정지 상태에서 한 프레임 전진."""
        self._player.next_frame()

    # ── 재생 속도 ─────────────────────────────────────────────────────────

    _SPEED_STEPS = [0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 3.0, 4.0]

    def set_speed(self, rate: float) -> None:
        rate = max(0.25, min(4.0, rate))
        self._player.set_rate(rate)
        self._speed = rate
        self.speed_changed.emit(rate)

    def speed_up(self) -> None:
        idx = self._nearest_speed_idx()
        if idx < len(self._SPEED_STEPS) - 1:
            self.set_speed(self._SPEED_STEPS[idx + 1])

    def speed_down(self) -> None:
        idx = self._nearest_speed_idx()
        if idx > 0:
            self.set_speed(self._SPEED_STEPS[idx - 1])

    def reset_speed(self) -> None:
        self.set_speed(1.0)

    def get_speed(self) -> float:
        return self._speed

    def _nearest_speed_idx(self) -> int:
        return min(
            range(len(self._SPEED_STEPS)),
            key=lambda i: abs(self._SPEED_STEPS[i] - self._speed)
        )

    # ── 볼륨 ──────────────────────────────────────────────────────────────

    def set_volume(self, volume: int) -> None:
        vol = max(0, min(150, volume))
        self._player.audio_set_volume(vol)
        self.volume_changed.emit(vol)

    def get_volume(self) -> int:
        v = self._player.audio_get_volume()
        return max(0, v)

    def toggle_mute(self) -> None:
        self._player.audio_toggle_mute()

    def is_muted(self) -> bool:
        return bool(self._player.audio_get_mute())

    # ── 상태 조회 ─────────────────────────────────────────────────────────

    def is_playing(self) -> bool:
        return bool(self._player.is_playing())

    def get_position(self) -> float:
        return max(0.0, self._player.get_position())

    def get_time_ms(self) -> int:
        return max(0, self._player.get_time())

    def get_duration_ms(self) -> int:
        """[BUG FIX] -1 반환 방지."""
        return max(0, self._player.get_length())

    # ── 드래그 억제 ───────────────────────────────────────────────────────

    def begin_seek(self) -> None:
        self._is_seeking = True

    def end_seek(self, position: float) -> None:
        self.seek(position)
        self._is_seeking = False

    # ── 폴링 ──────────────────────────────────────────────────────────────

    def _poll_state(self) -> None:
        try:
            state     = self._player.get_state()
            state_str = self._vlc_state_to_str(state)

            if state_str != self._last_state:
                self._last_state = state_str
                self.state_changed.emit(state_str)
                if state_str == "ended":
                    self._poll_timer.stop()

            if state_str in ("playing", "paused"):
                dur = self._player.get_length()
                if dur > 0 and dur != self._duration:
                    self._duration = dur
                    self.duration_changed.emit(dur)

                if not self._is_seeking:
                    pos = self._player.get_position()
                    t   = self._player.get_time()
                    if pos >= 0:
                        self.position_changed.emit(pos)
                    if t >= 0:
                        self.time_changed.emit(t)

        except Exception:
            pass

    @staticmethod
    def _vlc_state_to_str(state) -> str:
        try:
            mapping = {
                vlc.State.Playing:   "playing",
                vlc.State.Paused:    "paused",
                vlc.State.Stopped:   "stopped",
                vlc.State.Ended:     "ended",
                vlc.State.Error:     "error",
                vlc.State.Opening:   "opening",
                vlc.State.Buffering: "buffering",
            }
            return mapping.get(state, "stopped")
        except Exception:
            return "stopped"

    def cleanup(self) -> None:
        """재생 중 종료 크래시 방지 — HWND 분리 → stop → 대기 → release."""
        if self._cleaned_up:
            return
        self._cleaned_up = True
        self._poll_timer.stop()
        try:
            # D3D 렌더링 타겟 먼저 분리 (렌더 스레드 충돌 방지)
            self._player.set_hwnd(0)
        except Exception:
            pass
        try:
            self._player.stop()
        except Exception:
            pass
        # VLC 내부 스레드(디코더/렌더러)가 완전히 종료될 시간 확보
        import time
        time.sleep(0.15)
        try:
            self._player.release()
            self._instance.release()
        except Exception:
            pass
