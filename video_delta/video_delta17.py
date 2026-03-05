import os
import time
import json
import threading
from dataclasses import dataclass
from typing import Optional, Tuple, List, Callable, Dict

import cv2
import numpy as np

import tkinter as tk
from tkinter import ttk, filedialog, messagebox

from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

# ---------------- Matplotlib font fix (Korean glyph warnings) ----------------
try:
    from matplotlib import font_manager, rcParams
    _CANDIDATES = ["Malgun Gothic", "AppleGothic", "NanumGothic", "Noto Sans CJK KR", "Noto Sans KR"]
    _found = None
    for _name in _CANDIDATES:
        try:
            font_manager.findfont(_name, fallback_to_default=False)
            _found = _name
            break
        except Exception:
            pass
    if _found:
        rcParams["font.family"] = _found
        rcParams["axes.unicode_minus"] = False
except Exception:
    pass

# ---------------- Panda3D ----------------
try:
    from direct.showbase.ShowBase import ShowBase
    from panda3d.core import (
        Geom, GeomNode, GeomVertexData, GeomVertexFormat, GeomVertexWriter,
        GeomTriangles, NodePath, Vec4,
        DirectionalLight, AmbientLight,
    )
    PANDA3D_OK = True
except Exception:
    PANDA3D_OK = False

# OpenGL support flag (not implemented in this script)
OPENGL_OK = False


@dataclass
class ROI:
    x1: int = 0
    y1: int = 0
    x2: int = 0
    y2: int = 0

    def normalized(self) -> "ROI":
        xa, xb = sorted([self.x1, self.x2])
        ya, yb = sorted([self.y1, self.y2])
        return ROI(xa, ya, xb, yb)

    def is_valid(self) -> bool:
        r = self.normalized()
        return (r.x2 - r.x1) >= 5 and (r.y2 - r.y1) >= 5

    def to_dict(self) -> dict:
        return dict(x1=int(self.x1), y1=int(self.y1), x2=int(self.x2), y2=int(self.y2))

    @staticmethod
    def from_dict(d: dict) -> Optional["ROI"]:
        if not isinstance(d, dict):
            return None
        try:
            return ROI(int(d.get("x1", 0)), int(d.get("y1", 0)), int(d.get("x2", 0)), int(d.get("y2", 0)))
        except Exception:
            return None


@dataclass
class DrawRect:
    x: int
    y: int
    w: int
    h: int


# ---------------- Scrollable Frame (Tune panel) ----------------
class ScrollableFrame(ttk.Frame):
    def __init__(self, master, height=260, *args, **kwargs):
        super().__init__(master, *args, **kwargs)
        self.canvas = tk.Canvas(self, borderwidth=0, highlightthickness=0, height=height)
        self.vsb = ttk.Scrollbar(self, orient="vertical", command=self.canvas.yview)
        self.inner = ttk.Frame(self.canvas)

        self.inner_id = self.canvas.create_window((0, 0), window=self.inner, anchor="nw")
        self.canvas.configure(yscrollcommand=self.vsb.set)

        self.canvas.pack(side="left", fill="both", expand=True)
        self.vsb.pack(side="right", fill="y")

        self.inner.bind("<Configure>", self._on_frame_configure)
        self.canvas.bind("<Configure>", self._on_canvas_configure)

        # mouse wheel support
        self.canvas.bind_all("<MouseWheel>", self._on_mousewheel, add="+")
        self.canvas.bind_all("<Button-4>", self._on_mousewheel_linux_up, add="+")
        self.canvas.bind_all("<Button-5>", self._on_mousewheel_linux_dn, add="+")

    def _on_frame_configure(self, _e):
        self.canvas.configure(scrollregion=self.canvas.bbox("all"))

    def _on_canvas_configure(self, _e):
        try:
            self.canvas.itemconfig(self.inner_id, width=self.canvas.winfo_width())
        except Exception:
            pass

    def _on_mousewheel(self, e):
        if self.winfo_containing(e.x_root, e.y_root) is None:
            return
        self.canvas.yview_scroll(int(-1 * (e.delta / 120)), "units")

    def _on_mousewheel_linux_up(self, e):
        if self.winfo_containing(e.x_root, e.y_root) is None:
            return
        self.canvas.yview_scroll(-1, "units")

    def _on_mousewheel_linux_dn(self, e):
        if self.winfo_containing(e.x_root, e.y_root) is None:
            return
        self.canvas.yview_scroll(+1, "units")


# ---------------- Collapsible (Accordion) section ----------------
class Collapsible(ttk.Frame):
    def __init__(self, master, title="Advanced", expanded=False, *args, **kwargs):
        super().__init__(master, *args, **kwargs)
        self._expanded = bool(expanded)

        hdr = ttk.Frame(self)
        hdr.pack(fill="x")
        self.btn = ttk.Button(hdr, text=("▼ " if self._expanded else "► ") + title, command=self.toggle, width=20)
        self.btn.pack(side="left")
        self.body = ttk.Frame(self)
        if self._expanded:
            self.body.pack(fill="x", pady=(4, 0))

        self._title = title

    def toggle(self):
        self._expanded = not self._expanded
        if self._expanded:
            self.body.pack(fill="x", pady=(4, 0))
            self.btn.configure(text="▼ " + self._title)
        else:
            self.body.forget()
            self.btn.configure(text="► " + self._title)


# ---------------- OpenGL Heightmap Viewer ----------------
if OPENGL_OK:
    pass

if PANDA3D_OK:
    class HeightmapPanda3D(ShowBase):
        def __init__(self):
            # Ensure ShowBase does not create a default window; we'll open a parented window later.
            from panda3d.core import loadPrcFileData
            loadPrcFileData("", "window-type none")
            loadPrcFileData("", "audio-library-name null")
            loadPrcFileData("", "show-frame-rate-meter 0")

            ShowBase.__init__(self)
            try:
                self.disableMouse()
            except Exception:
                pass

            self.gray = None
            self.title = "Panda3D Heightmap"

            # camera state
            self.rot_x = -88.0
            self.rot_z = -90.0
            self.pan_x = 0.0
            self.pan_y = 0.0
            self.dist = 3.2

            self.zoom_mode = "scale"   # "scale" or "dist"
            self.obj_scale = 1.2
            self.z_scale = 1.0
            self.invert_lr = False
            self.invert_ud = False

            self._cam_presets = [
                dict(name="Top",       rot_x=-88.0, rot_z=-90.0, dist=3.2, obj_scale=1.2, z_scale=1.0, pan_x=0.0, pan_y=0.0),
                dict(name="SideRight", rot_x=-25.0, rot_z=-95.0, dist=3.2, obj_scale=1.2, z_scale=1.0, pan_x=0.0, pan_y=0.0),
                dict(name="Iso",       rot_x=-55.0, rot_z=-50.0, dist=3.2, obj_scale=1.2, z_scale=1.0, pan_x=0.0, pan_y=0.0),
            ]
            self._cam_idx = 0

            self.on_camera_changed = None

            self._geom_np = None
            self._last_shape = None
            self._last_hash = None

            # pending update (thread-safe)
            self._pending_lock = threading.Lock()
            self._pending_gray = None
            self._pending_title = ""

            # lighting
            dlight = DirectionalLight('dlight')
            dlight.setColor(Vec4(0.8, 0.8, 0.8, 1))
            dlnp = self.render.attachNewNode(dlight)
            dlnp.setHpr(0, -60, 0)
            self.render.setLight(dlnp)

            alight = AmbientLight('alight')
            alight.setColor(Vec4(0.25, 0.25, 0.25, 1))
            alnp = self.render.attachNewNode(alight)
            self.render.setLight(alnp)

            # mouse input (optional)
            self.accept('wheel_up',   self._zoom, [1])
            self.accept('wheel_down', self._zoom, [-1])
            self.accept('mouse1',     self._start_drag)
            self.accept('mouse1-up',  self._end_drag)
            self.accept('mouse3',     self._start_pan)
            self.accept('mouse3-up',  self._end_pan)

            self._dragging = False
            self._panning = False
            self._last_mouse = None

            self._update_camera()

            # tasks
            self.taskMgr.add(self._task_mouse, 'mouse_task')
            self.taskMgr.add(self._task_apply_pending, 'apply_pending_task')

        # -------- public API called from Tk thread --------
        def set_heightmap(self, gray, title: str = ""):
            with self._pending_lock:
                self._pending_gray = None if gray is None else np.array(gray, copy=True)
                self._pending_title = title or self._pending_title

        def cycle_camera(self):
            self.set_camera_preset(self._cam_idx + 1)

        def set_camera_preset(self, idx: int):
            idx = int(idx) % len(self._cam_presets)
            self._cam_idx = idx
            p = self._cam_presets[idx]
            self.rot_x = float(p["rot_x"])
            self.rot_z = float(p["rot_z"])
            self.dist = float(p["dist"])
            self.obj_scale = float(p["obj_scale"])
            self.z_scale = float(p["z_scale"])
            self.pan_x = float(p["pan_x"])
            self.pan_y = float(p["pan_y"])
            self._update_camera()
            self._notify_cam()

        def get_camera_preset_name(self) -> str:
            return str(self._cam_presets[self._cam_idx].get("name", f"Preset{self._cam_idx}"))

        def get_state(self) -> dict:
            return dict(
                cam_idx=self._cam_idx,
                rot_x=self.rot_x, rot_z=self.rot_z,
                dist=self.dist,
                obj_scale=self.obj_scale, z_scale=self.z_scale,
                pan_x=self.pan_x, pan_y=self.pan_y,
                zoom_mode=self.zoom_mode,
                invert_lr=self.invert_lr, invert_ud=self.invert_ud,
            )

        def set_state(self, s: dict):
            self._cam_idx = int(s.get("cam_idx", self._cam_idx)) % len(self._cam_presets)
            self.rot_x = float(s.get("rot_x", self.rot_x))
            self.rot_z = float(s.get("rot_z", self.rot_z))
            self.dist = float(s.get("dist", self.dist))
            self.obj_scale = float(s.get("obj_scale", self.obj_scale))
            self.z_scale = float(s.get("z_scale", self.z_scale))
            self.pan_x = float(s.get("pan_x", self.pan_x))
            self.pan_y = float(s.get("pan_y", self.pan_y))
            self.zoom_mode = str(s.get("zoom_mode", self.zoom_mode))
            self.invert_lr = bool(s.get("invert_lr", self.invert_lr))
            self.invert_ud = bool(s.get("invert_ud", self.invert_ud))
            self._update_camera()
            self._notify_cam()

        # -------- internal --------
        def _notify_cam(self):
            cb = self.on_camera_changed
            if cb:
                try:
                    cb()
                except Exception:
                    pass

        def _hash_gray(self, g: np.ndarray) -> int:
            if g is None or g.size == 0:
                return 0
            h, w = g.shape[:2]
            pts = [
                int(g[0, 0]), int(g[0, w // 2]), int(g[0, w - 1]),
                int(g[h // 2, 0]), int(g[h // 2, w // 2]), int(g[h // 2, w - 1]),
                int(g[h - 1, 0]), int(g[h - 1, w // 2]), int(g[h - 1, w - 1]),
            ]
            return int(h * 1000003 + w * 9176 + sum(pts))

        def _colormap_fast(self, z01: np.ndarray) -> np.ndarray:
            z = np.clip(z01, 0.0, 1.0).astype(np.float32)
            r = np.clip(1.5 * z - 0.2, 0, 1)
            g = np.clip(1.5 * (1 - np.abs(z - 0.5) * 2.0), 0, 1)
            b = np.clip(1.1 - 1.6 * z, 0, 1)
            return np.stack([r, g, b], axis=-1).astype(np.float32)

        def _task_apply_pending(self, task):
            # runs in Panda3D thread
            with self._pending_lock:
                g = self._pending_gray
                title = self._pending_title
                self._pending_gray = None

            if g is not None:
                self.title = title or self.title
                self._rebuild_mesh(g)
            else:
                # title-only change, don't rebuild mesh (avoid flicker when no gray provided)
                if title:
                    self.title = title or self.title
            return task.cont

        def _rebuild_mesh(self, gray: np.ndarray):
            # If no data provided, keep existing mesh (avoid flicker). Only remove when new data provided.
            if gray is None or gray.size == 0:
                return

            if gray.ndim != 2:
                gray = gray[:, :]

            h, w = gray.shape[:2]
            hsh = self._hash_gray(gray)

            # If identical data and a mesh already exists, only update scale and skip rebuild
            if self._last_shape == (h, w) and self._last_hash == hsh and self._geom_np is not None:
                try:
                    self._geom_np.setScale(self.obj_scale, self.obj_scale, self.z_scale)
                except Exception:
                    pass
                return

            # Otherwise remove old mesh and rebuild
            if self._geom_np:
                try:
                    self._geom_np.removeNode()
                except Exception:
                    pass
                self._geom_np = None

            self._last_shape = (h, w)
            self._last_hash = hsh

            z = gray.astype(np.float32) / 255.0
            xs = np.linspace(-1.0, 1.0, w, dtype=np.float32)
            ys = np.linspace(-1.0, 1.0, h, dtype=np.float32)
            xx, yy = np.meshgrid(xs, ys)
            zz = (z - 0.5) * 2.0

            verts = np.stack([xx, -yy, zz], axis=-1).reshape(-1, 3).astype(np.float32)
            cols = self._colormap_fast(z).reshape(-1, 3).astype(np.float32)

            vdata = GeomVertexData('heightmap', GeomVertexFormat.getV3c4(), Geom.UHStatic)
            vtxw = GeomVertexWriter(vdata, 'vertex')
            colw = GeomVertexWriter(vdata, 'color')

            for v, c in zip(verts, cols):
                vtxw.addData3f(float(v[0]), float(v[1]), float(v[2]))
                colw.addData4f(float(c[0]), float(c[1]), float(c[2]), 1.0)

            prim = GeomTriangles(Geom.UHStatic)

            y = np.arange(h - 1, dtype=np.int32)[:, None]
            x = np.arange(w - 1, dtype=np.int32)[None, :]
            v00 = y * w + x
            v01 = v00 + 1
            v10 = v00 + w
            v11 = v10 + 1
            t1 = np.stack([v00, v10, v11], axis=-1).reshape(-1, 3)
            t2 = np.stack([v00, v11, v01], axis=-1).reshape(-1, 3)
            idx = np.concatenate([t1, t2], axis=0)

            for a, b, c in idx:
                prim.addVertices(int(a), int(b), int(c))

            geom = Geom(vdata)
            geom.addPrimitive(prim)

            node = GeomNode('heightmap_node')
            node.addGeom(geom)
            self._geom_np = self.render.attachNewNode(node)
            self._geom_np.setScale(self.obj_scale, self.obj_scale, self.z_scale)

        # ---- mouse control ----
        def _start_drag(self):
            if self.mouseWatcherNode.hasMouse():
                self._dragging = True
                self._last_mouse = self.mouseWatcherNode.getMouse()

        def _end_drag(self):
            self._dragging = False
            self._last_mouse = None

        def _start_pan(self):
            if self.mouseWatcherNode.hasMouse():
                self._panning = True
                self._last_mouse = self.mouseWatcherNode.getMouse()

        def _end_pan(self):
            self._panning = False
            self._last_mouse = None

        def _zoom(self, direction):
            if self.zoom_mode == "dist":
                self.dist = max(0.6, min(25.0, self.dist / 1.08 if direction > 0 else self.dist * 1.08))
            else:
                self.obj_scale = max(0.25, min(10.0, self.obj_scale * 1.08 if direction > 0 else self.obj_scale / 1.08))
            self._update_camera()

        def _task_mouse(self, task):
            if not self.mouseWatcherNode.hasMouse():
                return task.cont

            mouse = self.mouseWatcherNode.getMouse()
            if self._last_mouse is None:
                self._last_mouse = mouse
                return task.cont

            if self._dragging:
                dx = mouse.getX() - self._last_mouse.getX()
                dy = mouse.getY() - self._last_mouse.getY()
                sx = -1.0 if self.invert_lr else 1.0
                sy = -1.0 if self.invert_ud else 1.0
                self.rot_z += sx * dx * 50.0
                self.rot_x += sy * dy * 50.0
                self.rot_x = max(-89.5, min(89.5, self.rot_x))
                self._update_camera()
                self._notify_cam()

            elif self._panning:
                dx = mouse.getX() - self._last_mouse.getX()
                dy = mouse.getY() - self._last_mouse.getY()
                sx = -1.0 if self.invert_lr else 1.0
                sy = -1.0 if self.invert_ud else 1.0
                zref = self.dist if self.zoom_mode == "dist" else max(0.8, 2.5 / max(0.2, self.obj_scale))
                s = 0.0035 * zref / max(0.2, self.z_scale)
                self.pan_x += sx * (-dx) * s
                self.pan_y += sy * (+dy) * s
                self._update_camera()
                self._notify_cam()

            self._last_mouse = mouse
            return task.cont

        def _update_camera(self):
            if getattr(self, 'camera', None) is None:
                return
            try:
                self.camera.setPos(self.pan_x, -self.dist, self.pan_y)
                self.camera.setHpr(self.rot_z, self.rot_x, 0)
            except Exception:
                pass


class VideoDeltaROIApp(tk.Tk):
    def __init__(self):
        super().__init__()
        # Debug: print runtime info to ensure we're running the edited file/environment
        try:
            import os, sys, traceback
            print("RUN:", os.path.abspath(__file__))
            print("CWD:", os.getcwd())
            print("PY :", sys.executable)

            def _tk_exc(exc, val, tb):
                traceback.print_exception(exc, val, tb)
            # Bind Tk's exception reporter to print full tracebacks to console
            self.report_callback_exception = _tk_exc.__get__(self, VideoDeltaROIApp)
        except Exception:
            pass
        self.title("Video + 2ROI Delta + Symmetry + OpenGL 3D (Tune: Smooth during playback)")
        self.geometry("1920x1080")

        self._alive = True
        self.data_lock = threading.Lock()

        # Video
        self.cap: Optional[cv2.VideoCapture] = None
        self.video_path: Optional[str] = None
        self.fps: float = 30.0
        self.frame_count: int = 0
        self.duration_sec: float = 0.0
        self.frame_idx: int = 0
        self.pos_msec: float = 0.0

        self.playing = False
        self.stop_flag = False
        self.worker: Optional[threading.Thread] = None
        self.cap_lock = threading.Lock()

        # Frames
        self.cur_bgr_raw: Optional[np.ndarray] = None
        self.cur_bgr_proc: Optional[np.ndarray] = None

        # Temporal EMA
        self.temporal_enabled_var = tk.BooleanVar(value=False)
        self.temporal_alpha = 0.20
        self._ema_prev_f32: Optional[np.ndarray] = None

        # ROI
        self.rois: List[Optional[ROI]] = [None, None]
        self.roi_enabled: List[bool] = [True, True]
        self.active_roi_idx: int = 0
        self.roi_dragging = False

        # Canvas (video)
        self.canvas_w = 860
        self.canvas_h = 484
        self.draw_rect: Optional[DrawRect] = None

        # OpenGL view size
        self.gl_w = 520
        self.gl_h = 300

        # Delta series
        self.delta_times: List[List[float]] = [[], []]
        self.delta_vals: List[List[float]] = [[], []]
        self.prev_roi_gray: List[Optional[np.ndarray]] = [None, None]

        # Symmetry series
        self.sym_times: List[float] = []
        self.sym_vals: List[float] = []

        # Metric
        self.metric_var = tk.StringVar(value="MeanAbsDiff")
        self.thresh_var = tk.IntVar(value=20)
        self.metrics: Dict[str, Callable[[np.ndarray, np.ndarray], float]] = {}
        self._init_metrics()

        # Seek
        self.seek_var = tk.DoubleVar(value=0.0)
        self.user_seeking = False

        # 3D throttle
        self._gl_tick = 0
        self._gl_every_n = 12
        self._gl_max_size = 40

        # Plot update throttle
        self._plot_update_counter = 0
        self._plot_update_every = 5  # Update plot every 5 frames during playback

        # Data size limit
        self._max_data_points = 10000  # Limit to prevent memory overflow
        self.roi_colors = ["#00ff00", "#00aaff"]

        # ---------------- Per-Graph tune vars ----------------
        self.rise1_var = tk.DoubleVar(value=2.0)
        self.fall1_var = tk.DoubleVar(value=2.0)
        self.min_on1_var = tk.IntVar(value=1)
        self.min_off1_var = tk.IntVar(value=1)
        self.spike1_var = tk.DoubleVar(value=3.0)

        self.rise2_var = tk.DoubleVar(value=2.0)
        self.fall2_var = tk.DoubleVar(value=2.0)
        self.min_on2_var = tk.IntVar(value=1)
        self.min_off2_var = tk.IntVar(value=1)
        self.spike2_var = tk.DoubleVar(value=3.0)

        self.rise3_var = tk.DoubleVar(value=2.0)
        self.fall3_var = tk.DoubleVar(value=2.0)
        self.min_on3_var = tk.IntVar(value=1)
        self.min_off3_var = tk.IntVar(value=1)
        self.spike3_var = tk.DoubleVar(value=3.0)

        self._max_spike_labels = 12

        # OpenGL UI state
        self.invert_lr_var = tk.BooleanVar(value=False)
        self.invert_ud_var = tk.BooleanVar(value=False)
        self.zoom_mode_var = tk.StringVar(value="scale")
        # backend selection (kept for state persistence; single backend now)
        self.backend_var = tk.StringVar(value="panda3d")

        # debounce
        self._tune_debounce_ms = 110
        self._tune_after_id = None
        self._tune_dirty = False

        script_dir = os.path.dirname(os.path.abspath(__file__))
        self._save_path = os.path.join(script_dir, "camera_state.json")

        self._build_ui()
        self._bind_events()

        self.after(250, lambda: self._load_all_state(show_error=False))
        self.after(250, self._ensure_focus)

    # ---------------- Metrics ----------------
    def _init_metrics(self):
        self.metrics = {
            "MeanAbsDiff": self._m_mean_abs,
            "SumAbsDiff": self._m_sum_abs,
            "RMSDiff": self._m_rms,
            "ThresholdPixelRatio": self._m_thresh_ratio,
            "EdgeMeanAbsDiff": self._m_edge_mean_abs,
            "SSIMLike(1-ssim)": self._m_ssim_like,
        }

    @staticmethod
    def _m_mean_abs(a: np.ndarray, b: np.ndarray) -> float:
        return float(np.mean(cv2.absdiff(a, b)))

    @staticmethod
    def _m_sum_abs(a: np.ndarray, b: np.ndarray) -> float:
        return float(np.sum(cv2.absdiff(a, b)))

    @staticmethod
    def _m_rms(a: np.ndarray, b: np.ndarray) -> float:
        diff = cv2.absdiff(a, b).astype(np.float32)
        return float(np.sqrt(np.mean(diff * diff)))

    def _m_thresh_ratio(self, a: np.ndarray, b: np.ndarray) -> float:
        thr = int(self.thresh_var.get())
        diff = cv2.absdiff(a, b)
        return float(np.mean(diff >= thr))

    @staticmethod
    def _m_edge_mean_abs(a: np.ndarray, b: np.ndarray) -> float:
        def edge(img):
            gx = cv2.Sobel(img, cv2.CV_32F, 1, 0, ksize=3)
            gy = cv2.Sobel(img, cv2.CV_32F, 0, 1, ksize=3)
            return cv2.magnitude(gx, gy)
        ea, eb = edge(a), edge(b)
        return float(np.mean(np.abs(ea - eb)))

    @staticmethod
    def _m_ssim_like(a: np.ndarray, b: np.ndarray) -> float:
        a = a.astype(np.float32)
        b = b.astype(np.float32)
        mu_a = cv2.GaussianBlur(a, (7, 7), 1.5)
        mu_b = cv2.GaussianBlur(b, (7, 7), 1.5)
        mu_a2 = mu_a * mu_a
        mu_b2 = mu_b * mu_b
        mu_ab = mu_a * mu_b
        sigma_a2 = cv2.GaussianBlur(a * a, (7, 7), 1.5) - mu_a2
        sigma_b2 = cv2.GaussianBlur(b * b, (7, 7), 1.5) - mu_b2
        sigma_ab = cv2.GaussianBlur(a * b, (7, 7), 1.5) - mu_ab
        C1 = (0.01 * 255) ** 2
        C2 = (0.03 * 255) ** 2
        num = (2 * mu_ab + C1) * (2 * sigma_ab + C2)
        den = (mu_a2 + mu_b2 + C1) * (sigma_a2 + sigma_b2 + C2)
        ssim = float(np.mean(num / (den + 1e-6)))
        return float(max(0.0, 1.0 - ssim))

    # ---------------- Tune debounce ----------------
    def _schedule_tune_refresh(self):
        self._tune_dirty = True
        if self._tune_after_id is not None:
            try:
                self.after_cancel(self._tune_after_id)
            except Exception:
                pass
            self._tune_after_id = None
        self._tune_after_id = self.after(self._tune_debounce_ms, self._apply_tune_refresh)

    def _apply_tune_refresh(self):
        self._tune_after_id = None
        if not self._tune_dirty:
            return
        self._tune_dirty = False
        self._refresh_plot(force_static_x=True)

    # ---------------- Presets ----------------
    def _apply_preset(self, which: int, preset: str):
        preset = preset.lower().strip()
        if preset == "sensitive":
            rise, fall, min_on, min_off, spike = 1.2, 1.2, 1, 1, 2.2
        elif preset == "robust":
            rise, fall, min_on, min_off, spike = 3.5, 3.5, 2, 3, 5.0
        else:
            rise, fall, min_on, min_off, spike = 2.0, 2.0, 1, 1, 3.0

        if which == 1:
            self.rise1_var.set(rise); self.fall1_var.set(fall)
            self.min_on1_var.set(min_on); self.min_off1_var.set(min_off)
            self.spike1_var.set(spike)
        elif which == 2:
            self.rise2_var.set(rise); self.fall2_var.set(fall)
            self.min_on2_var.set(min_on); self.min_off2_var.set(min_off)
            self.spike2_var.set(spike)
        else:
            self.rise3_var.set(rise); self.fall3_var.set(fall)
            self.min_on3_var.set(min_on); self.min_off3_var.set(min_off)
            self.spike3_var.set(spike)

        self.status_var.set(f"Preset applied: Graph{which} = {preset}")
        self._schedule_tune_refresh()

    # ---------------- UI ----------------
    def _build_ui(self):
        root = ttk.Frame(self)
        root.pack(fill="both", expand=True)

        left = ttk.Frame(root)
        left.pack(side="left", fill="y", expand=False, padx=8, pady=8)

        right = ttk.Frame(root)
        right.pack(side="right", fill="both", expand=True, padx=8, pady=8)

        # Video canvas
        self.canvas = tk.Canvas(left, width=self.canvas_w, height=self.canvas_h,
                                bg="#111111", highlightthickness=1, highlightbackground="#333")
        self.canvas.pack()

        # Controls row 1
        ctrl1 = ttk.Frame(left)
        ctrl1.pack(fill="x", pady=(8, 0))

        self.btn_open = ttk.Button(ctrl1, text="Open", command=self.on_open)
        self.btn_open.pack(side="left")

        self.btn_play = ttk.Button(ctrl1, text="Play", command=self.on_play_pause, state="disabled")
        self.btn_play.pack(side="left", padx=6)

        self.btn_stop = ttk.Button(ctrl1, text="Stop(↩)", command=self.on_stop_rewind, state="disabled")
        self.btn_stop.pack(side="left")

        self.chk_temporal = ttk.Checkbutton(ctrl1, text="Temporal 누적(EMA)",
                                            variable=self.temporal_enabled_var,
                                            command=self._on_temporal_toggle)
        self.chk_temporal.pack(side="left", padx=(16, 0))

        # Controls row 2
        ctrl2 = ttk.Frame(left)
        ctrl2.pack(fill="x", pady=(8, 0))

        ttk.Label(ctrl2, text="Metric:").pack(side="left")
        self.metric_combo = ttk.Combobox(ctrl2, textvariable=self.metric_var,
                                         values=list(self.metrics.keys()), state="readonly", width=18)
        self.metric_combo.pack(side="left", padx=6)
        self.metric_combo.bind("<<ComboboxSelected>>", lambda e: self._on_metric_changed())

        ttk.Label(ctrl2, text="Threshold:").pack(side="left", padx=(12, 0))
        self.thresh_spin = ttk.Spinbox(ctrl2, from_=0, to=255, textvariable=self.thresh_var, width=6)
        self.thresh_spin.pack(side="left", padx=6)
        self.thresh_spin.bind("<KeyRelease>", lambda e: self._on_metric_changed())
        self.thresh_spin.configure(command=self._on_metric_changed)

        self.metric_hint = ttk.Label(ctrl2, text="")
        self.metric_hint.pack(side="left", padx=10)

        self.lbl_active_roi = ttk.Label(ctrl2, text="Active ROI: 1 (ON)")
        self.lbl_active_roi.pack(side="right")

        # Seek
        seek = ttk.Frame(left)
        seek.pack(fill="x", pady=(10, 0))
        self.seek_scale = ttk.Scale(seek, from_=0.0, to=1.0, variable=self.seek_var)
        self.seek_scale.pack(fill="x", expand=True)
        self.seek_scale.bind("<Button-1>", self._on_seek_click_or_drag_start)
        self.seek_scale.bind("<B1-Motion>", self._on_seek_drag_motion)
        self.seek_scale.bind("<ButtonRelease-1>", lambda e: setattr(self, "user_seeking", False))

        # Info
        info = ttk.Frame(left)
        info.pack(fill="x", pady=(6, 0))
        self.lbl_info = ttk.Label(info, text="No video loaded.")
        self.lbl_info.pack(side="left")
        self.lbl_roi = ttk.Label(info, text="ROIs: none")
        self.lbl_roi.pack(side="right")

        info2 = ttk.Frame(left)
        info2.pack(fill="x", pady=(4, 0))
        self.lbl_frame = ttk.Label(info2, text="Frame: -/-  |  Time: -/-")
        self.lbl_frame.pack(side="left")

        # Help
        info3 = ttk.Frame(left)
        info3.pack(fill="x", pady=(4, 0))
        ttk.Label(
            info3,
            text=(
                "Keys: Space=Play/Pause | 1/2=ROI select | r=ROI ON/OFF | c=Camera preset | "
                "Ctrl+S=Save | Ctrl+R=Load | ←/→ step(pause)\n"
                "OpenGL: L-drag=rotate | Ctrl+L-drag=Z-rotate only | R-drag=pan | Wheel=zoom"
            )
        ).pack(side="left")

        # ---------------- OpenGL + Tune side-by-side ----------------
        gl_outer = ttk.LabelFrame(left, text="3D View + Tune (side-by-side)")
        gl_outer.pack(fill="x", pady=(10, 0))

        self.gl_row = ttk.Frame(gl_outer)
        self.gl_row.pack(fill="x", expand=False)

        # Backend selection
        # (Removed - now Panda3D only)

        # 3D view container
        self.gl_frame = ttk.Frame(self.gl_row)
        self.gl_frame.pack(side="left", fill="both", expand=True, padx=(4, 6), pady=4)

        # Debug: print the Tk handle/id for the gl container to diagnose bad parent-window values
        try:
            print("TK gl_frame winfo_id =", self.gl_frame.winfo_id(), "size =", self.gl_w, self.gl_h)
        except Exception:
            pass

        if PANDA3D_OK:
            try:
                self.gl_view = Panda3DEmbed(self.gl_frame, width=self.gl_w, height=self.gl_h)
                self.gl_view.pack(fill="both", expand=True)
            except Exception:
                import traceback
                traceback.print_exc()
                # Re-raise so the traceback clearly shows the failing line during debugging
                raise
        else:
            self.gl_view = None
            ttk.Label(self.gl_frame, text="⚠️ Panda3D not available: pip install panda3d").pack(anchor="w")

        # Tune notebook + scroll
        tune_side = ttk.Frame(self.gl_row)
        tune_side.pack(side="left", fill="both", expand=True, padx=(0, 4), pady=4)

        nb = ttk.Notebook(tune_side)
        nb.pack(fill="both", expand=True)
        nb.bind("<<NotebookTabChanged>>", lambda e: self._schedule_tune_refresh())

        tab1 = ttk.Frame(nb)
        tab2 = ttk.Frame(nb)
        tab3 = ttk.Frame(nb)
        nb.add(tab1, text="ROI1")
        nb.add(tab2, text="ROI2")
        nb.add(tab3, text="SYM")

        s1 = ScrollableFrame(tab1, height=240); s1.pack(fill="both", expand=True)
        s2 = ScrollableFrame(tab2, height=240); s2.pack(fill="both", expand=True)
        s3 = ScrollableFrame(tab3, height=240); s3.pack(fill="both", expand=True)

        def preset_bar(parent, which):
            bar = ttk.Frame(parent)
            bar.pack(fill="x", pady=(2, 6), padx=6)
            ttk.Label(bar, text="Preset:").pack(side="left")
            ttk.Button(bar, text="민감", command=lambda: self._apply_preset(which, "sensitive")).pack(side="left", padx=4)
            ttk.Button(bar, text="보통", command=lambda: self._apply_preset(which, "normal")).pack(side="left", padx=4)
            ttk.Button(bar, text="둔감", command=lambda: self._apply_preset(which, "robust")).pack(side="left", padx=4)
            ttk.Label(bar, text="(재생 중 변경 OK)").pack(side="right")

        preset_bar(s1.inner, 1)
        preset_bar(s2.inner, 2)
        preset_bar(s3.inner, 3)

        # BASIC sliders (+ Entry)
        b1 = ttk.LabelFrame(s1.inner, text="Basic"); b1.pack(fill="x", padx=6, pady=(0, 6))
        b2 = ttk.LabelFrame(s2.inner, text="Basic"); b2.pack(fill="x", padx=6, pady=(0, 6))
        b3 = ttk.LabelFrame(s3.inner, text="Basic"); b3.pack(fill="x", padx=6, pady=(0, 6))

        # NOTE: ROI2가 0.2 밑으로 안내려간다 → vmin을 0.01로 낮춤
        self._make_slider_row(b1, "Surge dY",   self.rise1_var, 0.01, 50.0, 0.01, is_int=False)
        self._make_slider_row(b1, "Release dY", self.fall1_var, 0.01, 50.0, 0.01, is_int=False)

        self._make_slider_row(b2, "Surge dY",   self.rise2_var, 0.01, 50.0, 0.01, is_int=False)
        self._make_slider_row(b2, "Release dY", self.fall2_var, 0.01, 50.0, 0.01, is_int=False)

        self._make_slider_row(b3, "Surge dY",   self.rise3_var, 0.01, 50.0, 0.01, is_int=False)
        self._make_slider_row(b3, "Release dY", self.fall3_var, 0.01, 50.0, 0.01, is_int=False)

        # ADVANCED collapsible
        adv1 = Collapsible(s1.inner, title="Advanced (Hold/Spike)", expanded=False)
        adv1.pack(fill="x", padx=6, pady=(0, 8))
        adv2 = Collapsible(s2.inner, title="Advanced (Hold/Spike)", expanded=False)
        adv2.pack(fill="x", padx=6, pady=(0, 8))
        adv3 = Collapsible(s3.inner, title="Advanced (Hold/Spike)", expanded=False)
        adv3.pack(fill="x", padx=6, pady=(0, 8))

        self._make_slider_row(adv1.body, "Min ON",   self.min_on1_var,  1, 30, 1,    is_int=True)
        self._make_slider_row(adv1.body, "Min OFF",  self.min_off1_var, 1, 80, 1,    is_int=True)
        self._make_slider_row(adv1.body, "Spike dY", self.spike1_var,   0.01, 80.0, 0.01, is_int=False)

        self._make_slider_row(adv2.body, "Min ON",   self.min_on2_var,  1, 30, 1,    is_int=True)
        self._make_slider_row(adv2.body, "Min OFF",  self.min_off2_var, 1, 80, 1,    is_int=True)
        self._make_slider_row(adv2.body, "Spike dY", self.spike2_var,   0.01, 80.0, 0.01, is_int=False)

        self._make_slider_row(adv3.body, "Min ON",   self.min_on3_var,  1, 30, 1,    is_int=True)
        self._make_slider_row(adv3.body, "Min OFF",  self.min_off3_var, 1, 80, 1,    is_int=True)
        self._make_slider_row(adv3.body, "Spike dY", self.spike3_var,   0.01, 80.0, 0.01, is_int=False)

        # GL controls under
        gl_ctrl = ttk.Frame(gl_outer)
        gl_ctrl.pack(fill="x", pady=(6, 4))

        self.btn_save = ttk.Button(gl_ctrl, text="Save (Ctrl+S)", command=self._save_all_state)
        self.btn_load = ttk.Button(gl_ctrl, text="Load (Ctrl+R)", command=lambda: self._load_all_state(show_error=True))
        self.btn_save.pack(side="left")
        self.btn_load.pack(side="left", padx=6)

        ttk.Label(gl_ctrl, text="Zoom:").pack(side="left", padx=(12, 2))
        self.zoom_combo = ttk.Combobox(gl_ctrl, textvariable=self.zoom_mode_var,
                                       values=["scale", "dist"], state="readonly", width=7)
        self.zoom_combo.pack(side="left")
        self.zoom_combo.bind("<<ComboboxSelected>>", lambda e: self._apply_gl_ui_to_view())

        self.chk_inv_lr = ttk.Checkbutton(gl_ctrl, text="Invert LR", variable=self.invert_lr_var,
                                          command=lambda: self._apply_gl_ui_to_view())
        self.chk_inv_ud = ttk.Checkbutton(gl_ctrl, text="Invert UD", variable=self.invert_ud_var,
                                          command=lambda: self._apply_gl_ui_to_view())
        self.chk_inv_lr.pack(side="left", padx=(12, 0))
        self.chk_inv_ud.pack(side="left", padx=(8, 0))

        # ---------------- Right plots (3) ----------------
        self.fig = Figure(figsize=(8.6, 8.2), dpi=100)
        self.fig.subplots_adjust(hspace=0.62, top=0.95, bottom=0.07)

        self.ax1 = self.fig.add_subplot(311)
        self.ax2 = self.fig.add_subplot(312)
        self.ax3 = self.fig.add_subplot(313)

        for ax in (self.ax1, self.ax2, self.ax3):
            ax.set_xlabel("Time (s)")
            ax.set_ylabel("Value")
            ax.grid(True, alpha=0.25)

        self.line1, = self.ax1.plot([], [], linewidth=1.8, color="tab:green")
        self.line2, = self.ax2.plot([], [], linewidth=1.8, color="tab:orange")
        self.line3, = self.ax3.plot([], [], linewidth=1.8, color="tab:purple")

        self.mean1 = self.ax1.axhline(0.0, linewidth=1.0, linestyle="--", color="tab:green", alpha=0.55)
        self.mean2 = self.ax2.axhline(0.0, linewidth=1.0, linestyle="--", color="tab:orange", alpha=0.55)
        self.mean3 = self.ax3.axhline(0.0, linewidth=1.0, linestyle="--", color="tab:purple", alpha=0.55)

        self.cursor1 = self.ax1.axvline(0.0, linewidth=1.0)
        self.cursor2 = self.ax2.axvline(0.0, linewidth=1.0)
        self.cursor3 = self.ax3.axvline(0.0, linewidth=1.0)

        self.spike_sc1, = self.ax1.plot([], [], linestyle="None", marker="o", color="red", markersize=6, zorder=6)
        self.spike_sc2, = self.ax2.plot([], [], linestyle="None", marker="o", color="red", markersize=6, zorder=6)
        self.spike_sc3, = self.ax3.plot([], [], linestyle="None", marker="o", color="red", markersize=6, zorder=6)

        self._spike_texts1: List = []
        self._spike_texts2: List = []
        self._spike_texts3: List = []

        self._roi1_red_patches: List = []
        self._roi1_blue_patches: List = []
        self._roi2_red_patches: List = []
        self._sym_red_patches: List = []
        self._detect_texts1: List = []
        self._surge_drop_texts1: List = []
        self._surge_drop_texts2: List = []
        self._surge_drop_texts3: List = []

        self.canvas_plot = FigureCanvasTkAgg(self.fig, master=right)
        self.canvas_plot.get_tk_widget().pack(fill="both", expand=True)

        self.status_var = tk.StringVar(value=f"Ready. state file: {os.path.basename(self._save_path)}")
        ttk.Label(self, textvariable=self.status_var, anchor="w").pack(fill="x", side="bottom")

        self._on_metric_changed()
        self._update_active_roi_label()
        self._update_roi_label()
        self._apply_gl_ui_to_view()
        self._refresh_plot(force_static_x=True)

    # ---------- Slider row (wider + Entry edit, 2 decimal) ----------
    def _make_slider_row(self, parent, label, var, vmin, vmax, step, is_int=False):
        row = ttk.Frame(parent)
        row.pack(fill="x", pady=3, padx=6)

        # grid로 폭 확보(너무 좁아지는 문제 완화)
        row.columnconfigure(1, weight=1)

        ttk.Label(row, text=label, width=12).grid(row=0, column=0, sticky="w")

        # Entry (소수 2째자리까지 직접 수정)
        entry_var = tk.StringVar()

        def fmt_from_var():
            try:
                if is_int:
                    entry_var.set(str(int(var.get())))
                else:
                    entry_var.set(f"{float(var.get()):.2f}")
            except Exception:
                pass

        def clamp_set_value(val):
            try:
                if is_int:
                    v = int(float(val))
                    v = max(int(vmin), min(int(vmax), v))
                    var.set(v)
                else:
                    v = float(val)
                    v = max(float(vmin), min(float(vmax), v))
                    # 표시/입력은 2자리, 내부는 float 그대로(원하면 round(.,2) 가능)
                    var.set(float(v))
            except Exception:
                return
            fmt_from_var()
            self._schedule_tune_refresh()

        def on_entry_commit(_e=None):
            s = entry_var.get().strip()
            if s == "":
                fmt_from_var()
                return
            clamp_set_value(s)

        ent = ttk.Entry(row, textvariable=entry_var, width=8)
        ent.grid(row=0, column=3, sticky="e", padx=(6, 0))
        ent.bind("<Return>", on_entry_commit)
        ent.bind("<FocusOut>", on_entry_commit)

        # Scale
        if is_int:
            sc = ttk.Scale(row, from_=vmin, to=vmax, orient="horizontal",
                           command=lambda s: (var.set(int(float(s))), fmt_from_var(), self._schedule_tune_refresh()))
        else:
            sc = ttk.Scale(row, from_=vmin, to=vmax, orient="horizontal",
                           command=lambda s: (var.set(float(s)), fmt_from_var(), self._schedule_tune_refresh()))
        sc.grid(row=0, column=1, sticky="ew", padx=(8, 4))

        # 작은 step 안내(시각적)
        ttk.Label(row, text=f"[{vmin}~{vmax}]", width=12).grid(row=0, column=2, sticky="e")

        fmt_from_var()

    # ---------------- Bind events ----------------
    def _bind_events(self):
        self.canvas.bind("<Button-1>", self.on_mouse_down)
        self.canvas.bind("<B1-Motion>", self.on_mouse_drag)
        self.canvas.bind("<ButtonRelease-1>", self.on_mouse_up)

        self.bind_all("<space>", lambda e: (self.on_play_pause(), "break"), add="+")
        self.bind_all("<Left>", lambda e: (self.on_step_frame(-1), "break"), add="+")
        self.bind_all("<Right>", lambda e: (self.on_step_frame(+1), "break"), add="+")
        self.bind_all("1", lambda e: (self._select_roi_slot(0), "break"), add="+")
        self.bind_all("2", lambda e: (self._select_roi_slot(1), "break"), add="+")
        self.bind_all("r", lambda e: (self._toggle_roi_enabled(), "break"), add="+")
        self.bind_all("R", lambda e: (self._toggle_roi_enabled(), "break"), add="+")
        self.bind_all("c", lambda e: (self._cycle_camera_preset(), "break"), add="+")
        self.bind_all("C", lambda e: (self._cycle_camera_preset(), "break"), add="+")
        self.bind_all("<Control-s>", lambda e: (self._save_all_state(), "break"), add="+")
        self.bind_all("<Control-S>", lambda e: (self._save_all_state(), "break"), add="+")
        self.bind_all("<Control-r>", lambda e: (self._load_all_state(show_error=True), "break"), add="+")
        self.bind_all("<Control-R>", lambda e: (self._load_all_state(show_error=True), "break"), add="+")
        self.protocol("WM_DELETE_WINDOW", self.on_close)

    def _ensure_focus(self):
        try:
            self.focus_force()
        except Exception:
            pass

    def _apply_gl_ui_to_view(self):
        if self.gl_view is None:
            return
        if hasattr(self.gl_view, 'zoom_mode'):
            try:
                self.gl_view.zoom_mode = str(self.zoom_mode_var.get())
                self.gl_view.invert_lr = bool(self.invert_lr_var.get())
                self.gl_view.invert_ud = bool(self.invert_ud_var.get())
            except Exception:
                pass

    def _switch_3d_backend(self):
        """Compatibility stub for legacy 'backend' state — no-op (Panda3D only)."""
        try:
            # previously allowed switching backends; now single Panda3D backend.
            return
        except Exception:
            pass



    def _cycle_camera_preset(self):
        if self.gl_view is not None:
            self.gl_view.cycle_camera()
            self.status_var.set(f"Camera preset: {self.gl_view.get_camera_preset_name()} (c)")

    def _select_roi_slot(self, idx: int):
        self.active_roi_idx = int(idx)
        self._update_active_roi_label()
        self.status_var.set(f"ROI 슬롯 {idx+1} 선택됨.")
        self._redraw()
        self._update_gl_view(force=True)

    def _toggle_roi_enabled(self):
        i = self.active_roi_idx
        self.roi_enabled[i] = not self.roi_enabled[i]
        state = "ON" if self.roi_enabled[i] else "OFF"
        self.status_var.set(f"ROI{i+1} {state}")
        self.prev_roi_gray[i] = None
        self._update_roi_label()
        self._update_active_roi_label()
        self._redraw()
        self._refresh_plot(force_static_x=True)
        self._update_gl_view(force=True)

    def _update_active_roi_label(self):
        i = self.active_roi_idx
        state = "ON" if self.roi_enabled[i] else "OFF"
        self.lbl_active_roi.configure(text=f"Active ROI: {i+1} ({state})")

    def _update_roi_label(self):
        parts = []
        for i in range(2):
            r = self.rois[i]
            en = "ON" if self.roi_enabled[i] else "OFF"
            if r is None or not r.is_valid():
                parts.append(f"{i+1}:{en} -")
            else:
                rr = r.normalized()
                parts.append(f"{i+1}:{en}({rr.x1},{rr.y1})-({rr.x2},{rr.y2})")
        self.lbl_roi.configure(text="ROIs: " + "  ".join(parts))

    def _on_metric_changed(self):
        m = self.metric_var.get()
        if m == "ThresholdPixelRatio":
            self.metric_hint.configure(text="(0~1) ratio of pixels >= threshold")
        elif m == "SSIMLike(1-ssim)":
            self.metric_hint.configure(text="0≈same, larger≈different")
        else:
            self.metric_hint.configure(text="")

        self.ax1.set_title(f"ROI1 Delta ({m})  + Detect(ROI1∩ROI2 surge)")
        self.ax2.set_title(f"ROI2 Delta ({m})")
        self.ax3.set_title("Symmetry Metric")

        if self.duration_sec > 0:
            self.ax1.set_xlim(0, self.duration_sec)
            self.ax2.set_xlim(0, self.duration_sec)
            self.ax3.set_xlim(0, self.duration_sec)

        self._refresh_plot(force_static_x=True)

    # ---------------- Save/Load ----------------
    def _gather_all_state(self) -> dict:
        gl_state = {}
        if self.gl_view is not None and hasattr(self.gl_view, 'get_state'):
            try:
                gl_state = self.gl_view.get_state()
            except Exception:
                gl_state = {}

        params = dict(
            backend=str(self.backend_var.get()),
            zoom_mode=str(self.zoom_mode_var.get()),
            invert_lr=bool(self.invert_lr_var.get()),
            invert_ud=bool(self.invert_ud_var.get()),

            rise1=float(self.rise1_var.get()), fall1=float(self.fall1_var.get()),
            min_on1=int(self.min_on1_var.get()), min_off1=int(self.min_off1_var.get()),
            spike1=float(self.spike1_var.get()),

            rise2=float(self.rise2_var.get()), fall2=float(self.fall2_var.get()),
            min_on2=int(self.min_on2_var.get()), min_off2=int(self.min_off2_var.get()),
            spike2=float(self.spike2_var.get()),

            rise3=float(self.rise3_var.get()), fall3=float(self.fall3_var.get()),
            min_on3=int(self.min_on3_var.get()), min_off3=int(self.min_off3_var.get()),
            spike3=float(self.spike3_var.get()),
        )

        rois_payload = [r.to_dict() if isinstance(r, ROI) else None for r in self.rois]
        roi_enabled_payload = [bool(x) for x in self.roi_enabled]

        return dict(
            version=8,
            gl=gl_state,
            params=params,
            rois=rois_payload,
            roi_enabled=roi_enabled_payload,
            active_roi_idx=int(self.active_roi_idx),
        )

    def _apply_all_state(self, payload: dict):
        params = payload.get("params", {}) if isinstance(payload, dict) else {}
        if isinstance(params, dict):
            if "backend" in params:
                self.backend_var.set(str(params["backend"]))
                self._switch_3d_backend()
            if "zoom_mode" in params: self.zoom_mode_var.set(str(params["zoom_mode"]))
            if "invert_lr" in params: self.invert_lr_var.set(bool(params["invert_lr"]))
            if "invert_ud" in params: self.invert_ud_var.set(bool(params["invert_ud"]))

            for k, v in [
                ("rise1", self.rise1_var), ("fall1", self.fall1_var),
                ("min_on1", self.min_on1_var), ("min_off1", self.min_off1_var), ("spike1", self.spike1_var),
                ("rise2", self.rise2_var), ("fall2", self.fall2_var),
                ("min_on2", self.min_on2_var), ("min_off2", self.min_off2_var), ("spike2", self.spike2_var),
                ("rise3", self.rise3_var), ("fall3", self.fall3_var),
                ("min_on3", self.min_on3_var), ("min_off3", self.min_off3_var), ("spike3", self.spike3_var),
            ]:
                if k in params:
                    try:
                        if isinstance(v, tk.IntVar):
                            v.set(int(params[k]))
                        else:
                            v.set(float(params[k]))
                    except Exception:
                        pass

        rois_payload = payload.get("rois", None)
        if isinstance(rois_payload, list) and len(rois_payload) >= 2:
            self.rois = [
                ROI.from_dict(rois_payload[0]) if rois_payload[0] is not None else None,
                ROI.from_dict(rois_payload[1]) if rois_payload[1] is not None else None,
            ]

        roi_enabled_payload = payload.get("roi_enabled", None)
        if isinstance(roi_enabled_payload, list) and len(roi_enabled_payload) >= 2:
            self.roi_enabled = [bool(roi_enabled_payload[0]), bool(roi_enabled_payload[1])]

        if "active_roi_idx" in payload:
            try:
                self.active_roi_idx = int(payload.get("active_roi_idx", 0)) % 2
            except Exception:
                self.active_roi_idx = 0

        gl = payload.get("gl", {}) if isinstance(payload, dict) else {}
        if self.gl_view is not None and isinstance(gl, dict) and gl:
            self.gl_view.set_state(gl)

        self._apply_gl_ui_to_view()
        self._update_active_roi_label()
        self._update_roi_label()
        self._redraw()
        self._update_gl_view(force=True)
        self._refresh_plot(force_static_x=True)

    def _save_all_state(self):
        try:
            payload = self._gather_all_state()
            with open(self._save_path, "w", encoding="utf-8") as f:
                json.dump(payload, f, ensure_ascii=False, indent=2)
            self.status_var.set(f"Saved: {self._save_path}")
        except Exception as e:
            self.status_var.set(f"Save failed: {e}")
            try:
                messagebox.showerror("Save failed", str(e))
            except Exception:
                pass

    def _load_all_state(self, show_error: bool = False):
        if not os.path.exists(self._save_path):
            msg = f"No state file: {self._save_path}"
            self.status_var.set(msg)
            if show_error:
                try:
                    messagebox.showwarning("Load", msg)
                except Exception:
                    pass
            return
        try:
            with open(self._save_path, "r", encoding="utf-8") as f:
                payload = json.load(f)
            self._apply_all_state(payload)
            self.status_var.set(f"Loaded: {self._save_path}")
        except Exception as e:
            self.status_var.set(f"Load failed: {e}")
            if show_error:
                try:
                    messagebox.showerror("Load failed", str(e))
                except Exception:
                    pass

    # ---------------- Temporal EMA ----------------
    def _reset_temporal_state(self):
        self._ema_prev_f32 = None

    def _on_temporal_toggle(self):
        self._reset_temporal_state()
        if self.cur_bgr_raw is not None:
            self._update_processed_frame(self.cur_bgr_raw)
            if self.cur_bgr_proc is not None:
                self._compute_draw_rect(self.cur_bgr_proc)
                self._draw_frame(self.cur_bgr_proc)
                self._update_gl_view(force=True)
        self.status_var.set("Temporal EMA toggled (state reset).")

    def _apply_temporal_ema(self, bgr: np.ndarray) -> np.ndarray:
        if not self.temporal_enabled_var.get():
            return bgr
        cur = bgr.astype(np.float32)
        if self._ema_prev_f32 is None or self._ema_prev_f32.shape != cur.shape:
            self._ema_prev_f32 = cur.copy()
            return bgr
        a = float(self.temporal_alpha)
        self._ema_prev_f32 = (1.0 - a) * self._ema_prev_f32 + a * cur
        return np.clip(self._ema_prev_f32, 0, 255).astype(np.uint8)

    def _update_processed_frame(self, raw_bgr: np.ndarray):
        self.cur_bgr_raw = raw_bgr
        self.cur_bgr_proc = self._apply_temporal_ema(raw_bgr)

    # ---------------- Video control ----------------
    def on_open(self):
        path = filedialog.askopenfilename(
            title="Select video",
            filetypes=[("Video files", "*.mp4 *.avi *.mov *.mkv *.wmv *.m4v"), ("All files", "*.*")]
        )
        if not path:
            return

        cap = cv2.VideoCapture(path)
        if not cap.isOpened():
            messagebox.showerror("Error", "Failed to open video.")
            return

        self._release_video()
        self.cap = cap
        self.video_path = path

        fps = float(self.cap.get(cv2.CAP_PROP_FPS) or 0.0)
        self.fps = fps if fps > 1e-6 else 30.0
        self.frame_count = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        self.duration_sec = (self.frame_count / self.fps) if self.frame_count > 0 else 0.0

        self.seek_scale.configure(from_=0.0, to=max(0.0, self.duration_sec))
        self.seek_var.set(0.0)

        self.playing = False
        self.stop_flag = True

        self.btn_play.configure(state="normal", text="Play")
        self.btn_stop.configure(state="normal")

        self._clear_timeseries()
        self._reset_temporal_state()

        self._seek_to_time_sec(0.0, reset_refs=True)
        self._setup_static_x_axis()
        self._refresh_plot(force_static_x=True)
        self._update_labels()
        self._redraw()
        self._update_gl_view(force=True)

        self.status_var.set("Loaded.")
        self._ensure_focus()

    def on_play_pause(self):
        if not self.cap:
            return
        self.playing = not self.playing
        self.btn_play.configure(text="Pause" if self.playing else "Play")
        self.status_var.set("Playing..." if self.playing else "Paused. (←/→ step)")

        if self.playing and (self.worker is None or not self.worker.is_alive()):
            self.stop_flag = False
            self.worker = threading.Thread(target=self._play_loop, daemon=True)
            self.worker.start()

    def on_stop_rewind(self):
        if not self.cap:
            return
        self.playing = False
        self.stop_flag = True
        self.btn_play.configure(text="Play")

        self._clear_timeseries()
        self.seek_var.set(0.0)
        self._reset_temporal_state()
        self._seek_to_time_sec(0.0, reset_refs=True)

        self._setup_static_x_axis()
        self._refresh_plot(force_static_x=True)
        self._update_labels()
        self._redraw()
        self._update_gl_view(force=True)

        self.status_var.set("Stopped: rewind + graphs reset (ROI kept).")
        self._ensure_focus()

    def _play_loop(self):
        frame_period = 1.0 / max(1e-6, self.fps)
        last = time.perf_counter()

        while not self.stop_flag and self.cap is not None and self._alive:
            if not self.playing:
                time.sleep(0.01)
                continue

            with self.cap_lock:
                ok, frame_raw = self.cap.read()

            if not ok or frame_raw is None:
                self.playing = False
                self.after(0, lambda: self.btn_play.configure(text="Play"))
                self.after(0, lambda: self.status_var.set("End of video."))
                break

            self._update_processed_frame(frame_raw)

            with self.cap_lock:
                self._update_pos_from_cap()

            self._accumulate_delta_for_frame(self.cur_bgr_proc)
            self._accumulate_symmetry(self.cur_bgr_proc)

            self.after(0, lambda f=self.cur_bgr_proc: self._draw_frame(f))
            self.after(0, self._update_labels)

            # Throttle plot updates during playback
            self._plot_update_counter += 1
            if self._plot_update_counter % self._plot_update_every == 0:
                self.after(0, lambda: self._refresh_plot(force_static_x=False))

            self._gl_tick += 1
            if (self._gl_tick % self._gl_every_n) == 0:
                self.after(0, lambda: self._update_gl_view(force=False))

            if not self.user_seeking:
                self.after(0, lambda: self.seek_var.set(self.pos_msec / 1000.0))

            now = time.perf_counter()
            elapsed = now - last
            sleep_t = frame_period - elapsed
            if sleep_t > 0:
                time.sleep(sleep_t)
            last = time.perf_counter()

    # ---------------- Seek ----------------
    def _scale_value_from_event(self, event) -> float:
        w = max(1, self.seek_scale.winfo_width())
        frac = event.x / w
        frac = max(0.0, min(1.0, frac))
        vmin = float(self.seek_scale.cget("from"))
        vmax = float(self.seek_scale.cget("to"))
        return vmin + frac * (vmax - vmin)

    def _on_seek_click_or_drag_start(self, event):
        if not self.cap:
            return
        self.user_seeking = True
        v = self._scale_value_from_event(event)
        self.seek_var.set(v)
        self._seek_to_time_sec(v, reset_refs=True)

    def _on_seek_drag_motion(self, event):
        if not self.cap:
            return
        v = self._scale_value_from_event(event)
        self.seek_var.set(v)
        self._seek_to_time_sec(v, reset_refs=True)

    def _update_pos_from_cap(self):
        self.pos_msec = float(self.cap.get(cv2.CAP_PROP_POS_MSEC) or 0.0)
        pf = float(self.cap.get(cv2.CAP_PROP_POS_FRAMES) or 0.0)
        idx = int(round(pf)) - 1
        if self.frame_count > 0:
            idx = max(0, min(idx, self.frame_count - 1))
        self.frame_idx = idx

    def _seek_to_time_sec(self, tsec: float, reset_refs: bool):
        if not self.cap:
            return None
        tsec = max(0.0, min(float(tsec), max(0.0, self.duration_sec)))

        if reset_refs:
            for i in range(2):
                self.prev_roi_gray[i] = None

        with self.cap_lock:
            self.cap.set(cv2.CAP_PROP_POS_MSEC, tsec * 1000.0)
            ok, frame_raw = self.cap.read()

        if not ok or frame_raw is None:
            return None

        self._update_processed_frame(frame_raw)
        self._compute_draw_rect(self.cur_bgr_proc)

        if self.fps > 1e-6:
            self.frame_idx = int(round(tsec * self.fps))
            if self.frame_count > 0:
                self.frame_idx = max(0, min(self.frame_idx, self.frame_count - 1))
            self.pos_msec = self.frame_idx * 1000.0 / self.fps

        self._draw_frame(self.cur_bgr_proc)
        return self.cur_bgr_proc

    def on_step_frame(self, delta_frames: int):
        if not self.cap or self.playing or self.frame_count <= 0:
            return

        target = int(self.frame_idx + delta_frames)
        target = max(0, min(target, self.frame_count - 1))

        with self.cap_lock:
            tsec = target / max(1e-6, self.fps)
            self.cap.set(cv2.CAP_PROP_POS_MSEC, tsec * 1000.0)
            ok, frame_raw = self.cap.read()

        if not ok or frame_raw is None:
            return

        self.frame_idx = target
        self.pos_msec = target * 1000.0 / max(1e-6, self.fps)

        self._update_processed_frame(frame_raw)
        self._compute_draw_rect(self.cur_bgr_proc)

        self._accumulate_delta_for_frame(self.cur_bgr_proc)
        self._accumulate_symmetry(self.cur_bgr_proc)

        self.seek_var.set(self.pos_msec / 1000.0)
        self._draw_frame(self.cur_bgr_proc)
        self._update_gl_view(force=True)
        self._update_labels()
        self._refresh_plot(force_static_x=True)

    # ---------------- Delta core ----------------
    def _compute_metric(self, a: np.ndarray, b: np.ndarray) -> float:
        name = self.metric_var.get()
        fn = self.metrics.get(name, self._m_mean_abs)
        return float(fn(a, b))

    def _extract_roi_gray(self, frame_bgr: np.ndarray, roi: ROI) -> Optional[np.ndarray]:
        if roi is None or not roi.is_valid():
            return None
        r = roi.normalized()
        roi_bgr = frame_bgr[r.y1:r.y2, r.x1:r.x2]
        if roi_bgr.size == 0:
            return None
        return cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2GRAY)

    def _accumulate_delta_for_frame(self, frame_bgr: np.ndarray):
        if frame_bgr is None:
            return
        tsec = self.pos_msec / 1000.0

        with self.data_lock:
            for i in range(2):
                if not self.roi_enabled[i]:
                    continue
                if self.rois[i] is None or not self.rois[i].is_valid():
                    continue

                roi_gray = self._extract_roi_gray(frame_bgr, self.rois[i])
                if roi_gray is None:
                    continue

                if self.prev_roi_gray[i] is None:
                    self.prev_roi_gray[i] = roi_gray
                    self.delta_times[i].append(tsec)
                    self.delta_vals[i].append(0.0)
                    continue

                dv = self._compute_metric(roi_gray, self.prev_roi_gray[i])
                self.prev_roi_gray[i] = roi_gray
                self.delta_times[i].append(tsec)
                self.delta_vals[i].append(float(dv))

                # Limit data size to prevent memory overflow
                if len(self.delta_times[i]) > self._max_data_points:
                    excess = len(self.delta_times[i]) - self._max_data_points
                    del self.delta_times[i][:excess]
                    del self.delta_vals[i][:excess]

    # ---------------- OpenGL 3D ----------------
    def _remove_center_cross_roi1_for_3d(self, gray: np.ndarray) -> np.ndarray:
        """Remove bright center cross from ROI1 for cleaner 3D rendering"""
        h, w = gray.shape[:2]
        cy, cx = h // 2, w // 2
        band = max(1, min(3, min(h, w) // 40))
        thr = 245

        mask = np.zeros((h, w), dtype=np.uint8)
        x1 = max(0, cx - band); x2 = min(w, cx + band + 1)
        y1 = max(0, cy - band); y2 = min(h, cy + band + 1)
        mask[:, x1:x2] = (gray[:, x1:x2] >= thr).astype(np.uint8) * 255
        mask[y1:y2, :] = np.maximum(mask[y1:y2, :], (gray[y1:y2, :] >= thr).astype(np.uint8) * 255)

        if int(np.sum(mask)) == 0:
            return gray
        return cv2.inpaint(gray, mask, 3, cv2.INPAINT_TELEA)

    def _get_gl_source_gray(self) -> Tuple[Optional[np.ndarray], str]:
        """Get grayscale image for OpenGL heightmap rendering"""
        if self.cur_bgr_proc is None:
            return None, ""
        roi = self.rois[self.active_roi_idx]
        if roi is not None and roi.is_valid():
            g = self._extract_roi_gray(self.cur_bgr_proc, roi)
            if g is None:
                return None, ""
            title = f"OpenGL Heightmap (ROI{self.active_roi_idx+1})"
            if self.active_roi_idx == 0:
                g = self._remove_center_cross_roi1_for_3d(g)
            return g, title
        return None, ""

    # ---------------- Symmetry ----------------
    def _get_sym_source_gray(self, frame_bgr: np.ndarray) -> Optional[np.ndarray]:
        roi = self.rois[self.active_roi_idx]
        if roi is not None and roi.is_valid():
            return self._extract_roi_gray(frame_bgr, roi)
        return cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)

    def _symmetry_value(self, gray: np.ndarray) -> float:
        if gray is None or gray.size == 0:
            return 0.0
        h, w = gray.shape[:2]
        target = 160
        if max(h, w) > target:
            s = target / max(h, w)
            gray = cv2.resize(gray, (max(2, int(w * s)), max(2, int(h * s))), interpolation=cv2.INTER_AREA)
        lr = cv2.flip(gray, 1)
        ud = cv2.flip(gray, 0)
        v1 = float(np.mean(cv2.absdiff(gray, lr)))
        v2 = float(np.mean(cv2.absdiff(gray, ud)))
        return 0.5 * (v1 + v2)

    def _accumulate_symmetry(self, frame_bgr: np.ndarray):
        g = self._get_sym_source_gray(frame_bgr)
        if g is None:
            return
        tsec = self.pos_msec / 1000.0
        v = self._symmetry_value(g)
        with self.data_lock:
            self.sym_times.append(tsec)
            self.sym_vals.append(float(v))

            # Limit data size to prevent memory overflow
            if len(self.sym_times) > self._max_data_points:
                excess = len(self.sym_times) - self._max_data_points
                del self.sym_times[:excess]
                del self.sym_vals[:excess]

    # ---------------- Surge mask / spikes ----------------
    def _compute_red_mask_no_mean(self, y: np.ndarray, rise_thr: float, fall_thr: float, min_on: int, min_off: int) -> Optional[np.ndarray]:
        if y.size < 3:
            return None

        rise = float(rise_thr)
        fall = float(fall_thr)
        min_on = max(1, int(min_on))
        min_off = max(1, int(min_off))

        dy = np.diff(y, prepend=np.nan)
        red = np.zeros_like(y, dtype=bool)

        in_red = False
        on_run = 0
        off_run = 0

        for i in range(len(y)):
            if not (np.isfinite(y[i]) and np.isfinite(dy[i])):
                in_red = False
                on_run = 0
                off_run = 0
                continue

            if not in_red:
                if dy[i] >= rise:
                    on_run += 1
                    if on_run >= min_on:
                        in_red = True
                        red[i] = True
                        off_run = 0
                else:
                    on_run = 0
            else:
                red[i] = True
                if dy[i] <= -fall:
                    off_run += 1
                    if off_run >= min_off:
                        in_red = False
                        on_run = 0
                        off_run = 0
                else:
                    off_run = 0

        return red

    def _find_spikes_by_dy(self, x: np.ndarray, y: np.ndarray, thr: float) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        if x.size < 3 or y.size < 3:
            return np.array([]), np.array([]), []
        thr = float(thr)
        dy = np.diff(y, prepend=np.nan)

        idxs = np.where(np.isfinite(dy) & (np.abs(dy) >= thr))[0]
        if idxs.size == 0:
            return np.array([]), np.array([]), []

        idxs = idxs[-self._max_spike_labels:]
        xs = x[idxs]
        ys = y[idxs]
        labels = []
        for i in idxs:
            if dy[i] >= thr:
                labels.append("SURGE")
            elif dy[i] <= -thr:
                labels.append("DROP")
            else:
                labels.append("JUMP")
        return xs, ys, labels

    # (요구사항) ROI1/ROI2에서도 surge/drop 텍스트가 잘 보이도록:
    # - bbox 추가
    # - surge/drop 별 색상 분리
    # - zorder 높임
    def _apply_spike_texts(self, ax, texts_store: List, xs: np.ndarray, ys: np.ndarray, labels: List[str]):
        while len(texts_store) < len(labels):
            t = ax.text(0, 0, "", fontsize=9, fontweight="bold", zorder=7)
            t.set_bbox(dict(facecolor="white", alpha=0.75, edgecolor="none", pad=1.6))
            texts_store.append(t)

        for i, t in enumerate(texts_store):
            if i < len(labels):
                lab = labels[i]
                t.set_text(lab)
                t.set_position((float(xs[i]), float(ys[i])))
                if lab == "DROP":
                    t.set_color("royalblue")
                else:
                    t.set_color("crimson")
                t.set_visible(True)
            else:
                t.set_visible(False)

    # ---------------- Coordinate mapping + ROI mouse + render ----------------
    def _compute_draw_rect(self, frame_bgr: np.ndarray):
        fh, fw = frame_bgr.shape[:2]
        cw, ch = self.canvas_w, self.canvas_h
        scale = min(cw / max(1, fw), ch / max(1, fh))
        dw = int(round(fw * scale))
        dh = int(round(fh * scale))
        dx = (cw - dw) // 2
        dy = (ch - dh) // 2
        self.draw_rect = DrawRect(dx, dy, dw, dh)

    def _canvas_to_frame(self, cx: int, cy: int) -> Optional[Tuple[int, int]]:
        if self.cur_bgr_proc is None or self.draw_rect is None:
            return None
        fh, fw = self.cur_bgr_proc.shape[:2]
        r = self.draw_rect
        if cx < r.x or cy < r.y or cx >= r.x + r.w or cy >= r.y + r.h:
            return None
        u = (cx - r.x) / max(1, r.w)
        v = (cy - r.y) / max(1, r.h)
        fx = int(round(u * (fw - 1)))
        fy = int(round(v * (fh - 1)))
        fx = max(0, min(fx, fw - 1))
        fy = max(0, min(fy, fh - 1))
        return fx, fy

    def _frame_to_canvas(self, fx: int, fy: int) -> Optional[Tuple[int, int]]:
        if self.cur_bgr_proc is None or self.draw_rect is None:
            return None
        fh, fw = self.cur_bgr_proc.shape[:2]
        r = self.draw_rect
        u = fx / max(1, (fw - 1))
        v = fy / max(1, (fh - 1))
        cx = int(round(r.x + u * (r.w - 1)))
        cy = int(round(r.y + v * (r.h - 1)))
        return cx, cy

    def on_mouse_down(self, event):
        if not self.cap:
            return
        p = self._canvas_to_frame(event.x, event.y)
        if p is None:
            return
        fx, fy = p
        self.roi_dragging = True
        self.rois[self.active_roi_idx] = ROI(fx, fy, fx, fy)
        self._redraw()

    def on_mouse_drag(self, event):
        if not self.roi_dragging:
            return
        roi = self.rois[self.active_roi_idx]
        if roi is None:
            return
        p = self._canvas_to_frame(event.x, event.y)
        if p is None:
            return
        fx, fy = p
        roi.x2 = fx
        roi.y2 = fy
        self._redraw()

    def on_mouse_up(self, _event):
        if not self.roi_dragging:
            return
        self.roi_dragging = False
        self._update_roi_label()
        self._redraw()
        self._update_gl_view(force=True)

    def _draw_frame(self, frame_bgr: np.ndarray):
        self.canvas.delete("all")
        if self.draw_rect is None:
            self._compute_draw_rect(frame_bgr)
        r = self.draw_rect

        self.canvas.create_rectangle(0, 0, self.canvas_w, self.canvas_h, fill="#111111", outline="")

        disp = cv2.resize(frame_bgr, (r.w, r.h), interpolation=cv2.INTER_AREA)
        disp_rgb = cv2.cvtColor(disp, cv2.COLOR_BGR2RGB)
        ppm = self._rgb_to_ppm_bytes(disp_rgb)
        self._tk_img = tk.PhotoImage(data=ppm)
        self.canvas.create_image(r.x, r.y, anchor="nw", image=self._tk_img)

        for i in range(2):
            roi = self.rois[i]
            if roi is not None and roi.is_valid():
                fr = roi.normalized()
                p1 = self._frame_to_canvas(fr.x1, fr.y1)
                p2 = self._frame_to_canvas(fr.x2, fr.y2)
                if p1 and p2:
                    color = self.roi_colors[i]
                    width = 3 if i == self.active_roi_idx else 2
                    self.canvas.create_rectangle(p1[0], p1[1], p2[0], p2[1], outline=color, width=width)
                    self.canvas.create_text(p1[0] + 2, max(0, p1[1] - 16),
                                            anchor="nw", text=f"ROI{i+1}", fill=color)

    @staticmethod
    def _rgb_to_ppm_bytes(rgb: np.ndarray) -> bytes:
        h, w = rgb.shape[:2]
        header = f"P6 {w} {h} 255\n".encode("ascii")
        return header + rgb.tobytes()

    def _redraw(self):
        if self.cur_bgr_proc is not None:
            self._draw_frame(self.cur_bgr_proc)

    def _update_gl_view(self, force: bool):
        if self.gl_view is None:
            return
        g, title = self._get_gl_source_gray()
        if g is None:
            self.gl_view.set_heightmap(None, title=title)
            return
        g = self._downsample_for_gl(g)
        self.gl_view.set_heightmap(g, title=title)

    def _downsample_for_gl(self, g: np.ndarray) -> np.ndarray:
        if g is None or g.size == 0:
            return g
        h, w = g.shape[:2]
        m = int(self._gl_max_size)
        if max(h, w) <= m:
            return g
        s = m / max(h, w)
        nh = max(2, int(round(h * s)))
        nw = max(2, int(round(w * s)))
        return cv2.resize(g, (nw, nh), interpolation=cv2.INTER_AREA)

    # ---------------- Plot refresh ----------------
    def _refresh_plot(self, force_static_x: bool = False):
        if not hasattr(self, "ax1") or not hasattr(self, "canvas_plot") or not hasattr(self, "cursor1"):
            return

        if force_static_x:
            self._setup_static_x_axis()

        cur_t = self.pos_msec / 1000.0
        self.cursor1.set_xdata([cur_t, cur_t])
        self.cursor2.set_xdata([cur_t, cur_t])
        self.cursor3.set_xdata([cur_t, cur_t])

        with self.data_lock:
            d0x = np.array(self.delta_times[0], dtype=np.float64) if self.delta_times[0] else np.array([], dtype=np.float64)
            d0y = np.array(self.delta_vals[0], dtype=np.float64) if self.delta_vals[0] else np.array([], dtype=np.float64)
            d1x = np.array(self.delta_times[1], dtype=np.float64) if self.delta_times[1] else np.array([], dtype=np.float64)
            d1y = np.array(self.delta_vals[1], dtype=np.float64) if self.delta_vals[1] else np.array([], dtype=np.float64)
            sx  = np.array(self.sym_times, dtype=np.float64) if self.sym_times else np.array([], dtype=np.float64)
            sy  = np.array(self.sym_vals, dtype=np.float64) if self.sym_vals else np.array([], dtype=np.float64)

        self.line1.set_data(d0x, d0y)
        self.line2.set_data(d1x, d1y)
        self.line3.set_data(sx,  sy)

        def mean_finite(arr: np.ndarray) -> float:
            if arr.size == 0:
                return 0.0
            aa = arr[np.isfinite(arr)]
            return float(np.mean(aa)) if aa.size else 0.0

        self.mean1.set_ydata([mean_finite(d0y)] * 2)
        self.mean2.set_ydata([mean_finite(d1y)] * 2)
        self.mean3.set_ydata([mean_finite(sy)] * 2)

        # ---------- span utils ----------
        def mask_to_spans(x: np.ndarray, mask: Optional[np.ndarray]) -> List[Tuple[float, float]]:
            if mask is None or x.size < 2:
                return []
            m = np.asarray(mask, dtype=bool)
            n = min(len(m), len(x))
            if n < 2:
                return []
            m = m[:n]
            x = x[:n]
            spans = []
            in_run = False
            t0 = None
            for i in range(n):
                if m[i] and not in_run:
                    in_run = True
                    t0 = float(x[i])
                elif (not m[i]) and in_run:
                    t1 = float(x[i])
                    if t0 is not None and t1 > t0:
                        spans.append((t0, t1))
                    in_run = False
                    t0 = None
            if in_run and t0 is not None:
                t1 = float(x[-1])
                if t1 > t0:
                    spans.append((t0, t1))
            return spans

        def merge_spans(spans: List[Tuple[float, float]], eps: float = 1e-6) -> List[Tuple[float, float]]:
            spans = [(float(a), float(b)) for a, b in spans if b > a]
            if not spans:
                return []
            spans.sort(key=lambda t: t[0])
            out = [spans[0]]
            for a, b in spans[1:]:
                la, lb = out[-1]
                if a <= lb + eps:
                    out[-1] = (la, max(lb, b))
                else:
                    out.append((a, b))
            return out

        def intersect_spans(a: List[Tuple[float, float]], b: List[Tuple[float, float]]) -> List[Tuple[float, float]]:
            a = merge_spans(a)
            b = merge_spans(b)
            i = j = 0
            out = []
            while i < len(a) and j < len(b):
                a0, a1 = a[i]
                b0, b1 = b[j]
                s = max(a0, b0)
                e = min(a1, b1)
                if e > s:
                    out.append((s, e))
                if a1 < b1:
                    i += 1
                else:
                    j += 1
            return merge_spans(out)

        def subtract_spans(a: List[Tuple[float, float]], b: List[Tuple[float, float]]) -> List[Tuple[float, float]]:
            a = merge_spans(a)
            b = merge_spans(b)
            if not a:
                return []
            if not b:
                return a
            out = []
            for a0, a1 in a:
                cur = a0
                for b0, b1 in b:
                    if b1 <= cur:
                        continue
                    if b0 >= a1:
                        break
                    if b0 > cur:
                        out.append((cur, min(b0, a1)))
                    cur = max(cur, b1)
                    if cur >= a1:
                        break
                if cur < a1:
                    out.append((cur, a1))
            return merge_spans(out)

        def clear_patches(patches: List):
            for p in patches:
                try:
                    p.remove()
                except Exception:
                    pass
            patches.clear()

        def draw_spans(ax, store: List, spans: List[Tuple[float, float]], color: str, alpha: float, zorder: int):
            clear_patches(store)
            for (a, b) in spans:
                store.append(ax.axvspan(a, b, color=color, alpha=alpha, zorder=zorder))

        def clear_texts(texts: List):
            for t in texts:
                try:
                    t.remove()
                except Exception:
                    pass
            texts.clear()

        def annotate_spans(ax, store: List, spans: List[Tuple[float, float]], label: str, color: str):
            clear_texts(store)
            if not spans:
                return
            y0, y1 = ax.get_ylim()
            y_pos = y0 + 0.94 * (y1 - y0)
            for (a, b) in spans:
                store.append(
                    ax.text((a + b) * 0.5, y_pos, label,
                            color=color, fontsize=10, fontweight="bold",
                            ha="center", va="top", zorder=8,
                            bbox=dict(facecolor="white", alpha=0.60, edgecolor="none", pad=1.8))
                )

        # masks per graph
        m1 = self._compute_red_mask_no_mean(d0y, self.rise1_var.get(), self.fall1_var.get(), self.min_on1_var.get(), self.min_off1_var.get()) if d0y.size >= 3 else None
        m2 = self._compute_red_mask_no_mean(d1y, self.rise2_var.get(), self.fall2_var.get(), self.min_on2_var.get(), self.min_off2_var.get()) if d1y.size >= 3 else None
        m3 = self._compute_red_mask_no_mean(sy,  self.rise3_var.get(), self.fall3_var.get(), self.min_on3_var.get(), self.min_off3_var.get()) if sy.size >= 3 else None

        spans1 = merge_spans(mask_to_spans(d0x, m1)) if (m1 is not None and d0x.size >= 2) else []
        spans2 = merge_spans(mask_to_spans(d1x, m2)) if (m2 is not None and d1x.size >= 2) else []
        spans3 = merge_spans(mask_to_spans(sx,  m3)) if (m3 is not None and sx.size  >= 2) else []

        detect_spans = intersect_spans(spans1, spans2)
        roi1_only_spans = subtract_spans(spans1, detect_spans)

        draw_spans(self.ax1, self._roi1_red_patches,  roi1_only_spans, color="red",        alpha=0.18, zorder=1)
        draw_spans(self.ax1, self._roi1_blue_patches, detect_spans,    color="dodgerblue", alpha=0.30, zorder=2)
        draw_spans(self.ax2, self._roi2_red_patches,  spans2,          color="red",        alpha=0.18, zorder=1)
        draw_spans(self.ax3, self._sym_red_patches,   spans3,          color="red",        alpha=0.12, zorder=1)

        xs1, ys1, lab1 = self._find_spikes_by_dy(d0x, d0y, self.spike1_var.get())
        xs2, ys2, lab2 = self._find_spikes_by_dy(d1x, d1y, self.spike2_var.get())
        xs3, ys3, lab3 = self._find_spikes_by_dy(sx,  sy,  self.spike3_var.get())

        self.spike_sc1.set_data(xs1.tolist(), ys1.tolist())
        self.spike_sc2.set_data(xs2.tolist(), ys2.tolist())
        self.spike_sc3.set_data(xs3.tolist(), ys3.tolist())

        self._apply_spike_texts(self.ax1, self._spike_texts1, xs1, ys1, lab1)
        self._apply_spike_texts(self.ax2, self._spike_texts2, xs2, ys2, lab2)
        self._apply_spike_texts(self.ax3, self._spike_texts3, xs3, ys3, lab3)

        for ax in (self.ax1, self.ax2, self.ax3):
            ax.relim()
            ax.autoscale_view(scalex=False, scaley=True)

        # DETECT text
        for t in self._detect_texts1:
            try:
                t.remove()
            except Exception:
                pass
        self._detect_texts1.clear()

        if detect_spans:
            y0, y1 = self.ax1.get_ylim()
            y_pos = y0 + 0.90 * (y1 - y0)
            for (a, b) in detect_spans:
                self._detect_texts1.append(
                    self.ax1.text((a + b) * 0.5, y_pos, "DETECT",
                                  color="dodgerblue", fontsize=10, fontweight="bold",
                                  ha="center", va="top", zorder=9,
                                  bbox=dict(facecolor="white", alpha=0.60, edgecolor="none", pad=1.8))
                )

        # 추가: ROI1/ROI2/SYM에서 surge 구간이 “동작 중”임을 확실히 보이게 span 중앙에 SURGE 라벨
        annotate_spans(self.ax1, self._surge_drop_texts1, spans1, "SURGE", "crimson")
        annotate_spans(self.ax2, self._surge_drop_texts2, spans2, "SURGE", "crimson")
        annotate_spans(self.ax3, self._surge_drop_texts3, spans3, "SURGE", "crimson")

        self.canvas_plot.draw_idle()

    def _setup_static_x_axis(self):
        if self.duration_sec > 0:
            self.ax1.set_xlim(0, self.duration_sec)
            self.ax2.set_xlim(0, self.duration_sec)
            self.ax3.set_xlim(0, self.duration_sec)

    def _update_labels(self):
        if not self.cap:
            self.lbl_info.configure(text="No video loaded.")
            self.lbl_frame.configure(text="Frame: -/-  |  Time: -/-")
            return

        base = os.path.basename(self.video_path) if self.video_path else "(video)"
        ema = "ON" if self.temporal_enabled_var.get() else "OFF"
        self.lbl_info.configure(text=f"{base} | FPS {self.fps:.2f} | EMA {ema} (a={self.temporal_alpha:.2f})")

        cur1 = self.frame_idx + 1
        tot = self.frame_count if self.frame_count > 0 else -1
        cur_t = self.pos_msec / 1000.0
        tot_t = self.duration_sec

        if tot > 0:
            self.lbl_frame.configure(text=f"Frame: {cur1}/{tot}  |  Time: {cur_t:.3f}/{tot_t:.3f} s")
        else:
            self.lbl_frame.configure(text=f"Frame: {cur1}/?  |  Time: {cur_t:.3f}/? s")

        if not self.user_seeking:
            self.seek_var.set(cur_t)

    def _clear_timeseries(self):
        with self.data_lock:
            for i in range(2):
                self.delta_times[i].clear()
                self.delta_vals[i].clear()
                self.prev_roi_gray[i] = None
            self.sym_times.clear()
            self.sym_vals.clear()
        self._plot_update_counter = 0

    def _release_video(self):
        if self.cap is not None:
            try:
                self.cap.release()
            except Exception:
                pass
        self.cap = None
        self.video_path = None

    def on_close(self):
        self._alive = False
        self.playing = False
        self.stop_flag = True
        try:
            self._save_all_state()
            self._release_video()
            # Panda3D cleanup
            if self.gl_view is not None:
                try:
                    self.gl_view.userExit()
                except Exception:
                    pass
        finally:
            self.destroy()


if PANDA3D_OK:
    import sys
    from panda3d.core import WindowProperties

    class Panda3DEmbed(tk.Frame):
        """
        Tk Frame 안에 Panda3D 창을 '부모 핸들'로 붙여 넣는 임베딩 래퍼.
        내부적으로 HeightmapPanda3D(ShowBase)를 구동하고, 필요한 API만 위임한다.
        """
        def __init__(self, master, width=520, height=300, **kwargs):
            super().__init__(master, **kwargs)

            self.configure(width=width, height=height)
            self.pack_propagate(False)

            self._width_px = int(width)
            self._height_px = int(height)

            # Panda3D는 시작 시 부모 윈도우 핸들을 알아야 함
            # Windows: HWND(int), X11: XID(hex string), macOS: NSView 포인터
            self.update_idletasks()

            # 플랫폼별 handle 준비
            if sys.platform.startswith("win"):
                parent_handle = int(self.winfo_id())  # HWND as int
            else:
                # X11 계열에서 Tk가 주는 id가 XID로 동작하는 경우가 많음
                parent_handle = str(self.winfo_id())

            # Panda3D 창 속성 세팅(부모에 임베드)
            wp = WindowProperties()
            wp.setOrigin(0, 0)
            wp.setSize(self._width_px, self._height_px)
            wp.setParentWindow(parent_handle)

            # ShowBase 생성 전에 windowType을 none으로 두고, openDefaultWindow로 생성
            # (환경에 따라 pipe 오류가 나면, 이 부분을 조정해야 할 수 있음)
            self._app = HeightmapPanda3D()
            try:
                self._app.openDefaultWindow(props=wp)
                # after opening the window, ensure camera is updated (ShowBase may set camera on window creation)
                try:
                    if hasattr(self._app, '_update_camera'):
                        self._app._update_camera()
                except Exception:
                    pass
                # mark embed as ready and set clear color if possible
                try:
                    if hasattr(self._app, 'win') and getattr(self._app, 'win') is not None:
                        try:
                            self._app.win.setClearColor((0, 0, 0, 1))
                        except Exception:
                            pass
                except Exception:
                    pass
                self._ready = True
                # flush any buffered set_heightmap call
                try:
                    if hasattr(self, '_buffered') and self._buffered is not None:
                        bg, bt = self._buffered
                        try:
                            self._app.set_heightmap(bg, title=bt)
                        except Exception:
                            pass
                        self._buffered = None
                except Exception:
                    pass
            except Exception:
                # 일부 환경에서 openDefaultWindow가 실패할 수 있음 — 무시하고 계속
                pass

            # Tk 리사이즈 이벤트 -> Panda3D 윈도우 리사이즈 반영
            self._resize_after = None
            self.bind("<Configure>", self._on_resize)

            # Start pumping Panda3D tasks periodically so taskMgr runs while Tk mainloop is active
            try:
                self._pump()
            except Exception:
                pass

        def _pump(self):
            try:
                # step Panda3D task manager to apply pending updates and advance tasks
                if hasattr(self._app, 'taskMgr'):
                    try:
                        self._app.taskMgr.step()
                    except Exception:
                        pass
                # optionally render a frame (some backends may need explicit render)
                if hasattr(self._app, 'graphicsEngine') and hasattr(self._app, 'win'):
                    try:
                        self._app.graphicsEngine.renderFrame()
                    except Exception:
                        pass
            except Exception:
                pass
            # schedule next pump (~16ms ~ 60Hz)
            try:
                self.after(16, self._pump)
            except Exception:
                pass

        # ---- VideoDeltaROIApp에서 쓰는 API 위임 ----
        def set_heightmap(self, gray, title: str = ""):
            # If embed is not ready yet, buffer the request to avoid drawing a default window first
            if not getattr(self, '_ready', False):
                try:
                    self._buffered = (None if gray is None else np.array(gray, copy=True), title or "")
                except Exception:
                    self._buffered = (gray, title or "")
                return
            return self._app.set_heightmap(gray, title=title)

        def cycle_camera(self):
            return self._app.cycle_camera()

        def set_camera_preset(self, idx: int):
            return self._app.set_camera_preset(idx)

        def get_camera_preset_name(self) -> str:
            return self._app.get_camera_preset_name()

        def get_state(self) -> dict:
            return self._app.get_state()

        def set_state(self, s: dict):
            return self._app.set_state(s)

        # HeightmapPanda3D가 접근하는 속성들(zoom_mode/invert_lr/invert_ud)
        @property
        def zoom_mode(self):
            return self._app.zoom_mode

        @zoom_mode.setter
        def zoom_mode(self, v):
            self._app.zoom_mode = v

        @property
        def invert_lr(self):
            return self._app.invert_lr

        @invert_lr.setter
        def invert_lr(self, v):
            self._app.invert_lr = bool(v)

        @property
        def invert_ud(self):
            return self._app.invert_ud

        @invert_ud.setter
        def invert_ud(self, v):
            self._app.invert_ud = bool(v)

        def _on_resize(self, event):
            # 너무 잦은 이벤트는 생략할 수도 있음(필요시 디바운스)
            w = max(10, int(event.width))
            h = max(10, int(event.height))
            if w == self._width_px and h == self._height_px:
                return
            self._width_px, self._height_px = w, h
            try:
                wp = WindowProperties()
                wp.setSize(self._width_px, self._height_px)
                self._app.win.requestProperties(wp)
            except Exception:
                pass

        def userExit(self):
            # VideoDeltaROIApp.on_close()에서 호출할 수 있게 제공
            try:
                self._app.userExit()
            except Exception:
                pass


if __name__ == "__main__":
    if not PANDA3D_OK:
        print("Panda3D not available. Install: pip install panda3d")
    app = VideoDeltaROIApp()
    app.mainloop()
