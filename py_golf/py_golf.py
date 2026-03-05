# =========================
# HARDTRACE (must be first)
# =========================
import os, sys, time
print("="*80)
print("[BUILD] 2026-01-19 TRACE-INTEGRATED v6 (golf physics: drag + magnus lift + rolling resistance)")
print("[HARDTRACE] START", time.strftime("%Y-%m-%d %H:%M:%S"))
print("[HARDTRACE] __file__ =", __file__)
print("[HARDTRACE] argv0    =", sys.argv[0])
print("[HARDTRACE] cwd      =", os.getcwd())
print("[HARDTRACE] exe      =", sys.executable)
print("[HARDTRACE] pid      =", os.getpid())
print("="*80)

# =========================
# Logging: file-only (.log)
#  - NO stdout/stderr redirection (prevents Spyder/IPython recursion)
# =========================
import json
import logging
from logging.handlers import RotatingFileHandler

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

def _safe_mkdir(p: str):
    try:
        os.makedirs(p, exist_ok=True)
    except Exception:
        pass

def _flush_all_handlers(logger: logging.Logger):
    for h in logger.handlers:
        try:
            h.flush()
        except Exception:
            pass

def _copy_latest(src: str, dst: str, logger: logging.Logger, initial: bool = False):
    try:
        if initial:
            with open(dst, "w", encoding="utf-8") as f:
                f.write(f"[LATEST_LOG] points to: {src}\n")
                f.write("[LATEST_LOG] file will be filled as program runs.\n")
        else:
            with open(src, "r", encoding="utf-8", errors="replace") as rf:
                content = rf.read()
            with open(dst, "w", encoding="utf-8") as wf:
                wf.write(f"[LATEST_LOG] points to: {src}\n\n")
                wf.write(content)
    except Exception:
        # If logging itself fails, just swallow to avoid recursion/secondary crashes.
        try:
            logger.exception("[log] Failed to update latest_log.txt")
        except Exception:
            pass

def setup_trace_logging(app_name: str = "py_golf"):
    ts = time.strftime("%Y%m%d_%H%M%S")
    logs_dir = os.path.join(BASE_DIR, "logs")
    _safe_mkdir(logs_dir)

    # file extension .log (as requested)
    log_path = os.path.join(logs_dir, f"{app_name}_{ts}.log")
    latest_path = os.path.join(BASE_DIR, "latest_log.log")

    logger = logging.getLogger(app_name)
    logger.setLevel(logging.DEBUG)

    # Prevent duplicate handlers in IDE reruns
    if logger.handlers:
        for h in list(logger.handlers):
            try:
                logger.removeHandler(h)
            except Exception:
                pass

    fmt = logging.Formatter(
        "%(asctime)s.%(msecs)03d [%(levelname)s] %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # FILE ONLY
    fh = RotatingFileHandler(
        log_path,
        maxBytes=10_000_000,
        backupCount=10,
        encoding="utf-8",
    )
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(fmt)
    logger.addHandler(fh)

    # Exception hooks (log to file)
    def excepthook(exc_type, exc, tb):
        try:
            logger.critical("UNHANDLED EXCEPTION", exc_info=(exc_type, exc, tb))
            _flush_all_handlers(logger)
            _copy_latest(log_path, latest_path, logger)
        except Exception:
            pass

    sys.excepthook = excepthook

    import threading
    if hasattr(threading, "excepthook"):
        def thread_excepthook(args):
            try:
                logger.critical(
                    "UNHANDLED THREAD EXCEPTION",
                    exc_info=(args.exc_type, args.exc_value, args.exc_traceback)
                )
                _flush_all_handlers(logger)
                _copy_latest(log_path, latest_path, logger)
            except Exception:
                pass
        threading.excepthook = thread_excepthook

    env_dump = {
        "argv": sys.argv,
        "python": sys.version,
        "executable": sys.executable,
        "cwd": os.getcwd(),
        "__file__": __file__,
        "base_dir": BASE_DIR,
        "CONDA_PREFIX": os.environ.get("CONDA_PREFIX"),
        "PYTHONPATH": os.environ.get("PYTHONPATH"),
        "platform": sys.platform,
    }
    logger.info("[boot] Program start")
    logger.debug("[env] %s", json.dumps(env_dump, ensure_ascii=False))
    logger.info("[boot] BASE_DIR=%s", BASE_DIR)
    logger.info("[boot] Log file: %s", log_path)

    try:
        _copy_latest(log_path, latest_path, logger, initial=True)
        logger.info("[boot] Latest log path: %s", latest_path)
    except Exception:
        try:
            logger.exception("[boot] Failed to init latest_log.log")
        except Exception:
            pass

    _flush_all_handlers(logger)
    return logger, log_path, latest_path

logger, LOG_PATH, LATEST_PATH = setup_trace_logging("py_golf")
logger.info("[boot] Starting GolfSimApp")

# =========================
# Panda3D imports
# =========================
from panda3d.core import loadPrcFileData
loadPrcFileData("", "load-display pandagl")

from panda3d.core import (
    Filename, getModelPath,
    Vec3, Point3,
    AmbientLight, DirectionalLight,
    CardMaker, TransparencyAttrib,
    WindowProperties,
    loadPrcFileData,
    TextNode,
    LineSegs,
    TextureStage,
)
from direct.showbase.ShowBase import ShowBase
from direct.task import Task
from direct.gui.OnscreenText import OnscreenText
from panda3d.bullet import (
    BulletWorld, BulletRigidBodyNode,
    BulletPlaneShape, BulletSphereShape, BulletBoxShape,
)

try:
    from panda3d.core import GeoMipTerrain
    HAS_TERRAIN = True
except Exception:
    HAS_TERRAIN = False

loadPrcFileData("", "notify-level info")
loadPrcFileData("", "notify-level-bullet info")

# =========================
# Config: Course scale / tiling
# =========================
# 1 unit ~ 1 meter
GROUND_HALF = 2000.0        # 4000m x 4000m
TILE_METERS = 8.0           # texture repeats every N meters
FAR_CLIP = 50000.0
NEAR_CLIP = 0.05

# =========================
# Path / Texture utilities
# =========================
logger.info("[init] Base dir resolved")

def ensure_model_path_has_base_dir():
    try:
        fn_dir = Filename.fromOsSpecific(BASE_DIR)
        fn_dir.makeTrueCase()
        fn_dir.makeAbsolute()
        fn_dir.standardize()
        getModelPath().appendDirectory(fn_dir)
        logger.info("[p3d] appended BASE_DIR to model-path: %s", fn_dir.getFullpath())
        logger.debug("[p3d] model-path now: %s", getModelPath().getValue())
    except Exception:
        logger.exception("[p3d] failed to append BASE_DIR to model-path")

ensure_model_path_has_base_dir()

def file_probe(abs_path: str, label: str):
    exists = os.path.exists(abs_path)
    isfile = os.path.isfile(abs_path)
    logger.info("[%s] exists=%s isfile=%s path=%s", label, exists, isfile, abs_path)
    if not exists:
        return
    try:
        size = os.path.getsize(abs_path)
        logger.debug("[%s] size=%s", label, size)
    except Exception:
        logger.exception("[%s] getsize failed", label)

def panda_fn_from_abs(abs_path: str) -> Filename:
    fn = Filename.fromOsSpecific(abs_path)
    fn.makeTrueCase()
    fn.makeAbsolute()
    fn.standardize()
    return fn

def load_texture_trace(loader, rel_name: str):
    abs_path = os.path.join(BASE_DIR, rel_name)
    file_probe(abs_path, f"tex:{rel_name}")
    fn = panda_fn_from_abs(abs_path)
    logger.info("[tex] try load: abs=%s | panda=%s", abs_path, fn.getFullpath())
    try:
        tex = loader.loadTexture(fn)
        logger.info("[tex] loaded=%s", bool(tex))
        return tex
    except Exception:
        logger.exception("[tex] loader.loadTexture crashed")
        return None

# =========================
# Core classes (Course / Ball)
# =========================
class CourseTiles:
    def __init__(self, app: ShowBase):
        logger.info("[init] CourseTiles initialization")
        self.app = app
        self.ground_np = None
        self.terrain = None

        self._build_ground_large_tiled()
        self._try_load_heightmap()

    def _build_ground_large_tiled(self):
        cm = CardMaker("ground")
        cm.setFrame(-GROUND_HALF, GROUND_HALF, -GROUND_HALF, GROUND_HALF)
        self.ground_np = self.app.render.attachNewNode(cm.generate())
        self.ground_np.setP(-90)
        self.ground_np.setZ(0.0)
        self.ground_np.setColor(0.15, 0.55, 0.20, 1.0)

        tex = load_texture_trace(self.app.loader, "ground.png")
        if not tex:
            logger.error("[error] Ground texture load failed | fallback color applied")
            return

        repeat = max(1.0, (2.0 * GROUND_HALF) / max(0.1, TILE_METERS))
        self.ground_np.setTexture(tex, 1)
        self.ground_np.setTexScale(TextureStage.getDefault(), repeat, repeat)

        logger.info("[ok] Ground texture tiled: repeat=%.2f (tile=%.1fm) ground=%.0fm x %.0fm",
                    repeat, TILE_METERS, 2*GROUND_HALF, 2*GROUND_HALF)

    def _try_load_heightmap(self):
        hm_abs = os.path.join(BASE_DIR, "heightmap.png")
        if os.path.isfile(hm_abs) and HAS_TERRAIN:
            try:
                file_probe(hm_abs, "heightmap")
                hm_fn = panda_fn_from_abs(hm_abs)
                terrain = GeoMipTerrain("terrain")
                terrain.setHeightfield(hm_fn)
                terrain.setBlockSize(32)
                terrain.setNear(120)
                terrain.setFar(6000)
                terrain.setFocalPoint(self.app.camera)
                terrain_root = terrain.getRoot()
                terrain_root.reparentTo(self.app.render)
                terrain_root.setSz(30.0)
                terrain_root.setPos(-GROUND_HALF, -GROUND_HALF, -2.0)
                terrain.generate()
                self.terrain = terrain
                logger.info("[ok] Heightmap loaded (terrain enabled)")
            except Exception:
                logger.exception("[error] Heightmap load failed")
        else:
            logger.info("[ok] Heightmap not used (missing or GeoMipTerrain unavailable)")

class BallSim:
    def __init__(self, app: ShowBase, world: BulletWorld):
        logger.info("[init] BallSim created")
        self.app = app
        self.world = world

        self.ball_np = None
        self.stroke_count = 0
        self.is_charging = False
        self.power = 0.0

        # Real-ish golf ball parameters
        self.ball_radius = 0.02135   # meters
        self.ball_mass = 0.0459      # kg

        # Shot tuning (impulse in N*s)
        # v = J/m => J ~ m*v. Typical v ~ 40~80 m/s => J ~ 1.8~3.7 N*s
        self.loft_deg = 14.0
        self.impulse_min = 1.3
        self.impulse_max = 4.2

        # Backspin (rad/s): 3000~5000 rpm => 314~524 rad/s
        self.spin_min_rads = 160.0
        self.spin_max_rads = 520.0

        self._create_ball()

    def _create_ball(self):
        shape = BulletSphereShape(self.ball_radius)
        body = BulletRigidBodyNode("Ball")
        body.addShape(shape)
        body.setMass(self.ball_mass)

        # Contact tuning
        body.setFriction(0.25)
        body.setRestitution(0.25)
        body.setLinearDamping(0.008)
        body.setAngularDamping(0.015)

        self.ball_np = self.app.render.attachNewNode(body)
        self.ball_np.setPos(0, -10, self.ball_radius + 0.01)
        self.world.attachRigidBody(body)

        vis = None
        try:
            vis = self.app.loader.loadModel("models/misc/sphere")
        except Exception:
            logger.exception("[ball] loadModel sphere failed")

        if vis:
            vis.setScale(self.ball_radius)
            vis.setColor(0.98, 0.98, 0.98, 1)
            vis.reparentTo(self.ball_np)
            logger.info("[ok] Ball visual: sphere model")
        else:
            logger.warning("[ball] sphere model missing -> no visual attached")

        logger.info("[ok] Ball created r=%.5fm mass=%.4fkg pos=%s",
                    self.ball_radius, self.ball_mass, self.ball_np.getPos())

    def ball_is_moving(self):
        v = self.ball_np.node().getLinearVelocity()
        return v.length() > 0.25

    def shot_direction(self):
        forward = self.app.camera.getQuat(self.app.render).getForward()
        flat = Vec3(forward.x, forward.y, 0.0)
        if flat.lengthSquared() < 1e-8:
            flat = Vec3(0, 1, 0)
        flat.normalize()

        import math
        loft = math.radians(self.loft_deg)
        dir3 = Vec3(flat.x * math.cos(loft), flat.y * math.cos(loft), math.sin(loft))
        if dir3.lengthSquared() < 1e-8:
            dir3 = Vec3(0, 1, 0.2)
        dir3.normalize()
        return dir3

    def apply_shot(self):
        if self.ball_is_moving():
            logger.info("[shot] blocked (ball moving)")
            return

        body = self.ball_np.node()
        body.setActive(True)
        try:
            body.wakeUp()
        except Exception:
            pass

        # reset existing motion
        body.clearForces()
        body.setLinearVelocity(Vec3(0, 0, 0))
        body.setAngularVelocity(Vec3(0, 0, 0))

        power01 = max(0.0, min(1.0, self.power))
        impulse = self.impulse_min + (self.impulse_max - self.impulse_min) * power01
        direction = self.shot_direction()

        # launch
        body.applyCentralImpulse(direction * impulse)

        # backspin: axis ~ camera right
        right = self.app.camera.getQuat(self.app.render).getRight()
        spin = self.spin_min_rads + (self.spin_max_rads - self.spin_min_rads) * power01
        body.setAngularVelocity(right * spin)

        logger.info(
            "[shot] power=%.3f impulse=%.3fN*s loft=%.1fdeg spin=%.1frad/s dir=(%.3f, %.3f, %.3f)",
            power01, impulse, self.loft_deg, spin, direction.x, direction.y, direction.z
        )

        self.stroke_count += 1
        self.power = 0.0

# =========================
# Main App
# =========================
class GolfSimApp(ShowBase):
    def __init__(self):
        super().__init__()

        self.disableMouse()
        self.setBackgroundColor(0.55, 0.75, 0.95, 1.0)

        # Lens far clip for huge course
        try:
            lens = self.cam.node().getLens()
            lens.setNear(NEAR_CLIP)
            lens.setFar(FAR_CLIP)
            logger.info("[cam] lens near=%.3f far=%.1f", NEAR_CLIP, FAR_CLIP)
        except Exception:
            logger.exception("[cam] lens setup failed")

        # Orbit params (mode 4)
        self.cam_yaw = 35.0
        self.cam_pitch = 18.0
        self.cam_dist = 55.0
        self.mouse_last = None

        # Camera safety
        self.camera_z_min = 1.2
        self.pitch_min = 8.0
        self.pitch_max = 70.0

        # Camera modes
        self.cam_mode = 1
        self.static_cam_pos = Vec3(0, -2600, 1400)
        self.static_look_at = Vec3(0, 0, 0)

        self.follow_offset = Vec3(0, -120, 55)
        self.follow_smooth = 0.10
        self._follow_cam_pos = None

        self.chase_dist = 70.0
        self.chase_height = 30.0
        self.chase_smooth = 0.14
        self._chase_cam_pos = None

        # Trail
        self.trail_enabled = True
        self.trail_max_points = 4000
        self.trail_min_dist = 1.5
        self._trail_points = []
        self._trail_last = None
        self._trail_np = None
        self._trail_dirty = True
        self._init_trail()

        # Aerodynamics / rolling resistance
        self.air_rho = 1.225          # kg/m^3
        self.Cd = 0.25                # drag coefficient
        self.Cl_cap = 0.75            # lift cap
        self.roll_mu = 0.020          # rolling resistance (tune)
        self.spin_decay = 0.22        # 1/s (angular velocity decay)
        self.stop_speed = 0.22        # m/s hard stop threshold

        self._setup_window()
        self._setup_lights()

        # Physics
        self.world = BulletWorld()
        self.world.setGravity(Vec3(0, 0, -9.81))
        self._setup_physics_ground()
        self._setup_obstacles()

        # Course + Ball
        self.course = CourseTiles(self)
        self.ball = BallSim(self, self.world)

        # UI + input
        self._setup_ui()
        self._setup_input()

        logger.info("[ok] Initialization complete, running main loop")
        self.taskMgr.add(self._update, "update")
        logger.info("[cam] initial mode=%d", self.cam_mode)
        _flush_all_handlers(logger)

    def _setup_window(self):
        props = WindowProperties()
        props.setTitle("Py Golf (Golf Physics: Drag + Magnus + Rolling)")
        props.setCursorHidden(False)
        self.win.requestProperties(props)

    def _setup_lights(self):
        amb = AmbientLight("amb"); amb.setColor((0.55, 0.55, 0.55, 1))
        amb_np = self.render.attachNewNode(amb); self.render.setLight(amb_np)

        sun = DirectionalLight("sun"); sun.setColor((0.9, 0.9, 0.85, 1))
        sun_np = self.render.attachNewNode(sun); sun_np.setHpr(45, -55, 0)
        self.render.setLight(sun_np)

        logger.info("[ok] Lights configured")

    def _setup_physics_ground(self):
        plane = BulletPlaneShape(Vec3(0, 0, 1), 0)
        ground_body = BulletRigidBodyNode("Ground")
        ground_body.addShape(plane)

        # ground contact tuning
        ground_body.setFriction(0.55)
        ground_body.setRestitution(0.15)

        self.ground_phys_np = self.render.attachNewNode(ground_body)
        self.world.attachRigidBody(ground_body)
        logger.info("[ok] Physics ground configured")

    def _setup_obstacles(self):
        # small reference wall near tee
        self._add_box_obstacle("wall1", Point3(60, 120, 2.0), (0, 0, 0), Vec3(12.0, 0.8, 2.0))
        logger.info("[ok] Obstacles configured")

    def _add_box_obstacle(self, name, pos, hpr, half_extents):
        shape = BulletBoxShape(half_extents)
        body = BulletRigidBodyNode(name)
        body.addShape(shape)
        body.setMass(0.0)
        body.setFriction(0.85)
        body.setRestitution(0.2)
        np = self.render.attachNewNode(body)
        np.setPos(pos); np.setHpr(*hpr)
        self.world.attachRigidBody(body)

    # ---------- UI ----------
    def _setup_ui(self):
        self.ui_status = OnscreenText(
            text="CAM 1:STATIC | 1..4 camera | SPACE charge/shoot | R reset | ESC quit",
            pos=(-1.32, 0.93), scale=0.045, fg=(1, 1, 1, 1),
            align=TextNode.ALeft, mayChange=True
        )
        self.ui_power = OnscreenText(
            text="Power: 0%",
            pos=(-1.32, 0.86), scale=0.05, fg=(1, 1, 1, 1),
            align=TextNode.ALeft, mayChange=True
        )
        self.ui_score = OnscreenText(
            text="Strokes: 0",
            pos=(1.15, 0.93), scale=0.06, fg=(1, 1, 1, 1),
            align=TextNode.ARight, mayChange=True
        )
        self.ui_pos = OnscreenText(
            text="Ball: (0.0, 0.0, 0.0) m",
            pos=(1.15, 0.86), scale=0.045, fg=(1, 1, 1, 1),
            align=TextNode.ARight, mayChange=True
        )
        self.ui_log = OnscreenText(
            text=f"LOG: {LATEST_PATH}",
            pos=(-1.32, 0.79), scale=0.04, fg=(1, 1, 1, 1),
            align=TextNode.ALeft, mayChange=True
        )

        cm = CardMaker("bar")
        cm.setFrame(0, 0.65, 0, 0.05)
        self.power_bg = self.aspect2d.attachNewNode(cm.generate())
        self.power_bg.setPos(-0.65, 0, 0.78)
        self.power_bg.setTransparency(TransparencyAttrib.M_alpha)
        self.power_bg.setColor(0, 0, 0, 0.35)

        self.power_fg = self.aspect2d.attachNewNode(cm.generate())
        self.power_fg.setPos(-0.65, 0, 0.78)
        self.power_fg.setTransparency(TransparencyAttrib.M_alpha)
        self.power_fg.setColor(0.2, 0.9, 0.25, 0.85)
        self.power_fg.setSx(0.001)

    # ---------- Input ----------
    def _setup_input(self):
        self.accept("escape", self._exit_safe)
        self.accept("r", self._reset_ball)

        self.accept("space", self._start_charge)
        self.accept("space-up", self._release_shot)

        # orbit mode controls (mode 4)
        self.accept("mouse1", self._mouse_down)
        self.accept("mouse1-up", self._mouse_up)
        self.accept("wheel_up", lambda: self._zoom(-8.0))
        self.accept("wheel_down", lambda: self._zoom(+8.0))

        # camera modes
        self.accept("1", lambda: self.set_camera_mode(1))
        self.accept("2", lambda: self.set_camera_mode(2))
        self.accept("3", lambda: self.set_camera_mode(3))
        self.accept("4", lambda: self.set_camera_mode(4))

    def set_camera_mode(self, mode: int):
        self.cam_mode = int(mode)
        self.mouse_last = None
        self._follow_cam_pos = None
        self._chase_cam_pos = None

        mode_name = {
            1: "STATIC_OVERVIEW",
            2: "FOLLOW_SMOOTH",
            3: "CHASE",
            4: "ORBIT_FREE",
        }.get(self.cam_mode, "UNKNOWN")

        self.ui_status.setText(
            f"CAM {self.cam_mode}:{mode_name} | 1..4 camera | LMB orbit(4) | SPACE shoot | R reset"
        )
        logger.info("[cam] switched mode=%d %s", self.cam_mode, mode_name)
        _flush_all_handlers(logger)

    # ---------- Trail ----------
    def _init_trail(self):
        try:
            segs = LineSegs("trail")
            segs.setThickness(2.0)
            self._trail_np = self.render.attachNewNode(segs.create(False))
            self._trail_np.setTransparency(TransparencyAttrib.M_alpha)
            self._trail_np.setColor(1, 1, 1, 0.85)
            logger.info("[trail] initialized")
        except Exception:
            logger.exception("[trail] init failed")
            self.trail_enabled = False

    def _trail_reset(self):
        self._trail_points = []
        self._trail_last = None
        self._trail_dirty = True
        try:
            if self._trail_np:
                self._trail_np.removeNode()
            segs = LineSegs("trail")
            segs.setThickness(2.0)
            self._trail_np = self.render.attachNewNode(segs.create(False))
            self._trail_np.setTransparency(TransparencyAttrib.M_alpha)
            self._trail_np.setColor(1, 1, 1, 0.85)
        except Exception:
            logger.exception("[trail] reset failed")

    def _trail_add_point(self, p: Vec3):
        if not self.trail_enabled:
            return
        if self._trail_last is None:
            self._trail_points.append(Vec3(p))
            self._trail_last = Vec3(p)
            self._trail_dirty = True
            return
        if (p - self._trail_last).length() >= self.trail_min_dist:
            self._trail_points.append(Vec3(p))
            self._trail_last = Vec3(p)
            if len(self._trail_points) > self.trail_max_points:
                self._trail_points = self._trail_points[-self.trail_max_points:]
            self._trail_dirty = True

    def _trail_rebuild(self):
        if not self._trail_dirty or not self.trail_enabled:
            return
        if len(self._trail_points) < 2:
            return
        try:
            if self._trail_np:
                self._trail_np.removeNode()

            segs = LineSegs("trail")
            segs.setThickness(2.0)
            segs.moveTo(self._trail_points[0])
            for pt in self._trail_points[1:]:
                segs.drawTo(pt)

            self._trail_np = self.render.attachNewNode(segs.create(False))
            self._trail_np.setTransparency(TransparencyAttrib.M_alpha)
            self._trail_np.setColor(1, 1, 1, 0.85)
            self._trail_dirty = False
        except Exception:
            logger.exception("[trail] rebuild failed")

    # ---------- Physics: Drag + Magnus + Rolling ----------
    def _apply_ball_aero_and_roll(self, dt: float):
        body = self.ball.ball_np.node()
        v = body.getLinearVelocity()
        speed = v.length()
        if speed < 1e-4:
            return

        m = self.ball.ball_mass
        r = self.ball.ball_radius
        A = 3.141592653589793 * (r * r)

        # Drag: Fd = -0.5*rho*Cd*A*|v|^2 * v_hat
        drag_mag = 0.5 * self.air_rho * self.Cd * A * speed * speed
        drag_dir = -v
        if drag_dir.lengthSquared() > 1e-12:
            drag_dir.normalize()
        Fd = drag_dir * drag_mag

        # Magnus lift (simplified)
        omega = body.getAngularVelocity()
        w = omega.length()

        # spin ratio SR = r*w / v
        SR = (r * w) / max(0.5, speed)

        # lift coefficient approx
        Cl = min(self.Cl_cap, 1.2 * SR)

        # lift direction: omega x v
        lift_dir = omega.cross(v)
        if lift_dir.lengthSquared() > 1e-12:
            lift_dir.normalize()
        lift_mag = 0.5 * self.air_rho * Cl * A * speed * speed
        Fl = lift_dir * lift_mag

        # Apply aerodynamic forces
        body.applyCentralForce(Fd + Fl)

        # Spin decay (air causes spin to drop)
        if w > 1e-6:
            decay = max(0.0, 1.0 - self.spin_decay * dt)
            body.setAngularVelocity(omega * decay)

        # Rolling resistance near ground
        pos = self.ball.ball_np.getPos(self.render)
        near_ground = pos.z <= (r + 0.02)
        vertical_small = abs(v.z) < 0.6

        if near_ground and vertical_small:
            vh = Vec3(v.x, v.y, 0.0)
            hs = vh.length()
            if hs > 1e-6:
                vh.normalize()
                Fr = -vh * (self.roll_mu * m * 9.81)
                body.applyCentralForce(Fr)

            # Hard stop to avoid infinite rolling on infinite plane
            if speed < self.stop_speed:
                body.setLinearVelocity(Vec3(0, 0, 0))
                body.setAngularVelocity(Vec3(0, 0, 0))
                body.clearForces()

    # ---------- Controls ----------
    def _exit_safe(self):
        logger.info("[ui] exit requested")
        _flush_all_handlers(logger)
        try:
            _copy_latest(LOG_PATH, LATEST_PATH, logger)
        except Exception:
            pass
        self.userExit()

    def _reset_ball(self):
        body = self.ball.ball_np.node()
        body.clearForces()
        body.setLinearVelocity(Vec3(0, 0, 0))
        body.setAngularVelocity(Vec3(0, 0, 0))
        self.ball.ball_np.setPos(0, -10, self.ball.ball_radius + 0.01)
        self.ball.stroke_count = 0
        self.ball.power = 0.0
        self.ui_score.setText("Strokes: 0")
        self._trail_reset()
        logger.info("[ui] reset ball")
        _flush_all_handlers(logger)

    def _start_charge(self):
        logger.info("[input] space down")
        if self.ball.ball_is_moving():
            logger.info("[input] charge blocked (ball moving)")
            return
        self.ball.is_charging = True

    def _release_shot(self):
        logger.info("[input] space up | charging=%s power=%.3f moving=%s",
                    self.ball.is_charging, self.ball.power, self.ball.ball_is_moving())
        if not self.ball.is_charging:
            return
        self.ball.is_charging = False
        if self.ball.ball_is_moving():
            self.ball.power = 0.0
            return
        self.ball.apply_shot()
        self.ui_score.setText(f"Strokes: {self.ball.stroke_count}")
        _flush_all_handlers(logger)

    def _mouse_down(self):
        if self.mouseWatcherNode.hasMouse():
            m = self.mouseWatcherNode.getMouse()
            self.mouse_last = (m.getX(), m.getY())

    def _mouse_up(self):
        self.mouse_last = None

    def _zoom(self, delta):
        self.cam_dist = max(15.0, min(1200.0, self.cam_dist + delta))

    # ---------- Camera ----------
    def _update_camera(self):
        ball_pos = self.ball.ball_np.getPos(self.render)

        if self.cam_mode == 1:
            self.camera.setPos(self.static_cam_pos)
            self.camera.lookAt(self.static_look_at)
            return

        if self.cam_mode == 2:
            desired = ball_pos + self.follow_offset
            if self._follow_cam_pos is None:
                self._follow_cam_pos = Vec3(desired)
            self._follow_cam_pos = self._follow_cam_pos + (desired - self._follow_cam_pos) * self.follow_smooth
            if self._follow_cam_pos.z < self.camera_z_min:
                self._follow_cam_pos.z = self.camera_z_min
            self.camera.setPos(self._follow_cam_pos)
            self.camera.lookAt(ball_pos + Vec3(0, 0, self.ball.ball_radius))
            return

        if self.cam_mode == 3:
            v = self.ball.ball_np.node().getLinearVelocity()
            if v.length() < 0.4:
                forward = self.camera.getQuat(self.render).getForward()
                flat = Vec3(forward.x, forward.y, 0)
                if flat.lengthSquared() < 1e-8:
                    flat = Vec3(0, 1, 0)
                flat.normalize()
                back_dir = -flat
            else:
                flat_v = Vec3(v.x, v.y, 0)
                if flat_v.lengthSquared() < 1e-8:
                    flat_v = Vec3(0, 1, 0)
                flat_v.normalize()
                back_dir = -flat_v

            desired = ball_pos + back_dir * self.chase_dist + Vec3(0, 0, self.chase_height)
            if self._chase_cam_pos is None:
                self._chase_cam_pos = Vec3(desired)
            self._chase_cam_pos = self._chase_cam_pos + (desired - self._chase_cam_pos) * self.chase_smooth
            if self._chase_cam_pos.z < self.camera_z_min:
                self._chase_cam_pos.z = self.camera_z_min
            self.camera.setPos(self._chase_cam_pos)
            self.camera.lookAt(ball_pos + Vec3(0, 0, self.ball.ball_radius))
            return

        # orbit mode 4
        if self.mouse_last is not None and self.mouseWatcherNode.hasMouse():
            mx, my = self.mouseWatcherNode.getMouse().getX(), self.mouseWatcherNode.getMouse().getY()
            lx, ly = self.mouse_last
            dx, dy = (mx - lx), (my - ly)
            self.mouse_last = (mx, my)
            self.cam_yaw -= dx * 110.0
            self.cam_pitch += dy * 90.0
            self.cam_pitch = max(self.pitch_min, min(self.pitch_max, self.cam_pitch))

        import math
        yaw_r = math.radians(self.cam_yaw)
        pitch_r = math.radians(self.cam_pitch)

        x = self.cam_dist * math.sin(yaw_r) * math.cos(pitch_r)
        y = -self.cam_dist * math.cos(yaw_r) * math.cos(pitch_r)
        z = self.cam_dist * math.sin(pitch_r)

        cam_pos = ball_pos + Vec3(x, y, z)
        if cam_pos.z < self.camera_z_min:
            cam_pos.z = self.camera_z_min

        self.camera.setPos(cam_pos)
        self.camera.lookAt(ball_pos + Vec3(0, 0, self.ball.ball_radius))

    # ---------- Update loop ----------
    def _update(self, task: Task):
        dt = globalClock.getDt()
        dt = max(0.0, min(1/30.0, dt))

        # Apply aero + rolling forces BEFORE stepping physics
        try:
            self._apply_ball_aero_and_roll(dt)
        except Exception:
            logger.exception("[aero] apply failed")

        self.world.doPhysics(dt, 4, 1.0/120.0)
        self._update_camera()

        # Charging power
        if self.ball.is_charging and not self.ball.ball_is_moving():
            self.ball.power = min(1.0, self.ball.power + 0.55 * dt)

        self.ui_power.setText(f"Power: {int(self.ball.power * 100)}%")
        self.power_fg.setSx(max(0.001, self.ball.power))

        bp = self.ball.ball_np.getPos(self.render)
        self.ui_pos.setText(f"Ball: ({bp.x:.1f}, {bp.y:.1f}, {bp.z:.1f}) m")

        try:
            if getattr(self.course, "terrain", None) is not None:
                self.course.terrain.update()
        except Exception:
            logger.exception("[terrain] update failed")

        # Trail update
        try:
            self._trail_add_point(bp)
            if int(task.time * 10) != int((task.time - dt) * 10):
                self._trail_rebuild()
        except Exception:
            logger.exception("[trail] update failed")

        # latest_log snapshot every ~2 sec
        if int(task.time * 0.5) != int((task.time - dt) * 0.5):
            try:
                _copy_latest(LOG_PATH, LATEST_PATH, logger)
            except Exception:
                pass

        return Task.cont

if __name__ == "__main__":
    try:
        app = GolfSimApp()
        app.run()
    except Exception:
        try:
            logger.critical("FATAL in main()", exc_info=True)
            _flush_all_handlers(logger)
            _copy_latest(LOG_PATH, LATEST_PATH, logger)
        except Exception:
            pass
        raise
