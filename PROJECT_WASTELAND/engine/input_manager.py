"""Logical action input layer.

Game code only asks for actions ("attack", "dig", ...). Physical keys / pad buttons are
resolved through bindings that can be changed at runtime (settings → controls).

Gamepad support uses the pygame Joystick API for discovery, hot-plug and rumble. When SDL
knows a GameController mapping for the device, the standardized Xbox layout is read through
it (robust across XInput / RawInput / Bluetooth drivers); otherwise the physical layout from
data/input_profiles.json is used.
"""
import math
import logging
import os

import pygame

log = logging.getLogger(__name__)

try:
    from pygame._sdl2 import controller as sdl_controller
except Exception:  # pragma: no cover - very old pygame builds
    sdl_controller = None

PAD_BUTTONS = ["A", "B", "X", "Y", "LB", "RB", "VIEW", "MENU", "L3", "R3",
               "DPAD_UP", "DPAD_DOWN", "DPAD_LEFT", "DPAD_RIGHT"]
PAD_VIRTUAL = ["LT", "RT", "LS_UP", "LS_DOWN", "LS_LEFT", "LS_RIGHT"]

_CTRL_BUTTONS = {
    "A": "CONTROLLER_BUTTON_A", "B": "CONTROLLER_BUTTON_B", "X": "CONTROLLER_BUTTON_X", "Y": "CONTROLLER_BUTTON_Y",
    "LB": "CONTROLLER_BUTTON_LEFTSHOULDER", "RB": "CONTROLLER_BUTTON_RIGHTSHOULDER",
    "VIEW": "CONTROLLER_BUTTON_BACK", "MENU": "CONTROLLER_BUTTON_START",
    "L3": "CONTROLLER_BUTTON_LEFTSTICK", "R3": "CONTROLLER_BUTTON_RIGHTSTICK",
    "DPAD_UP": "CONTROLLER_BUTTON_DPAD_UP", "DPAD_DOWN": "CONTROLLER_BUTTON_DPAD_DOWN",
    "DPAD_LEFT": "CONTROLLER_BUTTON_DPAD_LEFT", "DPAD_RIGHT": "CONTROLLER_BUTTON_DPAD_RIGHT",
}
_CTRL_AXES = {"LX": "CONTROLLER_AXIS_LEFTX", "LY": "CONTROLLER_AXIS_LEFTY", "RX": "CONTROLLER_AXIS_RIGHTX",
              "RY": "CONTROLLER_AXIS_RIGHTY", "LT": "CONTROLLER_AXIS_TRIGGERLEFT", "RT": "CONTROLLER_AXIS_TRIGGERRIGHT"}


def key_code(name: str):
    """'K_w' → pygame key int, 'MOUSE_1' → 'MOUSE_1' (kept as string)."""
    if name.startswith("MOUSE_"):
        return name
    return getattr(pygame, name, None)


def key_label(code) -> str:
    if isinstance(code, str):
        return {"MOUSE_1": "LMB", "MOUSE_2": "MMB", "MOUSE_3": "RMB"}.get(code, code)
    name = pygame.key.name(code)
    return {"left shift": "Shift", "space": "Space", "escape": "Esc", "return": "Enter", "tab": "Tab"}.get(name, name.upper())


def pad_button_label(name: str, pad_name: str = "") -> str:
    """Printed label for a logical pad button. Xbox 360 pads say BACK/START instead of VIEW/MENU."""
    if "360" in pad_name:
        return {"VIEW": "BACK", "MENU": "START"}.get(name, name)
    return name


def apply_radial_deadzone(x: float, y: float, deadzone: float) -> tuple[float, float]:
    mag = math.hypot(x, y)
    if mag < deadzone:
        return 0.0, 0.0
    scaled = min(1.0, (mag - deadzone) / (1.0 - deadzone))
    return x / mag * scaled, y / mag * scaled


class Pad:
    """One connected gamepad, polled into a dict of logical inputs."""

    def __init__(self, device_index: int, layout: dict):
        self.joystick = pygame.joystick.Joystick(device_index)
        self.instance_id = self.joystick.get_instance_id()
        self.name = self.joystick.get_name()
        self.layout = layout
        self.controller = None
        if sdl_controller is not None:
            try:
                if sdl_controller.is_controller(device_index):
                    self.controller = sdl_controller.Controller(device_index)
            except Exception as exc:  # mapping problems must never crash the game
                log.warning("GameController mapping unavailable for %s: %s", self.name, exc)
        self._trigger_neg_rest = None

    def close(self):
        try:
            if self.controller is not None:
                self.controller.quit()
        except Exception:
            pass
        try:
            self.joystick.quit()
        except Exception:
            pass

    def _norm_trigger(self, v: float) -> float:
        if self._trigger_neg_rest is None:
            self._trigger_neg_rest = v < -0.5
        elif v < -0.1:
            self._trigger_neg_rest = True
        if self._trigger_neg_rest:
            return max(0.0, (v + 1.0) * 0.5)
        return max(0.0, v)

    def poll(self) -> dict:
        state = {}
        try:
            if self.controller is not None:
                for name, const in _CTRL_BUTTONS.items():
                    state[name] = bool(self.controller.get_button(getattr(pygame, const)))
                for name, const in _CTRL_AXES.items():
                    raw = self.controller.get_axis(getattr(pygame, const)) / 32767.0
                    state[name] = max(-1.0, min(1.0, raw))
                return state
            js = self.joystick
            nb = js.get_numbuttons()
            for name, idx in self.layout["buttons"].items():
                state[name] = bool(js.get_button(idx)) if idx < nb else False
            na = js.get_numaxes()
            for name, idx in self.layout["axes"].items():
                v = js.get_axis(idx) if idx < na else 0.0
                state[name] = self._norm_trigger(v) if name in ("LT", "RT") else v
            hx, hy = (js.get_hat(self.layout.get("hat", 0)) if js.get_numhats() else (0, 0))
            state["DPAD_UP"], state["DPAD_DOWN"] = hy > 0, hy < 0
            state["DPAD_LEFT"], state["DPAD_RIGHT"] = hx < 0, hx > 0
        except pygame.error:
            return {}
        return state

    def rumble(self, low: float, high: float, ms: int):
        try:
            if self.controller is not None:
                self.controller.rumble(low, high, ms)
            else:
                self.joystick.rumble(low, high, ms)
        except Exception:
            pass


class InputManager:
    def __init__(self, profiles: dict, settings: dict):
        self.profiles = profiles
        self.settings = settings
        self.layout = profiles["gamepad_layouts"][profiles["default_layout"]]
        self.kb_bindings: dict[str, list] = {}
        self.pad_bindings: dict[str, list] = {}
        self.load_bindings(settings.get("bindings"))

        self.keys_down: set = set()
        self._pressed: set = set()
        self._released: set = set()
        self._held: set = set()
        self._prev_held: set = set()
        self.pads: dict[int, Pad] = {}
        self.active_pad: Pad | None = None
        self.pad_state: dict = {}
        self.prev_pad_digital: set = set()
        self.pad_digital: set = set()
        self.last_device = "kbm"
        self.mouse_pos = (0, 0)
        self.mouse_moved = False
        self.mouse_wheel = 0
        self.text_events: list = []
        self.move_smooth = [0.0, 0.0]
        self.aim_vec = (1.0, 0.0)  # last aim direction from right stick (screen space)
        self.aim_stick_active = False
        self.pad_connected_event = None  # "connected" / "disconnected" (consumed by UI)
        self.capture_device = None      # rebinding capture mode
        self.captured = None
        self.quit_requested = False
        self.resized = None

    # ------------------------------------------------------------------ bindings
    def load_bindings(self, custom: dict | None):
        defaults = self.profiles["default_bindings"]
        kb = dict(defaults["keyboard"])
        pad = dict(defaults["gamepad"])
        if custom:
            kb.update(custom.get("keyboard", {}))
            pad.update(custom.get("gamepad", {}))
        self.kb_bindings = {a: [key_code(n) for n in names if key_code(n) is not None]
                            for a, names in kb.items()}
        self.pad_bindings = {a: list(v) for a, v in pad.items() if isinstance(v, list)}
        self._kb_names = kb

    def export_bindings(self) -> dict:
        defaults = self.profiles["default_bindings"]
        kb_out = {}
        for action, codes in self.kb_bindings.items():
            kb_out[action] = [self._code_name(c) for c in codes]
        pad_out = dict(self.pad_bindings)
        return {"keyboard": {a: v for a, v in kb_out.items() if v != defaults["keyboard"].get(a)},
                "gamepad": {a: v for a, v in pad_out.items() if v != defaults["gamepad"].get(a)}}

    @staticmethod
    def _code_name(code):
        if isinstance(code, str):
            return code
        for attr in dir(pygame):
            if attr.startswith("K_") and getattr(pygame, attr) == code:
                return attr
        return str(code)

    def rebind(self, action: str, device: str, code):
        if device == "kbm":
            self.kb_bindings[action] = [code]
        else:
            self.pad_bindings[action] = [code]

    def reset_bindings(self):
        self.load_bindings(None)

    def binding_label(self, action: str, device: str | None = None) -> str:
        device = device or self.last_device
        if device == "pad":
            names = self.pad_bindings.get(action) or []
            return names[0] if names else "-"
        codes = self.kb_bindings.get(action) or []
        return key_label(codes[0]) if codes else "-"

    # ------------------------------------------------------------------ devices
    def init_pads(self):
        try:
            pygame.joystick.init()
            if sdl_controller is not None:
                sdl_controller.init()
            for i in range(pygame.joystick.get_count()):
                self._add_pad(i)
        except Exception as exc:
            log.warning("Gamepad init failed: %s", exc)

    def _add_pad(self, device_index: int):
        if os.environ.get("WASTELAND_NO_PAD"):  # automated tests must not read real hardware
            return
        # SDL also sends JOYDEVICEADDED for pads already opened by init_pads. Joystick objects for
        # one device share the SDL handle, so a duplicate must be dropped without quit()/close(),
        # which would close the live pad too.
        try:
            if pygame.joystick.Joystick(device_index).get_instance_id() in self.pads:
                return
            pad = Pad(device_index, self.layout)
        except pygame.error as exc:
            log.warning("Could not open pad %s: %s", device_index, exc)
            return
        self.pads[pad.instance_id] = pad
        if self.active_pad is None:
            self.active_pad = pad
        self.pad_connected_event = "connected"
        log.info("Gamepad connected: %s", pad.name)

    def _remove_pad(self, instance_id: int):
        pad = self.pads.pop(instance_id, None)
        if pad is None:
            return
        pad.close()
        if self.active_pad is pad:
            self.active_pad = next(iter(self.pads.values()), None)
            self.pad_state = {}
            self.pad_digital = set()
            if self.last_device == "pad":
                self.last_device = "kbm"
            self.pad_connected_event = "disconnected"
        log.info("Gamepad disconnected: %s", pad.name)

    @property
    def has_pad(self) -> bool:
        return self.active_pad is not None

    def rumble(self, strength: float, duration: float = 0.12):
        scale = float(self.settings.get("vibration", 0.8))
        if self.active_pad is None or scale <= 0 or self.last_device != "pad":
            return
        s = max(0.0, min(1.0, strength * scale))
        self.active_pad.rumble(s, s * 0.7, int(duration * 1000))

    # ------------------------------------------------------------------ frame
    def begin_frame(self):
        # Presses stay latched until a fixed update consumes them (see flush) so a press is
        # never lost on a frame with zero simulation steps, nor seen twice on a frame with two.
        self.mouse_moved = False
        self.mouse_wheel = 0
        self.text_events.clear()

    def process_event(self, e):
        t = e.type
        if t == pygame.QUIT:
            self.quit_requested = True
        elif t == pygame.KEYDOWN:
            self.last_device = "kbm"
            if self.capture_device == "kbm":
                self.captured = e.key
                return
            self.keys_down.add(e.key)
            self.text_events.append(e)
        elif t == pygame.KEYUP:
            self.keys_down.discard(e.key)
        elif t == pygame.MOUSEBUTTONDOWN:
            self.last_device = "kbm"
            if self.capture_device == "kbm" and e.button in (1, 2, 3):
                self.captured = f"MOUSE_{e.button}"
                return
            self.keys_down.add(f"MOUSE_{e.button}")
        elif t == pygame.MOUSEBUTTONUP:
            self.keys_down.discard(f"MOUSE_{e.button}")
        elif t == pygame.MOUSEMOTION:
            self.mouse_pos = e.pos
            if abs(e.rel[0]) + abs(e.rel[1]) > 2:
                self.last_device = "kbm"
                self.mouse_moved = True
        elif t == pygame.MOUSEWHEEL:
            self.mouse_wheel += e.y
            self.last_device = "kbm"
        elif t == pygame.JOYDEVICEADDED:
            self._add_pad(e.device_index)
        elif t == pygame.JOYDEVICEREMOVED:
            self._remove_pad(e.instance_id)
        elif t == pygame.WINDOWFOCUSLOST:
            self.keys_down.clear()
        elif t in (pygame.VIDEORESIZE,):
            self.resized = (e.w, e.h)

    def end_events(self):
        """Call after all events of the frame were processed. Resolves action states."""
        self._poll_pad()
        held = set()
        for action, codes in self.kb_bindings.items():
            for c in codes:
                if c in self.keys_down:
                    held.add(action)
                    break
        for action, names in self.pad_bindings.items():
            for n in names:
                if n in self.pad_digital:
                    held.add(action)
                    break
        self._pressed |= held - self._prev_held
        self._released |= self._prev_held - held
        self._held = held
        self._prev_held = held

    def _poll_pad(self):
        self.prev_pad_digital = self.pad_digital
        if self.active_pad is None:
            self.pad_digital = set()
            return
        state = self.active_pad.poll()
        self.pad_state = state
        dz = float(self.settings.get("stick_deadzone", 0.22))
        digital = {n for n in PAD_BUTTONS if state.get(n)}
        if state.get("LT", 0) > 0.45:
            digital.add("LT")
        if state.get("RT", 0) > 0.45:
            digital.add("RT")
        lx, ly = apply_radial_deadzone(state.get("LX", 0.0), state.get("LY", 0.0), dz)
        if ly < -0.6: digital.add("LS_UP")
        if ly > 0.6: digital.add("LS_DOWN")
        if lx < -0.6: digital.add("LS_LEFT")
        if lx > 0.6: digital.add("LS_RIGHT")
        rx, ry = apply_radial_deadzone(state.get("RX", 0.0), state.get("RY", 0.0), dz)
        self.aim_stick_active = (rx, ry) != (0.0, 0.0)
        if self.aim_stick_active:
            m = math.hypot(rx, ry)
            # Low-pass filter the aim direction to remove stick noise.
            ax, ay = self.aim_vec
            k = 0.55
            nx, ny = ax + (rx / m - ax) * k, ay + (ry / m - ay) * k
            n = math.hypot(nx, ny) or 1.0
            self.aim_vec = (nx / n, ny / n)
        if digital or (lx, ly) != (0, 0) or self.aim_stick_active:
            if self.last_device != "pad":
                self.last_device = "pad"
        if self.capture_device == "pad":
            new = digital - self.prev_pad_digital - {"LS_UP", "LS_DOWN", "LS_LEFT", "LS_RIGHT"}
            if new:
                self.captured = sorted(new)[0]
            digital = set()
        self.pad_digital = digital

    # ------------------------------------------------------------------ queries
    def held(self, action: str) -> bool:
        return action in self._held

    def pressed(self, action: str) -> bool:
        return action in self._pressed

    def released(self, action: str) -> bool:
        return action in self._released

    def flush(self):
        self._pressed.clear()
        self._released.clear()

    def consume(self, action: str):
        self._pressed.discard(action)

    def move_vector(self) -> tuple[float, float]:
        """Screen-space movement intent, magnitude in [0, 1]."""
        if self.active_pad is not None and self.pad_state:
            dz = float(self.settings.get("stick_deadzone", 0.22))
            lx, ly = apply_radial_deadzone(self.pad_state.get("LX", 0.0), self.pad_state.get("LY", 0.0), dz)
            if (lx, ly) != (0.0, 0.0):
                k = 0.5  # noise filter
                self.move_smooth[0] += (lx - self.move_smooth[0]) * k
                self.move_smooth[1] += (ly - self.move_smooth[1]) * k
                return tuple(self.move_smooth)
            self.move_smooth = [0.0, 0.0]
        x = (1 if self.held("move_right") else 0) - (1 if self.held("move_left") else 0)
        y = (1 if self.held("move_down") else 0) - (1 if self.held("move_up") else 0)
        if x and y:
            return x * 0.7071, y * 0.7071
        return float(x), float(y)

    def trigger(self, name: str) -> float:
        return float(self.pad_state.get(name, 0.0)) if self.pad_state else 0.0

    # ------------------------------------------------------------------ rebinding
    def begin_capture(self, device: str):
        self.capture_device = device
        self.captured = None

    def end_capture(self):
        self.capture_device = None
        value, self.captured = self.captured, None
        self._prev_held = set(self._held)
        return value
