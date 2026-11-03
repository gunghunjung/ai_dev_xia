import pygame

from config import DEFAULT_SETTINGS
from engine.camera import iso, inv_iso, screen_dir_to_world
from engine.input_manager import InputManager, apply_radial_deadzone


def test_deadzone():
    assert apply_radial_deadzone(0.1, 0.1, 0.2) == (0.0, 0.0)
    x, y = apply_radial_deadzone(1.0, 0.0, 0.2)
    assert abs(x - 1.0) < 1e-9 and y == 0
    x, _ = apply_radial_deadzone(0.6, 0.0, 0.2)
    assert 0.45 < x < 0.55                         # rescaled after the deadzone


def test_actions_latch_and_rebinding(data):
    pygame.init()
    inp = InputManager(data.input_profiles, dict(DEFAULT_SETTINGS))
    inp.begin_frame()
    inp.process_event(pygame.event.Event(pygame.KEYDOWN, key=pygame.K_f, mod=0, unicode="f", scancode=0))
    inp.end_events()
    assert inp.pressed("dig") and inp.held("dig")
    inp.begin_frame()
    inp.end_events()
    assert inp.pressed("dig")                      # still latched: no update consumed it yet
    inp.flush()
    assert inp.held("dig") and not inp.pressed("dig")
    inp.rebind("dig", "kbm", pygame.K_g)
    exported = inp.export_bindings()
    assert exported["keyboard"]["dig"] == ["K_g"]
    inp2 = InputManager(data.input_profiles, {"bindings": exported})
    assert inp2.kb_bindings["dig"] == [pygame.K_g]


def test_move_vector_normalized(data):
    inp = InputManager(data.input_profiles, dict(DEFAULT_SETTINGS))
    inp.keys_down = {pygame.K_w, pygame.K_d}
    inp.end_events()
    x, y = inp.move_vector()
    assert abs((x * x + y * y) ** 0.5 - 1.0) < 1e-3


def test_pad_bindings_are_logical(data):
    inp = InputManager(data.input_profiles, dict(DEFAULT_SETTINGS))
    inp.pad_digital = {"RT"}
    inp._prev_held = set()
    held = {a for a, names in inp.pad_bindings.items() if any(n in inp.pad_digital for n in names)}
    assert "attack" in held


def test_iso_roundtrip():
    for x, y in ((0, 0), (3.5, -2.25), (100, 42)):
        sx, sy = iso(x, y)
        bx, by = inv_iso(sx, sy)
        assert abs(bx - x) < 1e-9 and abs(by - y) < 1e-9
    wx, wy = screen_dir_to_world(0, -1)               # screen up
    sx, sy = iso(wx, wy)
    assert abs(sx) < 1e-9 and sy < 0


def test_duplicate_device_added_keeps_pad_open(data, monkeypatch):
    """SDL re-announces pads opened at startup; Joystick objects share one handle per device."""
    import engine.input_manager as im
    closed = set()

    class FakeJoystick:
        def __init__(self, index):
            self.index = index
        def get_instance_id(self): return 7
        def get_name(self): return "Fake Pad"
        def quit(self): closed.add(self.index)

    monkeypatch.setattr(im, "sdl_controller", None)
    monkeypatch.setattr(pygame.joystick, "Joystick", FakeJoystick)
    inp = InputManager(data.input_profiles, dict(DEFAULT_SETTINGS))
    inp._add_pad(0)
    pad = inp.active_pad
    inp.process_event(pygame.event.Event(pygame.JOYDEVICEADDED, device_index=0))
    assert inp.active_pad is pad and len(inp.pads) == 1
    assert not closed                               # the shared handle was not closed
