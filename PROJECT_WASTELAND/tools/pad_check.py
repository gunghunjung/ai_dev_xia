"""Guided gamepad check: walks through every button, trigger, stick direction and vibration
using the game's own InputManager, and writes the results to a JSON file.

Run:  .venv\\Scripts\\python tools\\pad_check.py [result.json]
Keyboard: S = skip step, Esc = quit.
"""
import json
import math
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pygame

from config import DEFAULT_SETTINGS
from engine.input_manager import InputManager, pad_button_label

DIRS = {"up": (0, -1), "down": (0, 1), "left": (-1, 0), "right": (1, 0)}


def build_steps(pad_name=""):
    steps = []
    for b in ["A", "B", "X", "Y", "LB", "RB", "VIEW", "MENU", "L3", "R3",
              "DPAD_UP", "DPAD_RIGHT", "DPAD_DOWN", "DPAD_LEFT", "LT", "RT"]:
        steps.append(("button", b, f"Press {pad_button_label(b, pad_name)}"))
    for d in DIRS:
        steps.append(("lstick", d, f"Push LEFT stick fully {d.upper()}"))
    for d in DIRS:
        steps.append(("rstick", d, f"Push RIGHT stick fully {d.upper()}"))
    steps.append(("rumble", None, "Did it vibrate?  A = yes, B = no"))
    return steps


def main(out_path):
    pygame.init()
    screen = pygame.display.set_mode((900, 420))
    pygame.display.set_caption("Pad check")
    font = pygame.font.SysFont("consolas", 22)
    big = pygame.font.SysFont("consolas", 40, bold=True)
    with open(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "input_profiles.json"),
              encoding="utf-8") as f:
        profiles = json.load(f)
    inp = InputManager(profiles, dict(DEFAULT_SETTINGS))
    inp.init_pads()

    steps = build_steps(inp.active_pad.name if inp.active_pad else "")
    results = {"pad": None, "uses_sdl_mapping": None, "steps": []}
    i, step_start, hold_since, rumble_sent = 0, time.time(), None, False
    extra_seen: set = set()
    clock = pygame.time.Clock()
    running = True
    while running and i < len(steps):
        inp.begin_frame()
        skip = False
        for e in pygame.event.get():
            inp.process_event(e)
            if e.type == pygame.KEYDOWN and e.key == pygame.K_ESCAPE:
                running = False
            if e.type == pygame.KEYDOWN and e.key == pygame.K_s:
                skip = True
        inp.end_events()
        if inp.quit_requested:
            break
        pad = inp.active_pad
        if pad and results["pad"] is None:
            results["pad"] = pad.name
            results["uses_sdl_mapping"] = pad.controller is not None

        kind, target, prompt = steps[i]
        st = inp.pad_state or {}
        ok, detail = False, {}
        if kind == "button":
            down = target in inp.pad_digital
            others = inp.pad_digital - {target, "LS_UP", "LS_DOWN", "LS_LEFT", "LS_RIGHT"}
            extra_seen |= others
            if down:
                acts = sorted(a for a in inp.pad_bindings if inp.held(a))
                ok, detail = True, {"actions": acts, "also_down": sorted(extra_seen)}
                if target in ("LT", "RT"):
                    detail["analog"] = round(inp.trigger(target), 3)
        elif kind in ("lstick", "rstick"):
            if kind == "lstick":
                vx, vy = inp.move_vector()
            else:
                vx, vy = (inp.aim_vec if inp.aim_stick_active else (0.0, 0.0))
            ex, ey = DIRS[target]
            mag = math.hypot(vx, vy)
            if mag > 0.7 and (vx * ex + vy * ey) / mag > 0.9:
                hold_since = hold_since or time.time()
                if time.time() - hold_since > 0.25:
                    ok = True
                    detail = {"vec": [round(vx, 3), round(vy, 3)],
                              "raw": [round(st.get("LX" if kind == "lstick" else "RX", 0), 3),
                                      round(st.get("LY" if kind == "lstick" else "RY", 0), 3)]}
            else:
                hold_since = None
        elif kind == "rumble":
            if not rumble_sent:
                inp.last_device = "pad"
                inp.rumble(1.0, 0.8)
                rumble_sent = True
            if inp.pressed("ui_confirm"):
                ok, detail = True, {"felt": True}
            elif inp.pressed("ui_back"):
                ok, detail = True, {"felt": False}
        inp.flush()

        if ok or skip:
            results["steps"].append({"kind": kind, "target": target, "ok": ok and not skip,
                                     "skipped": skip, "secs": round(time.time() - step_start, 2), **detail})
            i += 1
            step_start, hold_since, extra_seen = time.time(), None, set()
            # wait for release so one press can't complete the next step
            if i < len(steps) and steps[i][0] != "rumble":
                pygame.time.wait(150)

        screen.fill((28, 26, 24))
        head = f"Pad: {pad.name if pad else 'NOT CONNECTED'}   ({i}/{len(steps)})"
        screen.blit(font.render(head, True, (200, 200, 190)), (20, 20))
        if i < len(steps):
            screen.blit(big.render(steps[i][2], True, (255, 210, 120)), (20, 110))
        live = f"digital: {sorted(inp.pad_digital)}"
        screen.blit(font.render(live[:80], True, (160, 200, 160)), (20, 230))
        axes = "  ".join(f"{k}={st.get(k, 0):+.2f}" for k in ("LX", "LY", "RX", "RY", "LT", "RT"))
        screen.blit(font.render(axes, True, (160, 180, 220)), (20, 270))
        screen.blit(font.render("S = skip step   Esc = quit", True, (130, 130, 120)), (20, 370))
        pygame.display.flip()
        clock.tick(60)

    results["completed"] = i >= len(steps)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=1, ensure_ascii=False)
    print(json.dumps(results, ensure_ascii=False))
    pygame.quit()


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "pad_check_result.json")
