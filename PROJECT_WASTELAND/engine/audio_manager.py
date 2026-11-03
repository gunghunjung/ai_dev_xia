"""Sound playback with procedural placeholder SFX.

``assets/sounds/<name>.wav|.ogg`` overrides a synthesized sound of the same name.
If no audio device is available the manager silently disables itself.
"""
import array
import logging
import math
import os
import random

import pygame

from config import ASSET_DIR

log = logging.getLogger(__name__)
RATE = 22050


def _env(i, n, attack=0.005, decay_pow=2.0):
    t = i / n
    a = min(1.0, (i / RATE) / attack) if attack > 0 else 1.0
    return a * (1 - t) ** decay_pow


def _synth(name: str) -> array.array:
    rng = random.Random(name)
    out = array.array("h")

    def push(v):
        out.append(int(max(-1.0, min(1.0, v)) * 30000))

    def noise_burst(dur, lp=0.5, decay=2.0, gain=1.0, pitch_tone=0.0, tone_gain=0.0, sweep=1.0):
        n = int(RATE * dur)
        y = 0.0
        for i in range(n):
            y += (rng.uniform(-1, 1) - y) * lp
            e = _env(i, n, 0.002, decay)
            tone = math.sin(2 * math.pi * pitch_tone * (1 + (sweep - 1) * i / n) * i / RATE) * tone_gain if pitch_tone else 0
            push((y * gain + tone) * e)

    if name == "pistol":
        noise_burst(0.28, 0.75, 3.5, 1.0, 140, 0.6, 0.4)
    elif name == "revolver":
        noise_burst(0.38, 0.65, 3.0, 1.0, 110, 0.7, 0.4)
    elif name == "shotgun":
        noise_burst(0.55, 0.55, 2.5, 1.1, 80, 0.8, 0.5)
    elif name == "rifle":
        noise_burst(0.45, 0.85, 3.2, 1.0, 180, 0.5, 0.3)
    elif name == "dry_fire":
        noise_burst(0.05, 0.9, 6, 0.4, 2000, 0.2)
    elif name == "reload":
        for k in range(2):
            noise_burst(0.04, 0.9, 5, 0.5, 1800 + k * 400, 0.3)
            out.extend([0] * int(RATE * 0.12))
    elif name in ("swing_light", "swing_heavy"):
        dur = 0.18 if name == "swing_light" else 0.3
        n = int(RATE * dur)
        y = 0.0
        for i in range(n):
            t = i / n
            lp = 0.05 + 0.4 * math.sin(math.pi * t)
            y += (rng.uniform(-1, 1) - y) * lp
            push(y * math.sin(math.pi * t) * (0.5 if name == "swing_light" else 0.75))
    elif name == "hit_blunt":
        noise_burst(0.16, 0.25, 3, 0.8, 90, 0.9, 0.5)
    elif name == "hit_metal":
        n = int(RATE * 0.3)
        for i in range(n):
            e = _env(i, n, 0.001, 3)
            v = (math.sin(2 * math.pi * 620 * i / RATE) * 0.5 + math.sin(2 * math.pi * 1340 * i / RATE) * 0.3 + rng.uniform(-1, 1) * 0.3 * (1 - i / n) ** 8)
            push(v * e)
    elif name == "hit_sharp":
        noise_burst(0.14, 0.7, 4, 0.9, 260, 0.5, 0.6)
    elif name == "hit_heavy":
        noise_burst(0.32, 0.2, 2.5, 1.0, 55, 1.0, 0.4)
    elif name == "bullet_hit":
        noise_burst(0.1, 0.6, 4, 0.7, 300, 0.3, 0.5)
    elif name == "dig":
        noise_burst(0.18, 0.18, 2.0, 0.9, 70, 0.4, 0.8)
    elif name == "dig_find":
        for f in (660, 880, 1320):
            n = int(RATE * 0.09)
            for i in range(n):
                push(math.sin(2 * math.pi * f * i / RATE) * 0.4 * _env(i, n, 0.003, 1.5))
    elif name == "pickup":
        n = int(RATE * 0.12)
        for i in range(n):
            f = 500 + 700 * i / n
            push(math.sin(2 * math.pi * f * i / RATE) * 0.35 * _env(i, n, 0.003, 1.2))
    elif name == "craft":
        for k in range(3):
            noise_burst(0.06, 0.4, 3, 0.6, 400 + k * 80, 0.3)
            out.extend([0] * int(RATE * 0.06))
    elif name == "hurt":
        noise_burst(0.22, 0.3, 2, 0.6, 200, 0.6, 0.6)
    elif name in ("groan", "snarl", "shout", "squeak"):
        base = {"groan": 95, "snarl": 160, "shout": 180, "squeak": 1400}[name]
        dur = {"groan": 0.7, "snarl": 0.4, "shout": 0.35, "squeak": 0.12}[name]
        n = int(RATE * dur)
        ph = 0.0
        for i in range(n):
            t = i / n
            f = base * (1 + 0.25 * math.sin(t * 9) + rng.uniform(-0.05, 0.05))
            ph += 2 * math.pi * f / RATE
            v = math.sin(ph) + 0.5 * math.sin(ph * 2.01) + rng.uniform(-0.4, 0.4)
            push(v * 0.35 * math.sin(math.pi * t))
    elif name == "step":
        noise_burst(0.05, 0.15, 3, 0.25)
    elif name == "ui_move":
        n = int(RATE * 0.04)
        for i in range(n):
            push(math.sin(2 * math.pi * 900 * i / RATE) * 0.2 * _env(i, n, 0.002, 1))
    elif name == "ui_confirm":
        for f in (700, 1050):
            n = int(RATE * 0.06)
            for i in range(n):
                push(math.sin(2 * math.pi * f * i / RATE) * 0.25 * _env(i, n, 0.002, 1))
    elif name == "fire_crackle":
        noise_burst(0.2, 0.9, 6, 0.25)
    elif name == "dodge":
        noise_burst(0.2, 0.12, 1.5, 0.5)
    elif name == "death":
        noise_burst(0.5, 0.15, 1.5, 0.8, 70, 0.8, 0.5)
    else:
        noise_burst(0.1, 0.5, 3, 0.4)
    return out


class AudioManager:
    def __init__(self, settings: dict):
        self.settings = settings
        self.enabled = False
        self._sounds: dict[str, pygame.mixer.Sound] = {}
        self.listener = (0.0, 0.0)
        try:
            pygame.mixer.pre_init(RATE, -16, 1, 512)
            pygame.mixer.init(RATE, -16, 1, 512)
            pygame.mixer.set_num_channels(24)
            self.enabled = True
        except pygame.error as exc:
            log.warning("Audio disabled: %s", exc)

    def _get(self, name: str):
        snd = self._sounds.get(name)
        if snd is None:
            for ext in (".wav", ".ogg"):
                path = os.path.join(ASSET_DIR, "sounds", name + ext)
                if os.path.exists(path):
                    try:
                        snd = pygame.mixer.Sound(path)
                        break
                    except pygame.error as exc:
                        log.warning("Bad sound file %s: %s", path, exc)
            if snd is None:
                snd = pygame.mixer.Sound(buffer=_synth(name).tobytes())
            self._sounds[name] = snd
        return snd

    def preload(self, names):
        if not self.enabled:
            return
        for n in names:
            self._get(n)

    def play(self, name: str, volume: float = 1.0, pos=None, max_dist: float = 22.0):
        if not self.enabled or not name:
            return
        vol = volume * float(self.settings.get("master_volume", 0.8)) * float(self.settings.get("sfx_volume", 0.9))
        if pos is not None:
            d = math.hypot(pos[0] - self.listener[0], pos[1] - self.listener[1])
            if d > max_dist:
                return
            vol *= max(0.0, 1.0 - d / max_dist) ** 1.3
        if vol <= 0.01:
            return
        try:
            snd = self._get(name)
            ch = snd.play()
            if ch is not None:
                ch.set_volume(min(1.0, vol))
        except pygame.error:
            pass
