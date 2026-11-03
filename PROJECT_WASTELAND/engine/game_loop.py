"""Fixed-timestep game loop: input → N fixed updates → render.

Game logic always advances in FIXED_DT steps so simulation speed is independent of the
frame rate. Rendering happens once per frame afterwards.
"""
import time

import pygame

from config import FIXED_DT, MAX_FRAME_TIME


class GameLoop:
    def __init__(self, app, target_fps: int = 120):
        self.app = app
        self.target_fps = target_fps
        self.clock = pygame.time.Clock()
        self.running = False
        self.accumulator = 0.0
        self.max_steps = 5
        # Automated tests: every frame advances exactly one fixed step, no real-time pacing.
        self.deterministic = False

    def run(self, max_frames: int | None = None):
        self.running = True
        prev = time.perf_counter()
        frames = 0
        prof = self.app.profiler
        while self.running:
            now = time.perf_counter()
            frame_time = FIXED_DT if self.deterministic else min(MAX_FRAME_TIME, now - prev)
            prev = now
            prof.frame(frame_time)
            self.accumulator += frame_time

            prof.begin("input")
            self.app.handle_input()
            prof.end("input")

            prof.begin("update")
            steps = 0
            while self.accumulator >= FIXED_DT and steps < self.max_steps:
                self.app.update(FIXED_DT)
                self.accumulator -= FIXED_DT
                steps += 1
            if steps == self.max_steps:
                self.accumulator = 0.0  # spiral-of-death guard
            prof.end("update")

            prof.begin("render")
            self.app.render(self.accumulator / FIXED_DT)
            prof.end("render")
            pygame.display.flip()

            if not self.app.running:
                self.running = False
            frames += 1
            if max_frames is not None and frames >= max_frames:
                self.running = False
            if not self.deterministic:
                self.clock.tick(self.target_fps)
