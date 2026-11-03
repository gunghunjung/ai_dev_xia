"""Application shell: window, scenes (title / gameplay), overlays and save/load flow."""
import logging
import math

import pygame

from config import SCREEN_W, SCREEN_H, GAME_TITLE, DEFAULT_SETTINGS, FIXED_DT
from database.save_manager import SaveManager, SettingsStore
from engine.asset_manager import AssetManager
from engine.audio_manager import AudioManager
from engine.camera import Camera
from engine.game_loop import GameLoop
from engine.input_manager import InputManager
from engine.profiler import Profiler
from engine.renderer import Renderer
from game.session import GameData, GameSession
from ui.controller_ui import ButtonPrompts
from ui.game_menu import GameMenu
from ui.hud import HUD
from ui.menu import MainMenu, PauseMenu, DeathScreen
from ui.settings_ui import SettingsUI
from ui.widgets import UIContext

log = logging.getLogger(__name__)
SLOT = "slot1"


class App:
    def __init__(self, save_dir: str | None = None):
        pygame.init()
        pygame.display.set_caption(GAME_TITLE)
        self.settings_store = SettingsStore(save_dir) if save_dir else SettingsStore()
        self.settings = self.settings_store.load(DEFAULT_SETTINGS)
        flags = pygame.SCALED | pygame.RESIZABLE
        try:
            self.screen = pygame.display.set_mode((SCREEN_W, SCREEN_H), flags, vsync=1)
        except pygame.error:
            self.screen = pygame.display.set_mode((SCREEN_W, SCREEN_H), flags)
        if self.settings.get("fullscreen"):
            self._set_fullscreen(True)
        self.running = True
        self.assets = AssetManager()
        self.audio = AudioManager(self.settings)
        self.data = GameData(self.assets)
        self.input = InputManager(self.data.input_profiles, self.settings)
        self.input.init_pads()
        self.profiler = Profiler()
        self.camera = Camera(SCREEN_W, SCREEN_H)
        self.camera.shake_scale = float(self.settings.get("screen_shake", 1.0))
        self.renderer = Renderer(self.assets, self.camera)
        self.ui = UIContext(self.assets)
        self.prompts = ButtonPrompts(self.ui, self.input)
        self.hud = HUD(self.ui, self.prompts, self.assets)
        self.game_menu = GameMenu(self.ui, self.prompts, self.assets)
        self.saves = SaveManager(save_dir) if save_dir else SaveManager()
        self.settings_ui = SettingsUI(self.ui, self.prompts, self.assets, self.input, self.settings, self.on_settings_changed)
        self.session: GameSession | None = None
        self.scene = "title"
        self.overlay = None            # "pause" | "settings" | "death"
        self.settings_return = None
        self.main_menu = MainMenu(self.ui, self.prompts, self.saves.exists(SLOT))
        self.pause_menu = PauseMenu(self.ui, self.prompts)
        self.death_screen = None
        self.events: list = []
        self.frame_dt = FIXED_DT
        self.show_debug = bool(self.settings.get("show_fps"))
        self.death_timer = 0.0
        self._consumed = False
        self.loop = GameLoop(self)

    # ------------------------------------------------------------ settings
    def _set_fullscreen(self, on: bool):
        try:
            if bool(pygame.display.is_fullscreen()) != on:
                pygame.display.toggle_fullscreen()
        except (pygame.error, AttributeError):
            pass

    def on_settings_changed(self):
        self.camera.shake_scale = float(self.settings.get("screen_shake", 1.0))
        self.show_debug = bool(self.settings.get("show_fps"))
        self._set_fullscreen(bool(self.settings.get("fullscreen")))
        if self.session:
            self.session.effects.damage_numbers = self.settings.get("damage_numbers", True)
        self.settings_store.save(self.settings)

    # ------------------------------------------------------------ session flow
    def new_game(self, seed=None):
        self.session = GameSession(self.data, self.settings, self.audio, seed=seed)
        self._start_session()
        self.session.message("아무것도 없이 황무지에서 눈을 떴다.", (240, 220, 180), big=True)
        self.session.message("돌과 나뭇가지를 모아 도구를 만들자. [C] 제작", (220, 210, 190))

    def load_game(self) -> bool:
        try:
            _, meta = self.saves.read_meta(SLOT)
        except Exception as exc:
            self.main_menu.note = f"불러오기 실패: {exc}"
            return False
        session = GameSession(self.data, self.settings, self.audio, seed=int(meta["seed"]), new_game=False)
        if not self.saves.load(session, SLOT):
            self.main_menu.note = f"불러오기 실패: {self.saves.last_error}"
            return False
        self.session = session
        self._start_session()
        session.message("저장된 게임을 불러왔다", (200, 230, 255))
        return True

    def _start_session(self):
        s = self.session
        s.input = self.input
        self.camera.snap(s.player.x, s.player.y)
        self.scene = "game"
        self.overlay = None
        self.death_screen = None
        self.death_timer = 0.0
        if self.game_menu.is_open:
            self.game_menu.close()

    def save_game(self, quiet=False) -> bool:
        if self.session is None or self.session.player.dead:
            return False
        ok = self.saves.save(self.session, SLOT)
        if not quiet or not ok:
            self.hud.show_toast("저장 완료" if ok else f"저장 실패: {self.saves.last_error}")
        self.main_menu = MainMenu(self.ui, self.prompts, self.saves.exists(SLOT))
        return ok

    # ------------------------------------------------------------ loop hooks
    def handle_input(self):
        inp = self.input
        inp.begin_frame()
        self.events = pygame.event.get()
        for e in self.events:
            inp.process_event(e)
        inp.end_events()
        if inp.quit_requested:
            self.running = False
            return
        dt = self.profiler.frame_times[-1] if self.profiler.frame_times else FIXED_DT
        self.frame_dt = dt
        if inp.pad_connected_event:
            self.hud.show_toast("Xbox 패드 연결됨" if inp.pad_connected_event == "connected" else "패드 연결 해제됨")
            if inp.pad_connected_event == "disconnected" and self.scene == "game" and self.overlay is None:
                self.overlay = "pause"
            inp.pad_connected_event = None
        if inp.pressed("debug"):
            self.show_debug = not self.show_debug

        # Overlays / menus consume input per frame.
        if self.overlay == "settings":
            if self.settings_ui.handle(inp, self.events, dt) == "close":
                self.overlay = self.settings_return
            self._consume()
            return
        if self.scene == "title":
            res = self.main_menu.handle(inp, self.events, dt)
            if res == "new":
                self.new_game()
            elif res == "continue":
                self.load_game()
            elif res == "settings":
                self.settings_return = None
                self.overlay = "settings"
            elif res == "quit":
                self.running = False
            self._consume()
            return
        s = self.session
        if self.overlay == "pause":
            res = self.pause_menu.handle(inp, self.events, dt)
            if res == "resume":
                self.overlay = None
            elif res == "save":
                ok = self.save_game(quiet=True)
                self.pause_menu.note, self.pause_menu.note_t = ("저장 완료" if ok else "저장 실패"), 2.0
            elif res == "load":
                if self.saves.exists(SLOT):
                    self.load_game()
                else:
                    self.pause_menu.note, self.pause_menu.note_t = "저장된 게임이 없다", 2.0
            elif res == "settings":
                self.settings_return = "pause"
                self.overlay = "settings"
            elif res == "title":
                self.scene = "title"
                self.overlay = None
                self.session = None
                self.main_menu = MainMenu(self.ui, self.prompts, self.saves.exists(SLOT))
            elif res == "quit":
                self.running = False
            self._consume()
            return
        if self.overlay == "death":
            res = self.death_screen.handle(inp, self.events, dt)
            if res == "load":
                self.load_game()
            elif res == "new":
                self.new_game()
            elif res == "title":
                self.scene = "title"
                self.overlay = None
                self.session = None
                self.main_menu = MainMenu(self.ui, self.prompts, self.saves.exists(SLOT))
            self._consume()
            return
        if s.lore_popup:
            if inp.pressed("ui_confirm") or inp.pressed("ui_back") or inp.pressed("interact"):
                s.lore_popup = None
            self._consume()
            return
        if self.game_menu.is_open:
            self.game_menu.handle(inp, self.events, dt)
            if s.player.placing is not None and self.game_menu.is_open:
                self.game_menu.close()
            self._consume()
            return
        # Gameplay-level global actions.
        if inp.pressed("pause"):
            if s.player.placing is not None:
                s.player.placing = None
            else:
                self.overlay = "pause"
            self._consume()
            return
        if not s.player.dead:
            if inp.pressed("inventory"):
                self.game_menu.open(s, "inventory")
                self._consume()
                return
            if inp.pressed("crafting"):
                self.game_menu.open(s, "crafting")
                self._consume()
                return
        if inp.pressed("quicksave"):
            self.save_game()
        if inp.pressed("quickload"):
            if self.saves.exists(SLOT):
                self.load_game()
        self._consumed = False

    def _consume(self):
        self.input.flush()
        self._consumed = True

    def update(self, dt: float):
        if self.scene != "game" or self.session is None:
            return
        if self.overlay == "pause" or (self.overlay == "settings" and self.settings_return == "pause"):
            return  # paused: the world is frozen
        s = self.session
        ui_blocking = self.game_menu.is_open or bool(s.lore_popup) or self.overlay is not None
        inp = self.input
        s.update(dt, inp, self.camera, ui_blocking)
        inp.flush()
        p = s.player
        lead = (0.0, 0.0)
        if not p.dead:
            lead = (math.cos(p.aim_angle) * 1.2, math.sin(p.aim_angle) * 1.2)
        self.camera.follow(p.x, p.y, dt, lead)
        if s.pending_trauma:
            self.camera.add_trauma(s.pending_trauma)
            s.pending_trauma = 0.0
        self.camera.update_shake(dt)
        if s.ui_request is not None:
            kind, obj = s.ui_request
            s.ui_request = None
            if kind == "inventory":
                self.game_menu.open(s, "inventory", container=obj)
            else:
                self.game_menu.open(s, "crafting")
        if s.autosave_requested:
            s.autosave_requested = False
            self.save_game(quiet=True)
            self.hud.show_toast("자동 저장")
        if p.dead and self.overlay is None:
            self.death_timer += dt
            if self.death_timer > 2.0:
                if self.game_menu.is_open:
                    self.game_menu.close()
                self.death_screen = DeathScreen(self.ui, self.prompts, p.stats, self.saves.exists(SLOT))
                self.overlay = "death"

    def render(self, alpha: float):
        screen = self.screen
        dt = self.frame_dt
        if self.scene == "title":
            self.main_menu.draw(screen, self.input)
        else:
            s = self.session
            self.renderer.draw(screen, s, dt)
            self.profiler.count("tiles", self.renderer.stats["tiles"])
            self.profiler.count("sprites", self.renderer.stats["sprites"])
            self.profiler.begin("ui")
            self.hud.draw(screen, s, self.camera, self.input, dt, self.show_debug, self.profiler,
                          menu_open=self.game_menu.is_open or self.overlay is not None)
            if self.game_menu.is_open:
                self.game_menu.draw(screen, self.input)
            if self.overlay == "pause" or (self.overlay == "settings" and self.settings_return == "pause"):
                if self.overlay == "pause":
                    self.pause_menu.draw(screen, self.input)
            if self.overlay == "death" and self.death_screen:
                self.death_screen.draw(screen, self.input)
            self.profiler.end("ui")
        if self.overlay == "settings":
            self.settings_ui.draw(screen)
        if self.scene == "title" and self.hud.toast_t > 0:
            self.hud.toast_t -= dt
        if self.show_debug and self.scene == "title":
            t = self.ui.text(f"FPS {self.profiler.fps():.0f}", 14, (180, 255, 180))
            screen.blit(t, (10, 10))

    def run(self, max_frames=None):
        self.loop.run(max_frames)
        if self.session is not None and not self.session.player.dead and max_frames is None:
            self.save_game(quiet=True)  # exit autosave
        self.settings_store.save(self.settings)
        pygame.quit()
