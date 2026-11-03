"""Character status screen: every survival variable with what it currently does to you."""
import pygame

from ui.widgets import panel, bar, TEXT, DIM, ACCENT, BAD, GOOD


class StatusUI:
    def __init__(self, ui, prompts, assets):
        self.ui = ui
        self.prompts = prompts
        self.session = None

    def open(self, session):
        self.session = session

    def close(self):
        pass

    def handle(self, inp, events, dt):
        if inp.pressed("ui_back"):
            return "close"
        return None

    def _row(self, screen, x, y, label, frac, color, value, note):
        screen.blit(self.ui.text(label, 16, TEXT, True), (x, y))
        bar(screen, (x + 110, y + 3, 260, 18), frac, color)
        screen.blit(self.ui.text(value, 15, TEXT, True), (x + 382, y + 1))
        if note:
            screen.blit(self.ui.text(note, 13, DIM), (x + 460, y + 3))

    def draw(self, screen, area, inp):
        s = self.session
        p = s.player
        st = p.stats
        r = pygame.Rect(area.x + 80, area.y + 40, area.width - 160, area.height - 90)
        panel(screen, r, "상태", self.ui)
        x, y = r.x + 30, r.y + 30
        rows = [
            ("체력", p.hp / p.max_hp, (210, 64, 56), f"{int(p.hp)}/{int(p.max_hp)}",
             "회복 중" if st.hunger > 50 and st.thirst > 50 and st.infection < 30 and st.bleeding == 0 else "자연 회복 없음"),
            ("기력", st.stamina / 100, (226, 190, 70), f"{int(st.stamina)}/{int(st.max_stamina)}",
             "최대치 감소 (피로/부상)" if st.max_stamina < 100 else ""),
            ("허기", st.hunger / 100, (214, 150, 70), f"{int(st.hunger)}",
             "굶주림: 체력 감소" if st.hunger <= 0 else "기력 회복 저하" if st.hunger < 25 else ""),
            ("갈증", st.thirst / 100, (84, 160, 226), f"{int(st.thirst)}",
             "탈수: 체력 감소" if st.thirst <= 0 else "달리기 효율·기력 회복 감소" if st.thirst < 25 else ""),
            ("피로", st.fatigue / 100, (170, 150, 210), f"{int(st.fatigue)}",
             "침낭에서 잠을 자야 한다" if st.fatigue > 70 else ""),
            ("감염", st.infection / 100, (150, 210, 70), f"{int(st.infection)}",
             "진행 중 — 항생제 필요" if st.infection >= 25 else "면역이 버티는 중" if st.infection > 0 else ""),
            ("부상", st.injury / 100, (220, 140, 140), f"{int(st.injury)}",
             "이동 속도 감소" if st.injury > 40 else ""),
        ]
        for label, frac, col, val, note in rows:
            self._row(screen, x, y, label, frac, col, val, note)
            y += 40
        t = st.body_temp
        tcol = (120, 180, 255) if t < 35.8 else (255, 120, 80) if t > 38.0 else GOOD
        screen.blit(self.ui.text("체온", 16, TEXT, True), (x, y))
        screen.blit(self.ui.text(f"{t:.1f}°C", 18, tcol, True), (x + 110, y))
        screen.blit(self.ui.text(f"기온 {s.survival.ambient_temp:.0f}°C · 체감 {s.survival.feels_temp:.0f}°C · 의복 보온 {p.insulation():.2f}", 13, DIM), (x + 200, y + 3))
        y += 40
        bleed = "없음" if st.bleeding <= 0 else ["가벼움", "심함", "치명적"][min(2, int(st.bleeding - 0.01))]
        screen.blit(self.ui.text("출혈", 16, TEXT, True), (x, y))
        screen.blit(self.ui.text(bleed, 16, BAD if st.bleeding > 0 else GOOD, True), (x + 110, y))
        if st.bleeding > 0:
            screen.blit(self.ui.text("붕대로 지혈하세요", 13, DIM), (x + 200, y + 3))
        y += 50
        screen.blit(self.ui.text(f"생존 {st.days_survived:.1f}일 · 처치 {st.kills} · 운반 무게 {p.carry_weight():.1f}/{s.carry_capacity():.0f}kg", 15, ACCENT, True), (x, y))
