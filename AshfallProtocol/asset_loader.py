import pygame
from config import FONT_NAME


class AssetLoader:
    def __init__(self):
        self.font_cache = {}

    def font(self, size: int, bold: bool = False):
        key = (size, bold)
        if key not in self.font_cache:
            self.font_cache[key] = pygame.font.SysFont(FONT_NAME, size, bold=bold)
        return self.font_cache[key]
