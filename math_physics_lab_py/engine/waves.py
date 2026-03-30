import numpy as np


def wave(x, t, freq, amp, phase=0):
    return amp * np.sin(2 * np.pi * freq * (x - t) + phase)


def electric_field_2d(charges, x_grid, y_grid):
    """전기장 2D 계산
    charges: list of (x, y, q)
    """
    Ex = np.zeros_like(x_grid, dtype=float)
    Ey = np.zeros_like(y_grid, dtype=float)
    k = 8.99e9  # 단위 생략, 상대값만 사용
    for cx, cy, q in charges:
        dx = x_grid - cx
        dy = y_grid - cy
        r2 = dx ** 2 + dy ** 2
        r2 = np.where(r2 < 0.01, 0.01, r2)
        r3 = r2 ** 1.5
        Ex += k * q * dx / r3
        Ey += k * q * dy / r3
    return Ex, Ey
