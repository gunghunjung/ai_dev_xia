import numpy as np


def rk4_step(f, t, y, dt):
    k1 = f(t, y)
    k2 = f(t + dt / 2, y + dt / 2 * k1)
    k3 = f(t + dt / 2, y + dt / 2 * k2)
    k4 = f(t + dt, y + dt * k3)
    return y + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)


def simulate_projectile(v0, theta_deg, k, dt=0.01, max_t=30):
    """포물선 운동 시뮬레이션
    state: [x, y, vx, vy]
    """
    theta = np.radians(theta_deg)
    g = 9.81
    state = np.array([0.0, 0.0, v0 * np.cos(theta), v0 * np.sin(theta)])
    xs, ys = [state[0]], [state[1]]
    t = 0
    while state[1] >= 0 and t < max_t:
        v = np.sqrt(state[2] ** 2 + state[3] ** 2)

        def deriv(t_, s, _v=v, _k=k, _g=g):
            return np.array([s[2], s[3], -_k * s[2] * _v, -_g - _k * s[3] * _v])

        state = rk4_step(deriv, t, state, dt)
        t += dt
        xs.append(state[0])
        ys.append(state[1])
        if state[1] < 0:
            break
    return np.array(xs), np.array(ys)


def simulate_pendulum(L, theta0_deg, b, dt=0.02, n_steps=2000):
    """진자 운동 시뮬레이션"""
    g = 9.81
    theta0 = np.radians(theta0_deg)
    state = np.array([theta0, 0.0])
    m = 1.0
    thetas, omegas = [state[0]], [state[1]]
    for _ in range(n_steps):
        def deriv(t, s, _L=L, _b=b, _m=m, _g=g):
            return np.array([s[1], -(_g / _L) * np.sin(s[0]) - (_b / (_m * _L ** 2)) * s[1]])

        state = rk4_step(deriv, 0, state, dt)
        thetas.append(state[0])
        omegas.append(state[1])
    return np.array(thetas), np.array(omegas)


def simulate_spring(k, m, b, x0, dt=0.02, n_steps=2000):
    """용수철 진동 시뮬레이션"""
    state = np.array([x0, 0.0])
    xs, vs = [state[0]], [state[1]]
    for _ in range(n_steps):
        def deriv(t, s, _k=k, _m=m, _b=b):
            return np.array([s[1], -(_k / _m) * s[0] - (_b / _m) * s[1]])

        state = rk4_step(deriv, 0, state, dt)
        xs.append(state[0])
        vs.append(state[1])
    return np.array(xs), np.array(vs)
