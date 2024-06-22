import numpy as np
from functools import partial
from tqdm import trange


# random initial conditions for Lotka-Volterra equation
def generate_random_ics(n_ics=10000, h_min=3, h_max=4.5, canonical=True):
    initial_conditions = []
    for _ in range(n_ics):
        x0 = np.random.uniform(0, 1, size=2)
        if canonical:
            x0 = np.log(x0)
        h = H_lv(x0, canonical=canonical)
        while h < h_min or h > h_max:
            x0 = np.random.uniform(0, 1, size=2)
            if canonical:
                x0 = np.log(x0)
            h = H_lv(x0, canonical=canonical)
        initial_conditions.append(x0)
    return np.array(initial_conditions)


# Hamiltonian for Lotka-Volterra equation
def H_lv(x, a=2/3, b=4/3, c=1.0, d=1.0, canonical=True):
    if canonical:
        return c * np.exp(x[..., 0]) - d * x[..., 0] + b * np.exp(x[..., 1]) - a * x[..., 1]
    else:
        return c * x[..., 0] - d * np.log(x[..., 0]) + b * x[..., 1] - a * np.log(x[..., 1])


# derivative for Lotka-Volterra equation
def lotka_volterra(x, a=2/3, b=4/3, c=1.0, d=1.0, canonical=True):
    dx = np.zeros_like(x)
    if not canonical:
        dx[..., 0] = a * x[..., 0] - b * x[..., 0] * x[..., 1]
        dx[..., 1] = c * x[..., 0] * x[..., 1] - d * x[..., 1]
    else:
        dx[..., 0] = a - b * np.exp(x[..., 1])
        dx[..., 1] = c * np.exp(x[..., 0]) - d
    return dx


# simulate Lotka-Volterra equation from initial conditions
def simulate_lv(x0, dt=0.002, num_steps=10000, a=2/3, b=4/3, c=1.0, d=1.0, canonical=True):
    x = np.zeros((num_steps, *x0.shape))
    dx = np.zeros_like(x)
    x[0] = x0
    update_fn = partial(lotka_volterra, a=a, b=b, c=c, d=d, canonical=canonical)
    # use RK4 integration
    for i in trange(num_steps):
        dx1 = update_fn(x[i])
        dx[i] = dx1
        if i == num_steps - 1:
            break
        k1 = dt * dx1
        dx2 = update_fn(x[i] + 0.5 * k1)
        k2 = dt * dx2
        dx3 = update_fn(x[i] + 0.5 * k2)
        k3 = dt * dx3
        dx4 = update_fn(x[i] + k3)
        k4 = dt * dx4
        x[i + 1] = x[i] + (k1 + 2 * k2 + 2 * k3 + k4) / 6
    return x, dx


def get_lv_data(n_ics, dt=0.002, num_steps=10000, a=2/3, b=4/3, c=1.0, d=1.0, canonical=True):
    ics = generate_random_ics(n_ics, canonical=canonical)
    x, dx = simulate_lv(ics, dt=dt, num_steps=num_steps, a=a, b=b, c=c, d=d, canonical=canonical)
    x = np.transpose(x, (1, 0, 2))  # (n_ics, num_steps, dim)
    dx = np.transpose(dx, (1, 0, 2))  # (n_ics, num_steps, dim)
    return x, dx
