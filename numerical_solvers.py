import numpy as np
import matplotlib.pyplot as plt

from scipy.integrate import solve_ivp



def RiccatiSolver(x,f,u_0):
    ode_fn = lambda x, y: y**2 + y + f(x)
    x_0 = x[0]
    x_n = x[-1]
    method = 'RK45' #available methods: 'RK45', 'RK23', 'DOP853', 'Radau', 'BDF', 'LSODA'
    num_sol = solve_ivp(ode_fn, [x_0, x_n], [u_0], method=method, dense_output=True)
    return num_sol.sol(x).T


def NumIntegral(x,f,u_0):
    ode_fn = lambda x, y: f(x)
    x_0 = x[0]
    x_n = x[-1]
    method = 'RK45' #available methods: 'RK45', 'RK23', 'DOP853', 'Radau', 'BDF', 'LSODA'
    num_sol = solve_ivp(ode_fn, [x_0, x_n], [u_0], method=method, dense_output=True)
    return num_sol.sol(x).T


def HeatSolver(T0, x, t_max=1, alpha=0.1):
    L = x.max()-x.min()
    nx = x.shape[0]
    dx = L / (nx - 1)
    sigma = 0.5
    dt = sigma * dx**2 / alpha  # time-step size
    nt = int(t_max//dt)

    temp = []
    T = T0(x)
    sigma = alpha * dt / dx**2

    for n in range(nt):
        temp.append(T.copy())
        T[1:-1] = (T[1:-1] +
                   sigma * (T[2:] - 2.0 * T[1:-1] + T[:-2]))
        T[-1] = 0
        T[0] = 0
        
    return np.array(temp).T, np.linspace(0,t_max, nt)



def Poisson_solve(f, rho, h, stepper, atol=1.E-6, max_steps=10**5):
    for _ in range(max_steps):
        f_new = stepper(f, rho, h)
        if np.max(np.abs(f_new - f)) < atol:
            return f_new
        f = f_new


def jacobi(f, rho, h):
    f_new = f.copy()
    id = tuple([slice(1, -1)] * f.ndim)
    f_new[id] = -h**2 * rho[id]
    for axis in range(f.ndim):
        for offset in [slice(2, None), slice(None, -2)]:
            sl = list(id)
            sl[axis] = offset
            f_new[id] += f[tuple(sl)]
    f_new[id] /= (2 * f.ndim)
    return f_new