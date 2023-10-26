import numpy as np
from functools import partial
from scipy.integrate import solve_ivp

class Pendulum():
    def __init__(self, m=0.33, Ixx=1.395e-05, Iyy=1.436e-05, Izz=2.173e-05, g=9.81):
        self.dim_n = 2
        self.dim_m = 1
        self.measure_dim = 2
        self.x = np.zeros((self.dim_n))
        self.x_dot = np.zeros((self.dim_n))
        self.measure = np.zeros((self.measure_dim))
        self.Domain = [-2, 2, -2, 2]

    def integrate(self, u, t_interval):
        dx_dt = partial(self.dynamics, u=u)
        sol = solve_ivp(dx_dt, t_interval, self.x, method='RK45', t_eval=None, rtol=1e-6, atol=1e-6, dense_output=False,
                        events=None, vectorized=False)
        self.x = sol.y[..., -1]
        # return partial(self.dynamics, u=u)

    def dynamics(self, t, x, u):
        m = 0.1
        l = 0.5
        k = 0.1
        x1, x2 = x

        x1_d = x2
        x2_d = -4*x2 + 19.6*np.sin(x1) + 40*u
        # x2_d = -(9.8 / l) * (x1) - (k / m) * x2 + (1 / (m * l ** 2)) * u
        return [x1_d, x2_d]

    def Read_sensor(self):
        return self.x  # self.x[0]

    def Read_sensor_with_noise(self, sig):
        self.measure = (np.random.normal(0, sig, self.measure_dim) + np.ones((self.measure_dim))) * self.measure
        return self.measure

    def In_domain(self, x):
        in_dom = True
        for i in range(self.dim_n):
            if (x[i] < self.Domain[2 * i]) | (x[i] > self.Domain[2 * i + 1]):
                in_dom = False
#                 print("x[{}]={} is out of range[{},{}]".format(i, x[i], self.Domain[2 * i], self.Domain[2 * i + 1]))
        return in_dom

    def randomly_initialize(self):
        x_init = np.zeros(self.dim_n)
        for i in range(self.dim_n):
            x_init[i] = np.random.uniform(self.Domain[2 * i]*0.5, self.Domain[2 * i + 1]*0.5, size=1)
        self.x = np.copy(x_init)
        return x_init
