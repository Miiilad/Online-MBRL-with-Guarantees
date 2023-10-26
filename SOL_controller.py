import numpy as np

class SOL_controller:
    def __init__(self,env_name) -> None:
        self.P = np.load('SOL_trained/' + env_name + '/P.npy')
        self.W = np.load('SOL_trained/' + env_name + '/W.npy')
        self.N = len(self.P)
        self.r = 1
        if env_name == "Pendulum":
            self.W = self.W[:,5:10]
            self.select_control = self.call_pendulum
        elif env_name == "Cartpole":
            self.W = self.W[:,self.N:2*self.N]
            self.select_control = self.call_cartpole
        else:
            self.r = 0.5e-7*9
            self.select_control = self.call_quadrotor

    def call_pendulum(self, x) -> np.ndarray:
        phi = np.ones(5)
        phi[1:3] = x
        phi[3:5] = np.sin(x)
        partial_d = np.array([[0., 0.], [1., 0.], [0., 1.], [np.cos(x[0]), 0], [0, np.cos(x[1])]])
        u = np.array([- phi.T.dot(1/self.r*self.P).dot(partial_d).dot(self.W).dot(phi)])
        return u
    
    def call_cartpole(self, x) -> np.ndarray:
        phi = np.ones(self.N)
        phi[1:5] = x
        phi[5:9] = x**2
        phi[9:] = np.sin(x)
        partial_d = np.concatenate((np.zeros((1,4)),np.identity(4), np.diag(2*x), np.diag(np.cos(x))),axis=0)
        u = np.array([- phi.T.dot(1/self.r*self.P).dot(partial_d).dot(self.W).dot(phi)])
        return u

    def call_quadrotor(self, x) -> np.ndarray:
        u = np.zeros(4)
        phi = np.ones(self.N)
        phi[1:] = x
        partial_d = np.concatenate((np.zeros((1,self.N-1)),np.identity(self.N-1)),axis=0)
        for j in range(4):
            u[j] = - phi.T.dot(1/self.r*self.P).dot(partial_d).dot(self.W[:,self.N*(j+1):self.N*(j+2)]).dot(phi)
        return u
