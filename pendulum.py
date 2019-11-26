import numpy as np

class PendulumEnv:

    def __init__(self, g=10.0):
        self.max_speed=8
        self.max_torque=2.
        self.dt=.05
        self.g = g
        self.step_counter = 0

        high = np.array([1., 1., self.max_speed])
        self.min_state = -high
        self.max_state = high
        self.min_action = [-self.max_torque]
        self.max_action = [self.max_torque]
        self.step_limit = 200


    def step(self,u):
        th, thdot = self.state # th := theta

        g = self.g
        m = 1.
        l = 1.
        dt = self.dt

        u = np.clip(u, -self.max_torque, self.max_torque)[0]
        self.last_u = u # for rendering
        costs = angle_normalize(th)**2 + .1*thdot**2 + .001*(u**2)

        newthdot = thdot + (-3*g/(2*l) * np.sin(th + np.pi) + 3./(m*l**2)*u) * dt
        newth = th + newthdot*dt
        newthdot = np.clip(newthdot, -self.max_speed, self.max_speed) #pylint: disable=E1111

        self.state = np.array([newth, newthdot])

        self.step_counter += 1

        if self.step_counter == self.step_limit:
            done = True 
        else:
            done = False

        return self._get_obs(), -costs, done, {}

    def reset(self):
        high = np.array([np.pi, 1])
        self.state = np.random.uniform(low=-high, high=high)
        self.last_u = None
        self.step_counter = 0
        return self._get_obs()

    def _get_obs(self):
        theta, thetadot = self.state
        return np.array([np.cos(theta), np.sin(theta), thetadot])

def angle_normalize(x):
    return (((x+np.pi) % (2*np.pi)) - np.pi)