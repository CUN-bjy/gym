import numpy as np
from gym import utils
from gym.envs.mujoco import mujoco_env


class InvertedPendulumEnv5(mujoco_env.MujocoEnv, utils.EzPickle):
    def __init__(self):
        utils.EzPickle.__init__(self)
        mujoco_env.MujocoEnv.__init__(self, "inverted_pendulum.xml", 2)

    def step(self, a):
        self.do_simulation(a, self.frame_skip)
        ob = self._get_obs()
        notdone = np.isfinite(ob).all() and (np.abs(ob[1]) <= 1.5)
        done = not notdone
        reward = -ob[1] if ob[1] < 0 else 0
        return ob, reward, done, {}

    def reset_model(self):
        qpos = np.array([self.init_qpos[0] + self.np_random.uniform(low=-3.5, high=3.5), self.init_qpos[1] + self.np_random.uniform(low=-0.6, high=0.6)])
        qvel = self.init_qvel + self.np_random.uniform(
            size=self.model.nv, low=-0.01, high=0.01
        )
        self.set_state(qpos, qvel)
        return self._get_obs()

    def _get_obs(self):
        return np.concatenate([self.sim.data.qpos, self.sim.data.qvel]).ravel()

    def get_spaces(self):
        return self.observation_space, self.action_space
    
    def viewer_setup(self):
        v = self.viewer
        v.cam.trackbodyid = 0
        v.cam.distance = self.model.stat.extent*1.5