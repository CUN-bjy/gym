import numpy as np
from gym import utils
from gym.envs.mujoco import mujoco_env


class InvertedPendulumTargetEnv(mujoco_env.MujocoEnv, utils.EzPickle):
    def __init__(self):
        self.tpos=np.array([0,0])
        self.tvel=np.array([0,0])
        utils.EzPickle.__init__(self)
        mujoco_env.MujocoEnv.__init__(self, "inverted_pendulum.xml", 2)
               

    def step(self, a):
        self.do_simulation(a, self.frame_skip)
        obs = self._get_obs()
        
        ob = obs["observation"]
        g = obs["desired_goal"]
        is_succeeded = (np.abs(g[:1]-ob[:1])<0.1)
        notdone = np.isfinite(ob).all() and (np.abs(ob[1]) <= 1.5)
        done = not notdone
                
        reward = 0.5*(-np.abs(g[0]-ob[0])) + (10 if is_succeeded else 1)
        
        if self.viewer:
            del self.viewer._markers[:]
            self.viewer.add_marker(pos=np.concatenate([self.tpos,[0]]),
                        rgba=np.array([1.0, 0.0, 0.0, 0.5]), label="g")
        
        return obs, reward, done, {"is_succeed": is_succeeded}

    def reset_model(self):
        qpos = np.array([self.init_qpos[0] + self.np_random.uniform(low=-3.5, high=3.5), 
                         self.init_qpos[1] + self.np_random.uniform(low=-0.6, high=0.6)])
        qvel = self.init_qvel + self.np_random.uniform(
            size=self.model.nv, low=-0.01, high=0.01
        )
        self.set_state(qpos, qvel)
        self._sample_goal()
        return self._get_obs()
    
    def get_spaces(self):
        return self.observation_space, self.action_space

    def _get_obs(self):
        obs = np.concatenate([self.sim.data.qpos, self.sim.data.qvel]).ravel()
        goal = np.concatenate([self.tpos, self.tvel], dtype=np.float64).ravel()
        return {"observation": obs.copy(),
                "achieved_goal": obs.copy(),
                "desired_goal": goal.copy()}

    def _sample_goal(self):
        self.tpos = np.array([self.init_qpos[0] + self.np_random.uniform(low=-3.5, high=3.5), self.init_qpos[1]])
        self.tvel = self.init_qvel 
        
    
    def viewer_setup(self):
        v = self.viewer
        v.cam.trackbodyid = 0
        v.cam.distance = self.model.stat.extent*1.5