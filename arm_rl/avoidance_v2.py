"""Versioned 21-dimensional environment for the planned SAC retraining."""
from __future__ import annotations
import numpy as np
from env3link import LINK_LEN, fk
from validate_avoidance import AuditEnv

OBS_VERSION = 'full_circle_geometry_v2_21'


class AvoidanceEnvV2(AuditEnv):
    def __init__(self, tolerance=0.02, resolution=0.001):
        super().__init__(tolerance, resolution)
        self.obs_dim = 21

    def _obs(self, pts):
        ee, goal = pts[-1], np.asarray(self.task['goal'])
        length = float(LINK_LEN.sum())
        values = [self.q/np.pi, ee/length, goal/length, (goal-ee)/(2*length)]
        obstacles = self.task['obstacles']
        if len(obstacles) > 3:
            raise ValueError('Version 2 supports at most three obstacles.')
        for i in range(3):
            if i < len(obstacles):
                values.append(np.r_[(np.asarray(obstacles[i])-ee)/(2*length), self.ob_r/length, 1.0])
            else:
                values.append(np.zeros(4))
        return np.concatenate(values).astype(np.float32)

    def step(self, action):
        previous = float(np.linalg.norm(fk(self.q)[-1]-self.task['goal']))
        obs, status, detail = self.advance(action)
        u = np.clip(np.asarray(action, float), -1.0, 1.0)
        clearance = detail['min_clearance']
        safety = 0.0 if not self.task['obstacles'] else 0.1*max(0.0,(0.03-clearance)/0.03)**2
        reward = 5*(previous-detail['dist'])/float(LINK_LEN.sum())-0.01-0.001*float(u@u)-safety
        reward += {'success':20.0, 'collision':-20.0, 'timeout':-5.0}.get(status,0.0)
        info = {'status':status, 'success':status=='success', 'collision':status=='collision',
                'dist':detail['dist'], 'clearance':self.ep_min_clear, 'steps':self.steps,
                'path_length':self.ep_path}
        return obs, float(reward), status in ('success','collision'), status=='timeout', info


def curriculum(step):
    if step <= 50000:
        return np.array([0.5,0.5,0.0])
    if step <= 100000:
        return np.array([0.2,0.3,0.5])
    return np.array([0.1,0.2,0.7])
