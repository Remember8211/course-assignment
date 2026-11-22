"""Behavioral checks for the new observation, curriculum and reward."""
import unittest
import numpy as np
from avoidance_v2 import AvoidanceEnvV2, curriculum
from env3link import Q0, fk


def task(goal=(.65,.4),obstacles=(),q0=Q0):
    return {'q0':list(q0),'goal':list(goal),'obstacles':[list(c) for c in obstacles],'obstacle_radius':.08}


class V2Checks(unittest.TestCase):
    def test_obs_geometry_mask_and_no_reference_leak(self):
        env=AvoidanceEnvV2(); t=task(obstacles=[[-.6,.3],[-.6,-.3]])
        t['reference_q']=[[999,999,999]]; t['q_goal']=[999,999,999]
        o,_=env.reset(task=t); self.assertEqual(o.shape,(21,))
        np.testing.assert_allclose(o[:3],Q0/np.pi,atol=1e-7)
        ee=fk(Q0)[-1]
        np.testing.assert_allclose(o[9:11],(np.array([-.6,.3])-ee)/2,atol=1e-7)
        np.testing.assert_allclose(o[[11,12,15,16]],[.08,1,.08,1],atol=1e-7)
        np.testing.assert_array_equal(o[17:21],np.zeros(4))
        o,_=env.reset(task=task()); np.testing.assert_array_equal(o[9:],np.zeros(12))

    def test_reward_stationary_and_timeout(self):
        env=AvoidanceEnvV2(); env.reset(task=task())
        _,r,term,trunc,_=env.step(np.zeros(3)); self.assertAlmostEqual(r,-.01); self.assertFalse(term or trunc)
        for _ in range(119): _,r,term,trunc,info=env.step(np.zeros(3))
        self.assertAlmostEqual(r,-5.01); self.assertFalse(term); self.assertTrue(trunc); self.assertEqual(info['status'],'timeout')

    def test_collision_priority_and_negative_terminal_reward(self):
        env=AvoidanceEnvV2(); env.reset(task=task(goal=(1,0),obstacles=[[1.0999,0]],q0=[-.025,0,0]))
        _,r,term,trunc,info=env.step([1,0,0]); self.assertEqual(info['status'],'collision')
        self.assertFalse(info['success']); self.assertTrue(term); self.assertFalse(trunc); self.assertLess(r,-19.)

    def test_arrival_bonus_and_tolerance(self):
        env=AvoidanceEnvV2(); env.reset(task=task(goal=fk(Q0)[-1]))
        _,r,term,trunc,info=env.step(np.zeros(3)); self.assertAlmostEqual(r,19.99); self.assertTrue(term)
        self.assertTrue(info['success']); self.assertFalse(trunc)
        env.reset(task=task(goal=fk(Q0)[-1]+[.04,0])); _,_,term,_,_=env.step(np.zeros(3)); self.assertFalse(term)

    def test_curriculum_boundaries(self):
        for step,expected in [(1,[.5,.5,0]),(50000,[.5,.5,0]),(50001,[.2,.3,.5]),(100000,[.2,.3,.5]),(100001,[.1,.2,.7]),(300000,[.1,.2,.7])]:
            np.testing.assert_array_equal(curriculum(step),expected)

if __name__=='__main__': unittest.main()
