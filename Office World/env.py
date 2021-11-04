if __name__ == '__main__':
    # This is a terrible hack just to be able to execute this file directly
    import sys
    sys.path.insert(0, '../')

import sys
import cv2
import torch
import numpy as np
from PIL import Image, ImageDraw
from atari_py.ale_python_interface import ALEInterface

import gym
from torch.multiprocessing import Process, Pipe
from typing import List, Tuple, Union, Optional, Callable, Any

from tianshou.data import Batch
from tianshou.utils import tqdm_config
from tianshou.env import BaseVectorEnv
from tianshou.env.utils import CloudpickleWrapper
from runx.logx import logx

from collections import namedtuple

import matplotlib.pyplot as plt
from argParser import *

from PIL import Image
from office_world import *
import random, math, os
import numpy as np
from reward_machine import RewardMachine

# e-mail, f-coffee, g-office, n-plant
#idx2proposition={0:'getCoffee', 1:'getMail', 2:'getOffice', 3:'getA', 4:'getB', 5:'getC', 6:'getD'}
#idx2proposition={0:'getCoffee', 1:'getMail', 2:'getOffice'}
idx2proposition={0:'(haveCoffee)', 1:'(haveMail)', 2:'(deliveredCoffee)', 3:'(deliveredMail)'}
str_to_action = {"w":Actions.up.value,"d":Actions.right.value,"s":Actions.down.value,"a":Actions.left.value}
Num_subgoal = args.num_subgoal

u2numstate={(1,0):[0,0,0,0], (1,1):[1,0,0,0], (1,2):[0,0,1,0], \
            (2,0):[0,0,0,0], (2,1):[0,1,0,0], (2,2):[0,0,0,1], \
            (3,0):[0,0,0,0], (3,1):[1,0,0,0], (3,2):[1,1,0,0], (3,3):[0,0,1,1]}

class OfficeEnv():
    def __init__(self,task):
        params = OfficeWorldParams()
        self.game = OfficeWorld(params)
        self.task = task
        tmp = "experiments/office/reward_machines/t%d.txt"%task
        self.rm = RewardMachine(tmp,False,0)
        self.is_terminal = False
        self.getgoal = False
        self.u = self.rm.get_initial_state()
        self.s = self.game.get_state()
        self.numstate = [0,0,0,0]

    def isTerminal(self):
        return self.is_terminal

    def getU(self):
        return self.u

    def restart(self):
        params = OfficeWorldParams()
        self.game = OfficeWorld(params)
        self.numstate = [0,0,0,0]
        self.is_terminal = False
        tmp = "experiments/office/reward_machines/t%d.txt"%self.task
        self.rm = RewardMachine(tmp,False,0)
        self.u = self.rm.get_initial_state()
        self.s = self.game.get_state()
        self.getgoal = False

    def execute_action(self,a):
        if a in [0,1,2,3]:
            s1 = self.s
            u1 = self.u
            self.game.execute_action(a)
            event = self.game.get_true_propositions()
            
            u2 = self.rm.get_next_state(u1,event)
            s2 = self.game.get_state()
            r = self.rm.get_reward(u1,u2,s1,a,s2,False)
            is_done = self.rm.is_terminal_state(u2)
            self.u = u2
            self.s = s2
            if is_done:
                self.is_terminal = True
            if is_done and r>0:
                self.getgoal = True
            if is_done and r==0:
                is_done = 2
            if (self.task,self.u) in u2numstate:
                self.numstate = u2numstate[(self.task,self.u)]
            return u2,s2,r,is_done
        else:
            return None
  
    def getNumState(self):
        return self.numstate

    def getstate(self):
        x,y = self.game.get_state()
        return x*9+y

    def getmetastate(self):
        x,y = self.game.get_state()
        if self.getgoal:
            return 108*Num_subgoal
        elif self.is_terminal:
            return 108*Num_subgoal+1
        else:
            return (self.u)*108+x*9+y
        

    def getSymbolicState(self):
        numstate = self.numstate
        symstate = []
        for i in range(4):
            if numstate[i]==0:
                str = "(not "+idx2proposition[i]+")"
                symstate.append(str)
            else:
                symstate.append(idx2proposition[i])
        
        return symstate

    def get_features(self):
        x,y = self.game.get_state()
        N,M = 12,9
        ret = np.zeros(N*M+1, dtype=np.float64)
        ret[x*M+y]=1
        ret[-1]=self.getU()
        return ret

