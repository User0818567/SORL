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

from collections import namedtuple
Mask = namedtuple('Mask', 'l t r b')

import matplotlib.pyplot as plt
from argParser import *

from PIL import Image


actionMap = [0, 1, 2, 3, 4, 5, 11, 12]

Num_subgoal = args.num_subgoal
goal_to_train = [i for i in range(Num_subgoal)]
maxStepsPerEpisode = 500


# location of static objects
coord_RightDoor = [137, 71]
coord_MiddleLadder = [80, 112]
coord_LeftLadder = [24, 157]
coord_RightLadder = [136, 157]
coord_Key = [16, 106]
coord_Down1 = [25,60]
coord_Down2 = [50,60]

Coord = [[0,0], coord_RightDoor, coord_Key,
        coord_MiddleLadder,
        coord_LeftLadder, coord_RightLadder,coord_Down1,coord_Down2, [0,0]
        ]

#mask_LeftDoor = Mask(18, 51, 26, 91)
mask_RightDoor = Mask(133, 51, 141, 91)
mask_MiddleLadder = Mask(72, 73, 88, 137)
mask_LeftLadder = Mask(16, 138, 32, 182)
mask_RightLadder = Mask(128, 138, 144, 182)
#mask_Conveyer = Mask(60, 136, 100, 142)
#mask_Chain = Mask(110, 95, 114, 135)
mask_Key = Mask(13, 97, 19, 115)
mask_Down1 = Mask(40,138,60,182)
mask_Down2 = Mask(80,138,100,182)

loclist = ['RightDoor', 'Key', 'MiddleLadder', 'LeftLadder', 'RightLadder','Down1','Down2']


idx2loc, loc2idx = {}, {}

for i in range(len(loclist)):
    idx2loc[i] = loclist[i]

for key, value in idx2loc.items():
    loc2idx[value] = key

idx2predicate = {0 : 'ActorOnSpot', 1 : 'ActorWithKey', 2 : 'ActorWithoutKey', 3 : 'PathExist', 4 : 'Conditional', 5 : 'Near', 6 : 'Away'}

predicate2idx = {}
for key,value in idx2predicate.items():
    predicate2idx[value] = key

from collections import deque

def initialize_ale_env(rom_file, args):
    ale = ALEInterface()
    if args.display_screen:
      if sys.platform == 'darwin':
        import pygame
        pygame.init()
        ale.setBool('sound', False)
      elif sys.platform.startswith('linux'):
        ale.setBool('sound', False)
      ale.setBool('display_screen', False)
    
    ale.setInt('frame_skip', args.frame_skip)
    ale.setFloat('repeat_action_probability', 0.0)
    ale.setBool('color_averaging', args.color_averaging)

    ale.setInt('random_seed', 0) #hoang addition to fix the random seed across all environment
    ale.loadROM(rom_file)

    if args.minimal_action_set:
      actions = ale.getMinimalActionSet()
    else:
      actions = ale.getLegalActionSet()
    
    return ale, actions

class ALEEnvironment():

    def __init__(self, rom_file, args):

        self.ale, self.actions = initialize_ale_env(rom_file, args)

        self.histLen = 4
        self.screen_width = args.screen_width
        self.screen_height = args.screen_height

        self.restart()
        self.agentOriginLoc = [80, 83]
        self.agentLastX = 80
        self.agentLastY = 83

        self.devilLastX = 0
        self.devilLastY = 0

        self.reachedGoal = np.zeros((10,10))
        self.last_spot = 'MiddleLadder'
        self.mode = 'train'

    def getimg(self):
        return self.ale.getScreenRGB()

    def locF(self):
        idx_loc = self.reach_goal()
        if idx_loc == -1:
            predicate_loc = "None"
        else:
            predicate_loc = idx2loc[idx_loc]
        return idx_loc,predicate_loc
    
    def getSymbolicState(self):
        idx_loc,predicate_loc = self.locF()
        iskey = self.keyExist()
        return predicate_loc, iskey
    
    #return the num list presentation of state [the index of state, iskey exist]
    def getNumState(self):
        numState = []
        idx_loc = self.reach_goal()
        numState.append(idx_loc)
        iskey = self.keyExist()
        if iskey:
            numState.append(1)
        else:
            numState.append(0)
        return numState


    def getScreen(self, ScreenType='Gray'):
        if ScreenType == 'Gray':
            screen = self.ale.getScreenGrayscale()
        else:
            screen = self.ale.getScreenRGB()
        resized = cv2.resize(screen, (self.screen_width, self.screen_height))
        return resized
    
    def getState(self):
        return np.reshape(self.getScreen(), (1, self.screen_width, self.screen_height))
    
    def initializeHistState(self):
        self.histState = np.zeros((self.histLen, self.screen_width, self.screen_height))
        initialState = self.getState()[:,:,0]
        for num in range(self.histLen):
            self.histState[num,:,:] = initialState
    
    def restart(self):
        self.ale.reset_game()
        self.life_lost = False
        for _ in range(19):
            rew = self.ale.act(0)
        self.initializeHistState()

        self.initializeScreen = self.ale.getScreenRGB()

        # initialize logcY buf
        self.agent_locY_buf = deque(maxlen=5)
        self.agent_locY_buf.append(83)

    @property
    def numActions(self):
        return len(self.actions)
  
    def getLoc(self, img, obj='Man'):
        
        # img : RGB Array
        if obj == 'Skull':
            color = [236, 236, 236]
            mean_x = self.devilLastX
            mean_y = self.devilLastY
        else:
            color = [200, 72, 72]
            mean_x = self.agentLastX
            mean_y = self.agentLastY

        mask = np.zeros(np.shape(img))
        mask[:,:,0] = color[0]
        mask[:,:,1] = color[1]
        mask[:,:,2] = color[2]

        diff = img - mask
        indxs = np.where(diff == 0)
    
        if (np.shape(indxs[0])[0]):
            y = indxs[0][-1]
            x = indxs[1][-1]
            if obj == 'Man':
                self.agentLastX, self.agentLastY = x - 3, y - 6
                return [x - 3, y - 6]
            else:

                if y > 150:
                    self.devilLastX, self.devilLastY = x, y
                    return [x - 1, y - 6]
                else:
                    return [None, None]
        else:
            if obj == 'Man':
                x, y = self.agentLastX, self.agentLastY
                return [x, y]
            else:
                x, y = None, None
                return [x, y]

    def distanceReward(self, lastgoal, goal):
        if (lastgoal == None):
            lastgoal_x, lastgoal_y = self.agentOriginLoc
        else:
            lastgoal_x, lastgoal_y = Coord[loc2idx[lastgoal]]
        
        img = self.ale.getScreenRGB()
        man_x, man_y = self.getLoc(img, 'Man')

        goal_x, goal_y = Coord[loc2idx[goal]]

        if goal_x == None or goal_y == None:
            return 0

        dis = np.sqrt((man_x - goal_x) ** 2 + (man_y - goal_y) ** 2)
        disLast = np.sqrt((man_x - lastgoal_x) ** 2 + (man_y - lastgoal_y) ** 2)
        disGoals = np.sqrt((goal_x - lastgoal_x) ** 2 + (goal_y - lastgoal_y) ** 2)
        return 0.001 * (disLast - dis) / disGoals
    
    def getStackedState(self):

        return self.histState

    def isGameOver(self):
        return self.ale.game_over()

    def isTerminal(self):

        if self.mode == 'train':
            return self.ale.game_over() or self.life_lost
        return self.ale.game_over()

    def isLifeLost(self):
        return self.life_lost

    def skullExist(self, img):

        skull_x, skull_y = self.getLoc(img, 'Skull')
        if skull_y == None:
            return False
        return True

    def keyExist(self):

        img = self.ale.getScreenRGB()

        mask = np.ones((3,2,3))
        mask[:,:,0] = 232
        mask[:,:,1] = 204
        mask[:,:,2] = 99


        if (img[104:107, 15:17, :] == mask).all():
            return True
        return False

    def get_symbolic_tensor(self):
        
        '''
            change symbolic state into array
            the predicate of each channel : 
            0 : ActorOnSpot
            1 : ActorWithObject
            2 : ActorWithoutObject
            3 : PathExist
            4 : Conditional

            the object order of each dimension
            0 : Man 
            1 : RightDoor
            2 : Key
            3 : MiddleLadder
            4 : LeftLadder
            5 : RightLadder
            6 : SkullLeft
        '''
        
        img = self.ale.getScreenRGB()

        self.state_tensor = np.zeros((5, 7, 7))
        symbolic_state = set()

        x,y = self.getLoc(img, 'Man')
        a,b = self.getLoc(img, 'Skull')

        if self.keyExist() and self.reach_goal() != 2:
            self.state_tensor[2][0][2] = 1
            symbolic_state.add(('ActorWithOutKey'))
        else:
            self.state_tensor[1][0][2] = 1
            symbolic_state.add(('ActorWithKey'))

        spot = self.ActorOnSpot(img, (x,y))
        self.state_tensor[0][0][loc2idx[spot]] = 1
        symbolic_state.add(('ActorOnSpot', spot))

        connected_tuple = [(3,5), (5,6) ,(6,4), (4,2), (3,1)]

        for ple in connected_tuple:
            self.state_tensor[-2][ple[0]][ple[1]] = 1
            self.state_tensor[-2][ple[1]][ple[0]] = 1
        
        self.state_tensor[-1][2][1] = 1

        self.symbolic_state = symbolic_state

        return self.state_tensor.reshape((5, 7 ** 2))

    def action_to_symbolic(self, action):

        actionMap = {0 : 0,
                    1 : 1,
                    2 : 2,
                    3 : 3,
                    4 : 4,
                    5 : 5,
                    6 : 11,
                    7 : 12}

        actionExplain = {0 : 'noAction',
                        1 : 'jump',
                        2 : 'up',
                        3 : 'right',
                        4 : 'left',
                        5 : 'down',
                        11 : 'jumpRight',
                        12 : 'jumpLeft'}
        
        return actionExplain[actionMap[action]]
    
    def create_dynamic_object_mask(self, img, coord_man, coord_skull):

        man_x, man_y = coord_man
        skull_x, skull_y = coord_skull

        self.mask_man = Mask(man_x - 4, man_y - 10, man_x + 4, man_y + 10)

        if skull_x != None:
            self.mask_skull = Mask(skull_x - 4, skull_y - 6, skull_x + 4, skull_y + 6)
        else:
            self.mask_skull = Mask(None, None, None, None)

        return self.mask_man, self.mask_skull

    def draw_mask(self, img):

        coord_man = self.getLoc(img, 'Man')
        coord_skull = self.getLoc(img, 'Skull')
        mask_Man, mask_Skull = self.create_dynamic_object_mask(img, coord_man, coord_skull)
        mask_list = [mask_LeftDoor, mask_RightDoor, mask_MiddleLadder,
                    mask_LeftLadder, mask_RightLadder, mask_Conveyer,
                    mask_Chain, mask_Man]
        if mask_Skull.l != None:
            mask_list.append(mask_Skull)
        if self.keyExist():
            mask_list.append(mask_Key)
        for mask in mask_list:
            cv2.rectangle(img, (mask.l, mask.t), (mask.r, mask.b), (0,0,255), 1)
        return img


    def select_goal(self, goal, reach_goal):
        
        '''
        0 : Man 
        1 : RightDoor
        2 : Key
        3 : MiddleLadder
        4 : LeftLadder
        5 : RightLadder
        6 : Skull
        '''

        if reach_goal == 2:
            if goal == 4:
                return 3

        if reach_goal == 3 and self.keyExist():
            if goal == 5:
                return 0

        if reach_goal == 3 and not self.keyExist():
            if goal == 1:
                return 6

        if reach_goal == 4:
            if goal == 5:
                return 4

        if reach_goal == 5:

            if goal == 3 and not self.keyExist():
                return 5

            if goal == 6 and self.keyExist():
                return 1

        if reach_goal == 6:
            if goal == 2:
                return 2
        return -1

    def reach_goal(self):
        goal_reached = -1
        img = self.ale.getScreenRGB()
        man_x, man_y = self.getLoc(img, 'Man')
        #for idx in [1,3,4,5,6,2]:
        for idx in [0,2,3,4]:
            goal = idx2loc[idx]
            #print("index %s , goal %s " % (idx,goal))
            if self.goalReached(img, goal, (man_x, man_y)):
                goal_reached = idx
                break
        return goal_reached

    def goalReached(self, img, loc, coords=None):

        if self.isTerminal():
            return False

        if coords == None:
            man_x, man_y = self.getLoc(img, 'Man')
        else:
            man_x, man_y = coords
        if loc == 'RightDoor':
            if man_x >= 128 and man_y >= 51 and man_y <= 91:
                return True

        skull_x,skull_y = self.getLoc(img,'Skull')
        
        if loc == 'Down1':
            left = skull_x - 25
            right = skull_x -5
            top = 160
            bottom = 182
            if man_x >= left and man_x <=right and man_y >=top and man_y <= bottom:
                return True
        if loc == 'Down2':
            left = skull_x + 5
            right = skull_x + 25
            top = 160
            bottom = 182
            if man_x >= left and man_x <=right and man_y >=top and man_y <= bottom:
                return True         

        '''
        if loc == 'Down1':
            if man_x >=40 and man_x <= 60 and man_y>=160:
                return True
        
        if loc == 'Down2':
            if man_x >=80 and man_x <=100 and man_y>=160:
                return True
        '''

        if loc == 'Key':
            if not self.keyExist():
                if man_x >= 8 and man_x <= 35 and man_y >= 108 and man_y <= 134:
                    return True
            else:
                if man_x >= 8 and man_x <= 21 and man_y <= 120:
                    return True

            '''
            if self.keyExist():
                return False
            return True
            '''
        
        mask_ladder = None
        if loc == 'MiddleLadder':
            mask_ladder = mask_MiddleLadder
        if loc == 'LeftLadder':
            mask_ladder = mask_LeftLadder
        if loc == 'RightLadder':
            mask_ladder = mask_RightLadder
        if mask_ladder != None:
            left, top, right, bottom = mask_ladder
            if loc == 'LeftLadder':
                right = 29
                top = 160
            
            if loc == 'RightLadder':
                top = 160
            
            if loc == 'MiddleLadder':
                bottom = 110

            if man_x >= left and man_x <= right and man_y >= top and man_y <= bottom:
                return True
            
            if loc == 'MiddleLadder':
                '''
                if man_x >= 60 and man_x <= 100 and man_y >= 125 and man_y <= 127:
                    return True
                '''
                if abs(man_x - 80) <= 2 and abs(man_y - 82) <= 3:
                    return True

        if loc == 'Skull':
            mask_ladder = mask_LeftLadder
            left, top, right, bottom = mask_ladder
            left = 31
            right = 35
            top = 160
            if man_x >= left and man_x <= right and man_y >= top and man_y <= bottom:
                return True
        if loc == 'SkullRight':
            mask_ladder = mask_RightLadder
            left, top, right, bottom = mask_ladder
            left = 119
            right = 124
            top = 160
            if man_x >= left and man_x <= right and man_y >= top and man_y <= bottom:
                return True
        
        if loc == 'Conveyer':
            if man_x >= 60 and man_x <= 100 and man_y >= 125 and man_y <= 127:
                return True

        return False
    
    def ActorOnSpot(self, img, coords=None):
        
        if coords == None:
            man_x, man_y = self.getLoc(img, 'Man')
        else:
            man_x, man_y = coords
        
        spot = None
        for loc in loclist[1:]:
            if self.goalReached(img, loc, (man_x, man_y)):
                spot = loc
                self.last_spot = spot
                return spot
        return self.last_spot
    
    def step(self, action, lastgoal, goal):

        reward = self.ale.act(self.actions[action])
        reward = self.distanceReward(lastgoal, goal)

        if self.isLifeLost():
            done = True
        return self.histState, reward, done, {}

    def act(self, action):
        lives = self.ale.lives()
        reward = self.ale.act(self.actions[action])
        self.life_lost = (not lives == self.ale.lives())
        curState = self.getState()

        img = self.ale.getScreenRGB()
        x,y = self.getLoc(img)
        #spot = self.ActorOnSpot(img, (x,y))

        '''
        x,y = self.getLoc(self.ale.getScreenRGB())
        self.agent_locY_buf.append(y)
        '''

        self.histState = np.concatenate((self.histState[1:, :, :], curState), axis = 0)
        return reward

    def beginNextLife(self):
        self.life_lost = False
        for _ in range(19):
            rew = self.act(0)
        self.initializeHistState()

    def Near(self, object_loc1, object_loc2, idx1=None, idx2=None):

        x1,y1 = object_loc1
        x2,y2 = object_loc2
        
        if x1 == None or x2 == None or y1 == None or y2 == None:
            return 'None'

        if (idx1 == 0 and idx2 == 9) or (idx1 == 9 and idx2 == 0):
            if abs(x1 - x2) <= 20 and abs(y1 - y2) <= 3:
                '''
                for i in self.agent_locY_buf:
                    if i != 169 or i != 170:
                        return 'Away'
                '''
                return 'Near'
            return 'Away'
        
        if abs(x1 - x2) < 20 and abs(y1 - y2) < 33:
            return 'Near'
        return 'Away'
