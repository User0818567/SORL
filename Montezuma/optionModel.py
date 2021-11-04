from argParser import *
from agent import *
from env import *

import torch.optim as optim

from torch.multiprocessing import Process, Pipe
#from multiprocessing import Process, Pipe
from typing import List, Tuple, Union, Optional, Callable, Any

from tianshou.utils import tqdm_config
from tianshou.env import BaseVectorEnv
from tianshou.env.utils import CloudpickleWrapper

import gym
import tqdm
import time
import pickle
import numpy as np
from runx.logx import logx
import matplotlib.pyplot as plt
import os

import subprocess

numStatepair = []
symbolicOptions = []
gainreward = []


def num2symbolic(numState): #from numstate to symbolic state without quality
    symbolicState = []
    if numState[0]==-1:
        atpro = "(at None)"
        symbolicState.append(atpro)
    else:
        atpro = "(at "+ idx2loc[numState[0] ]+")"
        symbolicState.append(atpro)
    if numState[1]==1:
        symbolicState.append('(keyexist)')
    return symbolicState

def getSymbolicEffect(state1,state2):
    effect = []
    for proposition in state1:
        if proposition not in state2:
            tmp = "(not "+proposition+" )"
            effect.append(tmp)
    for proposition in state2:
        if proposition not in state1:
            effect.append(proposition)
    return effect

#plantrace is a list of numStates and the index of options
#in the process of getPlantrace,we also append the numStatepair and gainreward
def getPlantrace(stepbuffer):
    plantrace = []
    currentQuality = 0
    for i in range(len(stepbuffer)):
        plantrace.append(num2symbolic(stepbuffer[i]) )
        if i == (len(stepbuffer)-1):
            break
        if (stepbuffer[i][:-1],stepbuffer[i+1][:-1]) in numStatepair:
            tmpindex = numStatepair.index( (stepbuffer[i][:-1],stepbuffer[i+1][:-1]) )
            plantrace.append(numStatepair.index((stepbuffer[i][:-1],stepbuffer[i+1][:-1]) ))
            gainreward[tmpindex].append(stepbuffer[i][2]-currentQuality)
        else:
            plantrace.append(len(numStatepair))
            numStatepair.append((stepbuffer[i][:-1],stepbuffer[i+1][:-1]) )
            gainreward.append([stepbuffer[i][2]-currentQuality ])
        currentQuality = stepbuffer[i][2]
             
    return plantrace

#symbolic option is a tuple of (prestate,effects)
def getSymbolicOptions():
    symbolicOptions.clear()
    for i in range(len(numStatepair) ):
        (state1,state2) = numStatepair[i]
        state1 = num2symbolic(state1)
        state2 = num2symbolic(state2)
        effect = getSymbolicEffect(state1,state2)
        tmp = round(np.mean(gainreward[i]))
        ss = "(increase (quality) "+str(tmp)+" )"
        effect.append(ss)
        symbolicOptions.append( (state1,effect,state2) )
    return symbolicOptions

#given symbolic option model,generate domain file
def generateDomainFile():
    print("start generate domain file")
    domainfile = open(args.domain_file,"w")
    domainfile.truncate()
    init_domain = open(args.initial_domain_file,"r")
    for line in init_domain:
        domainfile.write(line)
    init_domain.close()
    for i in range(len(symbolicOptions)):
        action = symbolicOptions[i]
        domainfile.write("\t(:action %s\n" % i)
        domainfile.write("\t\t:precondition (and ")
        for precondition in action[0]:
            domainfile.write(precondition+" ")
        domainfile.write(")\n")
        domainfile.write("\t\t:effect (and ")
        for effect in action[1]:
            domainfile.write(effect+" ")
        domainfile.write(")\n\t)\n")
    domainfile.write(")")
    domainfile.close()


def testOptions():

    actionMap = [0, 1, 2, 3, 4, 5, 11, 12]

    env = ALEEnvironment(args.game, args)
    maxStepsPerEpisode = 500
    loclist = ['LeftDoor', 'RightDoor', 'Key', 'MiddleLadder', 
            'LeftLadder', 'RightLadder']
    idx2loc,loc2idx = {},{}
    for i in range(len(loclist)):
        idx2loc[i] = loclist[i]

    for key, value in idx2loc.items():
        loc2idx[value] = key

    episodebuffer = []
    for episodeCount in range(20):
        test_rew = 0
        env.restart()
        episodeStep = 0
        stepbuffer = []
        while not env.isTerminal() and episodeStep < maxStepsPerEpisode:
            numState = env.getNumState()
            numState.append(test_rew)
            if len(stepbuffer)==0 or (stepbuffer[-1]!=numState and numState[0]!=-1):
                stepbuffer.append(numState)
            episodeStep += 1
            act = random.randint(0,7)
            tmp_rew = env.act(actionMap[act])
            test_rew += tmp_rew
            if test_rew >= 400:
                break
        print(stepbuffer)
        episodebuffer.append(stepbuffer)
    for stepbuffer in episodebuffer:
        print(getPlantrace(stepbuffer))
    getSymbolicOptions()
    generateDomainFile()
    print("gain reward: ",gainreward)
    print("numStatepair: ",numStatepair)
    print("symbolicOptions: ",symbolicOptions)



testOptions()