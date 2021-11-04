from argParser import *
from agent import *
from env import *

import torch.optim as optim

import torch
import tqdm
import time
import pickle
import numpy as np
from runx.logx import logx

from collections import Counter

import os
import shutil
import pdb
import pickle

Num_subgoal = args.num_subgoal

goal_to_train = [i for i in range(Num_subgoal)]

maxOptionsPerEpisode = 10
maxStepsPerOption = 1000
logx.initialize(logdir=args.hrllogdir, hparams=vars(args), tensorboard=True)

env = OfficeEnv(1)
subgoal2goal = [1,2,3]

agent_list = [Q_table(i,108,4,10000) for i in range(Num_subgoal-1)]
metaAgent = Q_table(-1,108*Num_subgoal+2,Num_subgoal-1,100)


#record the random experience num of a subgoal
option_t = [0 for _ in range(Num_subgoal)]

success_tracker = [[] for _ in range(Num_subgoal)]

sample_t = [0 for _ in range(Num_subgoal)]
option_performance = [0 for _ in range(Num_subgoal)]

option_learned = [False for _ in range(Num_subgoal)]

print("10000  50000  5000")

Steps = 0
test_rew = 0
rew_num = 0
episodeCount = 0
cumulative_average_reward = 0
test_external_rew = 0
#random_exp = []

data = {}
data['Steps'] = []
data['episode'] = []
data['trainMeta/externalRewards'] = []
data['trainMeta/cumulative_average_rew'] = []

for i in range(Num_subgoal):
    data['trainGoal/success_ratio_'+str(i)] = []
    data['trainGoal/sample_used_'+str(i)] = []

data['test/Steps']=[]
data["test/episodeCount"]=[]
data["test/mean_rew"]=[]
data["test/var_rew"]=[]


#delete previous data
#rootdir = r+args.logdir
def deleteDir():
    filelist = os.listdir(rootdir)
    for f in filelist:
        filepath = os.path.join(rootdir,f)
        if os.path.isfile(filepath):
            os.remove(filepath)
            print(str(filepath)+" is removed!")
        elif os.path.isdir(filepath):
            shutil.rmtree(filepath,True)
            print("dir "+str(filepath)+" is removed!")
    shutil.rmtree(rootdir,True)
    print("successfully deleted!")
    
"""
if os.path.isdir(rootdir):
    deleteDir()
"""
#load model
'''
path_checkpoint = "./norldata/last_checkpoint_ep524.pth"
checkpoint = torch.load(path_checkpoint)
for goal in goal_to_train:
    agent_list[goal].model.load_state_dict(checkpoint['goal'+str(goal)]) #加载模型参数
    agent_list[goal].set_eps(checkpoint['eps'+str(goal)]) 
    agent_list[goal].optim.load_state_dict(checkpoint['goal'+str(goal)+'_optim']) #加载优化器参数

metaAgent.synchronization(checkpoint['meta'])


episodeCount = 524
'''

#start training



def test(testnum,Steps,episodeCount):
    data['test/Steps'].append(Steps)
    data["test/episodeCount"].append(episodeCount)
    rewards = []
    while testnum>0:
        testnum -= 1
        #logx.msg("testnum:{}".format(testnum))
        env.restart()
        externalRewards = 0
        episodeSteps = 0
        while not env.isTerminal() and episodeSteps <= maxStepsPerEpisode:
            #choose a subgoal
            meta_obs1 = env.getmetastate()
            subgoal = metaAgent.choose_action(meta_obs1)
            sub_externalReward = 0
            episodeSteps += 1
            while not env.isTerminal() and env.getU() != subgoal2goal[subgoal] and episodeSteps <= maxStepsPerOption:
                episodeSteps += 1
                obs1 = env.getstate()  
                act = agent_list[subgoal].choose_action(obs1)
                u2,s2,tmp_rew,done = env.execute_action(act)
                sub_externalReward += tmp_rew            
 
            #logx.msg("episodeSteps:{}".format(episodeSteps))
            #an subgoal finished
            externalRewards += sub_externalReward
        rewards.append(externalRewards)

    data["test/mean_rew"].append(np.mean(rewards))
    data["test/var_rew"].append(np.std(rewards,ddof=1))
    
    #print(data["test/mean_rew"])

with tqdm.tqdm(total=2000000,desc = 'Train', **tqdm_config) as t:
    while episodeCount <= t.total:
        episodeCount += 1
        env.restart()
        total_rew = 0
        episodeSteps = 0
        externalRewards = 0
        numstate_trace = []
        numstate = env.getNumState()
        numstate_trace.append(numstate)
        subgoal = Num_subgoal-1
        lastsubgoal = subgoal
        episodeOptions = 0
        while not env.isTerminal() and episodeOptions <= maxOptionsPerEpisode:
            #choose a subgoal
            meta_obs1 = env.getmetastate()
            subgoal = metaAgent.choose_action(meta_obs1)
            sub_externalReward = 0
            #logx.msg("metaAct: {}".format(subgoal))
            #excute planlastnumstate
            episodeSteps = 0
            episodeOptions += 1
            while not env.isTerminal() and env.getU() != subgoal2goal[subgoal] and episodeSteps <= maxStepsPerOption:
                episodeSteps += 1
                obs1 = env.getstate()  
                act = agent_list[subgoal].choose_action(obs1)
                u2,s2,tmp_rew,done = env.execute_action(act)
                sub_externalReward += tmp_rew
                total_rew += tmp_rew
                obs2 = env.getstate()
                numstate = env.getNumState()
                intrinsicRewards = agent_list[subgoal].criticize(u2==subgoal2goal[subgoal] ,done)
                agent_list[subgoal].learn(obs1,act,intrinsicRewards,obs2)

            Steps += episodeSteps
            option_t[subgoal]+=episodeSteps
            metareward = sub_externalReward 
            if env.isTerminal():
                metareward = sub_externalReward 
            
            if env.getU()==subgoal2goal[subgoal]:
                success_tracker[subgoal].append(1)
            else:
                success_tracker[subgoal].append(0)

            meta_obs2 = env.getmetastate()
            metaAgent.learn(meta_obs1,subgoal,metareward,meta_obs2)  
            option_t[-1]+=1    
            #an subgoal finished
            externalRewards += sub_externalReward

        if externalRewards > 0:
            success_tracker[-1].append(1)
        else:
            success_tracker[-1].append(0)

        #an episode finished
        rew_num += 1
        #compute average rewards
        cumulative_average_reward = (cumulative_average_reward * (rew_num -1)+externalRewards)/rew_num


        # save train metrics
        logx.add_scalar('trainMeta/train_rew/episode', externalRewards, episodeCount)
        logx.add_scalar('trainMeta/train_rew/steps', externalRewards, Steps)
        logx.add_scalar('trainMeta/cumulative_average_rew/episode', cumulative_average_reward, episodeCount)
        logx.add_scalar('trainMeta/cumulative_average_rew/steps', cumulative_average_reward, Steps)

        # save train weights
        save_dict = {}


        #save data
        data['Steps'].append(Steps)
        data['episode'].append(episodeCount)
        data['trainMeta/externalRewards'].append(externalRewards)
        data['trainMeta/cumulative_average_rew'].append(cumulative_average_reward)
        
        for i in range(Num_subgoal):
            data['trainGoal/success_ratio_'+str(i)].append(option_performance[i])
            data['trainGoal/sample_used_'+str(i)].append(option_t[i])

        for goal in goal_to_train:
            if goal!=Num_subgoal-1:
                save_dict['goal' + str(goal)] = agent_list[goal].table()
                save_dict['eps' + str(goal)] = agent_list[goal].geteps()
            else:
                save_dict['meta_table'] = metaAgent.table()
                save_dict['meta_eps']=metaAgent.geteps()

        logx.save_model(save_dict, cumulative_average_reward, episodeCount)

        # save best subgoal weights based on success ratio
        for goal in goal_to_train:
            if len(success_tracker[goal]) >= 50:
                option_performance[goal] = sum(success_tracker[goal][-50:]) / 50.

                if option_performance[goal] >= args.stop_threshold:
                    option_learned[goal] = True
                    if goal == Num_subgoal-1:
                        metaAgent.save_weights()
                    else:
                        agent_list[goal].save_weights()
                else:
                    option_learned[goal] = False
            else:
                option_performance[goal] = 0.
            # save best weights
            # save metrics
            logx.add_scalar('trainGoal/' + str(goal) + '/success_ratio/episodeCount', option_performance[goal], episodeCount)
            logx.add_scalar('trainGoal/' + str(goal) + '/success_ratio/Steps', option_performance[goal], Steps)
  

        #test
        if episodeCount%50 == 0:
            test(10,Steps,episodeCount)

        if episodeCount % 50 == 0:
            with open(args.hrllogdir + '/data.pkl', 'wb') as f:
                pickle.dump(data, f)        

        # anneal subgoal eps
        for goal in goal_to_train:
            if goal!=Num_subgoal-1:
                agent_list[goal].anneal_eps(option_t[goal], option_learned[goal])
            else:
                metaAgent.anneal_eps(option_t[-1],option_learned[-1])

        if episodeCount%200==0:
            logx.msg("episode: {}".format(episodeCount))
            logx.msg("Steps:{}".format(Steps))
            t.update(200)
            logx.msg("option_t: {}".format(option_t) )
            logx.msg("success_radio: {}".format(option_performance) )



