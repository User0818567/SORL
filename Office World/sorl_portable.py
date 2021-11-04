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

Num_subgoal = 3
goal_to_train = [i for i in range(Num_subgoal)]

maxStepsPerEpisode = 5000
logx.initialize(logdir=args.logdir, hparams=vars(args), tensorboard=True)

env = OfficeEnv(2)

agent_list = [Q_table(i,108,4,20000) for i in range(Num_subgoal)]

#record the random experience num of a subgoal
option_t = [180000,180000,0]

success_tracker = [[] for _ in range(Num_subgoal)]

sample_t = [0 for _ in range(Num_subgoal)]
option_performance = [0 for _ in range(Num_subgoal)]

option_learned = [False for _ in range(Num_subgoal)]


Steps = 0
test_rew = 0
rew_num = 0
episodeCount = 0
cumulative_average_reward = 0
test_external_rew = 0
random_exp = []

data = {}
data['Steps'] = []
data['episode'] = []
data['trainMeta/externalRewards'] = []
data['trainMeta/cumulative_average_rew'] = []
data['trainGoal/sample_used'] = []
for i in range(Num_subgoal):
    data['trainGoal/success_ratio_'+str(i)] = []

data['test/Steps'] = []
data["test/mean_rew"] = []
data["test/var_rew"] = []
data["test/episodeCount"] = []



#load options
#path_model = ["./Model/sorl/policy_subgoal_office.pth","./Model/sorl/policy_subgoal_mail.pth"]
path_model = ["./1_sorl_task3_1/policy_subgoal_0.pth","./1_sorl_task3_1/policy_subgoal_1.pth"]

for i in range(2):
    with open(path_model[i],'rb') as f:
        data1 = pickle.load(f)
        agent_list[i].load(data1["table"],0)

numStatepair = [[[0, 0, 0, 0], [1, 0, 0, 0]], [[1, 0, 0, 0], [1, 1, 0, 0]], [[1, 1, 0, 0], [0, 0, 1, 1]]]
gainreward = [[1],[1],[1]]
metaAgent = SymbolicModel(numStatepair=numStatepair,gainreward=gainreward)



#start training

def test(testnum,Steps,episodeCount):
    data['test/Steps'].append(Steps)
    data["test/episodeCount"].append(episodeCount)
    rewards = []
    while testnum:
        testnum-=1
        env.restart()
        
        symstate = env.getSymbolicState()
        quality = metaAgent.getQuality()
        metaAgent.generateDomainFile()
        metaAgent.generateProblemFile(symstate,quality)
        metaAgent.generatePlan()
        subgoalSteps = 0 #the index of the plan
        externalRewards = 0
        numstate_trace = []
        numstate = env.getNumState()
        numstate_trace.append(numstate)
        subgoal = Num_subgoal-1
        lastsubgoal = subgoal
        episodeSteps = 0
        while not env.isTerminal() and episodeSteps <= maxStepsPerEpisode:
            #choose a subgoal
            lastsubgoal = subgoal
            subgoal = metaAgent.act(subgoalSteps)
            subgoalSteps += 1
            sub_externalReward = 0
            numstate = env.getNumState()
            lastnumstate = numstate
            #excute planlastnumstate
            if subgoal != -1:
                episodeSteps = 0
                goalnumstate = metaAgent.getgoalnumstate(subgoal)
                numstate = env.getNumState()
                while not env.isTerminal() and numstate != goalnumstate and episodeSteps <= maxStepsPerEpisode:
                    episodeSteps += 1
                    obs1 = env.getstate()
                    
                    act = agent_list[subgoal].choose_action(obs1)
                    u2,s2,tmp_rew,done = env.execute_action(act)
                    sub_externalReward += tmp_rew
                    
                    obs2 = env.getstate()
                    numstate = env.getNumState()

                if numstate == goalnumstate:
                    lastnumstate = goalnumstate
                    episodeSteps = 0
                else:
                    metaAgent.clearPlan()

            else: #use the global option
                numstate = lastnumstate
                subgoal = Num_subgoal-1
                episodeSteps = 0
                while not env.isTerminal() and episodeSteps <= maxStepsPerEpisode:
                    episodeSteps += 1
                    obs1 = env.getstate()
                    act = agent_list[subgoal].choose_test_action(obs1)
                    u2,s2,tmp_rew,done = env.execute_action(act)
                    sub_externalReward += tmp_rew

            #an subgoal finished
            externalRewards += sub_externalReward
        rewards.append(externalRewards)

    data["test/mean_rew"].append(np.mean(rewards))
    data["test/var_rew"].append(np.std(rewards,ddof=1))

with tqdm.tqdm(total=200000,desc = 'Train', **tqdm_config) as t:
    while episodeCount <= t.total:
        episodeCount += 1
        env.restart()
        total_rew = 0
        episodeSteps = 0
        symstate = env.getSymbolicState()
        quality = metaAgent.getQuality()
        metaAgent.generateDomainFile()
        metaAgent.generateProblemFile(symstate,quality)
        metaAgent.generatePlan()

        #metaAgent.plan(predicate_loc,iskey,quality)
        subgoalSteps = 0 #the index of the plan
        externalRewards = 0

        if episodeCount%50==0:
            logx.msg("episode: {}".format(episodeCount))
            logx.msg("Steps: {}".format(Steps))
            logx.msg('plan: {}'.format(metaAgent.getPlan()))

        numstate_trace = []
        numstate = env.getNumState()
        numstate_trace.append(numstate)
        subgoal = Num_subgoal-1
        lastsubgoal = subgoal
        while not env.isTerminal() and episodeSteps <= maxStepsPerEpisode:
            #choose a subgoal
            lastsubgoal = subgoal
            subgoal = metaAgent.act(subgoalSteps)
            subgoalSteps += 1
            sub_externalReward = 0
            numstate = env.getNumState()
            lastnumstate = numstate
            #excute planlastnumstate
            if subgoal != -1:
                goalnumstate = metaAgent.getgoalnumstate(subgoal)
                numstate = env.getNumState()
                episodeSteps = 0
                while not env.isTerminal() and numstate != goalnumstate and episodeSteps <= maxStepsPerEpisode:
                    episodeSteps += 1
                    obs1 = env.getstate()
                    
                    act = agent_list[subgoal].choose_action(obs1)
                    u2,s2,tmp_rew,done = env.execute_action(act)
                    sub_externalReward += tmp_rew
                    total_rew += tmp_rew
                    obs2 = env.getstate()
                    numstate = env.getNumState()
                    intrinsicRewards = agent_list[subgoal].criticize(numstate==goalnumstate,done)
                    agent_list[subgoal].learn(obs1,act,intrinsicRewards,obs2)

                Steps += episodeSteps
                option_t[subgoal]+=episodeSteps
                metaAgent.add(subgoal,sub_externalReward)      

                if numstate == goalnumstate:
                    lastnumstate = goalnumstate
                    numstate_trace.append(goalnumstate)
                    success_tracker[subgoal].append(1)
                    episodeSteps = 0
                else:
                    success_tracker[subgoal].append(0)
                    metaAgent.clearPlan()

            else: #use the global option
                numstate = lastnumstate
                subgoal = Num_subgoal-1
                episodeSteps = 0
                obs1 = 0
                obs2 = 0
                act = 0
                random_exp.clear()
                while numstate==lastnumstate and not env.isTerminal() and episodeSteps <= maxStepsPerEpisode:
                    episodeSteps += 1
                    obs1 = env.getstate()
                    act = agent_list[subgoal].choose_action(obs1)
                    u2,s2,tmp_rew,done = env.execute_action(act)
                    sub_externalReward += tmp_rew
                    total_rew += tmp_rew
                    obs2 = env.getstate()
                    numstate = env.getNumState()
                    random_exp.append([obs1,act,tmp_rew,obs2])
                Steps += episodeSteps

                if numstate != lastnumstate:
                    numstate_trace.append(numstate)
                    subgoal = metaAgent.getsubgoal(lastnumstate,numstate)
                    episodeSteps = 0
                    random_exp[-1][2] = 30
                    for exp in random_exp:
                        agent_list[subgoal].learn(exp[0],exp[1],exp[2],exp[3])
                    random_exp.clear()
                    metaAgent.add(subgoal,sub_externalReward)
                    success_tracker[subgoal].append(1)

            #an subgoal finished
            externalRewards += sub_externalReward


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
        save_dict['meta'] = metaAgent.getall()

        #save data
        data['Steps'].append(Steps)
        data['episode'].append(episodeCount)
        data['trainMeta/externalRewards'].append(externalRewards)
        data['trainMeta/cumulative_average_rew'].append(cumulative_average_reward)
        data['trainGoal/sample_used'].append(option_t)
        for i in range(Num_subgoal):
            data['trainGoal/success_ratio_'+str(i)].append(option_performance[i])

        for goal in goal_to_train:
            save_dict['goal' + str(goal)] = agent_list[goal].table()
            save_dict['eps' + str(goal)] = agent_list[goal].eps

        logx.save_model(save_dict, cumulative_average_reward, episodeCount)

        # save best subgoal weights based on success ratio
        for goal in goal_to_train:
            if len(success_tracker[goal]) >= 50:
                option_performance[goal] = sum(success_tracker[goal][-50:]) / 50.

                if option_performance[goal] >= args.stop_threshold and goal < Num_subgoal-1:
                    option_learned[goal] = True
                    agent_list[goal].save_weights()
                else:
                    option_learned[goal] = False
            else:
                option_performance[goal] = 0.
            # save best weights
            # save metrics
            logx.add_scalar('trainGoal/' + str(goal) + '/success_ratio/episodeCount', option_performance[goal], episodeCount)
            logx.add_scalar('trainGoal/' + str(goal) + '/success_ratio/Steps', option_performance[goal], Steps)

        # train meta agent
        # metaAgent.setTabu(option_deadend)
        metaAgent.setSuccess(option_performance)
  

        if episodeCount % 50 == 0:
            with open(args.logdir + '/data.pkl', 'wb') as f:
                pickle.dump(data, f)        

        # anneal subgoal eps
        for goal in goal_to_train:
            agent_list[goal].anneal_eps(option_t[goal], option_learned[goal])
        
        #test
        if episodeCount%50 == 0:
            test(10,Steps,episodeCount)

        if episodeCount%50==0:
            logx.msg("numstatepair: {}".format(metaAgent.getnumStatepair() ) )
            logx.msg("numstate_trace: {}".format(numstate_trace))
            t.update(50)
            logx.msg("option_learned: {}".format(option_learned))
            logx.msg("option_t: {}".format(option_t) )
            logx.msg("success_radio: {}".format(option_performance) )





save_dict = {}
save_dict['meta'] = metaAgent.getall()                
for goal in goal_to_train:
    save_dict['goal' + str(goal)] = agent_list[goal].model.state_dict()
    save_dict['eps' + str(goal)] = agent_list[goal].eps
    save_dict['goal' + str(goal) + '_optim'] = agent_list[goal].optim.state_dict()


