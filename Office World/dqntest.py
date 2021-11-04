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


maxStepsPerEpisode = 100
logx.initialize(logdir=args.dqnlogdir, hparams=vars(args), tensorboard=True)

env = OfficeEnv(1)

agent_eps = 1.0 
agent_exploration_steps = args.explorationSteps

agent_buffer_size = args.buffer_size

device='cuda:'+str(args.gpu) if args.gpu >=0 and torch.cuda.is_available() else 'cpu'
model = Model().to(device)
opt = optim.RMSprop(model.parameters(), lr=args.lr, alpha=0.95, eps=1e-08)
agent = DDQNPolicy(model=model,optim=opt,goal = "0",train_freq=args.train_freq,discount_factor=args.gamma,
                    estimation_step=10,target_update_freq=args.target_update_freq if agent_buffer_size[indx] else -1,
                    reward_normalization=False,batch_size=args.batch,buffer_size=agent_buffer_size,
                    explorationSteps=agent_exploration_steps,random_play_steps=args.random_steps,device=device,
                    initial_eps=agent_eps,final_eps=0.02)


#record the random experience num of a subgoal
option_t = 0

success_tracker = []
option_performance = 0

option_learned = False

option_start_train_episode = None

Steps = 0
test_rew = 0
rew_num = 0
episodeCount = 0
cumulative_average_reward = 0

#start training

with tqdm.tqdm(total=args.episode_limit,desc = 'Train', **tqdm_config) as t:
    while episodeCount <= t.total:
        episodeCount += 1
        env.restart()
        total_rew = 0
        episodeSteps = 0
        logx.msg("episode: {}".format(episodeCount))
        #metaAgent.plan(predicate_loc,iskey,quality)
        subgoalSteps = 0 #the index of the plan
        externalRewards = 0

        numstate_trace = []
        numstate = env.getNumState()
        numstate_trace.append(numstate)
        
        while not env.isTerminal() and episodeSteps <= maxStepsPerEpisode:
            #choose a subgoal
            lastsubgoal = subgoal
            subgoal = metaAgent.act(subgoalSteps)
            subgoalSteps += 1
            sub_externalReward = 0
            lastnumstate = env.getNumState()
            obs = env.get_features()
            #excute plan
            if subgoal != -1:
                goalnumstate = metaAgent.getgoalnumstate(subgoal)
                numstate = env.getNumState()
                while not env.isTerminal() and numstate != goalnumstate and episodeSteps <= maxStepsPerEpisode:
                    episodeSteps += 1
                    obs = env.get_features()
                    batch_data = Batch(obs=obs.reshape(1,109))
                    with torch.no_grad():
                        result = agent_list[subgoal](batch_data)
                    act = result.act[0]
                    u2,s2,tmp_rew,done = env.execute_action(act)
                    sub_externalReward += tmp_rew
                    total_rew += tmp_rew
                    obs_next = env.get_features()
                    numstate = env.getNumState()
                    intrinsicRewards = agent_list[subgoal].criticize(numstate==goalnumstate,done,0, True)
                    intrinsicRewards += tmp_rew
                    agent_list[subgoal].add(obs=obs,act=act,rew=intrinsicRewards,done=done,obs_next=obs_next)

                Steps += episodeSteps
                option_t[subgoal]+=episodeSteps
                metaAgent.add(subgoal,sub_externalReward)      

                if numstate == goalnumstate:
                    option_deadend[lastsubgoal]=-1
                    numstate_trace.append(goalnumstate)
                    success_tracker[subgoal].append(1)
                    episodeSteps = 0
                    if numstate == [0,0]:
                        for _ in range(5):
                            tmp_rew += env.act(3)
                            total_rew += tmp_rew
                            sub_externalReward += tmp_rew    
         
                else:
                    success_tracker[subgoal].append(0)
                    metaAgent.clearPlan()

            else: #use the global option
                numstate = lastnumstate
                subgoal = Num_subgoal-1
                episodeSteps = 0
                while numstate==lastnumstate and not env.isTerminal() and episodeSteps <= maxStepsPerEpisode:
                    episodeSteps += 1
                    obs = env.get_features()
                    
                    batch_data = Batch(obs=obs.reshape(1,109))
                    
                    with torch.no_grad():
                        #pdb.set_trace()
                        result = agent_list[subgoal](batch_data)
                    act = result.act[0]
                    u2,s2,tmp_rew,done = env.execute_action(act)
                    total_rew += tmp_rew
                    sub_externalReward += tmp_rew

                    obs_next = env.get_features()
                    numstate = env.getNumState()
                    #calculate intrinsic rewards
                    if numstate == lastnumstate:
                        intrinsicRewards = -0.2
                    elif numstate in numstate_trace:
                        intrinsicRewards = -2
                    else:
                        intrinsicRewards = 10
                    if env.isTerminal():
                        intrinsicRewards-=50
                    intrinsicRewards += tmp_rew
                    
                    
                    #intrinsicRewards = agent_list[subgoal].criticize(numstate != lastnumstate, done, 0, True)
                    agent_list[subgoal].add(obs=obs, act=act, rew=intrinsicRewards, done=done, obs_next=obs_next)
                    random_exp.append([obs,act,intrinsicRewards,done,obs_next])
                

                option_t[subgoal] += episodeSteps
                if env.isTerminal():
                    if option_deadend[lastsubgoal]!=-1:
                        option_deadend[lastsubgoal]+=1
                elif numstate != lastnumstate:
                    '''
                    if numstate==[6,1]:
                        if numstate not in numstate_trace:
                            sub_externalReward += 30
                        else:
                            sub_externalReward -= 2
                    '''
                    numstate_trace.append(numstate)
                    option_deadend[lastsubgoal] = -1
                    subgoal = metaAgent.getsubgoal(lastnumstate,numstate)
                    
                    for experience in random_exp:
                        agent_list[subgoal].add(obs=experience[0], act=experience[1],rew=experience[2],done=experience[3],obs_next=experience[4])
                    random_exp.clear()
                    option_t[subgoal]+=episodeSteps
                    Steps+=episodeSteps
                    episodeSteps = 0
                    
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

        logx.msg("numstatepair: {}".format(metaAgent.getnumStatepair() ) )
        logx.msg("numstate_trace: {}".format(numstate_trace))
        logx.msg("deadend: {}".format(option_deadend ) )

        for goal in goal_to_train:
            save_dict['goal' + str(goal)] = agent_list[goal].model.state_dict()
            save_dict['eps' + str(goal)] = agent_list[goal].eps
            save_dict['goal' + str(goal) + '_optim'] = agent_list[goal].optim.state_dict()

        logx.save_model(save_dict, cumulative_average_reward, episodeCount)

        # save best subgoal weights based on success ratio
        for goal in goal_to_train:
            if len(success_tracker[goal]) >= 10:
                option_performance[goal] = sum(success_tracker[goal][-10:]) / 10.

                if option_performance[goal] >= args.stop_threshold and goal < Num_subgoal-1:
                    option_learned[goal] = True
                    agent_list[goal].save_weights(option_performance[goal])
                else:
                    option_learned[goal] = False
            else:
                option_performance[goal] = 0.
            # save best weights
            # save metrics
            logx.add_scalar('trainGoal/' + str(goal) + '/success_ratio/episodeCount', option_performance[goal], episodeCount)
            logx.add_scalar('trainGoal/' + str(goal) + '/success_ratio/Steps', option_performance[goal], Steps)

        option_loss = [[] for _ in range(Num_subgoal)]

        # train subgoal network
        for goal in goal_to_train:
            agent_list[goal].train()
            if option_t[goal] >= agent_list[goal].random_play_steps and not option_learned[goal] and len(
                    agent_list[goal].buffer) > agent_list[goal].batch_size:
                for _ in range(args.controller_train_times):
                    sample_t[goal] += agent_list[goal].batch_size
                    loss = agent_list[goal].update()
                    option_loss[goal].append(loss)
                if option_start_train_episode[goal] == None:
                    option_start_train_episode[goal] = episodeCount

                logx.add_scalar('trainGoal/' + str(goal) + '/loss/episodeCount', np.mean(option_loss[goal]), episodeCount)
                logx.add_scalar('trainGoal/' + str(goal) + '/loss/Steps', np.mean(option_loss[goal]), Steps)

        # train meta agent
        #metaAgent.setTabu(option_deadend)
        metaAgent.setSuccess(option_performance)
        

        # anneal subgoal eps
        for goal in goal_to_train:
            agent_list[goal].anneal_eps(option_t[goal], option_learned[goal])
        
        if episodeCount%50==0:

            logx.msg("option_learned: {}".format(option_learned))
            logx.msg("option_t: {}".format(option_t) )
            logx.msg("success_radio: {}".format(option_performance) )
            t.update(50)





save_dict = {}
save_dict['meta'] = metaAgent.getall()                
for goal in goal_to_train:
    save_dict['goal' + str(goal)] = agent_list[goal].model.state_dict()
    save_dict['eps' + str(goal)] = agent_list[goal].eps
    save_dict['goal' + str(goal) + '_optim'] = agent_list[goal].optim.state_dict()


