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

'''
seed = 1337
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
'''

data = {}
data['Steps'] = []
data['episode'] = []

data['trainMeta/externalRewards'] = []
data['trainMeta/cumulative_average_rew'] = []

data['trainGoal/sample_used'] = []
data['trainGoal/success_ratio'] = []
data['trainGoal/loss'] = []



actionMap = [0, 1, 2, 3, 4, 5, 11, 12]

Num_subgoal = args.num_subgoal
goal_to_train = [i for i in range(Num_subgoal)]

maxStepsPerEpisode = 500
logx.initialize(logdir=args.logdir, hparams=vars(args), tensorboard=True)

env = ALEEnvironment(args.game,args)

agent_eps = [1.0 for _ in range(Num_subgoal)]
agent_exploration_steps = [args.explorationSteps for _ in range(Num_subgoal)]

save_learned = [False for _ in range(Num_subgoal+1)]
agent_buffer_size = [args.buffer_size for _ in range(Num_subgoal)]

agent_list, metaAgent = create_pair_model_hrl(  agent_eps=agent_eps,
                                            meta_eps=1.0,
                                            agent_buffer_size=agent_buffer_size,
                                            agent_exploration_steps=agent_exploration_steps,
                                            meta_buffer_size=args.meta_buffer_size,
                                            meta_exploration_steps=args.meta_explorationSteps,
                                            device='cuda:'+str(args.gpu) if args.gpu >=0 and torch.cuda.is_available() else 'cpu')



#record the random experience num of a subgoal
option_t = [0 for _ in range(Num_subgoal)]

success_tracker = [[] for _ in range(Num_subgoal)]
option_deadend = [0 for _ in range(Num_subgoal)]
sample_meta = 0
sample_t = [0 for _ in range(Num_subgoal)]
option_performance = [0 for _ in range(Num_subgoal)]

option_learned = [False for _ in range(Num_subgoal)]

option_start_train_episode = [None for _ in range(Num_subgoal)]
meta_start_train_episode = None



Steps = 0
test_rew = 0
rew_num = 0
episodeCount = 0
cumulative_average_reward = 0
test_external_rew = 0
random_exp = []


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

with tqdm.tqdm(total=args.episode_limit,desc = 'Train', **tqdm_config) as t:
    while episodeCount <= t.total:
        try:
            episodeCount += 1
            env.restart()
            total_rew = 0
            episodeSteps = 0
            predicate_loc, iskey = env.getSymbolicState()
            quality = metaAgent.getQuality()
            metaAgent.generateDomainFile()
            metaAgent.generateProblemFile(predicate_loc,iskey,quality)
            metaAgent.generatePlan()
            logx.msg("episode: {}".format(episodeCount))
            logx.msg('plan: {}'.format(metaAgent.getPlan()))
            #metaAgent.plan(predicate_loc,iskey,quality)
            subgoalSteps = 0 #the index of the plan
            externalRewards = 0

            numstate_trace = []
            numstate = env.getNumState()
            numstate_trace.append(numstate)
            subgoal = -1
            lastsubgoal = subgoal
            option = Num_subgoal-1
            #clear buffer which has learned well
            #for i in range(Num_subgoal):
            #    agent_list[i].operate_buffer(option_learned[i])

            while not env.isTerminal() and episodeSteps <= maxStepsPerEpisode:
                #choose a subgoal
                lastsubgoal = subgoal
                subgoal = metaAgent.act(subgoalSteps)
                subgoalSteps += 1
                sub_externalReward = 0
                lastnumstate = env.getNumState()
                option = metaAgent.getaction2option(subgoal)
                #excute plan
                if subgoal != -1:
                    agent_list[option].anneal_eps(option_t[subgoal], option_learned[subgoal])

                    goalnumstate = metaAgent.getgoalnumstate(subgoal)
                    numstate = env.getNumState()
                    while not env.isTerminal() and numstate != goalnumstate and episodeSteps <= maxStepsPerEpisode:
                        episodeSteps += 1
                        obs = env.getStackedState()
                        batch_data = Batch(obs=obs.reshape((1, 4, 84, 84)))

                        with torch.no_grad():
                            result = agent_list[option](batch_data)
                        act = result.act[0]
                        tmp_rew = env.act(actionMap[act])
                        sub_externalReward += tmp_rew
                        total_rew += tmp_rew
                        obs_next = env.getStackedState()
                        numstate = env.getNumState()
                        done = env.isTerminal()
                        intrinsicRewards = agent_list[option].criticize(numstate==goalnumstate,done,0, True)
                        #intrinsicRewards += tmp_rew
                        agent_list[option].add(obs=obs,act=act,rew=intrinsicRewards,done=done,obs_next=obs_next)

                    Steps += episodeSteps
                    option_t[subgoal]+=episodeSteps
                    metaAgent.add(subgoal,sub_externalReward)      

                    if numstate == goalnumstate:
                        option_deadend[lastsubgoal]=-1
                        numstate_trace.append(goalnumstate)
                        success_tracker[subgoal].append(1)
                        episodeSteps = 0
                        lastnumstate = goalnumstate
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
                    option = Num_subgoal-1
                    episodeSteps = 0
                    while numstate==lastnumstate and not env.isTerminal() and episodeSteps <= maxStepsPerEpisode:
                        episodeSteps += 1
                        obs = env.getStackedState()
                        batch_data = Batch(obs=obs.reshape((1, 4, 84, 84)))
                        with torch.no_grad():
                            result = agent_list[option](batch_data)
                        act = result.act[0]
                        tmp_rew = env.act(actionMap[act])
                        total_rew += tmp_rew
                        sub_externalReward += tmp_rew

                        obs_next = env.getStackedState()
                        numstate = env.getNumState()
                        if numstate[0]==-1:
                            numstate = lastnumstate
                        done = env.isTerminal()
                        
                        # if numstate != lastnumstate:
                        #     intrinsicRewards = 20
                        # else:
                        #     intrinsicRewards = 0
                        
                        #calculate intrinsic rewards
                        # if numstate == lastnumstate:
                        #     intrinsicRewards = -0.2
                        # elif numstate in numstate_trace:
                        #     intrinsicRewards = -5
                        # else:
                        #     intrinsicRewards = 20
                        # if env.isLifeLost():
                        #     intrinsicRewards-=100
                        #intrinsicRewards += tmp_rew
                        
                        
                        #intrinsicRewards = agent_list[subgoal].criticize(numstate != lastnumstate, done, 0, True)
                        #agent_list[subgoal].add(obs=obs, act=act, rew=intrinsicRewards, done=done, obs_next=obs_next)

                    #option_t[subgoal] += episodeSteps
                    if env.isTerminal():
                         if option_deadend[lastsubgoal]!=-1:
                             option_deadend[lastsubgoal]+=1
                    if numstate != lastnumstate:
                        '''
                        if numstate==[6,1]:
                            if numstate not in numstate_trace:
                                sub_externalReward += 30
                            else:
                                sub_externalReward -= 2
                        '''
                        numstate_trace.append(numstate)
                        option_deadend[lastsubgoal] = -1
                        subgoal,option = metaAgent.getsubgoal(lastnumstate,numstate)
                    
                        #option_t[subgoal]+=episodeSteps
                        Steps+=episodeSteps
                        episodeSteps = 0
                        if numstate == [0,0]:
                            for _ in range(30):
                                tmp_rew += env.act(3)
                                total_rew += tmp_rew
                                sub_externalReward += tmp_rew
                                if env.isTerminal():
                                    break
                        
                        metaAgent.add(subgoal,sub_externalReward)
                        #success_tracker[subgoal].append(1)

                #an subgoal finished
                externalRewards += sub_externalReward


            #an episode finished
            rew_num += 1
            #compute average rewards
            cumulative_average_reward = (cumulative_average_reward * (rew_num -1)+externalRewards)/rew_num
            data['Steps'].append(Steps)
            data['episode'].append(episodeCount)
            data['trainMeta/externalRewards'].append(externalRewards)
            data['trainMeta/cumulative_average_rew'].append(cumulative_average_reward)
            # save train metrics
            logx.add_scalar('trainMeta/train_rew/episode', externalRewards, episodeCount)
            logx.add_scalar('trainMeta/train_rew/steps', externalRewards, Steps)
            logx.add_scalar('trainMeta/cumulative_average_rew/episode', cumulative_average_reward, episodeCount)
            logx.add_scalar('trainMeta/cumulative_average_rew/steps', cumulative_average_reward, Steps)

            # save train weights
            save_dict = {}
            save_dict['meta'] = metaAgent.getall()


            options,action2options = metaAgent.getoptions()
            logx.msg("options: {}".format(options) )
            logx.msg("action2options: {}".format(action2options) )
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
                option = metaAgent.getaction2option(goal)
                if len(success_tracker[goal]) >= 100:
                    option_performance[goal] = sum(success_tracker[goal][-100:]) / 100.
                    
                    #if len(success_tracker[goal]) > 500 and option_performance[goal]<0.02:
                    #    option_deadend[goal] = args.tabu_threshold
                    if option_performance[goal] >= args.stop_threshold:
                        option_learned[goal] = True
                        agent_list[option].save_weights(option_performance[goal])
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
                
                option = metaAgent.getaction2option(goal)
                agent_list[option].train()
                if option_t[goal] >= agent_list[option].random_play_steps and not option_learned[goal] and len(
                        agent_list[option].buffer) > agent_list[option].batch_size:
                    for _ in range(args.controller_train_times):
                        sample_t[goal] += agent_list[option].batch_size
                        loss = agent_list[option].update()
                        option_loss[goal].append(loss)
                    if option_start_train_episode[goal] == None:
                        option_start_train_episode[goal] = episodeCount

                    logx.add_scalar('trainGoal/' + str(goal) + '/loss/episodeCount', np.mean(option_loss[goal]), episodeCount)
                    logx.add_scalar('trainGoal/' + str(goal) + '/loss/Steps', np.mean(option_loss[goal]), Steps)

            # train meta agent
            #metaAgent.setTabu(option_deadend)
            metaAgent.setSuccess(option_performance)
            

            # anneal subgoal eps
            # for goal in goal_to_train:
            #     agent_list[goal].anneal_eps(option_t[goal], option_learned[goal])
            

            # save metrics
            data['trainGoal/sample_used'].append(option_t)
            data['trainGoal/success_ratio'].append(option_performance)
            data['trainGoal/loss'].append(option_loss)
            

            if episodeCount % 50 == 0:
                with open(args.logdir + '/data.pkl', 'wb') as f:
                    pickle.dump(data, f)

            if episodeCount%100==0:
                logx.msg("option_learned: {}".format(option_learned))
                logx.msg("option_t: {}".format(option_t) )
                logx.msg("success_radio: {}".format(option_performance) )
                t.update(100)
        except OSError as exception:
            episodeCount -= 1
            print(str(exception))
            time.sleep(600)

save_dict = {}
save_dict['meta'] = metaAgent.getall()                
for goal in goal_to_train:
    save_dict['goal' + str(goal)] = agent_list[goal].model.state_dict()
    save_dict['eps' + str(goal)] = agent_list[goal].eps
    save_dict['goal' + str(goal) + '_optim'] = agent_list[goal].optim.state_dict()

torch.save(save_dict, args.logdir + '/sorl.pth')