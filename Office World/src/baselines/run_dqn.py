import random, time
import tensorflow as tf
from worlds.game import *
from tester.saver import Saver
from common.schedules import LinearSchedule
from baselines.policy_bank import PolicyBank

def run_dqn_baseline(sess, rm_file, policy_bank, tester, curriculum, show_print):
    """
    This code runs one training episode. 
        - rm_file: It is the path towards the RM machine to solve on this episode
    """
    # Initializing parameters and the game
    learning_params = tester.learning_params
    testing_params = tester.testing_params
    rm_id = tester.get_reward_machine_id_from_file(rm_file)
    task_params = tester.get_task_params(rm_file)
    task = Game(task_params)
    actions = task.get_actions()
    num_features = policy_bank.get_number_features(rm_id)
    num_steps = learning_params.max_timesteps_per_task
    rm = tester.get_reward_machines()[rm_id]
    training_reward = 0
    # Getting the initial state of the environment and the reward machine
    s1, s1_features = task.get_state_and_features()
    u1 = rm.get_initial_state()

    # Starting interaction with the environment
    if show_print: print("Executing", num_steps)
    for t in range(num_steps):

        # Choosing an action to perform
        if random.random() < 0.1:
            a = random.choice(actions)
        else: 
            a = policy_bank.get_best_action(rm_id, s1_features, u1)

        # updating the curriculum
        curriculum.add_step()
        policy_bank.add_step(rm_id)
        
        # Executing the action
        task.execute_action(a)
        s2, s2_features = task.get_state_and_features()
        events = task.get_true_propositions()
        u2 = rm.get_next_state(u1, events)
        reward = rm.get_reward(u1,u2,s1,a,s2,is_training=True)
        done = task.is_env_game_over() or rm.is_terminal_state(u2)
        training_reward += reward

        # Saving this transition
        policy_bank.add_experience(rm_id, s1_features, u1, a, reward, s2_features, u2, float(done))

        # Learning
        if policy_bank.get_step(rm_id) > learning_params.learning_starts and policy_bank.get_step(rm_id) % learning_params.train_freq == 0:
            policy_bank.learn(rm_id)

        # Updating the target network
        if policy_bank.get_step(rm_id) > learning_params.learning_starts and policy_bank.get_step(rm_id) % learning_params.target_network_update_freq == 0:
            policy_bank.update_target_network(rm_id)

        # Testing
        if testing_params.test and curriculum.get_current_step() % testing_params.test_freq == 0:
            tester.run_test(curriculum.get_current_step(), sess, run_dqn_test, policy_bank)

        # Restarting the environment (Game Over)
        if done:
            # Restarting the game
            task = Game(task_params)
            s2, s2_features = task.get_state_and_features()
            u2 = rm.get_initial_state()

            if curriculum.stop_task(t):
                break
        
        # checking the steps time-out
        if curriculum.stop_learning():
            break

        # Moving to the next state
        s1, s1_features, u1 = s2, s2_features, u2

    if show_print: print("Done! Total reward:", training_reward)


def run_dqn_test(sess, reward_machines, task_params, rm_id, learning_params, testing_params, policy_bank):
    # Initializing parameters
    task = Game(task_params)
    rm = reward_machines[rm_id]
    s1, s1_features = task.get_state_and_features()
    u1 = rm.get_initial_state()

    # Starting interaction with the environment
    r_total = 0
    for t in range(testing_params.num_steps):
        # Choosing an action to perform
        a = policy_bank.get_best_action(rm_id, s1_features, u1)

        # Executing the action
        task.execute_action(a)
        s2, s2_features = task.get_state_and_features()
        u2 = rm.get_next_state(u1, task.get_true_propositions())
        r = rm.get_reward(u1,u2,s1,a,s2,is_training=False)

        r_total += r * learning_params.gamma**t
        
        # Restarting the environment (Game Over)
        if task.is_env_game_over() or rm.is_terminal_state(u2):
            break
        
        # Moving to the next state
        s1, s1_features, u1 = s2, s2_features, u2

    return r_total


def run_dqn_experiments(alg_name, tester, curriculum, num_times, show_print):
    # Setting up the saver
    saver = Saver(alg_name, tester, curriculum)
    learning_params = tester.learning_params

    # Running the tasks 'num_times'
    time_init = time.time()
    for t in range(num_times):
        # Setting the random seed to 't'
        random.seed(t)
        sess = tf.Session()

        # Reseting default values
        curriculum.restart()

        # Creating policy bank
        task_aux = Game(tester.get_task_params(curriculum.get_current_task()))
        num_features = len(task_aux.get_features())
        num_actions  = len(task_aux.get_actions())

        policy_bank = PolicyBank(sess, num_actions, num_features, learning_params, curriculum, tester.get_reward_machines())

        # Task loop
        while not curriculum.stop_learning():
            if show_print: print("Current step:", curriculum.get_current_step(), "from", curriculum.total_steps)
            rm_file = curriculum.get_next_task()
            run_dqn_baseline(sess, rm_file, policy_bank, tester, curriculum, show_print)
        tf.reset_default_graph()
        sess.close()
        
        # Backing up the results
        saver.save_results()

    # Showing results
    tester.show_results()
    print("Time:", "%0.2f"%((time.time() - time_init)/60), "mins")

