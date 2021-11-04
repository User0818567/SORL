import random, time, argparse, os.path
from qrm.qrm import run_qrm_experiments
from baselines.run_dqn import run_dqn_experiments
from baselines.run_hrl import run_hrl_experiments
from tester.tester import Tester
from tester.saver import Saver
from tester.tester_params import TestingParameters
from common.curriculum import CurriculumLearner
from qrm.learning_params import LearningParameters
# The pickle library is asking me to have access to Ball and BallAgent from the main...
from worlds.water_world import Ball, BallAgent

def get_params_craft_world(experiment, use_rs):
    step_unit = 1000

    # configuration of testing params
    testing_params = TestingParameters()
    testing_params.test = True
    testing_params.test_freq = 10*step_unit 
    testing_params.num_steps = 1000

    # configuration of learning params
    learning_params = LearningParameters()
    learning_params.gamma = 0.9
    learning_params.print_freq = step_unit
    learning_params.train_freq = 1
    learning_params.tabular_case = True
    learning_params.max_timesteps_per_task = testing_params.num_steps

    # This are the parameters that tabular q-learning would use to work as 'tabular q-learning'
    learning_params.lr = 1
    learning_params.batch_size = 1
    learning_params.learning_starts = 1
    learning_params.buffer_size = 1

    # Setting the experiment
    tester = Tester(learning_params, testing_params, experiment, use_rs)

    # Setting the curriculum learner
    curriculum = CurriculumLearner(tester.get_task_rms())
    curriculum.num_steps = testing_params.num_steps 
    curriculum.total_steps = 1000*step_unit
    curriculum.min_steps = 1

    print("Craft World ----------")
    print("TRAIN gamma:", learning_params.gamma)
    print("Total steps:", curriculum.total_steps)
    print("tabular_case:", learning_params.tabular_case)
    print("num_steps:", testing_params.num_steps)
    print("total_steps:", curriculum.total_steps)


    return testing_params, learning_params, tester, curriculum

def get_params_office_world(experiment, use_rs):
    step_unit = 500

    # configuration of testing params
    testing_params = TestingParameters()
    testing_params.test = True
    testing_params.test_freq = 1*step_unit 
    testing_params.num_steps = 1000

    # configuration of learning params
    learning_params = LearningParameters()
    learning_params.gamma = 0.9
    learning_params.tabular_case = True
    learning_params.max_timesteps_per_task = testing_params.num_steps

    # This are the parameters that tabular q-learning would use to work as 'tabular q-learning'
    learning_params.lr = 1
    learning_params.batch_size = 1
    learning_params.learning_starts = 1
    learning_params.buffer_size = 1

    # Setting the experiment
    tester = Tester(learning_params, testing_params, experiment, use_rs)

    # Setting the curriculum learner
    curriculum = CurriculumLearner(tester.get_task_rms())
    curriculum.num_steps = testing_params.num_steps #100
    curriculum.total_steps = 100*step_unit
    curriculum.min_steps = 1

    print("Office World ----------")
    print("TRAIN gamma:", learning_params.gamma)
    print("tabular_case:", learning_params.tabular_case)
    print("num_steps:", testing_params.num_steps)
    print("total_steps:", curriculum.total_steps)


    return testing_params, learning_params, tester, curriculum

def get_params_water_world(experiment, use_rs):
    step_unit = 1000

    # configuration of testing params
    testing_params = TestingParameters()
    testing_params.test = True
    testing_params.test_freq = 10*step_unit 
    testing_params.num_steps = 600 # I'm giving one minute to the agent to solve the task

    # configuration of learning params
    learning_params = LearningParameters()
    learning_params.lr = 1e-5 # 5e-5 seems to be better than 1e-4
    learning_params.gamma = 0.9
    learning_params.max_timesteps_per_task = testing_params.num_steps
    learning_params.buffer_size = 50000
    learning_params.print_freq = step_unit
    learning_params.train_freq = 1
    learning_params.batch_size = 32
    learning_params.target_network_update_freq = 100 # obs: 500 makes learning more stable, but slower
    learning_params.learning_starts = 1000
    
    # Tabular case
    learning_params.tabular_case    = False # it is not possible to use tabular RL in the water world
    learning_params.use_random_maps = False
    learning_params.use_double_dqn  = True
    learning_params.prioritized_replay = True
    learning_params.num_hidden_layers = 6
    learning_params.num_neurons = 64

    # Setting the experiment
    tester = Tester(learning_params, testing_params, experiment, use_rs)

    # Setting the curriculum learner
    curriculum = CurriculumLearner(tester.get_task_rms())
    curriculum.num_steps = 300
    curriculum.total_steps = 2000*step_unit
    curriculum.min_steps = 1


    print("Water World ----------")
    print("lr:", learning_params.lr)
    print("batch_size:", learning_params.batch_size)
    print("num_hidden_layers:", learning_params.num_hidden_layers)
    print("target_network_update_freq:", learning_params.target_network_update_freq)
    print("TRAIN gamma:", learning_params.gamma)
    print("Total steps:", curriculum.total_steps)
    print("tabular_case:", learning_params.tabular_case)
    print("use_double_dqn:", learning_params.use_double_dqn)
    print("prioritized_replay:", learning_params.prioritized_replay)
    print("use_random_maps:", learning_params.use_random_maps)

    return testing_params, learning_params, tester, curriculum


def run_experiment(world, alg_name, experiment, num_times, use_rs, show_print):
    if world == 'officeworld':
        testing_params, learning_params, tester, curriculum = get_params_office_world(experiment, use_rs)
    if world == 'craftworld':
        testing_params, learning_params, tester, curriculum = get_params_craft_world(experiment, use_rs)
    if world == 'waterworld':
        testing_params, learning_params, tester, curriculum = get_params_water_world(experiment, use_rs)
        
    # Baseline 1 (standard DQN with Michael Littman's approach)
    if alg_name == "dqn":
        run_dqn_experiments(alg_name, tester, curriculum, num_times, show_print)

    # Baseline 2 (Hierarchical RL)
    if alg_name == "hrl":
        run_hrl_experiments(alg_name, tester, curriculum, num_times, show_print, use_rm = False)

    # Baseline 3 (Hierarchical RL with DFA constraints)
    if alg_name == "hrl-rm":
        run_hrl_experiments(alg_name, tester, curriculum, num_times, show_print, use_rm = True)

    # QRM
    if alg_name in ["qrm","qrm-rs"]:
        run_qrm_experiments(alg_name, tester, curriculum, num_times, show_print)


if __name__ == "__main__":

    # EXAMPLE: python3 run.py --algorithm="qrm" --world="craft" --map=0 --num_times=1

    # Getting params
    algorithms = ["dqn", "hrl", "hrl-rm", "qrm", "qrm-rs"]
    worlds     = ["office", "craft", "water"]

    parser = argparse.ArgumentParser(prog="run_experiments", description='Runs a multi-task RL experiment over a particular environment.')
    parser.add_argument('--algorithm', default='qrm', type=str, 
                        help='This parameter indicated which RL algorithm to use. The options are: ' + str(algorithms))
    parser.add_argument('--world', default='office', type=str, 
                        help='This parameter indicated which world to solve. The options are: ' + str(worlds))
    parser.add_argument('--map', default=0, type=int, 
                        help='This parameter indicated which map to use. It must be a number between 0 and 10.')
    parser.add_argument('--num_times', default=1, type=int, 
                        help='This parameter indicated which map to use. It must be a number greater or equal to 1')
    parser.add_argument("--verbosity", help="increase output verbosity")

    args = parser.parse_args()
    if args.algorithm not in algorithms: raise NotImplementedError("Algorithm " + str(args.algorithm) + " hasn't been implemented yet")
    if args.world not in worlds: raise NotImplementedError("World " + str(args.world) + " hasn't been defined yet")
    if not(0 <= args.map <= 10): raise NotImplementedError("The map must be a number between 0 and 10")
    if args.num_times < 1: raise NotImplementedError("num_times must be greater than 0")

    # Running the experiment
    alg_name   = args.algorithm
    world      = args.world
    map_id     = args.map
    num_times  = args.num_times
    show_print = args.verbosity is not None
    use_rs     = alg_name.endswith("-rs")

    if world == "office": experiment = "../experiments/office/tests/office.txt"
    else: experiment = "../experiments/%s/tests/%s_%d.txt"%(world, world, map_id)
    world += "world"

    print("world: " + world, "alg_name: " + alg_name, "experiment: " + experiment, "num_times: " + str(num_times), show_print)
    run_experiment(world, alg_name, experiment, num_times, use_rs, show_print)
