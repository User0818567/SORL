import argparse
import time

def str2bool(v):
    return v.lower() in ("yes", "true", "t", "1")

parser = argparse.ArgumentParser()


#params for office
parser.add_argument("--task",type=int, default = 1)

#params for tabu
parser.add_argument("--tabu_threshold",type=str, default= 100)

#params for planner
parser.add_argument("--domain_file",type=str, default= "Metric/Test/testDomain.pddl")
parser.add_argument("--problem_file",type=str, default="Metric/Test/testProblem.pddl")
parser.add_argument("--initial_problem_file",type=str, default="Metric/Test/initial_prob.pddl")
parser.add_argument("--initial_domain_file",type=str, default="Metric/Test/initial_domain.pddl")
parser.add_argument("--template_problem_file",type=str, default="Metric/Test/templateProblem.pddl")
parser.add_argument("--template_domain_file",type=str, default="Metric/Test/templateDomain.pddl")

parser.add_argument("--num_subgoal",type=int,default=3)

# params for training
parser.add_argument("--episode_limit", type=int, default=100000)
parser.add_argument("--steps_limit", type=int, default=10000000)
parser.add_argument("--gpu", type=int, default=0)
parser.add_argument("--model", type=str, default='nsrl', help='hdqn | mlp | nsrl')
parser.add_argument("--num_process", type=int, default=8)
parser.add_argument("--meta_train_times", type=int, default=5)
parser.add_argument("--controller_train_times", type=int, default=20)


# params for the game 'Montezuma'
parser.add_argument("--game", default="montezuma_revenge.bin")
parser.add_argument("--display_screen", type=str2bool, default=False)
parser.add_argument("--frame_skip", default=4)
parser.add_argument("--color_averaging", default=True)
parser.add_argument("--random_seed", default=0)
parser.add_argument("--minimal_action_set", default=False)
parser.add_argument("--screen_width", default=84)
parser.add_argument("--screen_height", default=84)
parser.add_argument("--load_weight", default=False)
parser.add_argument("--use_sparse_reward", type=str2bool, default=True)
parser.add_argument("--stop_threshold", type=float, default=0.95)

# params for dqn
parser.add_argument("--lr", type=float, default=1e-4)
parser.add_argument("--n_step", type=int, default=1)
parser.add_argument("--gamma", type=float, default=0.99)

parser.add_argument("--rho", type=float, default=0.95)
parser.add_argument("--eps", type=float, default=1e-8)

# params for eps
parser.add_argument("--initial_eps", type=float, default=1.0)
parser.add_argument("--final_eps", type=float, default=0.02)

parser.add_argument("--meta_initial_eps", type=float, default=1.0)
parser.add_argument("--meta_final_eps", type=float, default=0.02)

# params for prioritized replay buffer
parser.add_argument("--max_timesteps", type=int, default=1000000)
parser.add_argument("--alpha", type=float, default=0.6)
parser.add_argument("--beta", type=float, default=0.4)
parser.add_argument("--replay_eps", type=float, default=1e-6)
parser.add_argument("--replay_beta_iters", type=int, default=int(100000 * 0.5))
parser.add_argument("--replay_beta", type=float, default=0.4)

# params for meta controller
parser.add_argument("--meta_batch", type=int, default=64)
parser.add_argument("--meta_buffer_size", type=int, default=20000)
parser.add_argument("--meta_explorationSteps", type=int, default=200000)
parser.add_argument("--meta_target_update_freq", type=int, default=500)
parser.add_argument("--meta_random_steps", type=int, default=2000)
parser.add_argument("--meta_train_freq", type=int, default=1)
parser.add_argument("--meta_type", type=int, default=0)

# params for controller
parser.add_argument("--batch", type=int, default=64)
parser.add_argument("--buffer_size", type=int, default=20000)
parser.add_argument("--explorationSteps", type=int, default=64)
parser.add_argument("--target_update_freq", type=int, default=100)
parser.add_argument("--random_steps", type=int, default=64)
parser.add_argument("--train_freq", type=int, default=1)

# params for symbolic logic
parser.add_argument("--predicate_num", type=int, default=5)
parser.add_argument("--arity_num", type=int, default=7)
parser.add_argument("--path_length", type=int, default=4)
parser.add_argument("--embedding_size", type=int, default=128)

# params for transformer
parser.add_argument("--n_layers", type=int, default=2)
parser.add_argument("--n_head", type=int, default=4)

# params for log
parser.add_argument("--logdir", type=str, default='./1_hrl_task1_1')
parser.add_argument("--hrllogdir", type=str, default='./1_hrl_task1_1')

parser.add_argument("--test_model", type=str, default='mlp')

args = parser.parse_args()
