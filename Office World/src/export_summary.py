from tester.tester import Tester
from tester.test_utils import get_precentiles
import numpy as np
import os, argparse


def export_results_water_world():
    # NOTE: We do not report performance on Map 0 because we used it as our validation map
    maps = ["water_%d"%i for i in range(1,11)]

    world = "water"    
    tmp_folder = "../tmp/"
    algs = ["dqn", "hrl", "hrl-rm", "qrm", "qrm-rs"]

    results = {}
    for alg in algs:
        results[alg] = []
    
    for map_name in maps:
        # computing best known performance
        optimal = {}
        for alg in algs:
            result_file = os.path.join(tmp_folder, world, map_name, alg + ".json")
            tester = Tester(None, None, None, None, result_file)

            f_optimal = tester.get_best_performance_per_task()
            for t in f_optimal:
                if t not in optimal:
                    optimal[t] = f_optimal[t]
                else:
                    optimal[t] = max([optimal[t], f_optimal[t]])
        
        # adding results for this map to the result summary
        for alg in algs:
            result_file = os.path.join(tmp_folder, world, map_name, alg + ".json")
            tester = Tester(None, None, None, None, result_file)
            tester.world.optimal = optimal
            alg_results = tester.get_result_summary()["all"]

            for i in range(len(alg_results)):
                step, reward = alg_results[i]
                if len(results[alg]) == i:
                    results[alg].append((step, []))
                results[alg][i][1].append(reward)

    # Compute final stats and export summary file
    for alg in algs:
        folder_out = os.path.join("../tmp/results", world)
        if not os.path.exists(folder_out): os.makedirs(folder_out)
        f_out = open(os.path.join(folder_out, alg + ".txt"), "w")
        for i in range(len(results[alg])):
            step, reward = results[alg][i]
            p25, p50, p75 = get_precentiles(np.concatenate(reward))
            f_out.write(str(step) + "\t" + str(p25) + "\t" + str(p50) + "\t" + str(p75) + "\n")
        f_out.close()


def export_results_tabular_world(world, maps):
    tmp_folder = "../tmp/"
    algs = ["dqn", "hrl", "hrl-rm", "qrm", "qrm-rs"]

    results = {}
    for alg in algs:
        results[alg] = []
        
    for map_name in maps:
        # adding results for this map to the result summary
        for alg in algs:
            result_file = os.path.join(tmp_folder, world, map_name, alg + ".json")            
            tester = Tester(None, None, None, None, result_file)

            alg_results = tester.get_result_summary()["all"]

            for i in range(len(alg_results)):
                step, reward = alg_results[i]
                if len(results[alg]) == i:
                    results[alg].append((step, []))
                results[alg][i][1].append(reward)

    # Compute final stats and export summary file
    for alg in algs:
        folder_out = os.path.join("../tmp/results", world)
        if not os.path.exists(folder_out): os.makedirs(folder_out)
        f_out = open(os.path.join(folder_out, alg + ".txt"), "w")
        for i in range(len(results[alg])):
            step, reward = results[alg][i]
            p25, p50, p75 = get_precentiles(np.concatenate(reward))
            f_out.write(str(step) + "\t" + str(p25) + "\t" + str(p50) + "\t" + str(p75) + "\n")
        f_out.close()

def export_results_office_world():
    export_results_tabular_world("office", ["office"])

def export_results_craft_world():
    # NOTE: We do not report performance on Map 0 because we used it as our validation map
    export_results_tabular_world("craft", ["craft_%d"%i for i in range(1,11)])

if __name__ == "__main__":


    # EXAMPLE: python3 export_summary.py --world="craft"

    # Getting params
    worlds     = ["office", "craft", "water"]

    parser = argparse.ArgumentParser(prog="export_summary", description='After running the experiments, this algorithm computes a summary of the results.')
    parser.add_argument('--world', default='office', type=str, 
                        help='This parameter indicated which world to solve. The options are: ' + str(worlds))

    args = parser.parse_args()
    if args.world not in worlds: raise NotImplementedError("World " + str(args.world) + " hasn't been defined yet")

    # Computing the experiment summary
    world = args.world
    if world == "office":
        export_results_office_world()
    if world == "craft":
        export_results_craft_world()
    if world == "water":
        export_results_water_world()
