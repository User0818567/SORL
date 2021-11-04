from worlds.game import GameParams
from worlds.water_world import WaterWorldParams

class TesterWaterWorld:
    def __init__(self, experiment, use_random_maps = False, data = None):
        if data is None:
            # Reading the file
            self.experiment = experiment
            f = open(experiment)
            lines = [l.rstrip() for l in f]
            f.close()
            # setting the test attributes
            self.max_x     = eval(lines[1])
            self.max_y     = eval(lines[2])
            self.b_num_per_color = eval(lines[3])
            self.b_radius  = eval(lines[4])
            self.use_velocities = eval(lines[5])
            self.ball_disappear = eval(lines[6])
            self.maps      = eval(lines[7])
            self.reward_machine_files = eval(lines[8])
            print("-------------")
            print("max_x", self.max_x)
            print("max_y", self.max_y)
            print("b_num_per_color", self.b_num_per_color)
            print("b_radius", self.b_radius)
            print("use_velocities", self.use_velocities)
            print("ball_disappear", self.ball_disappear)
            print("-------------")            
        else:
            self.experiment = data["experiment"]
            self.maps      = data["maps"]
            self.reward_machine_files = data["reward_machine_files"]
        self.use_random_maps = use_random_maps
        self.tasks = [(t,m) for t in self.reward_machine_files for m in self.maps]
        self.current_map = 0
        # NOTE: Update the optimal value per task when you know it...
        self.optimal = {}
        for i in range(len(self.tasks)):
            self.optimal[self.tasks[i]] = 1

    def get_dictionary(self):
        d = {}
        d["experiment"] = self.experiment
        d["maps"] = self.maps
        d["reward_machine_files"] = self.reward_machine_files
        return d

    def get_reward_machine_files(self):
        return self.reward_machine_files

    def get_task_specifications(self):
        return self.tasks

    def get_task_params(self, task_specification):
        if type(task_specification) == tuple: _, map_file = task_specification
        if type(task_specification) == str: 
            if self.use_random_maps:
                map_file = None 
            else:
                # I'm returning one map from the testing set
                map_file = self.maps[self.current_map]
                self.current_map = (self.current_map+1)%len(self.maps)

        params = WaterWorldParams(map_file, max_x = self.max_x, max_y = self.max_y, b_radius = self.b_radius, 
                                  b_num_per_color = self.b_num_per_color, 
                                  use_velocities = self.use_velocities, 
                                  ball_disappear = self.ball_disappear)
        return GameParams("waterworld", params)

    def get_task_rm_file(self, task_specification):
        if type(task_specification) == tuple: 
            rm_task, _ = task_specification
        if type(task_specification) == str: 
            rm_task = task_specification
        return rm_task

