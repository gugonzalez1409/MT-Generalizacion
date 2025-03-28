import csv
from stable_baselines3.common.vec_env import VecEnvWrapper


""""

Wrapper para obtener un .csv con cantidad de steps por 
cada nivel de entrenamiento en SuperMarioBrosRandomStages-v0

"""

trys_path = "./models/statistics/steps_per_lvl.csv"

with open(trys_path, 'w') as file:
    writer = csv.writer(file)
    writer.writerow(['level', 'steps'])


trys_per_lvl = {}


class LevelMonitor(VecEnvWrapper):
    def __init__(self, venv):
        super().__init__(venv)


    def step_async(self, actions):

        self.venv.step_async(actions)

    def step_wait(self):
        obs, rewards, dones, infos = self.venv.step_wait()

        for info in infos:

            world = info['world']
            stage = info['stage']

            if world is not None and stage is not None:
                level = f"{world}-{stage}"
                if level not in trys_per_lvl:
                    trys_per_lvl[level] = 0
                trys_per_lvl[level] += 1

        with open(trys_path, 'w', newline="") as file:
            writer = csv.writer(file)
            writer.writerow(['level', 'steps'])
            for level, steps in trys_per_lvl.items():
                writer.writerow([level, steps])

        return obs, rewards, dones, infos
    
    def reset(self):
        return self.venv.reset()