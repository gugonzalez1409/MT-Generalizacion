import os
import gym
import neat
import pickle
import visualize
import numpy as np
import multiprocessing
import gym_super_mario_bros
from nes_py.wrappers import JoypadSpace
from pureples.shared.substrate import Substrate
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT
from pureples.es_hyperneat.es_hyperneat import ESNetwork
from stable_baselines3.common.atari_wrappers import WarpFrame


TRAINING_LEVEL_LIST = ["1-2", "1-4", "2-1", "2-3","3-2", "3-4", "4-1", "4-3","5-1", "5-4", "6-2", "6-4","7-1", "8-2"]

input_width, input_height = 32, 32

input_coords = [(x, y) for x in np.linspace(-1, 1, input_width) for y in np.linspace(-1, 1, input_height)]

output_coords = [(x, 0.0) for x in np.linspace(-1, 1, len(SIMPLE_MOVEMENT))]

substrate = Substrate(input_coords, output_coords)

def preprocess_observation(obs):

    screen = np.ndarray.flatten(obs)
    screen = (screen / 255.0) * 2.0 - 1.0
    return screen

def params(): 
    
    return {   
        "initial_depth": 1,
        "max_depth": 3,
        
        "variance_threshold": 0.05,
        "band_threshold": 0.3,
        "division_threshold": 0.5,
   
        "iteration_level": 1,
        
        "max_weight": 5.0,
        "activation": "tanh"
    }

class fitness(object):

    def __init__(self, genome, config):

        self.genome = genome
        self.config = config
        
    def eval(self):

        total_fitness = 0
        num_levels = 1

        for _ in range(num_levels):

            level_fitness = self.evaluate_single_level()
            total_fitness += level_fitness
            
        return total_fitness // num_levels

    def evaluate_single_level(self):

        lvl = np.random.choice(TRAINING_LEVEL_LIST)
        env_id = f'SuperMarioBros-{lvl}-v1'
        env = gym.make(env_id)
        env = JoypadSpace(env, SIMPLE_MOVEMENT)
        env = WarpFrame(env, width=32, height=32)

        try:

            cppn = neat.nn.FeedForwardNetwork.create(self.genome, self.config)
            network = ESNetwork(substrate, cppn, params())
            net = network.create_phenotype_network()
        
        except Exception as e:

            env.close()
            return 0

        try:
            obs = env.reset()
            done = False
            patience = 0
            max_x_pos = 0
            steps = 0
            max_steps = 4000

            net.reset()

            while not done and steps < max_steps:

                obs_processed = preprocess_observation(obs)
                
                for _ in range(2):
                    output = net.activate(obs_processed)
                
                action = np.argmax(output)
                obs, reward, done, info = env.step(action)
                steps += 1

                current_x_pos = info.get("x_pos", 0)
                if current_x_pos > max_x_pos:
                    max_x_pos = current_x_pos
                    patience = 0
                else:
                    patience += 1
                
                if patience > 100:
                    break
                    
                if info.get("flag_get", False):
                    fitness_score = max_x_pos + 1000
                    env.close()
                    return fitness_score

            fitness_score = max_x_pos
            env.close()
            return int(fitness_score)
            
        except Exception as e:
            print(f"Error en evaluación: {e}")
            env.close()
            return 0
    
def eval_genomes(genome, config):

    eval_obj = fitness(genome, config)
    return eval_obj.eval()


if __name__ == '__main__':
    
    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                        neat.DefaultSpeciesSet, neat.DefaultStagnation,
                        'hyper-neat-config-feedforward')

    pop = neat.Population(config)
    pop.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    pop.add_reporter(stats)
    if not os.path.exists('checkpoints'):
        os.makedirs('checkpoints')
    pop.add_reporter(neat.Checkpointer(generation_interval=50, filename_prefix='checkpoints/mario-'))

    pe = neat.ParallelEvaluator(multiprocessing.cpu_count()-1, eval_genomes)

    winner = pop.run(pe.evaluate, 200)

    visualize.plot_stats(stats, ylog=False, view=True, 
                        filename="hyperneat-mario-fitness.svg", 
                        smooth_best=True, window_size=10)