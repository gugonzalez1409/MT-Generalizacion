import gym
import neat
import pickle
import logging
import numpy as np
import multiprocessing as mp
import gym_super_mario_bros
import matplotlib.pyplot as plt
from reward import customReward
from gym_utils import SMBRamWrapper
from nes_py.wrappers import JoypadSpace
from pureples.shared.visualize import draw_net
from pureples.shared.substrate import Substrate
from pureples.hyperneat.hyperneat import create_phenotype_network
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT

# Definicion del substrate
input_width, input_height = 16, 13

input_coords = [(x,y) for x in np.linspace(-1,1, input_width) for y in np.linspace(-1, 1, input_height)]

hidden_coords = [[(x,y) for x in np.linspace(1,-1, 5) for y in np.linspace(1,-1, 5)]]

output_coords = [(x, 0.0) for x in np.linspace(-1, 1, len(SIMPLE_MOVEMENT))]


substrate = Substrate(input_coords, output_coords, hidden_coords)

def make_env():

    env = gym.make('SuperMarioBros-1-1-v0')
    env = JoypadSpace(env, SIMPLE_MOVEMENT)
    env = customReward(env)
    x0 = 0
    x1 = 16
    y0 = 0
    y1 = 13
    n_stack = 1
    n_skip = 1
    env = SMBRamWrapper(env, [x0, x1, y0, y1], n_stack=n_stack, n_skip=n_skip)
    return env


def eval_genome_fitness(genome, config):

    local_env = make_env()
    
    cppn = neat.nn.FeedForwardNetwork.create(genome, config)
    net = create_phenotype_network(cppn, substrate)
    
    fitnesses = []
    for _ in range(config.trials):
        ob = local_env.reset()
        net.reset()
        total_reward = 0
        stag_counter = 0
        last_x_pos = 0

        for _ in range(config.max_steps):
            for _ in range(config.activations):
                o = net.activate(ob.flatten())

            action = np.argmax(o)

            ob, reward, done, info = local_env.step(action)
            total_reward += reward

            current_x_pos = info['x_pos']

            if current_x_pos <= last_x_pos:
                stag_counter += 1
            else:
                stag_counter = 0
                last_x_pos = current_x_pos
            
            if done or stag_counter > 100:
                break
        fitnesses.append(total_reward)
    
    local_env.close()

    return np.array(fitnesses).mean()


def ini_pop(state, stats, config, output):
    """
    Inicializa la poblacion y le añade el reporte de estadisticas.
    """
    pop = neat.population.Population(config, state)
    if output:
        pop.add_reporter(neat.reporting.StdOutReporter(True))
    pop.add_reporter(stats)
    return pop


def run_hyper_parallel(gens, max_steps, config, activations, max_trials=100, output=True):

    config.max_steps = max_steps
    config.activations = activations
    config.activation = 'sigmoid'
    config.trials = 1
    

    pe = neat.ParallelEvaluator(mp.cpu_count(), eval_genome_fitness)
    
   
    stats_one = neat.statistics.StatisticsReporter()
    pop = ini_pop(None, stats_one, config, output)
    winner_one = pop.run(pe.evaluate, gens)
    
    stats_ten = neat.statistics.StatisticsReporter()
    pop = ini_pop((pop.population, pop.species, 0), stats_ten, config, output)
    config.trials = 10
    winner_ten = pop.run(pe.evaluate, gens)
    
    if max_trials == 0:
        return winner_ten, (stats_one, stats_ten)
    
    stats_hundred = neat.statistics.StatisticsReporter()
    pop = ini_pop((pop.population, pop.species, 0), stats_hundred, config, output)
    config.trials = max_trials
    winner_hundred = pop.run(pe.evaluate, gens)
    
    return winner_hundred, (stats_one, stats_ten, stats_hundred)

gens = 150
max_steps = 3000
activations = len(hidden_coords) + 2

CONFIG = neat.Config(
    neat.DefaultGenome,
    neat.DefaultReproduction,
    neat.DefaultSpeciesSet,
    neat.DefaultStagnation,
    'hyper-neat-config-feedforward'
)

def run(gens):
    winner, stats = run_hyper_parallel(
        gens=gens, max_steps=max_steps, config=CONFIG, activations=activations)
    return winner, stats

if __name__ == '__main__':
    LOGGER = logging.getLogger()
    LOGGER.setLevel(logging.INFO)

    WINNER, STATS = run(150)
    
    cppn = neat.nn.FeedForwardNetwork.create(WINNER, CONFIG)
    NET = create_phenotype_network(cppn, substrate)
    draw_net(cppn, filename="hyperneat_mario_cppn")
    draw_net(NET, filename="hyperneat_mario_winner")
    with open('hyperneat_mario_cppn.pkl', 'wb') as output:
        pickle.dump(cppn, output, pickle.HIGHEST_PROTOCOL)