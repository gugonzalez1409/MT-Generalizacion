import pickle
import logging
import neat
import gym
from nes_py.wrappers import JoypadSpace
from gym.wrappers import GrayScaleObservation, ResizeObservation
from pureples.shared.visualize import draw_net
from pureples.hyperneat.hyperneat import create_phenotype_network
from pureples.shared.gym_runner import run_hyper
from pureples.shared.substrate import Substrate
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT


"""
Definicion del substrate

"""

input_width, input_height = 8, 8
coord_input = []
for y in range(input_height):
    for x in range(input_width):
        x_coord = (x / input_width) * 2 - 1
        y_coord = (y / input_height) * 2 - 1
        coord_input.append((x_coord, y_coord))

coord_ocultas = []
size_ocultas = [(6,6), (4,4), (4,4)]
z_coords = [-0.5, 0 , 0.5]

for layer_idx, (width, height) in enumerate(size_ocultas):
    layer = []
    z = z_coords[layer_idx]
    for y in range(height):
        for x in range(width):
            x_coord = (x/(width - 1 )) * 2 - 1
            y_coord = (y/(height -1 )) * 2 - 1
            layer.append((x_coord, y_coord))
    coord_ocultas.append(layer)

coord_output = [(i / (len(SIMPLE_MOVEMENT) - 1) * 2 - 1, 1.0) for i in range(len(SIMPLE_MOVEMENT))]

substrate = Substrate(coord_input, coord_output, coord_ocultas)

"""
Crea el entorno de gym en escala de grises y observaciones de 8x8

"""

def make_env():

    env = gym.make('SuperMarioBros-1-1-v0')
    env = JoypadSpace(env, SIMPLE_MOVEMENT)
    env = GrayScaleObservation(env, keep_dim=False)
    env = ResizeObservation(env, shape=8)

    return env



gens = 150 # numero de generaciones
max_steps = 2000
activations = 3 # activaciones de red CPPN

"""
Creacion del config (NEAT-Python)

"""

CONFIG = neat.Config(
    neat.DefaultGenome,
    neat.DefaultReproduction,
    neat.DefaultSpeciesSet,
    neat.DefaultStagnation,
    'hyper-neat-config-feedforward'
)


def run(gens, env):

    winner, stats = run_hyper(
        gens=gens, env=env, config= CONFIG, substrate=substrate)

    return winner, stats

if __name__ == '__main__':

    LOGGER = logging.getLogger()
    LOGGER.setLevel(logging.INFO)

    WINNER, STATS = run(150, make_env)

    cppn = neat.nn.FeedForwardNetwork.create(WINNER, CONFIG)
    NET = create_phenotype_network(cppn, substrate)
    draw_net(cppn, filename="hyperneat_mario_cppn")
    draw_net(NET, filename="hyperneat_mario_winner")
    with open('hyperneat_mario_cppn.pkl', 'wb') as output:
        pickle.dump(cppn, output, pickle.HIGHEST_PROTOCOL)

