import gym
import time
from nes_py.wrappers import JoypadSpace
import random
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT

env = gym.make('SuperMarioBros-1-1-v0')
env = JoypadSpace(env, SIMPLE_MOVEMENT)

"""

Probando metodos para Domain Randomization

"""

def randomize_background(env): # DONE
    """ Modifica la paleta de colores del nivel """

    palette_options = [0x00, 0x01, 0x02, 0x03, 0x04]
    new_palette = random.choice(palette_options)
    env.unwrapped.ram[0x0773] = new_palette


def randomize_speed(env): # DONE
    """modifica el movimiento horizontal de Mario"""

    current_speed = env.unwrapped.ram[0x0057]

    if current_speed > 0x00: # derecha
        bound = min(current_speed + random.randint(0, 15), 0x28)
        new_speed = random.randint(1, bound)

    elif current_speed < 0x00: # izquierda
    
        current_speed_signed = current_speed - 0x100
        bound = max(current_speed_signed - random.randint(0, 15), -40)
        new_speed_signed = random.randint(bound, -1)
        new_speed = new_speed_signed & 0xFF

    else:
        new_speed = current_speed

    env.unwrapped.ram[0x0057] = new_speed


def randomize_enemies_speed(env): # DONE
    """modifica el movimiento de los enemigos en pantalla"""
    
    for i in range(5):
        enemy_speed_address = 0x0058 + i
        current_speed = env.unwrapped.ram[enemy_speed_address]

        if current_speed > 0x00:
            bound = min(current_speed + random.randint(0, 10), 0x28)
            new_speed = random.randint(1, bound)
        
        elif current_speed < 0x00:
    
            current_speed_signed = current_speed - 0x100
            bound = max(current_speed_signed - random.randint(0, 10), -40)
            new_speed_signed = random.randint(bound, -1)
            
            new_speed = new_speed_signed & 0xFF  

        else:
            new_speed = current_speed

        env.unwrapped.ram[enemy_speed_address] = new_speed



for episode in range(1000):
    obs = env.reset()
    for i in range(10000):
        obs, reward, done, info = env.step(env.action_space.sample())
        if(i % 50 == 0): randomize_enemies_speed(env)
        time.sleep(0.00001)
        env.render()
        if done:
            break

env.close()


"""APLICAR RANDOMIZACIONES EN SB3"""

class RandomizationWrapper(gym.Wrapper):
    def __init__(self, env, randomize_fn, every_n_frames=100):
        super().__init__(env)
        self.randomize_fn = randomize_fn
        self.every_n_frames = every_n_frames
        self.frame_count = 0

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        self.frame_count += 1

        if self.frame_count % self.every_n_frames == 0:
            self.randomize_fn(self.env)

        return obs, reward, done, info

# llamar a las funciones hechas
def my_randomization_function(env):
    pass


