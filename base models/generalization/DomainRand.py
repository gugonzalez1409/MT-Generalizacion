import gym
from nes_py.wrappers import JoypadSpace
import time
import random
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT

env = gym.make('SuperMarioBros-1-2-v0')
env = JoypadSpace(env, SIMPLE_MOVEMENT)


"""

Probando metodos para Domain Randomization hechas 
especificamente para Super Mario Bros

"""

def randomize_background(env): # DONE
    """ Modifica la paleta de colores del nivel """

    palette_options = [0x00, 0x01, 0x02, 0x03, 0x04]
    new_palette = random.choice(palette_options)
    env.unwrapped.ram[0x0773] = new_palette

def randomize_speed(env): # DONE
    """modifica el movimiento horizontal del jugador"""

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


def randomize_enemies_speed(env):
    """Modifica la velocidad de los enemigos y cambia su dirección al toparse con una pared."""
    
    for i in range(5): 
        enemy_speed_address = 0x0058 + i
        enemy_state = env.unwrapped.ram[0x03B0 + i] 
        current_speed = env.unwrapped.ram[enemy_speed_address]

        if enemy_state == 0x02:

            new_speed = -current_speed & 0xFF
        else:
            new_speed = current_speed
        env.unwrapped.ram[enemy_speed_address] = new_speed + random.randint(0, 7)


class DomainRandom(gym.Wrapper):
    def __init__(self, env, enemy_random_frames):
        super().__init__(env)
        self.current_step = 0
        self.enemy_random_frames = enemy_random_frames

    def reset(self):
        obs = self.env.reset()
        randomize_background(self.env)
        self.current_step = 0
        return obs

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        self.current_step += 1

        if(self.current_step % 30 == 0): randomize_speed(self.env)
        if(self.current_step % self.enemy_random_frames == 0): randomize_enemies_speed(self.env)
        
        return obs, reward, done, info  

