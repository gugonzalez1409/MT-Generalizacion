import gym
import random

"""

Probando metodos para Domain Randomization

"""

# lista de enemigos a no randomizar, posible crash del juego
not_use_list = [0x0D, 0x011, 0x07, 0x08, 0x09, 0x0A, 0x0B, 0x3C, 0x3B, 0x3A, 0x38, 0x37, 0x36, 0x35, 0x34, 0x32, 0x31, 0x30, 0x2F, 0x2E, 
                0x2D, 0x2B, 0x2A, 0x28, 0x29, 0x24, 0x25, 0x26, 0x27, 0x1F, 0x1B, 0X1C, 0X1D, 0X1E, 0x18, 0x17, 0x16, 0x14, 0x13, 0x2C ]




easy_list = [0x00, 0x06, 0x01, 0x02, 0x03, 0x04] # goomba, koopas verde, koopa rojo
normal_list = [0x33, 0x0E, 0x15, 0x05, 0x12] # tortuga con espinas, koopas con alas, tortuga con martillo

def change_enemy(enemy_id):

    if enemy_id in easy_list:
        return random.choice(easy_list + normal_list[:2])
    
    elif enemy_id in normal_list:
        return random.choice(normal_list + easy_list[:2])
    
    else:
        return random.choice(normal_list + easy_list)
    
  
def randomize_enemies(env):

    ram = env.unwrapped.env.ram # acceder a la ram
    
    for i in range(5):

        active_enemy = ram[0x000F + i]
        enemy_id = ram[0x0015 + i]

        if enemy_id in not_use_list or active_enemy != 1 or enemy_id not in easy_list + normal_list:
            continue

        if random.random() < 0.5:

            ram[0x0015 + i] = change_enemy(enemy_id)


def randomize_enemies_speed(env):

    ram = env.unwrapped.env.ram
    
    for i in range(5):

        active_enemy = ram[0x00F + i]
        enemy_id = ram[0x0015 + i]
        enemy_speed_address = 0x0058 + i

        if enemy_id in not_use_list or active_enemy != 1 or enemy_id not in easy_list + normal_list:

            continue

        if random.random() < 0.5:

            current_speed = ram[enemy_speed_address]
            random_boost = random.randint(-1, 2)
            new_speed = (current_speed + random_boost) & 0xFF
            ram[enemy_speed_address] = new_speed


class DomainRandom(gym.Wrapper):
    def __init__(self, env, enemy_random_frames = 500, render=False):

        super().__init__(env)
        self.current_step = 0
        self.enemy_random_frames = enemy_random_frames
        self.render_enabled = render # solo para dummyvecenv

    def reset(self):

        obs = self.env.reset()
        self.current_step = 0
        return obs

    def step(self, action):

        if self.render_enabled:
            self.env.render()
        
        obs, reward, done, info = self.env.step(action)
        self.current_step += 1

        # hacer la randomizacion de velocidad más veces que la de enemigos
        if self.current_step > 200 and not done and self.current_step % self.enemy_random_frames == 0:
            randomize_enemies_speed(self.env)

        if self.current_step > 200 and not done and self.current_step % (self.enemy_random_frames * 2) == 0:
            randomize_enemies(self.env)
        
        return obs, reward, done, info

