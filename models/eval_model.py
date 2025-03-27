import gym
from stable_baselines3 import PPO, DQN
from nes_py.wrappers import JoypadSpace
from icm.reward import customReward
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT
from stable_baselines3.common.atari_wrappers import AtariWrapper

"""
Entorno para evaluacion de modelo en SMB
el agente tiene 10 intentos a cada nivel del juego
donde se mide la recompensa obtenida y cuanto del nivel se logró completar

"""


path = r'./statistics/log_dir/baseDQN/best_model.zip'
model = DQN.load(path)

levels = [f"SuperMarioBros-{w}-{s}-v0" for w in range(1,9) for s in range(1,5)]

results = {}

for level in levels:

    print(f"Evaluando en {level}")

    env = gym.make(level)
    env = JoypadSpace(env, SIMPLE_MOVEMENT)
    env = customReward(env)
    env = AtariWrapper(env= env, noop_max=30, frame_skip=4, screen_size=84, terminal_on_life_loss=False, clip_reward=False)

    total_rewards = []
    completition_rates = []
    max_completion = 0
    lvl_length = 3000 # valor aproximado, en caso de nunca llegar al final


    for i in range(10):

        obs = env.reset()
        total_reward = 0
        max_x_pos = 0

        while True:

            env.render()
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, info = env.step(action)
            total_reward += reward
            max_x_pos = max(max_x_pos, info['x_pos'])

            if done:
                if info['flag_get'] == True:
                    lvl_length = info['x_pos']
                    print("x_pos final: ", lvl_length)
                break

        total_rewards.append(total_reward)

        completition_rate = (max_x_pos / lvl_length) * 100
        completition_rates.append(completition_rate)
        max_completion = max(max_completion, completition_rate)
        print(f"Intento {i+1}: Recompensa total = {total_reward}, Completado = {completition_rate:.2f}%")

    env.close()


    avg_reward = sum(total_rewards) / len(total_rewards)
    avg_completition = sum(completition_rates) / len(completition_rates)

    results[level] = {
        "avg_reward": avg_reward,
        "avg_completition": avg_completition,
        "max_completion": max_completion
    }


    print(f"Nivel {level}: Recompensa promedio: {avg_reward}\n")
    print(f"Nivel {level}: Completacion promedio: {avg_completition}\n")
    print(f"Nivel {level}: Máxima completacion: {max_completion}\n")

print("\nResumen de pruebas:")
for level, stats in results.items():
    print(f"{level}: Recompensa Promedio = {stats['avg_reward']}, Completación Promedio = {stats['avg_completion']:.2f}%, Máxima Completación = {stats['max_completion']:.2f}%")
