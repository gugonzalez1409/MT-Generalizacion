import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import gym
import csv
import numpy as np
import matplotlib.pyplot as plt
from .utils.reward import customReward
from stable_baselines3 import PPO, DQN
from sb3_contrib import QRDQN, RecurrentPPO
from nes_py.wrappers import JoypadSpace
from .utils.envs import EVALUATION_LEVEL_LIST
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT
from stable_baselines3.common.atari_wrappers import AtariWrapper
from stable_baselines3.common.vec_env import VecFrameStack, DummyVecEnv, VecMonitor, VecVideoRecorder

"""
Entorno para evaluacion de modelo en SMB
el agente tiene 10 intentos a cada nivel del juego
donde se mide la recompensa obtenida y cuanto del nivel se logró completar

"""


model_name = {
    'PPO': PPO,
    'DQN' : DQN,
    'RDQN' : QRDQN,
    'RPPO' : RecurrentPPO
}

def evaluate_model(algo_name, model_path, csv_filename, video_prefix):
    model = model_name[algo_name].load(model_path)

    levels = [f"SuperMarioBros-{lvl}-v0" for lvl in EVALUATION_LEVEL_LIST]
    keys = EVALUATION_LEVEL_LIST.copy()

    with open(csv_filename, 'w') as file:
        writer = csv.writer(file)
        writer.writerow(['level', 'avg_reward', 'avg_completion', 'max_completion'])

        results = {}
        for i, level in enumerate(levels):
            
            env = gym.make(level)
            env = JoypadSpace(env, SIMPLE_MOVEMENT)
            env = customReward(env)
            env = AtariWrapper(env=env, noop_max=30, frame_skip=4, screen_size=84, terminal_on_life_loss=False, clip_reward=False)
            env = DummyVecEnv([lambda: env])
            env = VecFrameStack(env, n_stack=4, channels_order='last')
            #env = VecVideoRecorder(env, f'{video_prefix}_videos/{level}', record_video_trigger=lambda x: True, name_prefix=f'{video_prefix}_{level}', video_length=20000)
            env = VecMonitor(env)

            total_rewards = []
            completition_rates = []
            max_completion = 0
            lvl_length = 3000 # aproximado, en caso de no llegar al final

            try:
                for j in range(10):
                    obs = env.reset()
                    total_reward = 0
                    max_x_pos = 0

                    while True:
                        action, _ = model.predict(obs, deterministic=True)
                        obs, reward, done, info = env.step(action)
                        env.render()
                        total_reward += reward[0]
                        max_x_pos = max(max_x_pos, info[0]['x_pos'])

                        if done[0]:
                            if info[0]['flag_get'] == True:
                                lvl_length = info[0]['x_pos']
                                print("x_pos final: ", lvl_length)
                            break

                    total_rewards.append(total_reward)
                    completition_rate = (max_x_pos / lvl_length) * 100
                    completition_rates.append(completition_rate)
                    max_completion = max(max_completion, completition_rate)
            finally:
                env.close()

            avg_reward = sum(total_rewards) / len(total_rewards)
            avg_completition = sum(completition_rates) / len(completition_rates)

            results[keys[i]] = {
                "avg_reward": avg_reward,
                "avg_completition": avg_completition,
                "max_completion": max_completion
            }

            writer.writerow([level, avg_reward, avg_completition, max_completion])

    # graficar resultados
    levels_plot = list(results.keys())
    avg_rewards = [results[l]['avg_reward'] for l in levels_plot]
    avg_completions = [results[l]['avg_completition'] for l in levels_plot]
    max_completions = [results[l]['max_completion'] for l in levels_plot]

    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.bar(levels_plot, avg_rewards, color='blue')
    plt.xticks(rotation=90)
    plt.ylabel('Avg reward')
    plt.title('Recompensa promedio por nivel')

    plt.subplot(1, 2, 2)
    x = np.arange(len(levels_plot))
    width = 0.35
    plt.bar(x - width/2, avg_completions, width, color='green', label='Promedio')
    plt.bar(x + width/2, max_completions, width, color='red', alpha=0.7, label='Maximo')
    plt.xticks(x, levels_plot, rotation=90)
    plt.ylabel('Completion %')
    plt.title('Completado por Nivel')
    plt.legend()

    plt.tight_layout()
    plt.savefig(os.path.join(os.path.dirname(csv_filename), 'eval_results.png'))
    plt.show()

if __name__ == "__main__":

    output_dir = 'statistics/evaluations'
    os.makedirs(output_dir, exist_ok=True)
    algo_name = 'PPO'
    model_path = f'./models/statistics/log_dir/{algo_name}_mario'
    csv_filename = os.path.join(output_dir, f'{algo_name}_evaluation.csv')
    video_prefix = os.path.join(output_dir, algo_name)
    evaluate_model(algo_name, model_path, csv_filename, video_prefix)