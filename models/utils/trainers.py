from typing import Callable
from sb3_contrib import RecurrentPPO
from stable_baselines3 import PPO, DQN
from ..rainbow.rainbow import Rainbow
from ..rainbow.policies import RainbowPolicy
from ..generalization.ImpalaCNN import ImpalaCNN
from stable_baselines3.common.callbacks import EvalCallback
from models.utils.envs import make_single_env, vectorizedEnv, eval_env



tensorboard_log = r'./models/statistics/tensorboard_log/'
log_dir = r'./models/statistics/log_dir/'


"""

Funciones de entrenamiento de agentes

"""

# obtenido de docs de sb3
def linear_schedule(initial_value: float) -> Callable[[float], float]:

    def func(progress_remaining: float) -> float:

        return progress_remaining * initial_value

    return func


def trainPPO(explore, random, custom, vectorized, impala, icm):

    if impala:
        policy_kwargs = dict(
            features_extractor_class=ImpalaCNN,
            features_extractor_kwargs=dict(
                features_dim=512,
                depths=[16, 32, 32],
                scale=2
            )
        )

    else:
        policy_kwargs = {}

    model = PPO(
        'CnnPolicy',
        learning_rate=linear_schedule(5e-4),
        env = vectorizedEnv(explore, random, custom, icm) if vectorized else make_single_env(explore, random, custom),
        policy_kwargs=policy_kwargs,
        ent_coef=0.03,
        gamma=0.99,
        verbose=1,
        tensorboard_log = tensorboard_log
        )
    
    callback = EvalCallback(eval_env=eval_env(custom), n_eval_episodes=10, eval_freq=100000,log_path=log_dir, best_model_save_path=log_dir)
    model.learn(total_timesteps=75e6, callback=callback)

    model_name = 'PPO'

    if explore:
        model_name += "_explore"
    if random:
        model_name += "_random"
    if impala:
        model_name += "_impala"

    model.save(f"{model_name}_mario")


def trainDQN(explore, random, custom, vectorized, impala, icm):

    if impala:
        policy_kwargs = dict(
            features_extractor_class=ImpalaCNN,
            features_extractor_kwargs=dict(
                features_dim=512,
                depths=[16, 32, 32],
                scale=2
            )
        )

    else:
        policy_kwargs = {}
    
    model = DQN(
        "CnnPolicy",
        learning_rate = linear_schedule(1e-5),
        env = vectorizedEnv(explore, random, custom, icm) if vectorized else make_single_env(explore, random, custom),
        policy_kwargs= policy_kwargs,
        buffer_size=100000,
        batch_size=64,
        learning_starts=50000,
        exploration_final_eps= 0.05,
        exploration_fraction=0.2,
        verbose=1,
        tensorboard_log = tensorboard_log
        )
    
    callback = EvalCallback(eval_env=eval_env(custom), n_eval_episodes=10, eval_freq=100000,log_path=log_dir, best_model_save_path=log_dir)
    model.learn(total_timesteps=75e6, callback=callback)

    model_name = "DQN"

    if explore:
        model_name += "_explore"
    if random:
        model_name += "_random"
    if impala:
        model_name += "_impala"

    model.save(f"{model_name}_mario")


def trainRecurrentPPO(explore, random, custom, vectorized, impala, icm):

    recurrent = True

    if impala:
        policy_kwargs = dict(
            features_extractor_class=ImpalaCNN,
            features_extractor_kwargs=dict(
                features_dim=512,
                depths=[16, 32, 32],
                scale=2
            )
        )

    else:
        policy_kwargs = {}

    model = RecurrentPPO(
        'CnnLstmPolicy',
        learning_rate = linear_schedule(1e-5),
        env = vectorizedEnv(explore, random, custom, icm, recurrent) if vectorized else make_single_env(explore, random, custom, recurrent),
        policy_kwargs= policy_kwargs,
        ent_coef=0.03,
        gamma=0.95,
        verbose=1,
        tensorboard_log = tensorboard_log
    )

    callback = EvalCallback(eval_env = eval_env(custom), n_eval_episodes=10, eval_freq=100000, log_path=log_dir, best_model_save_path=log_dir)
    model.learn(total_timesteps=75e6, callback=callback)

    model_name = "RPPO"

    if explore:
        model_name += "_explore"
    if random:
        model_name += "_random"
    if impala:
        model_name += "_impala"

    model.save(f"{model_name}_mario")


def trainRainbow(explore, random, custom, vectorized, impala, icm):

    if impala:
        policy_kwargs = dict(
            n_quantiles=200,
            net_arch=[256, 256],
            dueling=True,
            noisy=True,
            noisy_kwargs={
                'sigma': 0.5
            },
            features_extractor_class=ImpalaCNN,
            features_extractor_kwargs=dict(
                features_dim=512,
                depths=[16, 32, 32],
                scale=2
            )
        )

    else:
        policy_kwargs = dict(
            n_quantiles=200,
            net_arch=[256, 256],
            dueling=True,
            noisy=True,
            noisy_kwargs={
                'sigma': 0.5
            }
            )


    model = Rainbow(
        RainbowPolicy,
        env = vectorizedEnv(explore, random, custom, icm) if vectorized else make_single_env(explore, random, custom),
        learning_rate=linear_schedule(2.5e-4),
        learning_starts=50000,
        policy_kwargs=policy_kwargs,
        buffer_size=100000,
        exploration_initial_eps=1.0,
        exploration_final_eps=0.01,
        exploration_fraction=0.2,
        batch_size=64,
        gamma=0.99,
        train_freq=4,
        gradient_steps=1,
        target_update_interval=10000,
        verbose=1,
        tensorboard_log = tensorboard_log
    )

    callback = EvalCallback(eval_env = eval_env(custom), n_eval_episodes=10, eval_freq=100000, log_path=log_dir, best_model_save_path=log_dir)
    model.learn(total_timesteps=75e6, callback=callback)

    model_name = "RDQN"
    if explore:
        model_name += "_explore"
    if random:
        model_name += "_random"
    if impala:
        model_name += "_impala"

    model.save(f"{model_name}_mario")

    