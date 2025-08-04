from typing import Callable
from sb3_contrib import RecurrentPPO
from ..rainbow.rainbow import Rainbow
from stable_baselines3 import PPO, DQN
from ..rainbow.policies import RainbowPolicy
from ..generalization.ImpalaCNN import ImpalaCNN
from models.utils.envs import make_single_env, vectorizedEnv
from stable_baselines3.common.callbacks import CheckpointCallback

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

    print("Trainning PPO")
    if random: print("Using Domain Randomization")
    if custom: print("Using Custom Reward")
    if impala: print("Using Impala CNN")
    if explore: print("Using ExploreGo")

    if impala:
        policy_kwargs = dict(
            features_extractor_class=ImpalaCNN,
            features_extractor_kwargs=dict(
                features_dim=512,
                depths=[16, 32, 32],
                scale=1
            )
        )

    else:
        policy_kwargs = {}

    model = PPO(
        'CnnPolicy',
        learning_rate=linear_schedule(1.75e-4 if impala else 1e-4),
        env = vectorizedEnv(explore, random, custom, icm) if vectorized else make_single_env(explore, random, custom),
        policy_kwargs=policy_kwargs,
        n_steps=512,
        batch_size=256,
        clip_range=0.2,
        ent_coef=0.01,
        gamma=0.99,
        gae_lambda=0.95,
        verbose=1,
        n_epochs=3,
        max_grad_norm=0.5,
        tensorboard_log = tensorboard_log
        )

    save_freq = max(10e6 // 11, 1)

    save_callback = CheckpointCallback(save_freq=save_freq, save_path=log_dir, name_prefix='PPO_checkpoint', verbose=1) 
    model.learn(total_timesteps=75e6, callback=save_callback)


    model_name = 'PPO'

    if explore:
        model_name += "_explore"
    if random:
        model_name += "_random"
    if impala:
        model_name += "_impala"

    model.save(f"{model_name}_mario")


def trainDQN(explore, random, custom, vectorized, impala, icm):

    print("Trainning DQN")
    if random: print("Using Domain Randomization")
    if custom: print("Using Custom Reward")
    if impala: print("Using Impala CNN")
    if explore: print("Using ExploreGo")

    if impala:
        policy_kwargs = dict(
            features_extractor_class=ImpalaCNN,
            features_extractor_kwargs=dict(
                features_dim=512,
                depths=[16, 32, 32],
                scale=1
            )
        )

    else:
        policy_kwargs = {}
    
    model = DQN(
        "CnnPolicy",
        learning_rate = linear_schedule(5e-5 if impala else 1e-4),
        env = vectorizedEnv(explore, random, custom, icm) if vectorized else make_single_env(explore, random, custom),
        policy_kwargs= policy_kwargs,
        buffer_size=100000,
        batch_size=64,
        learning_starts=10000,
        exploration_final_eps= 0.05,
        exploration_fraction=0.2,
        verbose=1,
        train_freq=4,
        tensorboard_log = tensorboard_log
        )
    
    save_freq = max(10e6 // 11, 1)

    save_callback = CheckpointCallback(save_freq=save_freq, save_path=log_dir, name_prefix='DQN_checkpoint')
    model.learn(total_timesteps=75e6, callback=save_callback)

    model_name = "DQN"

    if explore:
        model_name += "_explore"
    if random:
        model_name += "_random"
    if impala:
        model_name += "_impala"

    model.save(f"{model_name}_mario")


def trainRecurrentPPO(explore, random, custom, vectorized, impala, icm):

    print("Trainning Recurrent PPO")
    if random: print("Using Domain Randomization")
    if custom: print("Using Custom Reward")
    if impala: print("Using Impala CNN")
    if explore: print("Using ExploreGo")

    

    recurrent = True

    if impala:
        policy_kwargs = dict(
            enable_critic_lstm=False,
            features_extractor_class=ImpalaCNN,
            features_extractor_kwargs=dict(
                features_dim=512,
                depths=[16, 32, 32],
                scale=1
            )
        )

    else:
        policy_kwargs = dict(
            enable_critic_lstm=False,
        )

    model = RecurrentPPO(
        'CnnLstmPolicy',
        learning_rate=linear_schedule(1.75e-4 if impala else 1e-4),
        env = vectorizedEnv(explore, random, custom, icm, recurrent) if vectorized else make_single_env(explore, random, custom),
        policy_kwargs=policy_kwargs,
        n_steps=2048,
        batch_size=512,
        clip_range=0.2,
        ent_coef=0.01,
        gamma=0.99,
        gae_lambda=0.95,
        verbose=1,
        n_epochs=4,
        max_grad_norm=0.5,
        tensorboard_log = tensorboard_log
    )

    print("critic lstm:", model.policy.enable_critic_lstm)

    save_freq = max(10e6 // 11, 1)

    save_callback = CheckpointCallback(save_freq=save_freq, save_path=log_dir, name_prefix='RPPO_checkpoint')
    model.learn(total_timesteps=75e6, callback=save_callback)


    model_name = "RPPO"

    if explore:
        model_name += "_explore"
    if random:
        model_name += "_random"
    if impala:
        model_name += "_impala"

    model.save(f"{model_name}_mario")


def trainRainbow(explore, random, custom, vectorized, impala, icm):

    print("Trainning Rainbow DQN")
    if random: print("Using Domain Randomization")
    if custom: print("Using Custom Reward")
    if impala: print("Using Impala CNN")
    if explore: print("Using ExploreGo")

    if impala:
        policy_kwargs = dict(
            n_quantiles=200,
            net_arch=[256, 256],
            dueling=True,
            noisy=True,
            noisy_kwargs={
                'sigma': 0.2
            },
            features_extractor_class=ImpalaCNN,
            features_extractor_kwargs=dict(
                features_dim=512,
                depths=[16, 32, 32],
                scale=1
            )
        )

    else:
        policy_kwargs = dict(
            n_quantiles=200,
            net_arch=[256, 256],
            dueling=True,
            noisy=True,
            noisy_kwargs={
                'sigma': 0.2
            }
        )


    model = Rainbow(
        RainbowPolicy,
        env = vectorizedEnv(explore, random, custom, icm) if vectorized else make_single_env(explore, random, custom, icm=False),
        learning_rate=linear_schedule(5e-5 if impala else 1e-4),
        learning_starts=10000,
        policy_kwargs=policy_kwargs,
        buffer_size=100000,
        exploration_initial_eps=1.0,
        exploration_final_eps=0.05,
        exploration_fraction=0.05,
        batch_size=64,
        gamma=0.99,
        train_freq=4,
        gradient_steps=1,
        target_update_interval=10000,
        verbose=1,
        tensorboard_log = tensorboard_log
    )

    save_freq = max(10e6 // 11, 1)

    save_callback = CheckpointCallback(save_freq=save_freq, save_path=log_dir, name_prefix='RDQN_checkpoint', verbose=1)
    model.learn(total_timesteps=75e6, callback=save_callback)

    model_name = "RDQN"
    if explore:
        model_name += "_explore"
    if random:
        model_name += "_random"
    if impala:
        model_name += "_impala"

    model.save(f"{model_name}_mario")