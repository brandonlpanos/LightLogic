import os
import numpy as np
from stable_baselines3 import SAC
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.noise import NormalActionNoise
from stable_baselines3.common.callbacks import EvalCallback, StopTrainingOnRewardThreshold, BaseCallback
from callbacks import ParameterHistoryCallback
from rl_env import NlteEnvParam 

# RL solution is extremely sensitive to the initial parameters and requres careful tuning.
# Action Noise 
# Discount factor
# Discretization of the action space
# Learning rate
# Entropy coefficient
# Batch and buffer size
# Termination conditions

# --- Configuration ---
LOG_DIR = "nlte_sac_param_logs/" 
MODEL_SAVE_PATH = os.path.join(LOG_DIR, "sac_nlte_param_model")
TOTAL_TIMESTEPS = 100000 # probably needs fewer steps
N_ENVS = 4 # number of parallel environments
EVAL_FREQ = 2000 # eval more often initially
S_CONVERGED_PATH = "S_converged_ali.npz" # use to debug the env
MSE_TERMINATION_THRESHOLD = 1e-6
MAX_ITERATIONS_PER_EPISODE = 50
REWARD_THRESHOLD = -1e-6 # Stop if average reward is better than this
os.makedirs(LOG_DIR, exist_ok=True)
PARAMS_HISTORY_SAVE_PATH = os.path.join(LOG_DIR, "param_history.npy") # Save path for parameter history

# Create vectorized environments for training
env_kwargs = {'max_iterations': MAX_ITERATIONS_PER_EPISODE, 'S_converged_ali_path': S_CONVERGED_PATH, 'mse_termination_threshold': MSE_TERMINATION_THRESHOLD}
# Wrap each env instance with Monitor
vec_env = make_vec_env(lambda: Monitor(NlteEnvParam(**env_kwargs)), n_envs=N_ENVS)
# Create a separate evaluation environment
eval_env = Monitor(NlteEnvParam(**env_kwargs))

# --- Callbacks ---
# Custom callback to save parameter history
param_history_callback = ParameterHistoryCallback(save_path=PARAMS_HISTORY_SAVE_PATH, verbose=1)
callback_on_best = StopTrainingOnRewardThreshold(reward_threshold=REWARD_THRESHOLD, verbose=1)
eval_callback = EvalCallback(eval_env,
                             best_model_save_path=os.path.join(LOG_DIR, 'best_model'),
                             log_path=LOG_DIR,
                             eval_freq=max(EVAL_FREQ // N_ENVS, 1),
                             n_eval_episodes=5,
                             deterministic=True,
                             render=False,
                             callback_on_new_best=callback_on_best)

# combine callbacks
callback_list = [param_history_callback, eval_callback]

# --- Agent Definition ---
# Action noise might need adjustment if stuck in local minima, higher sigma promotes exploration
n_actions = vec_env.action_space.shape[-1]
action_noise = NormalActionNoise(mean=np.zeros(n_actions), sigma=0.15 * np.ones(n_actions))
model = SAC("MlpPolicy",
            vec_env,
            verbose=1,
            tensorboard_log=LOG_DIR,
            learning_rate=3e-4, # Default might be okay
            buffer_size=50_000, # Can maybe reduce buffer size
            learning_starts=1000,
            batch_size=256,
            tau=0.005,
            gamma=0.99, # Discount factor - maybe lower gamma (e.g., 0.95) for faster convergence focus?
            action_noise=action_noise,
            seed=42,
            policy_kwargs=dict(net_arch=[64, 64]) # Smaller network might work 
            )

# --- Training ---
model.learn(total_timesteps=TOTAL_TIMESTEPS, callback=callback_list, log_interval=10)

# --- Save Final Model ---
model.save(MODEL_SAVE_PATH)
vec_env.close()
eval_env.close()

# --- Optional: Evaluate the Trained Agent ---
# Load best model
model = SAC.load(os.path.join(LOG_DIR, 'best_model/best_model'), env=eval_env)
n_eval_episodes = 5
total_reward = 0
total_steps = 0
eval_single_env = NlteEnvParam(**env_kwargs, render_mode='human') # Watch evaluation

for episode in range(n_eval_episodes):
    obs, info = eval_single_env.reset()
    done = False
    truncated = False
    ep_reward = 0
    ep_steps = 0
    while not (done or truncated):
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, done, truncated, info = eval_single_env.step(action)
        ep_reward += reward
        ep_steps += 1
    total_reward += ep_reward
    total_steps += ep_steps
    final_mse = info.get('mse', np.inf)
    final_params = info.get('params', [])
    print(f"Eval Ep {episode+1}: Steps={ep_steps}, Reward={ep_reward:.4f}, Done={done}, Trunc={truncated}, Final MSE={final_mse:.4e}")
    print(f"  Final Params: floor={final_params[0]:.3f}, amp={final_params[1]:.3f}, cen={final_params[2]:.3f}, wid={final_params[3]:.3f}")

avg_reward = total_reward / n_eval_episodes
avg_steps = total_steps / n_eval_episodes
print(f"\nAverage Eval Results: Steps={avg_steps:.2f}, Reward={avg_reward:.4f}")

eval_single_env.close()