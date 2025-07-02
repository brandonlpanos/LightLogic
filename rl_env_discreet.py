import numpy as np
import gymnasium as gym
from gymnasium import spaces
from lvl_nlte import calculate_J_L

# Define the parameterized source function
def sigmoid(x):
    x_clipped = np.clip(x, -500, 500)
    return 1.0 / (1.0 + np.exp(-x_clipped))

# Calculates S based on 4 parameters
def calculate_S_parameterized(logtau, params):
    floor, amplitude, center, width = params
    width = max(width, 1e-6)
    logtau_norm = (logtau - center) / width
    S = floor + amplitude * sigmoid(logtau_norm)
    return np.clip(S, 1e-10, 1.5)

class NlteEnvParam(gym.Env):
    """
    RL environment for discrete delta action to mimic ALI optimization.
    """

    N_PARAMS = 4

    def __init__(self, max_iterations=8, S_converged_ali_path="S_converged_ali.npz", mse_termination_threshold=1e-6):
        super().__init__()
        # Load target solution
        data = np.load(S_converged_ali_path)
        self.S_converged_ali = data[data.files[0]]

        # target parameters
        self.target_params = np.array([0.012653, 1.028329, -0.028090, 0.571760], dtype=np.float64)
        
        # initial parameters
        self.initial_params = np.array([0.9, 0.1, -2.5, 2.0], dtype=np.float64)

        # Atmosphere setup
        self.ND = self.S_converged_ali.size
        self.logtau = np.linspace(-7, 2, self.ND)
        self.tau = 10.0 ** self.logtau
        self.B = np.ones(self.ND, dtype=np.float64)
        self.eps = np.full(self.ND, 1e-4, dtype=np.float64)
        self.line_ratio = 1e3

        # Frequency and angular quadrature
        NL = 21
        x = np.linspace(-4, 4, NL)
        self.profile = (1.0/np.sqrt(np.pi) * np.exp(-x**2)).astype(np.float64)
        wx = np.zeros(NL, np.float64)
        wx[0] = 0.5*(x[1]-x[0]); wx[-1] = 0.5*(x[-1]-x[-2])
        wx[1:-1] = 0.5*(x[2:]-x[:-2])
        self.wx = wx/np.sum(self.profile*wx)
        self.mu = np.array([0.887298335, 0.5, 0.112701665], np.float64)
        self.wmu = np.array([0.2777777778, 0.4444444444, 0.2777777778], np.float64)

        # RL settings
        self.max_iterations = max_iterations
        self.mse_termination_threshold = mse_termination_threshold

        # Calculate incremental action bounds such that at end of max_iterations we reach target parameters
        # Each parameter needs (target - initial) / max_iterations
        delta_per_step = (self.target_params - self.initial_params) / self.max_iterations
        
        # Action bounds --> allow the agent to move up to one step size in either direction
        self.low_actions = -np.abs(delta_per_step)
        self.high_actions = np.abs(delta_per_step)
        self.action_space = spaces.Box(self.low_actions, self.high_actions, shape=(self.N_PARAMS,), dtype=np.float64)

        # Observation --> current S profile
        self.observation_space = spaces.Box(low=0.0, high=1.5, shape=(self.ND,), dtype=np.float64)

        # Internal state
        self.current_params = None
        self.S = None
        self.iteration_count = 0
        self.current_mse = np.inf

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        # Reset to initial parameters
        self.current_params = self.initial_params.copy()
        self.S = calculate_S_parameterized(self.logtau, self.current_params)
        self.iteration_count = 0
        self.current_mse = np.mean((self.S - self.S_converged_ali)**2)
        return self.S.copy(), {"iteration": 0, "mse": self.current_mse, "params": self.current_params.copy()}

    def step(self, action):

        # 1) Apply incremental action
        self.current_params = self.current_params + action

        # 2) clip to reasonable bounds to prevent numerical issues
        param_bounds_low = np.array([0.0, 0.0, -5.0, 0.1], dtype=np.float64)
        param_bounds_high = np.array([2.0, 2.0, 2.0, 10.0], dtype=np.float64)
        self.current_params = np.clip(self.current_params, param_bounds_low, param_bounds_high)
        
        # 3) Calculate the new S profile based on these parameters
        self.S = calculate_S_parameterized(self.logtau, self.current_params)

        # 4) Real situation without knowing the parameters (the mean radiation field integrated over angles and frequencies at each depth point)
        J, _ = calculate_J_L(self.tau, self.S, self.mu, self.wmu, self.profile, self.wx, self.line_ratio, self.B[-1])

        # 5) Calculate the equilibrium-implied source function S_implied based on the calculated J_current
        S_implied = (1.0 - self.eps) * J + self.eps * self.B
        
        # 6) Calculate the reward as the negative mean squared error (MSE) of the residual, i.e., how far from equilibrium
        reward = -np.mean((self.S - S_implied)**2)
        
        # Counters and termination
        self.iteration_count += 1
        self.current_mse = np.mean((self.S - self.S_converged_ali)**2)
        
        # Check if close to target parameters
        param_distance = np.linalg.norm(self.current_params - self.target_params)
        
        terminated = self.current_mse < self.mse_termination_threshold
        truncated = self.iteration_count >= self.max_iterations
        
        # bonus reward for reaching target
        if terminated:
            reward += 10.0
        
        info = {"iteration": self.iteration_count,
                "mse": self.current_mse,
                "params": self.current_params.copy(),
                "param_distance": param_distance,
                "target_params": self.target_params}
        
        return self.S.copy(), reward, terminated, truncated, info