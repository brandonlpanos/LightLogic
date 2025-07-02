import numpy as np
import gymnasium as gym
from gymnasium import spaces
from lvl_nlte import calculate_J_L

# Define the parameterized source function
def sigmoid(x):
    x_clipped = np.clip(x, -500, 500)  # Add clipping for numerical stability with large exponents
    return 1.0 / (1.0 + np.exp(-x_clipped))

# Calculates S based on 4 parameters
def calculate_S_parameterized(logtau, params):
    floor, amplitude, center, width = params
    width = max(width, 1e-6)
    logtau_norm = (logtau - center) / width
    S = floor + amplitude * sigmoid(logtau_norm)
    S = np.clip(S, 1e-10, 1.5)
    return S

class NlteEnvParam(gym.Env):
    """
    Action: A vector of 4 values in [-1, 1] which are scaled to define the parameters (floor, amplitude, center, width) of S. Note that all optimal settings are only used for plotting and debugging.
    Observation: The current source function S calculated from parameters
    Reward: Based on MSE fit
    """

    metadata = {"render_modes": ["human"], "render_fps": 4}

    # Number of parameters for the source function
    N_PARAMS = 4

    def __init__(self, render_mode=None, max_iterations=50, S_converged_ali_path="S_converged_ali.npz", mse_termination_threshold=1e-6):
        super().__init__()

        # --- Load Converged Solution which is needed for reward/termination ---
        fhand = np.load(S_converged_ali_path)
        self.S_converged_ali = fhand[fhand.files[0]]

        # --- Initialize Atmospheric Parameters ---
        self.ND = self.S_converged_ali.shape[0] # Get ND from loaded array
        self.logtau = np.linspace(-7, 2, self.ND)
        self.tau = 10.0**self.logtau
        self.B = np.ones(self.ND, dtype=np.float64) # Assumes B=1 used for S_converged_ali
        self.eps = np.full(self.ND, 1E-4, dtype=np.float64)
        self.line_ratio = 1E3

        # --- Radiative Transfer Setup ---
        NL = 21
        x = np.linspace(-4, 4, NL)
        self.profile = (1./np.sqrt(np.pi) * np.exp(-(x**2.0))).astype(np.float64)
        wx = np.zeros(NL, dtype=np.float64)
        wx[0] = (x[1] - x[0]) * 0.5
        wx[-1] = (x[-1] - x[-2]) * 0.5
        wx[1:-1] = (x[2:NL] - x[0:-2]) * 0.5
        norm = (np.sum(self.profile * wx))
        self.wx = wx / norm
        self.mu = np.array([0.887298335, 0.5, 0.112701665], dtype=np.float64)
        self.wmu = np.array([0.2777777778, 0.4444444444, 0.2777777778], dtype=np.float64)
        # --- End Atmospheric Parameters ---

        self.max_iterations = max_iterations
        self.mse_termination_threshold = mse_termination_threshold

        # --- Action Space: 4 parameters, output in [-1, 1] ---
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(self.N_PARAMS,), dtype=np.float64)

        # --- Observation Space: The S profile itself ---
        self.observation_space = spaces.Box(low=0.0, high=1.5, shape=(self.ND,), dtype=np.float64)

        # --- Internal state variables ---
        self.current_params = None # Holds the 4 current parameters
        self.S = None # Holds the S profile calculated from parameters
        self.iteration_count = 0
        self.current_mse = np.inf

    # Maps agent action [-1, 1] to tailored physical parameter ranges based on initial and target values
    def _scale_action_to_params(self, action):

        # action shape: (4,)
        scaled_params = np.zeros(self.N_PARAMS)

        # Parameter: Range needed -> Proposed Range [min, max] -> center + half_width * action

        # Floor: [0.02, 0.90] -> [0.0, 1.0]
        center_floor = 0.5
        half_width_floor = 0.5
        scaled_params[0] = center_floor + half_width_floor * action[0]

        # Amplitude: [0.10, 1.00] -> [0.0, 1.1]
        center_amp = 0.55
        half_width_amp = 0.55
        scaled_params[1] = center_amp + half_width_amp * action[1]

        # Center: [-2.50, -0.06] -> [-3.0, 0.5]
        center_cen = -1.25
        half_width_cen = 1.75
        scaled_params[2] = center_cen + half_width_cen * action[2]

        # Width: [0.54, 2.00] -> [0.2, 2.3]
        center_wid = 1.25
        half_width_wid = 1.05
        scaled_params[3] = center_wid + half_width_wid * action[3]

        # Clip parameters incase action + noise goes out of bounds [-1, 1]
        scaled_params[0] = np.clip(scaled_params[0], 0.0, 1.0)
        scaled_params[1] = np.clip(scaled_params[1], 0.0, 1.1)
        scaled_params[2] = np.clip(scaled_params[2], -3.0, 0.5)
        scaled_params[3] = np.clip(scaled_params[3], 0.2, 2.3)

        return scaled_params

    def _get_obs(self):
        # Observation is the current S profile
        return self.S.astype(np.float64)

    def _get_info(self):
        # Report MSE and current parameters
        return {"iteration": self.iteration_count,"mse": self.current_mse,"params": self.current_params.copy()}

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        # Initialize parameters to S = B
        # S=B requires amplitude=0, floor=1. Center/width less important
        initial_params = np.array([
            0.9,  # floor near 1
            0.1,  # small amplitude
            -2.5, # center (arbitrary initial)
            2.0   # width (arbitrary initial)
        ])
        self.current_params = initial_params
        self.S = calculate_S_parameterized(self.logtau, self.current_params)

        self.iteration_count = 0
        diff = self.S - self.S_converged_ali
        self.current_mse = np.mean(diff**2)

        observation = self._get_obs()
        info = self._get_info()

        return observation, info

    def step(self, action):

        # 1. Scale agent action to physical parameters
        self.current_params = self._scale_action_to_params(action)

        # 2. Calculate the new S profile based on these parameters
        self.S = calculate_S_parameterized(self.logtau, self.current_params)

        # 3. Real siuation without knowing the parameters (the mean radiation field integrated over angles and frequencies at each depth point)
        J_current, _ = calculate_J_L(self.tau, self.S, self.mu, self.wmu, self.profile, self.wx, self.line_ratio, self.B[-1]) # (ND,)

        # 4. Calculate the equilibrium-implied source function S_implied based on the calculated J_current
        S_implied = (1.0 - self.eps) * J_current + self.eps * self.B

        # 5. Calculate the residual between the agent's S and the S implied by the physics
        residual = self.S - S_implied

        # 6. Calculate the reward as the negative mean squared error (MSE) of the residual, i.e., how far from equilibrium
        reward = -np.mean(residual**2)

        # Update internal state
        self.iteration_count += 1

        # For stopping criteria
        diff = self.S - self.S_converged_ali
        self.current_mse = np.mean(diff**2)

        # Check Termination/Truncation based on MSE
        terminated = self.current_mse < self.mse_termination_threshold
        truncated = self.iteration_count >= self.max_iterations

        # Bonus reward for convergence (not sure if this will ever trigger)
        if terminated:
            reward += 10.0

        # Get Observation and Info
        observation = self._get_obs()
        info = self._get_info()

        return observation, reward, terminated, truncated, info