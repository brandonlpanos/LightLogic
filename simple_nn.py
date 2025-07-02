import os
import torch
import itertools
import numpy as np
import torch.nn as nn
from tqdm import tqdm
import torch.optim as optim
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.colors as mcolors

from lvl_nlte import calculate_J_L

# --- Configuration ---
LOG_DIR = "nlte_fnn_param_logs/" # New directory for FNN results
PARAMS_HISTORY_SAVE_PATH = os.path.join(LOG_DIR, "fnn_parameter_history.npy")
OUTPUT_PLOT_PATH = os.path.join(LOG_DIR, "fnn_parameter_pairwise_evolution.png")
os.makedirs(LOG_DIR, exist_ok=True)

# Optimal parameters [floor, amplitude, center, width] - TARGET/GOAL
OPTIMAL_PARAMS = np.array([0.02, 1.00, -0.06, 0.54]) # To show target in the plot
PARAM_NAMES = ['Floor', 'Amplitude', 'Center', 'Width']
N_PARAMS = len(PARAM_NAMES)

# --- FNN Training Configuration ---
N_TRAINING_STEPS = 20000
LEARNING_RATE = 1e-4
HIDDEN_SIZES = [128, 64]

# --- Load Atmospheric Parameters ---
# Get the number of depth points from the file
S_CONVERGED_PATH = "S_converged_ali.npz"
fhand = np.load(S_CONVERGED_PATH)
_S_converged_ali = fhand[fhand.files[0]]
ND = _S_converged_ali.shape[0]
del _S_converged_ali

logtau_np = np.linspace(-7, 2, ND)
tau_np = 10.0**logtau_np
B_np = np.ones(ND, dtype=np.float64)
eps_np = np.full(ND, 1E-4, dtype=np.float64)
line_ratio = 1E3

# --- Radiative Transfer Setup ---
NL = 21
x = np.linspace(-4, 4, NL)
profile_np = (1./np.sqrt(np.pi) * np.exp(-(x**2.0))).astype(np.float64)
wx_np = np.zeros(NL, dtype=np.float64)
wx_np[0] = (x[1] - x[0]) * 0.5
wx_np[-1] = (x[-1] - x[-2]) * 0.5
wx_np[1:-1] = (x[2:NL] - x[0:-2]) * 0.5
norm_prof = (np.sum(profile_np * wx_np))
wx_np = wx_np / norm_prof
mu_np = np.array([0.887298335, 0.5, 0.112701665], dtype=np.float64)
wmu_np = np.array([0.2777777778, 0.4444444444, 0.277777778], dtype=np.float64)

# --- Define the simple Feedforward Neural Network ---
class ParameterPredictor(nn.Module):
    def __init__(self, input_size, hidden_sizes, output_size):
        super(ParameterPredictor, self).__init__()
        layers = []
        prev_size = input_size
        for hidden_size in hidden_sizes:
            layers.append(nn.Linear(prev_size, hidden_size))
            layers.append(nn.ReLU())
            prev_size = hidden_size
        layers.append(nn.Linear(prev_size, output_size))
        layers.append(nn.Tanh()) # Output in [-1, 1]
        self.network = nn.Sequential(*layers)

    def forward(self, s_profile):
        return self.network(s_profile)

# --- Define PyTorch Parameter Scaling Function ---
def scale_nn_output_to_params_torch(nn_output_t):
    # nn_output_t is of shape (N_PARAMS,)
    nn_output_flat = nn_output_t.flatten()

    # Define centers and half-widths as tensors
    centers = torch.tensor([0.5, 0.55, -1.25, 1.25], device=nn_output_t.device, dtype=torch.float32)
    half_widths = torch.tensor([0.5, 0.55, 1.75, 1.05], device=nn_output_t.device, dtype=torch.float32)

    # Scale: center + half_width * action
    scaled_params_t = centers + half_widths * nn_output_flat

    # Define bounds as tensors
    min_bounds = torch.tensor([0.0, 0.0, -3.0, 0.2], device=nn_output_t.device, dtype=torch.float32)
    max_bounds = torch.tensor([1.0, 1.1, 0.5, 2.3], device=nn_output_t.device, dtype=torch.float32)

    # Clip using torch.clamp
    clipped_params_t = torch.clamp(scaled_params_t, min=min_bounds, max=max_bounds)

    return clipped_params_t # Returns a tensor requiring grad if nn_output_t did

# --- Define the S-parameterization function ---
def sigmoid_torch(x):
    x_clipped = torch.clamp(x, -500, 500)
    return 1.0 / (1.0 + torch.exp(-x_clipped))

def calculate_S_parameterized_torch(logtau_t, params_t):
    # logtau_t: Tensor (ND,)
    # params_t: Tensor (N_PARAMS,) requiring grad
    floor = params_t[0]
    amplitude = params_t[1]
    center = params_t[2]
    # Ensure width is positive, use clamp or minimum with a small epsilon
    width = torch.clamp(params_t[3], min=1e-6) # Get width from params

    logtau_norm = (logtau_t - center) / width
    S_raw = floor + amplitude * sigmoid_torch(logtau_norm)

    # Clip S to physically reasonable bounds
    S_t = torch.clamp(S_raw, min=1e-10, max=1.5)
    return S_t # Returns tensor (ND,) requiring grad

# --- Setup for Training ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Instantiate network and optimizer
model = ParameterPredictor(ND, HIDDEN_SIZES, N_PARAMS).to(device)
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
loss_fn = nn.MSELoss()

# Convert atmospheric params to tensors 
logtau_t = torch.tensor(logtau_np, dtype=torch.float32).to(device)
B_t = torch.tensor(B_np, dtype=torch.float32).to(device)
eps_t = torch.tensor(eps_np, dtype=torch.float32).to(device)

# --- Training Loop ---
parameter_history_fnn = []
losses = []

# Initial state S=B
current_S_np = np.copy(B_np)

for step in tqdm(range(N_TRAINING_STEPS)):
 
    current_S_t_input = torch.tensor(current_S_np, dtype=torch.float32).to(device)

    # 1. Predict raw parameters [-1, 1] 
    params_pred_raw_t = model(current_S_t_input)

    # 2. Scale raw parameters to physical ranges 
    params_pred_scaled_t = scale_nn_output_to_params_torch(params_pred_raw_t)

    # --- Store parameters for history (convert to NumPy) ---
    parameter_history_fnn.append(params_pred_scaled_t.detach().cpu().numpy().copy())

    # 3. Calculate S based on predicted scaled parameters
    S_pred_t = calculate_S_parameterized_torch(logtau_t, params_pred_scaled_t)

    # --- Calculate J and S_implied ---
    S_pred_np_for_J = S_pred_t.detach().cpu().numpy()
    J_current_np, _ = calculate_J_L(tau_np, S_pred_np_for_J, mu_np, wmu_np, profile_np, wx_np, line_ratio, B_np[-1])
    S_implied_np = (1.0 - eps_np) * J_current_np + eps_np * B_np

    # 4. Convert S_implied to tensor for loss calculation
    S_implied_t = torch.tensor(S_implied_np, dtype=torch.float32).to(device)

    # 5. Calculate the loss
    loss = loss_fn(S_pred_t, S_implied_t)

    # 6. Backpropagation
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # Store loss
    losses.append(loss.item())

    # 7. Update the input S for the next iteration
    current_S_np = S_pred_t.detach().cpu().numpy()
    # current_S_np = S_implied_t.detach().cpu().numpy() # Same result if I use implied as input (NN doesnt converge to target)

# Save parameter history
parameter_history_fnn_np = np.array(parameter_history_fnn)
np.save(PARAMS_HISTORY_SAVE_PATH, parameter_history_fnn_np)