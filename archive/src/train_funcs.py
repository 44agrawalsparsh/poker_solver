from reservoir_sampling import RegretReservoir, PolicyReservoir, LinearReservoir
from gen_simulations import gen_simulation, sim_mock_games #, play_human
from config import POLICY_RESERVOIR_PATH, EV_RESERVOIR_PATH, TRAIN_CONFIG, FUNCTION_CONFIG, SIM_CONFIG
from approximators import PolicyNetwork, EVNetwork
import numpy as np
import os
import pickle
import torch
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
from itertools import cycle

from tqdm import tqdm, trange
import pdb
device = FUNCTION_CONFIG["device"]


def jensen_shannon_divergence(log_p, q, eps=1e-12):
    """
    Compute Jensen-Shannon divergence using log-probabilities.
    :param log_p: Log of predicted probability distribution (batch, n_discrete)
    :param q: Target probability distribution (batch, n_discrete)
    :param eps: Small value to avoid log(0)
    :return: Scalar JSD loss
    """
    p = torch.exp(log_p)  # Convert back to probabilities
    q = q + eps  # Avoid log(0)
    q = q / q.sum(dim=-1, keepdim=True)  # Ensure proper probability distribution
    
    m = 0.5 * (p + q)  # Mixture distribution
    log_m = torch.log(m + eps)  # Log of mixture to avoid log(0)

    kl_pm = F.kl_div(log_p, m, reduction="batchmean")  # KL(p || m)
    kl_qm = F.kl_div(log_m, q, reduction="batchmean")  # KL(q || m)

    return 0.5 * (kl_pm + kl_qm)



def train_policy_network(reservoir):
    data = reservoir.get_data()
    model = PolicyNetwork(FUNCTION_CONFIG["obs_dim"]).to(device)
    model.train()

    if len(data) == 0:
        print("Reservoir is empty. No training data available. Sending untrained model.")
        return model

    states, target_policies, _ = zip(*data)

    # Convert observations into tensor format
    states = torch.tensor(states, dtype=torch.float32, device=device)
    target_policies = torch.tensor(target_policies, dtype=torch.float32, device=device)

    dataset = TensorDataset(states, target_policies)
    batch_size = TRAIN_CONFIG["batch_size"]
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=TRAIN_CONFIG["lr"])

    max_batches = TRAIN_CONFIG["iterations"]
    patience = TRAIN_CONFIG.get("patience", 500)
    min_delta = TRAIN_CONFIG.get("min_delta", 1e-4)
    best_loss = float('inf')
    wait = 0

    batch_iterator = cycle(dataloader)
    progress_bar = trange(max_batches, desc="Training Policy Network", leave=True)
    for batch_idx in progress_bar:
        batch_states, batch_targets = next(batch_iterator)
        optimizer.zero_grad()
        log_preds = model(batch_states)
        loss = jensen_shannon_divergence(log_preds, batch_targets)
        loss.backward()
        optimizer.step()

        loss_val = loss.item()
        progress_bar.set_postfix(loss=f"{loss_val:.4f}")

        if loss_val < best_loss - min_delta:
            best_loss = loss_val
            wait = 0
        else:
            wait += 1

        if wait >= patience:
            print(f"Early stopping triggered after {batch_idx+1} batches with no significant improvement.")
            break

    return model


def train_ev_network(reservoir):
    data = reservoir.get_data()
    model = EVNetwork(FUNCTION_CONFIG["obs_dim"] + FUNCTION_CONFIG["n_discrete"]).to(device)
    model.train()

    if len(data) == 0:
        print("Reservoir is empty. No training data available. Sending untrained model.")
        return model

    states, actions, rewards, reaches = zip(*data)

    # Convert observations into tensor format
    states = torch.tensor(states, dtype=torch.float32, device=device)
    
    # One-hot encode actions
    actions = torch.tensor(actions, dtype=torch.long, device=device)
    actions_one_hot = torch.nn.functional.one_hot(actions, num_classes=FUNCTION_CONFIG["n_discrete"]).float()

    # Stack state and one-hot encoded actions
    input_tensor = torch.cat((states, actions_one_hot), dim=1)

    rewards = torch.tensor(rewards, dtype=torch.float32, device=device)
    reaches = torch.tensor(reaches, dtype=torch.float32, device=device)
    weights = 1.0 / (reaches + 1e-6)
    weights /= weights.sum()

    dataset = TensorDataset(input_tensor, rewards, weights)
    batch_size = TRAIN_CONFIG["batch_size"]
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    max_batches = TRAIN_CONFIG["iterations"]
    patience = TRAIN_CONFIG.get("patience", 500)
    min_delta = TRAIN_CONFIG.get("min_delta", 1e-4)
    best_loss = float('inf')
    wait = 0

    batch_iterator = cycle(dataloader)
    progress_bar = trange(max_batches, desc="Training EV Network", leave=True)
    for batch_idx in progress_bar:
        batch_inputs, batch_rewards, batch_weights = next(batch_iterator)
        optimizer.zero_grad()
        preds = model(batch_inputs).squeeze(1)
        loss = torch.mean(batch_weights * len(weights) * (preds - batch_rewards) ** 2)
        loss.backward()
        optimizer.step()

        loss_val = loss.item()
        progress_bar.set_postfix(loss=f"{loss_val:.4f}")

        if loss_val < best_loss - min_delta:
            best_loss = loss_val
            wait = 0
        else:
            wait += 1

        if wait >= patience:
            print(f"Early stopping triggered after {batch_idx+1} batches with no significant improvement.")
            break

    return model

def iteration(iter, num_sims=SIM_CONFIG["sims_per_iter"]):
    #epsilon = SIM_CONFIG["base_epsilon"] * SIM_CONFIG["epsilon_decay"]**iter
    p_random =  SIM_CONFIG["random_games"] * np.exp(-1*SIM_CONFIG["random_decay_rate"]*iter)

    if not os.path.exists(POLICY_RESERVOIR_PATH):
        policy_reservoir = LinearReservoir(size=TRAIN_CONFIG["policy_reservoir_size"])
        ev_reservoir = LinearReservoir(size=TRAIN_CONFIG["ev_reservoir_size"])
    else:
        with open(POLICY_RESERVOIR_PATH, "rb") as f:
            policy_reservoir = pickle.load(f)

            if isinstance(policy_reservoir, PolicyReservoir):
                print("Loaded object is a PolicyReservoir")
                data = policy_reservoir.data
                policy_reservoir = LinearReservoir(size=TRAIN_CONFIG["policy_reservoir_size"])
                policy_reservoir.add(data)
        with open(EV_RESERVOIR_PATH, "rb") as f:
            ev_reservoir = pickle.load(f)

            if isinstance(ev_reservoir, RegretReservoir):
                print("Loaded object is a RegretReservoir")
                data = ev_reservoir.data
                ev_reservoir = LinearReservoir(size=TRAIN_CONFIG["regret_reservoir_size"])
                ev_reservoir.add(data)
    
    print(f"Policy Reservoir Size is: {len(policy_reservoir)}, Regret Reservoir Size is: {len(ev_reservoir)}")
    regret_network = train_ev_network(ev_reservoir)
    policy_network = train_policy_network(policy_reservoir)


    
    sim_mock_games(policy_network, iter, n=25)

    for _ in tqdm(range(num_sims), desc=f"Iteration {iter}", leave=True, ascii=True):
        
        regret_samples, policy_samples = gen_simulation(policy_network, regret_network, p_random, SIM_CONFIG["prob_random"])

        ev_reservoir.add(regret_samples)
        policy_reservoir.add(policy_samples)
    
    #pdb.set_trace()

    with open(POLICY_RESERVOIR_PATH, "wb") as f:
        pickle.dump(policy_reservoir, f)

    with open(EV_RESERVOIR_PATH, "wb") as f:
        pickle.dump(ev_reservoir, f)

    

    

def train_and_play_human():
    if not os.path.exists(POLICY_RESERVOIR_PATH):
        raise Exception("No training data available. Please run iteration() first.")
    with open(POLICY_RESERVOIR_PATH, "rb") as f:
        policy_reservoir = pickle.load(f)

    policy_network = train_policy_network(policy_reservoir)

    while True:
        play_human(policy_network)
    

