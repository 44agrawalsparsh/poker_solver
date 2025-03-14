import numpy as np
from config import gen_env, DISCRETE_ACTION_INDICES, DISCRETE_ACTIONS, FUNCTION_CONFIG, LOGS_PATH
import torch
import os
import sys
import pdb
import torch

env = gen_env() #game environment
device = FUNCTION_CONFIG["device"]

def predict_counter_factuals(ev_network, observation):
    """
    Predict counterfactual values for all discrete actions for a single observation.
    
    Args:
        ev_network: The neural network that takes (obs_dim + n_discrete) input features and outputs a scalar.
        observation: A dictionary containing state information with a concatenated observation vector under key "obs".
    
    Returns:
        A NumPy array of predicted counterfactual values for each action.
    """
    # Convert the observation vector to a torch tensor.
    obs = torch.tensor(observation, dtype=torch.float32, device=device)  # shape: (obs_dim,)
    
    # Number of discrete actions.
    n_actions = FUNCTION_CONFIG["n_discrete"]
    
    # Expand the observation so we have one copy per action.
    obs_batch = obs.unsqueeze(0).expand(n_actions, -1)  # shape: (n_actions, obs_dim)
    
    # Create a one-hot encoding for each discrete action.
    action_one_hots = torch.eye(n_actions, dtype=torch.float32, device=device)  # shape: (n_actions, n_actions)
    
    # Concatenate the observation with each one-hot vector.
    # Each network input now has dimension: obs_dim + n_discrete.
    input_batch = torch.cat([obs_batch, action_one_hots], dim=1)  # shape: (n_actions, obs_dim + n_actions)
    
    # Forward pass through the EV network.
    predicted_values = ev_network(input_batch).squeeze(1)  # shape: (n_actions,)
    
    return predicted_values.detach().cpu().numpy()


def predict_policy(policy_network, observation):
    """
    Predict action probabilities from the policy network for a single observation.
    
    Args:
        policy_network: The network that takes an observation and outputs a log probability distribution.
        observation: A dictionary containing state information with a concatenated observation vector under key "obs".
    
    Returns:
        A NumPy array of predicted action probabilities.
    """
    # Convert the observation vector to a torch tensor.
    obs = torch.tensor(observation, dtype=torch.float32, device=device)  # shape: (obs_dim,)
    
    # Add a batch dimension.
    obs = obs.unsqueeze(0)  # shape: (1, obs_dim)
    
    # Forward pass through the policy network. It outputs log probabilities.
    log_probs = policy_network(obs)  # shape: (1, n_discrete)
    
    # Exponentiate to get actual probabilities.
    action_probs = torch.exp(log_probs)
    
    return action_probs.squeeze(0).detach().cpu().numpy()

'''
def predict_counter_factuals(ev_network, observation):
    """
    Predict counterfactual values for all discrete actions for a single observation.
    
    Args:
        ev_network: The neural network that takes (obs_dim + 1) input features and outputs a scalar.
        observation: A dictionary containing state information as described earlier.
        device: The torch device to use (e.g., "mps" or "cpu").
    
    Returns:
        A NumPy array of predicted counterfactual values for each action.
    """
    # Extract non-card and card data from the observation.
    non_card_state = torch.tensor(observation["non_card_state"], dtype=torch.float32, device=device)
    rank_data = torch.tensor(observation["rank_data"], dtype=torch.long, device=device)
    suit_data = torch.tensor(observation["suit_data"], dtype=torch.long, device=device)
    card_data = torch.tensor(observation["card_data"], dtype=torch.long, device=device)
    
    n_actions = len(DISCRETE_ACTION_INDICES)

    # Expand observation for all possible actions
    non_card_state_batch = non_card_state.unsqueeze(0).expand(n_actions, -1)
    rank_data_batch = rank_data.unsqueeze(0).expand(n_actions, -1)
    suit_data_batch = suit_data.unsqueeze(0).expand(n_actions, -1)
    card_data_batch = card_data.unsqueeze(0).expand(n_actions, -1)

    # Create a tensor for discrete actions
    action_batch = torch.tensor(DISCRETE_ACTION_INDICES, dtype=torch.float32, device=device).unsqueeze(1)  # (n_actions, 1)

    # Forward pass through the EV network (which takes state + action as input)
    predicted_values = ev_network(non_card_state_batch, rank_data_batch, suit_data_batch, card_data_batch, action_batch).squeeze(1)

    return predicted_values.detach().cpu().numpy()


def predict_policy(policy_network, observation):
    """
    Predict action probabilities from the policy network for a single observation.
    
    Args:
        policy_network: The network that takes an observation and outputs a probability distribution.
        observation: A dictionary containing state information as described earlier.
        device: The torch device to use (e.g., "mps" or "cpu").
    
    Returns:
        A NumPy array of predicted action probabilities.
    """
    # Extract non-card and card data from the observation.
    non_card_state = torch.tensor(observation["non_card_state"], dtype=torch.float32, device=device)
    rank_data = torch.tensor(observation["rank_data"], dtype=torch.long, device=device)
    suit_data = torch.tensor(observation["suit_data"], dtype=torch.long, device=device)
    card_data = torch.tensor(observation["card_data"], dtype=torch.long, device=device)

    # Add batch dimension -> (1, obs_dim)
    non_card_state = non_card_state.unsqueeze(0)
    rank_data = rank_data.unsqueeze(0)
    suit_data = suit_data.unsqueeze(0)
    card_data = card_data.unsqueeze(0)

    # Forward pass through the policy network
    log_probs = policy_network(non_card_state, rank_data, suit_data, card_data)  # Already outputs softmax probs
    action_probs = torch.exp(log_probs)
    return action_probs.squeeze(0).detach().cpu().numpy()
'''

def gen_simulation(policy_network, ev_network, p_random, proportion_random):
    ''' 
    Inputs:
      policy_network -> network used to predict the average strategy (trained offline)
      ev_network -> network used to predict the value/regret of taking an action in a given state: (state, action) -> scalar
      iter_epsilon -> probability parameter for choosing a random action in the main branch
    Outputs:
      ev_samples -> list of [(State, Action, Value, Reach)] samples to train the regret network
      policy_samples -> list of [(State, Probabilities, Reach)] samples to train the policy network
    '''
    ev_samples = []
    policy_samples = []

    if np.random.random() > p_random:
        proportion_random = 0
    
    env.reset()
    
    while env.round_alive():
        current_agent = env.get_cur_agent()
        obs = env.get_state()
        #pdb.set_trace()

        action_to_play = ''

        if current_agent == 0:
            expected_outcomes = predict_counter_factuals(ev_network, obs)
            cur_policy = predict_policy(policy_network, obs)
            baseline_value = np.dot(expected_outcomes, cur_policy)
            regrets = expected_outcomes - baseline_value
            pos_regrets = np.maximum(regrets, 0)
            sum_pos_regrets = np.sum(pos_regrets)
            if sum_pos_regrets > 1e-8:
                probabilities = pos_regrets / sum_pos_regrets
            else:
                probabilities = np.ones(len(DISCRETE_ACTION_INDICES)) / len(DISCRETE_ACTION_INDICES)

            policy_samples.append([obs, probabilities, 1])


            main_idx = np.argmax(probabilities)

            # --- Branching simulation at the current decision point ---
            branch_actions = []    # actions chosen at this decision point for each branch
            branch_rewards = []    # outcome reward from simulating each branch
            branch_reaches = []    # corresponding reach values (computed from delta_reach)

            for idx,action in enumerate(DISCRETE_ACTIONS):
                branch_env = env.copy()

                branch_env.step(*action)

                while branch_env.round_alive():
                    b_agent = branch_env.get_cur_agent()
                    obs_q = branch_env.get_state()
                    if b_agent == 0:
                        exp_outcomes_q = predict_counter_factuals(ev_network, obs_q)
                        pol_q = predict_policy(policy_network, obs_q)
                        baseline_q = np.dot(exp_outcomes_q, pol_q)
                        regrets_q = exp_outcomes_q - baseline_q
                        pos_regrets_q = np.maximum(regrets_q, 0)
                        sum_pos_regrets_q = np.sum(pos_regrets_q)
                        if sum_pos_regrets_q > 1e-8:
                            probs_q = pos_regrets_q / sum_pos_regrets_q
                        else:
                            probs_q = np.ones(len(DISCRETE_ACTION_INDICES)) / len(DISCRETE_ACTION_INDICES)
                        
                        step_action = DISCRETE_ACTIONS[np.argmax(probs_q)]
                    else:
                        pol_q = predict_policy(policy_network, obs_q)
                        act_idx_q = np.random.choice(len(DISCRETE_ACTION_INDICES), p=pol_q)
                        step_action = DISCRETE_ACTIONS[act_idx_q]
                    branch_env.step(*step_action)


                branch_actions.append(idx)
                branch_reward = branch_env.get_reward()[0]
                branch_rewards.append(branch_reward)
                branch_reaches.append(1)
            
            for a, r, reach in zip(branch_actions, branch_rewards, branch_reaches):
                ev_samples.append([obs, a, r, reach])
            action_to_play = DISCRETE_ACTIONS[main_idx]
        else:
            cur_policy = predict_policy(policy_network, obs)
            if np.random.random() < proportion_random:
                act_idx = np.random.choice(DISCRETE_ACTION_INDICES)
            else:   
                act_idx = np.random.choice(DISCRETE_ACTION_INDICES, p=cur_policy)
            action_to_play = DISCRETE_ACTIONS[act_idx]
        
        env.step(*action_to_play)
    
    return ev_samples, policy_samples

def sim_mock_games(policy_network, iter, n=100):
    #### Simulates a game based off a given policy network (assumes all players are playing with it)
    for round in range(n):
        env.reset()

        while env.round_alive():
            if env.state.status == False:
                pdb.set_trace()
            obs = env.get_state()
            cur_policy = predict_policy(policy_network, obs)
            action_idx = np.random.choice(DISCRETE_ACTION_INDICES, p=cur_policy)

            action = DISCRETE_ACTIONS[action_idx]
            env.step(*action)

        env.save_log(os.path.join(LOGS_PATH, f"{iter}_{round+1}.txt"))

    '''
    
    log = env.state #will do smth better later

    with open(os.path.join(LOGS_PATH, f"{iter}.txt"), "w") as f:
        f.write(str(log))
    '''


def clear():
    if sys.platform == "win32":
        os.system("cls")
    else:
        os.system("clear")
    
'''
def play_human(policy_network):
    env.reset()

    while not env.game_over:
        clear()
        print(env.get_current_state())
        actions = {}
        for p in range(env.num_players):
            obs = env.get_observation_for_agent(p)
            if p == 0:
                print(f"Player {p}'s observation: {obs}")
                #print(f"Player {p}'s policy: {predict_policy(policy_network, obs)}")
                action = float(input("Enter your bid: "))/100
            else:
                cur_policy = predict_policy(policy_network, obs)
                action_idx = np.random.choice(len(DISCRETE_ACTION_SPACE), p=cur_policy)
                action = DISCRETE_ACTION_SPACE[action_idx]
            actions[f"agent_{p}"] = action
        env.step(actions)
    clear()
    print(env.get_game_log())
    input("Press Enter to exit...")
'''



