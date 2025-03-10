import numpy as np
from config import gen_env, DISCRETE_ACTION_SPACE, FUNCTION_CONFIG, LOGS_PATH
import torch
import os
import sys
import pdb

env = gen_env() #game environment
device = FUNCTION_CONFIG["device"]
def predict_counter_factuals(regret_network, observation):
    """
    Predict counterfactual values for all 40 discrete actions for a single observation.
    Args:
        regret_network: The neural network that takes (obs_dim + 1) input features and outputs a scalar.
        observation: A single observation (1D tensor or convertible to one) of shape (obs_dim,).
        device: The torch device to use (e.g., mps or cpu).
    Returns:
        A NumPy array of predicted counterfactual values for each of the 40 actions.
    """
    import torch
    if not isinstance(observation, torch.Tensor):
        observation = torch.tensor(observation, dtype=torch.float32)

    observation = observation.to(device)
    
    # Ensure observation is a 1D tensor.
    if observation.dim() != 1:
        raise ValueError("Expected a single observation (1D tensor).")
    
    n_actions = len(DISCRETE_ACTION_SPACE)
    
    # Repeat observation for each action: (n_actions, obs_dim)
    observation_batch = observation.unsqueeze(0).expand(n_actions, -1)
    
    # Create a tensor for discrete actions and move it to the correct device
    action_batch = torch.tensor(DISCRETE_ACTION_SPACE, dtype=torch.float32, device=device).unsqueeze(1)  # (40, 1)
    
    # Concatenate along the feature dimension -> (40, obs_dim + 1)
    input_batch = torch.cat((observation_batch, action_batch), dim=1)
    
    # Forward pass through the regret network
    predicted_values = regret_network(input_batch)  # Expected output shape: (40,)
    return predicted_values.detach().cpu().numpy()


def predict_policy(policy_network, observation):
    """
    Predict action probabilities from the policy network for a single observation.
    Args:
        policy_network: The network that takes an observation (obs_dim,) and outputs a probability distribution.
        observation: A single observation (1D tensor or convertible to one) of shape (obs_dim,).
        device: The torch device to use (e.g., mps or cpu).
    Returns:
        A NumPy array of predicted action probabilities.
    """
    import torch
    if not isinstance(observation, torch.Tensor):
        observation = torch.tensor(observation, dtype=torch.float32)

    observation = observation.to(device)
    
    # Ensure observation is a 1D tensor.
    if observation.dim() != 1:
        raise ValueError("Expected a single observation (1D tensor).")
    
    # Add batch dimension -> (1, obs_dim)
    observation = observation.unsqueeze(0)
    
    action_probs = torch.exp(policy_network(observation) ) # -> (1, n_discrete)
    
    return action_probs.squeeze(0).detach().cpu().numpy()

def gen_simulation(policy_network, regret_network, n_per_step, p_random, proportion_random):
    ''' 
    Inputs:
      policy_network -> network used to predict the average strategy (trained offline)
      regret_network -> network used to predict the value/regret of taking an action in a given state: (state, action) -> scalar
      iter_epsilon -> probability parameter for choosing a random action in the main branch
      n_per_step -> number of simulation branches to generate at each decision point for player 0
    Outputs:
      regret_samples -> list of [(State, Action, Value, Reach)] samples to train the regret network
      policy_samples -> list of [(State, Probabilities, Reach)] samples to train the policy network
    '''
    regret_samples = []
    policy_samples = []

    if np.random.random() > p_random:
        proportion_random = 0
    
    env.reset()
    
    while not env.game_over:
        actions = {}
        for p in range(env.num_players):
            obs = env.get_observation_for_agent(p)
            if p == 0:
                # --- Compute instantaneous strategy for player 0 ---
                expected_outcomes = predict_counter_factuals(regret_network, obs)
                cur_policy = predict_policy(policy_network, obs)
                baseline_value = np.dot(expected_outcomes, cur_policy)
                regrets = expected_outcomes - baseline_value
                pos_regrets = np.maximum(regrets, 0)
                sum_pos_regrets = np.sum(pos_regrets)
                if sum_pos_regrets > 1e-8:
                    probabilities = pos_regrets / sum_pos_regrets
                else:
                    probabilities = np.ones(len(DISCRETE_ACTION_SPACE)) / len(DISCRETE_ACTION_SPACE)

                #set all probabilities of invalid bids to 0 
                max_bid = env.get_max_bid(0)
                probabilities[DISCRETE_ACTION_SPACE > max_bid] = 0
                if np.sum(probabilities) == 0:
                    probabilities[0] = 1
                probabilities /= np.sum(probabilities)
                
                # Record the policy sample for training the policy network.
                policy_samples.append([obs, probabilities, 1])
                
                # Determine the main (argmax) action.
                main_action = DISCRETE_ACTION_SPACE[np.argmax(probabilities)]
                
                # --- Branching simulation at the current decision point ---
                branch_actions = []    # actions chosen at this decision point for each branch
                branch_rewards = []    # outcome reward from simulating each branch
                branch_reaches = []    # corresponding reach values (computed from delta_reach)
                
                # For each branch simulate one full rollout.
                for branch in range(n_per_step):
                    # Decide the action for this branch:
                    if branch == 0:
                        # Main branch: use main_action unless a random check perturbs it.
                        #if np.random.random() < iter_epsilon:
                        #    chosen_action = np.random.choice(DISCRETE_ACTION_SPACE)
                        #else:
                        #    chosen_action = main_action
                        chosen_action = main_action
                    else:
                        # Alternative branch: force a deviation by picking a random action (typically not the main one).
                        alt_actions = [a for a in DISCRETE_ACTION_SPACE if a != main_action and a <= max_bid]
                        chosen_action = np.random.choice(alt_actions) if alt_actions else main_action

                    equivalent_actions = []


                    
                    #branch_actions.append(chosen_action)
                    
                    # --- Simulate this branch ---
                    # Make a copy of the current environment state.
                    branch_env = env.copy()
                    
                    # Prepare the actions for the current step in this branch.
                    branch_act = {}
                    branch_act["agent_0"] = chosen_action
                    # For other players, choose actions using the fixed policy.

                    #highest_op = -100
                    for op in range(1, branch_env.num_players):
                        op_obs = branch_env.get_observation_for_agent(op)
                        op_policy = predict_policy(policy_network, op_obs)
                        op_action_idx = np.random.choice(len(DISCRETE_ACTION_SPACE), p=op_policy)
                        branch_act[f"agent_{op}"] = DISCRETE_ACTION_SPACE[op_action_idx]
                        #highest_op = max(DISCRETE_ACTION_SPACE[op_action_idx], highest_op)
                    '''
                    if branch_act["agent_0"] > highest_op and env.get_max_bid(0) > highest_op and False:
                        for i in range(int(np.ceil(highest_op + 1)), 100):
                            equivalent_actions.append(i)
                    else:
                        equivalent_actions = [chosen_action]
                    '''
                        
                        
                    equivalent_actions = [chosen_action]
                    
                    branch_env.step(branch_act)
                    
                    # Roll out the remainder of the game deterministically:
                    # For player 0 we always pick the argmax action from the current regret-based probabilities.
                    while not branch_env.game_over:
                        step_actions = {}
                        for q in range(branch_env.num_players):
                            obs_q = branch_env.get_observation_for_agent(q)
                            if q == 0:
                                exp_outcomes_q = predict_counter_factuals(regret_network, obs_q)
                                pol_q = predict_policy(policy_network, obs_q)
                                baseline_q = np.dot(exp_outcomes_q, pol_q)
                                regrets_q = exp_outcomes_q - baseline_q
                                pos_regrets_q = np.maximum(regrets_q, 0)
                                sum_pos_regrets_q = np.sum(pos_regrets_q)
                                if sum_pos_regrets_q > 1e-8:
                                    probs_q = pos_regrets_q / sum_pos_regrets_q
                                else:
                                    probs_q = np.ones(len(DISCRETE_ACTION_SPACE)) / len(DISCRETE_ACTION_SPACE)
                                step_actions[f"agent_{q}"] = DISCRETE_ACTION_SPACE[np.argmax(probs_q)]
                            else:
                                pol_q = predict_policy(policy_network, obs_q)
                                act_idx_q = np.random.choice(len(DISCRETE_ACTION_SPACE), p=pol_q)
                                step_actions[f"agent_{q}"] = DISCRETE_ACTION_SPACE[act_idx_q]
                        branch_env.step(step_actions)
                    
                    # Get the final reward from this branch.
                    branch_reward = branch_env.get_reward()[0]
                    branch_actions.append(equivalent_actions[0])
                    branch_rewards.append(branch_reward)
                    branch_reaches.append(1)
                    '''
                    for eq in equivalent_actions:
                        branch_actions.append(eq)
                        branch_rewards.append(branch_reward)
                        branch_reaches.append(1/len(equivalent_actions))
                    '''
                    
                    # --- Compute reach probability weight ---
                    # Here we mimic the original logic: if the chosen action is close to the main action,
                    # we assign a higher reach probability.
                    '''if np.abs(chosen_action - main_action) < 0.01:
                        delta_reach = np.log(iter_epsilon / len(DISCRETE_ACTION_SPACE) + 1 - iter_epsilon)
                    else:
                        delta_reach = np.log(iter_epsilon / len(DISCRETE_ACTION_SPACE))
                    branch_reaches.append(np.exp(delta_reach))'''
                    
                
                # --- Record a regret sample for each branch ---
                for a, r, reach in zip(branch_actions, branch_rewards, branch_reaches):
                    regret_samples.append([obs, a, r, reach])
                
                # For the main simulation, continue along branch 0.
                actions["agent_0"] = branch_actions[0]
            else:
                # For players other than player 0, act using the fixed policy.
                cur_policy = predict_policy(policy_network, obs)
                if np.random.random() < proportion_random:
                    act_idx = np.random.choice(len(DISCRETE_ACTION_SPACE))
                else:   
                    act_idx = np.random.choice(len(DISCRETE_ACTION_SPACE), p=cur_policy)
                actions[f"agent_{p}"] = DISCRETE_ACTION_SPACE[act_idx]
        env.step(actions)
    
    return regret_samples, policy_samples

def sim_mock_game(policy_network, iter):
    #### Simulates a game based off a given policy network (assumes all players are playing with it)
    env.reset()

    while not env.game_over:
        actions = {}
        for p in range(env.num_players):
            obs = env.get_observation_for_agent(p)
            cur_policy = predict_policy(policy_network, obs)
            action_idx = np.random.choice(len(DISCRETE_ACTION_SPACE), p=cur_policy)

            action = DISCRETE_ACTION_SPACE[action_idx]
            actions[f"agent_{p}"] = action
        env.step(actions)
    
    log = env.get_game_log()

    with open(os.path.join(LOGS_PATH, f"{iter}.txt"), "w") as f:
        f.write(log)


def clear():
    if sys.platform == "win32":
        os.system("cls")
    else:
        os.system("clear")
    
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




