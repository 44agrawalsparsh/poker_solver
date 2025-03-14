
from round_env import PokerEnv
import numpy as np
import torch
import os





GAME_CONFIG = {
    "num_players" : 2,
    "small_blind" : 1,
    "big_blind" : 2,
    "starting_stacks" : 100,
    "ante" : 0,
}

def gen_env():
    return PokerEnv(
        num_players=GAME_CONFIG["num_players"],
        sb=GAME_CONFIG["small_blind"],
        bb=GAME_CONFIG["big_blind"],
        stack_size=GAME_CONFIG["starting_stacks"],
        ante=GAME_CONFIG["ante"]
    )


SIM_CONFIG = {
    "random_games" : 1, #probability of the game allowing for opponent randomness
    "random_decay_rate" : 0.01,
    "prob_random" : 1, #in games with random behaviour, probability of the opponent going off the rails
    "greedy" : True,
    "sims_per_iter" : 1000,
    "actions_per_step" : 10
}

''' 

Actions we'll allow:

Fold
Check/Call
Min Raise
Raise 33% Pot - make the bet size whatever 33% of the pot is currently
Raise 50% Pot - make the bet size whatever 50% of the pot is currently
Raise 100% Pot - make the bet size whatever 100% of the pot is currently
Shove - go all in!

For raise x% pot, it is taking the existing pot multiplying by x% then adding to the existing bet size

e.g. opponent bets $50 into a $100 pot, we raise 33% means 0.33*$150 + $50 ~= $100 bet
'''
FUNCTION_CONFIG = {
    "n_discrete" : 7,
    "obs_dim" : len(gen_env().get_state()),
    "device" : torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")

}

TRAIN_CONFIG = {
    "policy_reservoir_size" : 500_000, #keep a 100000 training samples always
    "ev_reservoir_size" : 700_000,  #keep a 100000 training samples always
    "batch_size" : 2048,
    "lr" : 1e-3,
    "iterations" : 10000,
    "patience": 500,      # number of batches with no improvement before stopping
    "min_delta": 1e-4
}


DISCRETE_ACTION_INDICES = np.arange(7) #np.linspace(GAME_CONFIG["action_space_start"],GAME_CONFIG["action_space_stop"],FUNCTION_CONFIG["n_discrete"])
DISCRETE_ACTIONS = [('fold', 0), ('check/call', 0), ('raise', 0), ('raise', 0.33), ('raise', 0.5), ('raise', 1.0), ('shove', 0)]

EXAMPLES_PATH = 'latest_examples'
POLICY_RESERVOIR_PATH = f'{EXAMPLES_PATH}/policy_reservoir.pkl'
EV_RESERVOIR_PATH = f'{EXAMPLES_PATH}/ev_reservoir.pkl'
LOGS_PATH = "game_logs"


if not os.path.exists(EXAMPLES_PATH):
    os.mkdir(EXAMPLES_PATH)
if not os.path.exists(LOGS_PATH):
    os.mkdir(LOGS_PATH)