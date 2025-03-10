
from game_env import PokerEnv
import numpy as np
import torch
import os





GAME_CONFIG = {
    "num_players" : 2,
    "small_blind" : 1,
    "big_blind" : 2,
    "starting_stacks" : 200,
    "ante" : 0
}

def gen_env():
    return PokerEnv(
        num_players=GAME_CONFIG["num_players"],
        small_blind=GAME_CONFIG["small_blind"],
        big_blind=GAME_CONFIG["big_blind"],
        starting_stacks=GAME_CONFIG["starting_stacks"],
        ante=GAME_CONFIG["ante"]
    )


SIM_CONFIG = {
    "random_games" : 0.1, #probability of the game allowing for opponent randomness
    "prob_random" : 0.1, #in games with random behaviour, probability of the opponent going off the rails
    "greedy" : True,
    "sims_per_iter" : 100,
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
    "obs_dim" : gen_env().OBS_DIM,
    "device" : torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")

}

TRAIN_CONFIG = {
    "policy_reservoir_size" : 25_000, #keep a 100000 training samples always
    "regret_reservoir_size" : 1_000_000,  #keep a 100000 training samples always
    "batch_size" : 2048,
    "lr" : 1e-3,
    "iterations" : 10000
}


DISCRETE_ACTION_SPACE = np.linspace(GAME_CONFIG["action_space_start"],GAME_CONFIG["action_space_stop"],FUNCTION_CONFIG["n_discrete"])


EXAMPLES_PATH = 'latest_examples'
POLICY_RESERVOIR_PATH = f'{EXAMPLES_PATH}/policy_reservoir.pkl'
REGRET_RESERVOIR_PATH = f'{EXAMPLES_PATH}/regret_reservoir.pkl'
LOGS_PATH = "game_logs"


if not os.path.exists(EXAMPLES_PATH):
    os.mkdir(EXAMPLES_PATH)
if not os.path.exists(LOGS_PATH):
    os.mkdir(LOGS_PATH)