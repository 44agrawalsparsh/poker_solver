import numpy as np
from enum import Enum
from scipy.stats import gamma
import random
import copy
import gymnasium as gym
import numpy as np
import json
import pdb
import pokerkit
from pokerkit import Automation, NoLimitTexasHoldem

automations = (Automation.ANTE_POSTING,
        Automation.BET_COLLECTION,
        Automation.BLIND_OR_STRADDLE_POSTING,
        Automation.BOARD_DEALING,
        Automation.CARD_BURNING,
        Automation.HOLE_DEALING,
        Automation.RUNOUT_COUNT_SELECTION,
        Automation.HOLE_CARDS_SHOWING_OR_MUCKING,
        Automation.HAND_KILLING,
        Automation.CHIPS_PUSHING,
        Automation.CHIPS_PULLING)

class PokerEnv():
    '''
    Wrapper for pokerkit TexasHoldEm
    '''

    def __init__(self, num_players=2, sb=1, bb=2, ante=0):
        ### Note we'll always train player 0 and have the rest use the existing policy - just randomize the position player 0 is in
        self.num_players = num_players
        self.sb = sb
        self.bb = bb
        self.ante = ante

        self.state = NoLimitTexasHoldem.create_state(automations=automations, 
                                                     ante_trimming_status=False, 
                                                     raw_antes=ante, 
                                                     raw_blinds_or_straddles=[sb, bb], 
                                                     min_bet=bb, 
                                                     raw_starting_stacks=100*bb, 
                                                     player_count=num_players, 
                                                     mode=pokerkit.state.Mode.CASH_GAME)
        
        self.agents = list(range(num_players))
        random.shuffle(self.agents) #so we get a variety of starting positions
        self.index_to_agent = {pos : a for pos,a in zip(self.state.acted_player_indices, self.agents)}


        print(self.starting_positions)

        #TODO figure out game representation etc with betting histories and what not

    def game(self):
        pass 

PokerEnv(num_players=6)