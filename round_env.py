import numpy as np
from enum import Enum
import random
import copy
import gymnasium as gym
import json
import pdb
import pokerkit
from pokerkit import Automation, NoLimitTexasHoldem, HandHistory
from treys import Card, Evaluator
import re
from utils import Position, PREFLOP_STRATEGY_PATH, PREFLOP_RANGES_PATH, GAME_CODES, solve_postflop, find_existing_file
import json
import os

def extract_parentheses(s):
	if s == None:
		return None
	print(s)
	return re.search(r'\((.*?)\)', s).group(1)

automations = (
	Automation.ANTE_POSTING,
	Automation.BET_COLLECTION,
	Automation.BLIND_OR_STRADDLE_POSTING,
	Automation.CARD_BURNING,
	Automation.HOLE_CARDS_SHOWING_OR_MUCKING,
	Automation.HAND_KILLING,
	Automation.CHIPS_PUSHING,
	Automation.CHIPS_PULLING,
	Automation.RUNOUT_COUNT_SELECTION
)


RANK_DICT = {'2' : 1, '3' : 2, '4' : 3, '5' : 4, '6' : 5, '7' : 6, '8' : 7, '9' : 8, 'T' : 9, 'J' : 10, 'Q' : 11, 'K' : 12, 'A' : 13}
SUIT_DICT = {'c' : 1, 'd' : 2, 'h' : 3, 's' : 4}


class RoundEnv():
	'''
	Class object to play a round of poker. We also 'solve' the game throughout using our preflop strategy and solver for flop,turn and river. 
	'''

	def __init__(self, sb=1, bb=2, ante=0, starting_stacks=''):
		self.num_players = 6
		self.sb = sb
		self.bb = bb
		self.ante = ante

		self.game = NoLimitTexasHoldem(
			automations,
			ante_trimming_status=False,
			raw_antes=ante, 
			raw_blinds_or_straddles=[sb, bb], 
			min_bet=bb,
			mode=pokerkit.state.Mode.CASH_GAME
		)
		self.state = self.game(raw_starting_stacks=starting_stacks, 
						  player_count=self.num_players)
		
		with open(PREFLOP_STRATEGY_PATH, 'r') as file:
			self.preflop_strategy =  json.load(file)

		self.cur_preflop_strategy = self.preflop_strategy

		self.path_to_preflop_range = PREFLOP_RANGES_PATH

		self.ip_range = ''
		self.oop_range = ''

		self.ip_idx = -1
		self.oop_idx = -1
		
		while self.state.can_deal_hole():
			self.state.deal_hole()

		self.original_hole_cards = copy.deepcopy(self.state.hole_cards)

	def round_alive(self):
		return self.state.status
	
	def card_to_string(card):
		rank = str(card.rank)
		suit = str(card.suit)

		return rank + suit
	
	def cur_actor(self):
		idx = self.state.actor_index
		position = Position(idx)
		return position.name, idx
	
	def get_player_hole_cards(self, player=None):
		if player == None:
			player = self.state.actor_index
		if isinstance(player, Position):
			return self.state.hole_cards[(player.value + 2) % 6]
		elif type(player) == int: ###here it is probably the pokerkit index
			return self.state.hole_cards[player]
		else:
			return self.state.hole_cards[Position[player].value]
		
	def extract_card(card):
		return (str(card.rank), str(card.suit))
		
	def convert_cards_to_hand_range(self, hand):
		card1, card2 = hand
		
		card1 = RoundEnv.extract_card(card1)
		card2 = RoundEnv.extract_card(card2)
		rank1, rank2 = RANK_DICT[card1[0]], RANK_DICT[card2[0]]
		if rank1 < rank2:
			temp = card2
			card2 = card1
			card1 = temp
		
		if rank1 == rank2:
			return f"{card1[0]}{card2[0]}"
		
		if card1[1] == card2[1]:
			return f"{card1[0]}{card2[0]}s"
		return f"{card1[0]}{card2[0]}o"
	
	def get_state(self):
		output = {}
		if self.round_alive():
			output["small_blind"] = self.sb
			output["big_blind"] = self.bb
			output["street_idx"] = self.state.street_index

			board = [RoundEnv.extract_card(c[0]) for c in self.state.board_cards]
			board = [f"{c[0]}{c[1]}" for c in board]
			output["board"] =  board

			output["players_active"] = self.state.statuses
			output["current_bets"] = self.state.bets
			output["pot_amount"] = self.state.total_pot_amount
			output["stacks"] = self.state.stacks
			try:
				pos,_ = self.cur_actor()
			except Exception as e:
				print(e)
				pdb.set_trace()

			output["player_turn"] = pos

			hole_cards = [(RoundEnv.extract_card(c[0]), RoundEnv.extract_card(c[1])) for c in self.original_hole_cards]
			hole_cards = [(f"{c[0][0]}{c[0][1]}",f"{c[1][0]}{c[1][1]}")  for c in hole_cards]
			output["hole_cards"] = hole_cards
			output["player_strategy"] = self.get_gto_strategy()
			output["random_num"] = 100*np.random.random()
			
		else:
			output["small_blind"] = self.sb
			output["big_blind"] = self.bb
			board = [RoundEnv.extract_card(c[0]) for c in self.state.board_cards]
			board = [f"{c[0]}{c[1]}" for c in board]
			output["board"] =  board
			hole_cards = [(RoundEnv.extract_card(c[0]), RoundEnv.extract_card(c[1])) for c in self.original_hole_cards]
			hole_cards = [(f"{c[0][0]}{c[0][1]}",f"{c[1][0]}{c[1][1]}")  for c in hole_cards]
			output["hole_cards"] = hole_cards
			output["stacks"] = self.state.stacks


		return copy.deepcopy(output)

		
	
	def get_gto_strategy(self):
		if self.state.street_index == 0:
			#pdb.set_trace()
			cur_player_name, cur_player_idx = self.cur_actor()
			private_hand = self.get_player_hole_cards(cur_player_name)
			hand = self.convert_cards_to_hand_range(private_hand)
			#print(self.cur_preflop_strategy["strategy"])
			#print(hand, self.cur_preflop_strategy["strategy"][hand])
			strategy = self.cur_preflop_strategy["next_action"]["strategy"][hand]
			return strategy

		cur_player_name, cur_player_idx = self.cur_actor()
		private_hand = self.get_player_hole_cards(cur_player_name)
		cards = [RoundEnv.extract_card(c) for c in private_hand]
		hand1 = cards[0][0] + cards[0][1] + cards[1][0] + cards[1][1]
		hand2 = cards[1][0] + cards[1][1] + cards[0][0] + cards[0][1]

		if hand1 in self.cur_postflop_strategy["strategy"]:
			strategy = self.cur_postflop_strategy["strategy"][hand1]
		elif hand2 in self.cur_postflop_strategy["strategy"]:
			strategy = self.cur_postflop_strategy["strategy"][hand2]
		else:
			raise NotImplementedError
		return strategy
		
	def process_preflop_action(self, action):
		pos,cur_index = self.cur_actor()
		if action == "fold":
			self.state.fold()
		elif action == "allin":
			# Calculate how much to raise to
			
			amount = self.state.stacks[cur_index] + self.state.bets[cur_index]
			if amount < self.state.stacks[cur_index]:
				self.state.check_or_call()
			else:
				self.state.complete_bet_or_raise_to(amount)
		elif action == "call" or action == "check":
			self.state.check_or_call()
		else:
			assert 'bb' in action, (f"{action} not recognized")
			#pdb.set_trace()
			amount = int(float(action[:-2])*self.bb)
			if amount < self.state.stacks[cur_index]:
				self.state.check_or_call()
			else:
				self.state.complete_bet_or_raise_to(amount)

		self.path_to_preflop_range = os.path.join(self.path_to_preflop_range, pos, action)
		
		self.cur_preflop_strategy = self.cur_preflop_strategy["next_action"]["child_nodes"][action]

	def range_to_str(dict):
		output = [f"{key}:{value:.2f}" for key,value in dict.items()]
		return ",".join(output)

	def process_postflop_action(self, action):
		_, cur_index = self.cur_actor()
		if action == "Fold":
			self.state.fold()
		elif action == "Check" or action == "Call":
			self.state.check_or_call()
		elif "Raise" in action or "Bet" in action:
			amount = int(action.split("(")[1].split(")")[0])
			if amount < self.state.stacks[cur_index]:
				self.state.check_or_call()
			else:
				self.state.complete_bet_or_raise_to(amount)
		elif "All" in action:
			amount = self.state.stacks[cur_index] + self.state.bets[cur_index]
			if amount < self.state.stacks[cur_index]:
				self.state.check_or_call()
			else:
				self.state.complete_bet_or_raise_to(amount)
		else:
			raise ValueError("Unrecognized action")
		
		self.ip_range = RoundEnv.range_to_str(self.cur_postflop_strategy["ip_range"])
		self.oop_range = RoundEnv.range_to_str(self.cur_postflop_strategy["oop_range"])

		if action in self.cur_postflop_strategy["children"]:
			self.cur_postflop_strategy = self.cur_postflop_strategy["children"][action]

	def play_move(self, action):
		cur_strat = self.get_gto_strategy()
		if action not in cur_strat:
			raise ValueError("Must be an action in the strategy set :(")

		if self.state.street_index == 0:
			self.process_preflop_action(action)
		else:
			self.process_postflop_action(action)

		### Check for all in
		if self.state.all_in_status:
			while self.state.can_deal_board():
				self.state.deal_board()

		if self.state.status == False:
			return GAME_CODES.FINISHED
		
		if self.state.can_deal_board():
			if self.state.street_index == 1:
				#print(self.path_to_preflop_range)
				active_players = np.array(self.state.statuses)
				positions = np.array([Position(x) for x in self.state.player_indices])

				active_players = positions[active_players]

				if len(active_players) > 2:
					return GAME_CODES.TOO_MANY_POSTFLOP
				
				path_to_oop_range = os.path.join(self.path_to_preflop_range, f"{active_players[0].name}.txt")
				path_to_oop_range = find_existing_file(path_to_oop_range)
				path_to_ip_range = os.path.join(self.path_to_preflop_range, f"{active_players[1].name}.txt")
				path_to_ip_range = find_existing_file(path_to_ip_range)

				with open(path_to_oop_range, 'r') as f:
					self.oop_range = f.read()

				with open(path_to_ip_range, 'r') as f:
					self.ip_range = f.read()

				effective_stack = self.state.get_effective_stack(active_players[0].value)
				pot = self.state.total_pot_amount
				while self.state.can_deal_board():
					self.state.deal_board()

				board = [RoundEnv.extract_card(c[0]) for c in self.state.board_cards]
				board = [f"{c[0]}{c[1]}" for c in board]
				#pdb.set_trace()
				self.postflop_strategy = solve_postflop(self.oop_range, self.ip_range, board, pot, effective_stack)
				self.cur_postflop_strategy = self.postflop_strategy
			elif self.state.street_index >= 2:
				active_players = np.array(self.state.statuses)
				positions = np.array([Position(x) for x in self.state.player_indices])

				active_players = positions[active_players]
				effective_stack = self.state.get_effective_stack(active_players[0].value)
				pot = self.state.total_pot_amount

				self.state.deal_board()

				board = [RoundEnv.extract_card(c[0]) for c in self.state.board_cards]
				board = [f"{c[0]}{c[1]}" for c in board]
				#pdb.set_trace()
				self.postflop_strategy = solve_postflop(self.oop_range, self.ip_range, board, pot, effective_stack)
				self.cur_postflop_strategy = self.postflop_strategy
		
		return GAME_CODES.ALIVE
	
	def save_log(self, filepath):
		hh = HandHistory.from_game_state(self.game, self.state)
		hh.players = [Position(i).name for i in range(self.num_players)]

		# Dump hand
		with open(filepath, "wb") as file:
			hh.dump(file)


'''
round = RoundEnv()

while round.state.status:
	pos, _ = round.cur_actor()
	print(f"{pos} to act with card {round.convert_cards_to_hand_range(round.get_player_hole_cards())}")
	print(f"GTO Strategy is: {round.get_gto_strategy()}")
	action = input("Which action?\n")
	round.play_move(action)

print(f"Final stacks: {round.state.stacks}")'
'''