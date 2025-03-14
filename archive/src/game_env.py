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
from pokerkit import Automation, NoLimitTexasHoldem, HandHistory
from treys import Card, Evaluator
import re

def extract_parentheses(s):
	if s == None:
		return None
	print(s)
	return re.search(r'\((.*?)\)', s).group(1)
'''
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
'''
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


class PokerEnv():
	'''
	Wrapper for pokerkit TexasHoldEm
	'''

	def __init__(self, num_players=2, sb=1, bb=2, stack_size=100,ante=0):
		### Note we'll always train player 0 and have the rest use the existing policy - just randomize the position player 0 is in
		self.num_players = num_players
		self.sb = sb
		self.bb = bb
		self.ante = ante
		self.stack_size = stack_size*bb

		self.game = NoLimitTexasHoldem(
			automations,
			ante_trimming_status=False,
			raw_antes=ante, 
			raw_blinds_or_straddles=[sb, bb], 
			min_bet=bb,
			mode=pokerkit.state.Mode.CASH_GAME
		)
		self.state = self.game(raw_starting_stacks=self.stack_size, 
						  player_count=num_players)

		'''self.state = NoLimitTexasHoldem.create_state(automations=automations, 
													 ante_trimming_status=False, 
													 raw_antes=ante, 
													 raw_blinds_or_straddles=[sb, bb], 
													 min_bet=bb, 
													 raw_starting_stacks=self.stack_size, 
													 player_count=num_players, 
													 mode=pokerkit.state.Mode.CASH_GAME)'''
		
		self.agents = list(range(num_players))
		random.shuffle(self.agents) #so we get a variety of starting positions

		#print(self.state.player_indices, self.agents, self.state.actor_index)
		self.index_to_agent = {pos : a for pos,a in zip(self.state.player_indices, self.agents)}
		self.agent_to_index = {a:pos for pos,a in zip(self.state.player_indices, self.agents)}
		#TODO figure out game representation etc with betting histories and what not

		self.betting_history = [np.zeros(self.num_players + 2) for _ in range(3)]

		while self.state.can_deal_hole():
			self.state.deal_hole()
		#self.state.deal_hole()

	def get_card_index(card=None):
		if card == None:
			return 0,0,0
		rank, suit = card
		rank_idx = RANK_DICT[rank]
		suit_idx = SUIT_DICT[suit]

		card_idx = (suit_idx - 1)*13 + rank_idx

		return suit_idx, rank_idx, card_idx
	
	def round_alive(self):
		return self.state.status

	def get_cur_agent(self):
		return self.index_to_agent[self.state.actor_index]
	
	def evaluate_preflop(self, hand):
		"""
		Evaluate the strength of a two-card starting hand using Chen's formula
		and return a normalized score between 0 and 1.
		The hand is a list of two card strings (e.g. "Ah", "Kd").
		"""

		# Mapping for ordering ranks
		rank_order = {
			'A': 14, 'K': 13, 'Q': 12, 'J': 11, 'T': 10,
			'9': 9, '8': 8, '7': 7, '6': 6, '5': 5,
			'4': 4, '3': 3, '2': 2
		}

		# Helper to get Chen base value for a single card
		def card_base(ch):
			if ch == 'A':
				return 10.0
			elif ch == 'K':
				return 8.0
			elif ch == 'Q':
				return 7.0
			elif ch == 'J':
				return 6.0
			elif ch == 'T':
				return 5.0
			else:
				# For numeric cards, half the card's value works neatly
				return int(ch) / 2.0

		# Extract rank and suit from each card.
		# We assume each card is represented as "<rank><suit>", e.g., "Ah"
		card1, card2 = hand[0], hand[1]
		rank1, suit1 = card1[0], card1[-1]
		rank2, suit2 = card2[0], card2[-1]

		# Determine which card is higher by Chen ranking (using rank_order)
		if rank_order[rank1] >= rank_order[rank2]:
			high_rank, low_rank = rank1, rank2
		else:
			high_rank, low_rank = rank2, rank1

		base_high = card_base(high_rank)
		# (The low card’s base isn’t used directly for scoring.)

		# If the hand is a pair, double the base value
		if rank1 == rank2:
			chen = 2 * base_high
			# A pair is never rated below 5 in Chen's method
			if chen < 5:
				chen = 5.0
		else:
			chen = base_high

			# If suited, add 2 bonus points
			if suit1 == suit2:
				chen += 2.0

			# Compute gap: difference in rank numbers minus 1.
			gap = rank_order[high_rank] - rank_order[low_rank] - 1

			# Subtract points for gaps:
			if gap == 1:
				chen -= 1.0
			elif gap == 2:
				chen -= 2.0
			elif gap == 3:
				chen -= 4.0
			elif gap >= 4:
				chen -= 5.0

			# Optional bonus: if the cards are connected (gap 0) and the lower card is a 2,
			# Chen's formula sometimes awards a +1 bonus.
			if gap == 0 and rank_order[low_rank] == 2:
				chen += 1.0

			# Round the result to the nearest half-point
			chen = round(chen * 2) / 2.0

			# Ensure a minimum score of 1
			if chen < 1:
				chen = 1.0

		# For Chen's formula, the best possible score is 20 (AA) and the worst is 1.
		# Normalize to a 0-1 range.
		normalized = (chen - 1.0) / (20.0 - 1.0)
		return normalized
	
	def card_to_string(card):
		rank = str(card.rank)
		suit = str(card.suit)

		return rank + suit



	
	def get_hand_scores(self):
		# Get the string representations of hole and board cards
		hand_cards = [PokerEnv.card_to_string(x) for x in self.state.hole_cards[self.state.actor_index]]
		board_cards = [PokerEnv.card_to_string(x[0])  for x in self.state.board_cards]  # May contain 0, 3, 4, or 5 cards

		# Convert to Card objects for the evaluator
		hand = [Card.new(x) for x in hand_cards]
		evaluator = Evaluator()

		# Preflop: Use a custom evaluation method because Evaluator doesn't support just 2 cards
		preflop_score = self.evaluate_preflop(hand_cards)  # You must implement this

		# Initialize scores for later streets
		flop_score = 0
		turn_score = 0
		river_score = 0

		# Define normalization parameters based on Evaluator's range
		best_possible = 1      # Best hand (e.g., royal flush)
		worst_possible = 7462  # Worst hand (verify with your Evaluator)

		# Flop evaluation (requires 3 board cards)
		if len(board_cards) >= 3:
			flop_cards = [Card.new(x) for x in board_cards[:3]]
			score = evaluator.evaluate(hand, flop_cards)
			# Normalize so that 1 is best and 0 is worst
			flop_score = 1 - (score - best_possible) / (worst_possible - best_possible)

		# Turn evaluation (requires 4 board cards)
		if len(board_cards) >= 4:
			turn_cards = [Card.new(x) for x in board_cards[:4]]
			score = evaluator.evaluate(hand, turn_cards)
			turn_score = 1 - (score - best_possible) / (worst_possible - best_possible)

		# River evaluation (requires 5 board cards)
		if len(board_cards) == 5:
			river_cards = [Card.new(x) for x in board_cards]
			score = evaluator.evaluate(hand, river_cards)
			river_score = 1 - (score - best_possible) / (worst_possible - best_possible)

		return {
			"preflop": preflop_score,
			"flop": flop_score,
			"turn": turn_score,
			"river": river_score
		}




	def get_state(self):
		'''
		Each card will be defined via an embedding of suit, value, and all 52 card position (extra index to each for blanks/unseen)

		So we have hole cards

		then flop cards

		then turn card

		then river card 

		We sort hole cards by suit and then value as well as for flop cards

		We need to pass in information on the current pot size (normalized by starting stack)

		We also need to pass information on the player's position one hot encoding from SB to Button

		Then for each player from SB to Button we pass in

		Whether the player is still active or not

		How much the player has bet in the current street

		We need to pass in information on how much to call

		Then we need to pass in information on previous betting history

		For now let's keep it simple!

		For each round of betting (pre-flop, flop, turn, river)

		We code in:

		- number of raises in total
		- remaining active players at completion
		- last agressor
		- final bet size

		'''

		state_info = {}


		cur_actor = self.state.actor_index
		if cur_actor == None:
			pdb.set_trace()
		state_info["hole_cards"] = sorted([(str(card.rank), str(card.suit)) for card in self.state.hole_cards[cur_actor]], key = lambda card : card[0] + card[1])
		state_info["flop_cards"] = [None,None,None]
		state_info["turn_cards"] = [None]
		state_info["river_cards"] = [None]

		if len(self.state.board_cards) <= 3 and len(self.state.board_cards) > 0:
			state_info["flop_cards"] = sorted([(str(card[0].rank), str(card[0].suit)) for card in self.state.board_cards], key = lambda card : card[0] + card[1])
		elif len(self.state.board_cards) > 3:
			state_info["flop_cards"] = sorted([(str(card[0].rank), str(card[0].suit)) for card in self.state.board_cards[:3]], key = lambda card : card[0] + card[1])
			state_info["turn_cards"] = [(str(self.state.board_cards[3][0].rank), str(self.state.board_cards[3][0].suit))]
			if len(self.state.board_cards) >= 5:
				state_info["river_cards"] = [(str(self.state.board_cards[4][0].rank), str(self.state.board_cards[4][0].suit))]

		###### EVALUATE CARD STRENGTH HERE ######

		hand_scores = self.get_hand_scores()

		state_info["hole_score"] = hand_scores["preflop"]
		state_info["flop_score"] = hand_scores["flop"]
		state_info["turn_score"] = hand_scores["turn"]
		state_info["river_score"] = hand_scores["river"]

		#state_info["hole_cards"] = [PokerEnv.get_card_index(c) for c in state_info["hole_cards"]]
		#state_info["flop_cards"] = [PokerEnv.get_card_index(c) for c in state_info["flop_cards"]]
		#state_info["turn_cards"] = [PokerEnv.get_card_index(c) for c in state_info["turn_cards"]]
		#state_info["river_cards"] = [PokerEnv.get_card_index(c) for c in state_info["river_cards"]]

		street = np.zeros(4)
		street[self.state.street_index] = 1
		state_info["street"] = street


		state_info["pot_size"] = self.state.total_pot_amount/self.stack_size
		
		state_info["calling_cost"] = self.state.checking_or_calling_amount/self.stack_size

		#state_info["agent_idx"] = self.index_to_agent[cur_actor]
		position_vec = np.zeros(self.num_players)
		position_vec[cur_actor] = 1
		state_info["position"] = position_vec

		state_info["statuses"] = np.array([float(x) for x in self.state.statuses])

		state_info["street_bets"] = np.array(self.state.bets) / self.stack_size

		state_info["stacks"] = np.array(self.state.stacks) / self.stack_size

		opener_vec = np.zeros(self.num_players)
		opener_vec[self.state.opener_index] = 1
		state_info["latest_opener"] = opener_vec
		state_info["latest_betting_count"] = self.state.completion_betting_or_raising_count
		state_info["latest_betting_amount"] = np.max(self.state.bets)
		state_info["betting_history"] = np.concatenate(self.betting_history)

		#### The above are features - now we reorganize into smth more condusive for our NN

		def card_to_vec(card):
			suit, rank, _ = PokerEnv.get_card_index(card)

			suit_vec = np.zeros(4)
			if suit > 0:
				suit_vec[suit-1] = 1
			rank_vec = np.zeros(13)
			if rank > 0:
				rank_vec[rank-1] = 1

			return np.concatenate([suit_vec, rank_vec])
		
		nn_data = np.concatenate([
			state_info["street"],
			[state_info["pot_size"]],
			[state_info["calling_cost"]],
			[state_info["hole_score"], state_info["flop_score"], state_info["turn_score"], state_info["river_score"]],
			state_info["position"],
			state_info["statuses"],
			state_info["street_bets"],
			state_info["stacks"],
			state_info["latest_opener"],
			[state_info["latest_betting_count"]],
			[state_info["latest_betting_amount"]],
			state_info["betting_history"],
			np.concatenate([card_to_vec(card) for card in state_info["hole_cards"]]),
			np.concatenate([card_to_vec(card) for card in state_info["flop_cards"]]),
			np.concatenate([card_to_vec(card) for card in state_info["turn_cards"]]),
			np.concatenate([card_to_vec(card) for card in state_info["river_cards"]])
		]).astype(np.float32)


		'''



		nn_data = {}
		
		non_card_data = np.concatenate([
			state_info["street"],
			[state_info["pot_size"]],
			[state_info["calling_cost"]],
			[state_info["hole_score"], state_info["flop_score"], state_info["turn_score"], state_info["river_score"]],
			state_info["position"],
			state_info["statuses"],
			state_info["street_bets"],
			state_info["stacks"],
			state_info["latest_opener"],
			[state_info["latest_betting_count"]],
			[state_info["latest_betting_amount"]],
			state_info["betting_history"]
		]).astype(np.float32)


		rank_data = np.concatenate([[x[0] for x in state_info["hole_cards"]],
			[x[0] for x in state_info["flop_cards"]],
			[x[0] for x in state_info["turn_cards"]],
			[x[0] for x in state_info["river_cards"]]
		]).astype(np.float32)

		suit_data = np.concatenate([
			[x[1] for x in state_info["hole_cards"]],
			[x[1] for x in state_info["flop_cards"]],
			[x[1] for x in state_info["turn_cards"]],
			[x[1] for x in state_info["river_cards"]]
		]).astype(np.float32)

		card_data = np.concatenate([
			[x[2] for x in state_info["hole_cards"]],
			[x[2] for x in state_info["flop_cards"]],
			[x[2] for x in state_info["turn_cards"]],
			[x[2] for x in state_info["river_cards"]]
		]).astype(np.float32)


		nn_data["non_card_state"] = non_card_data
		nn_data["rank_data"] = rank_data
		nn_data["suit_data"] = suit_data
		nn_data["card_data"] = card_data

		'''


		return nn_data

	def step(self, action_type, bet_proportion=0.0):
		''' 

		Conducts a betting action for the current player.

		Params:
			action_type(string) - one of ["fold", "check/call", "raise", "shove"]
			bet_proportion(float) - positive - if raise it's the percent of existing pot to add on to raise by 
		
		'''

		while self.state.can_deal_hole():
			self.state.deal_hole()

		cur_highest_bid = np.max(self.state.bets)


		if action_type == "fold":
			self.state.fold()
		elif action_type == "check/call":
			self.state.check_or_call()
		else:
			if action_type == "raise":
				base_bet = np.max(self.state.bets)
				cur_pot = self.state.total_pot_amount

				target_bet_size = base_bet + cur_pot*bet_proportion
				#print("Bet Size", base_bet, cur_pot, target_bet_size)
			else:
				target_bet_size = self.state.stacks[self.state.actor_index]
		
			target_bet_size = np.clip(target_bet_size, self.state.min_completion_betting_or_raising_to_amount, self.state.max_completion_betting_or_raising_to_amount)
			target_bet_size = np.floor(target_bet_size)

			try:
				self.state.complete_bet_or_raise_to(target_bet_size)
			except ValueError:
				#usually because we are betting when we should be calling not sure
				self.state.check_or_call()
		
		while self.state.can_deal_board():
			#print("Current Bets", self.state.bets)
			street_index = self.state.street_index - 1
			open_vec = np.zeros(self.num_players)
			open_vec[self.state.opener_index] = 1
			open_vec = list(open_vec)
			open_vec.append(cur_highest_bid/self.stack_size)
			open_vec.append(self.state.completion_betting_or_raising_count)

			self.betting_history[street_index] = open_vec

			self.state.deal_board()

	def get_reward(self, agent=0):
		if self.state.status:
			raise Exception('Game Not Complete')
		
		rewards = (np.array(self.state.stacks) - self.stack_size)/self.stack_size

		output = []

		for agent in range(self.num_players):
			output.append(rewards[self.agent_to_index[agent]])
		return output
	def reset(self):
		self.__init__(num_players=self.num_players, sb=self.sb, bb=self.bb, stack_size=self.stack_size/self.bb,ante=self.ante)

	def copy(self):
		return copy.deepcopy(self)
	
	def save_log(self, filepath):
		hh = HandHistory.from_game_state(self.game, self.state)
		hh.players = [f"Agent {i}" for i in range(self.num_players)]

		# Dump hand
		with open(filepath, "wb") as file:
			hh.dump(file)

'''
env = PokerEnv(num_players=2)
state = env.state

print(state)
while state.status:
	pdb.set_trace()
'''