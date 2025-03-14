import os
import numpy as np
from round_env import RoundEnv
from utils import Position, clear_screen
from random import shuffle
from time import sleep
import sys
import termios
import tty
import re

# 1) Regex to remove ANSI escape codes
ANSI_PATTERN = re.compile(r'\x1B\[[0-9;]*[A-Za-z]')

def strip_ansi_codes(text: str) -> str:
    """Remove ANSI escape codes for length calculations."""
    return ANSI_PATTERN.sub('', text)

def ansi_ljust(text: str, width: int) -> str:
    """Left-justify, ignoring ANSI codes for length."""
    visible_length = len(strip_ansi_codes(text))
    return text + ' ' * max(0, width - visible_length)

def ansi_rjust(text: str, width: int) -> str:
    """Right-justify, ignoring ANSI codes for length."""
    visible_length = len(strip_ansi_codes(text))
    return ' ' * max(0, width - visible_length) + text

def ansi_center(text: str, width: int) -> str:
    """Center text, ignoring ANSI codes for length."""
    visible_length = len(strip_ansi_codes(text))
    left_spaces = (width - visible_length) // 2
    right_spaces = width - visible_length - left_spaces
    return ' ' * max(0, left_spaces) + text + ' ' * max(0, right_spaces)

class GameEnv:

	def __init__(self, sb=1, bb=2, ante=0):
		self.num_players = 6
		self.sb = sb
		self.bb = bb
		self.ante = ante
		
		#id, is human controlled!
		ids = list(range(self.num_players))
		shuffle(ids)

		self.human_player = ids[0]

		self.players = [(ids[0],False)] + [(ids[i],False) for i in range(1,self.num_players)]
		self.player_positions = {player_id : Position(player_id) for player_id,_ in self.players}
		self.position_to_player = {val.value : player_id for player_id, val in self.player_positions.items()}
		self.stacks = {player_id : 50*self.bb for player_id,_ in self.players}

		self.correct_decisions = 0
		self.incorrect_decisions = 0

	def order_actions_by_aggression(self, actions):
		"""
		Orders a list of poker actions from least aggressive to most aggressive.
		
		:param actions: List of action strings
		:param bb: Big blind value to scale bets/raises
		:return: Ordered list of actions
		"""
		def aggression_level(action):
			if action.lower() == "fold":
				return 0
			elif action.lower() in ["check", "call"]:
				return 1
			elif "all" in action.lower():
				return 4
			elif "raise" in action.lower() or "bet" in action.lower():
				try:
					amount = int(action.split("(")[1].split(")")[0])
				except (IndexError, ValueError):
					amount = 0  # If malformed, assume lowest aggression for safety
				return 2 + (amount / self.bb)  # Scale aggression by bet size
			elif "bb" in action:
				try:
					amount = int(float(action[:-2]) * self.bb)
				except ValueError:
					amount = 0
				return 2 + (amount / self.bb)
			else:
				raise ValueError(f"Unrecognized action: {action}")
		
		return sorted(actions, key=aggression_level)
	
	def select_action(self, actions, gto_strat, rand_value):
		"""
		Selects an action based on a given random value using the strategy probabilities.
		
		:param actions: List of action strings
		:param gto_strat: Dictionary mapping actions to probabilities
		:param rand_value: Integer between 1-100 representing aggressiveness
		:param bb: Big blind value for scaling
		:return: Selected action
		"""
		ordered_actions = self.order_actions_by_aggression(actions)
		
		# Compute cumulative distribution function (CDF)
		cdf = {}
		cumulative = 0
		for action in ordered_actions:
			if action in gto_strat:
				cumulative += gto_strat[action] * 100  # Convert to percentage scale
				cdf[action] = cumulative
		
		# Select the least aggressive action with CDF >= rand_value
		for action in ordered_actions:
			if cdf.get(action, 0) >= rand_value:
				return action
		
		return ordered_actions[-1] 

	def collect_action(self, state):
		### For now let's just play according to strategy
		player = state["player_turn"]
		entropy = state["random_num"]
		player_id = self.position_to_player[Position[player].value]
		gto_strat = state["player_strategy"]
		actions = list(gto_strat.keys())
		actions = self.order_actions_by_aggression(actions)
		probs = np.array([gto_strat[a] for a in actions])
		probs = probs / np.sum(probs)

		theo_action = self.select_action(actions, gto_strat, entropy)

		if player_id == self.human_player:
			#TODO
			print("Actions to choose from:")
			for i,a in enumerate(actions):
				print(f"Press {i+1} for {a}")

			# Wait for button press and get the action index
			action_index = self.wait_for_button_press(len(actions)) - 1
			decision = actions[action_index]

			##TODO - look at entropy to tell it which action it expected

			text_print = "\n"

			for action in actions:
				text_print += f"\n{action} : {(gto_strat[action]*100):.2f}%"
			if decision == theo_action:
				self.correct_decisions += 1
				text_print += f"\n\nBased off the entropy of {entropy:.2f} you correctly picked {theo_action}."
			else:
				self.incorrect_decisions += 1
				text_print += f"\n\nBased off the entropy of {entropy:.2f} you should have picked {theo_action}."
			print(text_print)
		else:
			
			decision = theo_action #np.random.choice(actions, p=probs)
		return decision

	def wait_for_button_press(self, num_actions):
		print("Press a key corresponding to your action choice (1 to {}):".format(num_actions))
		while True:
			key = self.get_key_press()
			if key.isdigit():
				action_index = int(key)
				if 1 <= action_index <= num_actions:
					return action_index

	def get_key_press(self):
		"""
		Capture a single key press on Unix-like systems (Linux, macOS).
		"""
		fd = sys.stdin.fileno()
		old_settings = termios.tcgetattr(fd)
		try:
			tty.setraw(sys.stdin.fileno())
			ch = sys.stdin.read(1)
		finally:
			termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
		return ch
	
	def render_screen(self, state):
		clear_screen()
		# --- Determine round status ---
		round_alive = 'current_bets' in state

		human_pos = self.player_positions[self.human_player]
		human_idx, human_pos = Position(human_pos).value, Position(human_pos).name
		# --- Extract meta information ---
		sb = state['small_blind']
		bb = state['big_blind']
		entropy = state.get('random_num', np.random.randint(0, 100))
		if round_alive and state["player_turn"] == human_pos:
			meta_line = f"Small Blind: {sb} | Big Blind: {bb} | Entropy: {entropy:.2f}"
		else:
			meta_line = f"Small Blind: {sb} | Big Blind: {bb}"

		score_line = f" | GTO Movement Rate: {self.correct_decisions}/{self.correct_decisions + self.incorrect_decisions}"
		meta_line += score_line

		# --- Suit symbols and color mapping ---
		suit_symbols = {
			'hearts': '♥',
			'diamonds': '♦',
			'clubs': '♣',
			'spades': '♠'
		}

		suit_colors = {
			'hearts': '\033[91m',  # Red
			'diamonds': '\033[94m',  # Blue
			'clubs': '\033[92m',  # Green
			'spades': '\033[93m'  # Yellow
		}

		def format_card(card):
			if len(card) < 2:
				return "[**]"
			rank, suit = card[:-1], card[-1]
			suit_name = {'h': 'hearts', 'd': 'diamonds', 'c': 'clubs', 's': 'spades'}.get(suit, 'spades')
			color = suit_colors[suit_name]
			symbol = suit_symbols[suit_name]
			return f"{color}[{rank}{symbol}]\033[0m"

		# --- Community cards and pot ---
		board = state.get('board', [])
		board_display = ' '.join(format_card(card) for card in board) if board else ""
		pot = state.get('pot_amount', 0)
		community_line = f"{board_display} | Pot: {pot}" if board else f"Pot: {pot}"

		# --- Position names mapping ---
		position_names = ['SB', 'BB', 'UTG', 'HJ', 'CO', 'BTN']

		# --- Player display order: UTG, HJ, CO, BTN, SB, BB ---
		display_order = [2, 3, 4, 5, 0, 1]

		# --- Prepare player data ---
		player_data = []
		for idx in display_order:
			position = position_names[idx]
			active = state['players_active'][idx] if round_alive else False
			stack = state['stacks'][idx]
			bet = state['current_bets'][idx] if round_alive else 0
			hole_cards = state['hole_cards'][idx]

			# Determine card visibility
			show_cards = (not round_alive) or (position == human_pos)
			card1 = format_card(hole_cards[0]) if show_cards and hole_cards else "[**]"
			card2 = format_card(hole_cards[1]) if show_cards and hole_cards else "[**]"
			cards_line = card1 + card2

			# Bold active positions
			position = f"{position}*" if round_alive and position == state["player_turn"] else position
			position_str = f"\033[1m{position}\033[0m" if active and round_alive else position

			info_line = f"{position_str} ${stack} (${bet})"

			player_data.extend([cards_line, info_line])

		# --- Unpack player data ---
		utg_cards, utg_info = player_data[0], player_data[1]
		hj_cards, hj_info   = player_data[2], player_data[3]
		co_cards, co_info   = player_data[4], player_data[5]
		btn_cards, btn_info = player_data[6], player_data[7]
		sb_cards, sb_info   = player_data[8], player_data[9]
		bb_cards, bb_info   = player_data[10], player_data[11]

		# --- Build ASCII art ---
		max_width = 100
		left_width = max_width // 2   # 40
		right_width = max_width - left_width  # 40

		lines = [
			ansi_center(meta_line, max_width),
			"",
			ansi_center(utg_cards, max_width),
			ansi_center(utg_info, max_width),
			"",
			ansi_ljust(bb_cards, left_width) + ansi_rjust(hj_cards, right_width),
			ansi_ljust(bb_info, left_width) + ansi_rjust(hj_info, right_width),
			"",
			ansi_center(community_line, max_width),
			"",
			ansi_ljust(sb_cards, left_width) + ansi_rjust(co_cards, right_width),
			ansi_ljust(sb_info, left_width) + ansi_rjust(co_info, right_width),
			"",
			ansi_center(btn_cards, max_width),
			ansi_center(btn_info, max_width)
		]

		text = '\n'.join(lines)
		print(text)

	def play_round(self):
		position_to_player = {val.value : player_id for player_id, val in self.player_positions.items()}
		starting_stacks = [self.stacks[position_to_player[i]] for i in range(self.num_players)]
		round = RoundEnv(sb=self.sb, bb=self.bb, ante=self.ante, starting_stacks=starting_stacks)

		while round.round_alive():
			state = round.get_state()
			self.render_screen(state)
			action = self.collect_action(state)
			print(f"\n\n{state['player_turn']} chose to {action}\n\n")
			round.play_move(action)
			input("Press enter to continue to next turn")
			if not round.round_alive():
				state = round.get_state()
				self.render_screen(state)
				input("Press enter to continue to next hand")
			

		#update stacks and cycle players
		self.stacks = {player_id : state["stacks"][self.player_positions[player_id].value] for player_id,_ in self.players}
		self.player_positions = {player_id : Position((self.player_positions[player_id].value + 1) % self.num_players) for player_id,_ in self.players}
		self.position_to_player = {val.value : player_id for player_id, val in self.player_positions.items()}

		if np.sum(np.array(list(self.stacks.values())) <= 0) > 0:
			clear_screen()
			print("One player has busted! Restarting all stacks...")
			self.stacks = {player_id : 50*self.bb for player_id,_ in self.players}
			input("Press enter to continue to next hand")