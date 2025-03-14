import numpy as np
from enum import Enum
from scipy.stats import gamma
import random
import copy
import gymnasium as gym
import numpy as np
import json
import pdb
#ATHLETES = sorted([150, 120, 100, 90, 85, 80, 75, 70, 62, 61, 60, 45], reverse=True)
#ATHLETES = sorted([5, 10, 75, 85,  155, 160], reverse=True)

#######################
# RLlib-compatible Auction Environment
#######################
'''

Will try a single position game with CFR

We will fix the list of players - so that we limit the scope of possibilities - and then our function we train will be:

f(B_self, B_opp, Mean_self, Mean_opp, Slots_self, Slots_opp, Cur Player Mean) -> distribution over 1 -> 100 of how much to bid

We will use Deep CFR to solve this


'''
class AuctionEnv():

	def __init__(self, forwards, defensemen, goalies, num_players=2):
		# For simplicity, we assume state=None (i.e. fresh game)
		self.agents = [f"agent_{i}" for i in range(num_players)]
		self.num_players = num_players
		self.GAME_BUDGET = 100

		self.goalies_bkup = copy.deepcopy(goalies)
		self.forwards_bkup = copy.deepcopy(forwards)
		self.defensemen_bkup = copy.deepcopy(defensemen)

		self.MAX_VAL = sum(goalies + forwards + defensemen)

		self.athletes = [(x, "G") for x in goalies] + [(x, "D") for x in defensemen] + [(x, "F") for x in forwards]
		self.athletes = sorted(self.athletes, key = lambda x : x[0], reverse=True)

		self.goalies_needed = np.ones(self.num_players) * len(self.goalies_bkup)/self.num_players
		self.defensemen_needed = np.ones(self.num_players) * len(self.defensemen_bkup)/self.num_players
		self.forwards_needed = np.ones(self.num_players) * len(self.forwards_bkup)/self.num_players
		self.TEAM_SIZE = len(self.athletes)/2

		self.budgets = np.ones(self.num_players) * self.GAME_BUDGET
		# Array to hold bids from each agent (initialized to -1)
		self.bids_placed = -1 * np.ones(self.num_players)
		# Start with a nominated player (pop from list)
		self.nominated_player = self.athletes.pop(0)

		self.members_means = np.zeros(self.num_players)
		self.last_winner = -1
		self.last_price = -1
		self.prev_bids = np.zeros(self.num_players)

		self.game_over = False
		self.history = {}  # Auction history

		self.illegal_bid_penalties = np.zeros(self.num_players)

		# Define the action space for each agent.
		per_agent_action_space = gym.spaces.Box(low=0.0, high=1.0, shape=(1,), dtype=np.float32)
		self.action_space = gym.spaces.Dict({
			agent: per_agent_action_space for agent in self.agents
		})

		# Define an observation space. (You’ll need to adjust the bounds/shapes to your needs.)
		obs_dim = (
			1 #self budget 
			+ self.num_players - 1 #opp budget
			+ 1 #self mean
			+ self.num_players - 1 #opp mean
			+ 1 #self slots forward
			+ self.num_players - 1 #opp slots forward
			+ 1 #self slots defense
			+ self.num_players - 1 #opp slots defense
			+ 1 #self slots goalie
			+ self.num_players - 1 #opp slots goalie
			+ 1 #current athlete mean
			+ 3 #current athlete position
		)

		self.OBS_DIM = obs_dim

		per_agent_obs_space = gym.spaces.Box(low=0.0, high=1.0, shape=(obs_dim,), dtype=np.float32)
		self.observation_space = gym.spaces.Dict({
			agent: per_agent_obs_space for agent in self.agents
		})
	
	def get_must_bid(self, agent):
		"""
		Returns 1 if the agent is forced to bid on the current nominated player,
		based on their remaining roster needs.
		"""

		if self.nominated_player[1] == "G" and self.goalies_needed[agent] == 0:
			return 1
		elif self.nominated_player[1] == "D" and self.defensemen_needed[agent] == 0:
			return 1
		elif self.nominated_player[1] == "F" and self.forwards_needed[agent] == 0:
			return 1
		
		return 0

	def get_max_bid(self, player):
		"""
		Returns the maximum bid an agent can make.
		(Here we use a fixed range, e.g., 0 to 1000.)
		"""

		if self.nominated_player[1] == "G" and self.goalies_needed[player] <= 0:
			return -1e10
		elif self.nominated_player[1] == "D" and self.defensemen_needed[player] <= 0:
			return -1e10
		elif self.nominated_player[1] == "F" and self.forwards_needed[player] <= 0:
			return -1e10

		max_val = self.budgets[player] #- self.athletes_needed[player] + 1
		return max_val
	
	# --- RLlib Required Methods: reset and step ---

	def reset(self, seed=None, options=None):
		"""
		Reset the environment to the initial state and return observations for all agents.
		"""
		# For simplicity, we reinitialize a new game.
		self.__init__(self.forwards_bkup, self.defensemen_bkup, self.goalies_bkup, self.num_players)
		obs = {}
		for i in range(self.num_players):
			obs[f"agent_{i}"] = self.get_observation_for_agent(i)
		return obs, {}
	
	def get_game_log(self):
		output = ""
		output += f"Final Means: {self.members_means}\n"
		output += f"Final Budgets: {self.budgets}\n"
		output += "\n\n\nHistory:\n"
		for x in self.history:
			output += f"{str(x)} : {str(self.history[x])}\n"
		return output
	
	def get_current_state(self):
		output = ""
		output += f"Current Means: {self.members_means}\n"
		output += f"Current Budgets: {self.budgets}\n"
		output += f"Current Athlete: {self.nominated_player}\n"
		defense_left = [x for x in self.athletes if x[1] == "D"]
		goalies_left = [x for x in self.athletes if x[1] == "G"]
		forwards_left = [x for x in self.athletes if x[1] == "F"]
		output += f"Forwards Left: {forwards_left}\n"
		output += f"Defensemen Left: {defense_left}\n"
		output += f"Goalies Left: {goalies_left}\n" 
		#output += f"Athletes Left: {self.athletes}\n"
		output += f"Forwards Needed: {self.forwards_needed}\n"
		output += f"Defensemen Needed: {self.defensemen_needed}\n"
		output += f"Goalies Needed: {self.goalies_needed}\n"
		output += "\n\n\nHistory:\n"
		for x in self.history:
			output += f"{str(x)} : {str(self.history[x])}\n"
		return output
	
	def step(self, action_dict):
		"""
		Expects a dictionary of actions from all agents.
		Each agent's action is assumed to be a continuous value in [0, 1] that
		is scaled to the bid range [min_bid, max_bid].
		Processes all bids simultaneously and advances the auction by one nominated player.
		"""
		#pdb.set_trace()
		#print(action_dict)
		# Process bids for each agent.
		for agent_id, action in action_dict.items():
			#action = action_dict["action"]
			# Extract agent index from string (assumes "agent_0", "agent_1", etc.)
			agent_index = int(agent_id.split("_")[1])
			min_bid = self.get_must_bid(agent_index)
			max_bid = self.get_max_bid(agent_index)

			bid = action*self.GAME_BUDGET #(max_bid - min_bid) + min_bid

			self.illegal_bid_penalties[agent_index] = min(0, bid - min_bid, max_bid - bid)

			bid = np.clip(bid, min_bid, max(min_bid, max_bid))
			if max_bid < 0:
				bid = max_bid
			# Scale the action to the bid range.
			#bid = action * (max_bid - min_bid) + min_bid
			bid = np.round(bid)
			self.bids_placed[agent_index] = bid

		# Determine the winner using your Vickrey auction rule.
		scores = self.bids_placed
		max_score = np.max(scores)
		winners = np.where(scores == max_score)[0]
		winner = np.random.choice(winners)
		# Second-highest bid: use partitioning.
		if len(self.bids_placed) > 1:
			price_paid = max(0,np.partition(self.bids_placed, -2)[-2])
		else:
			price_paid = self.bids_placed[0]

		self.last_winner = winner
		self.last_price = price_paid
		self.prev_bids = copy.deepcopy(list(self.bids_placed))
		#print("storing old bids:", self.prev_bids)

		# Update winner's budget and history.
		self.budgets[winner] -= price_paid
		self.history[self.nominated_player] = {"price": price_paid, "winner": winner, "bids" : copy.deepcopy(self.prev_bids), "cur_budgets" : copy.deepcopy(list(self.budgets))}
		self.members_means[winner] += self.nominated_player[0]

		if self.nominated_player[1] == "G":
			self.goalies_needed[winner] -= 1
		elif self.nominated_player[1] == "D":
			self.defensemen_needed[winner] -= 1
		else:
			self.forwards_needed[winner] -= 1

		# Reset bids for the next round.
		self.bids_placed = -1 * np.ones(self.num_players)

		# Advance to the next nominated player.
		if len(self.athletes) == 0:
			self.game_over = True
		else:
			self.nominated_player = self.athletes.pop(0)

		# Build observations for all agents.
		obs = {}
		for i in range(self.num_players):
			obs[f"agent_{i}"] = self.get_observation_for_agent(i)

		# Get rewards—here we use your get_reward method.
		rewards_arr = self.get_reward()
		rewards = {}
		for i in range(self.num_players):
			rewards[f"agent_{i}"] = rewards_arr[i]

		# The environment is done if the game is over.
		dones = {"__all__": self.game_over}
		truncated = {"__all__" : False}
		info = {}
		return obs, rewards, dones, truncated, info
	
	def get_observation_for_agent(self, agent):
		"""
		Returns a normalized, flattened observation vector for the given agent.
		"""

		def position_encode(position):

			#return np.array([position == "F", position == "D", position == "G"]).astype(np.float32)
			if position == "F":
				return [1, 0, 0]
			elif position == "D":
				return [0, 1, 0]
			else:
				return [0, 0, 1]

		obs = np.concatenate([
			[self.budgets[agent] / self.GAME_BUDGET], #our remaining budget
			[x/self.GAME_BUDGET for i,x in enumerate(self.budgets) if i != agent], #opponent remaining budget
			[self.members_means[agent] / self.MAX_VAL], #our current mean
			[x/self.MAX_VAL for i,x in enumerate(self.members_means) if i != agent], #opponent current mean
			[self.forwards_needed[agent]/self.TEAM_SIZE], #our forwards needed
			[x/self.TEAM_SIZE for i,x in enumerate(self.forwards_needed) if i != agent], #opponent forwards needed
			[self.defensemen_needed[agent]/self.TEAM_SIZE], #our defensemen needed
			[x/self.TEAM_SIZE for i,x in enumerate(self.defensemen_needed) if i != agent], #opponent defensemen needed
			[self.goalies_needed[agent]/self.TEAM_SIZE], #our goalies needed
			[x/self.TEAM_SIZE for i,x in enumerate(self.goalies_needed) if i != agent], #opponent goalies needed
			[self.nominated_player[0]/self.MAX_VAL], #current athlete mean
			position_encode(self.nominated_player[1]) #current athlete position
		])

		
		obs = obs.astype(np.float32)

		return obs


	def get_reward(self):
		rewards = np.zeros(self.num_players)

		if self.game_over:
			# Final score components


			# Budget utilization penalty (lose points for leftover budget)
			#budget_left_ratios = self.budgets / self.GAME_BUDGET
			#budget_penalty = budget_left_ratios * 0.5  # Max 0.5 penalty for full budget

			final_scores = (self.members_means) # - budget_penalty
			mean_score = np.mean(final_scores)

			# Ranking reward [-1 to 1]
			#rankings = np.argsort(final_scores)[::-1]
			#ranking_reward = np.linspace(1.0, -1.0, num=self.num_players)
			#rewards += ranking_reward[rankings]

			# Relative performance bonus
			rewards += (final_scores - mean_score)/5000
			win_rewards = (rewards == np.max(rewards)).astype(np.float32)
			win_rewards /= np.sum(win_rewards)
			rewards += win_rewards
			rewards -= 1/self.num_players #make sure its 0-sum
			rewards = np.clip(rewards, -1, 1)

		return rewards
	
	def copy(self):
		return copy.deepcopy(self)