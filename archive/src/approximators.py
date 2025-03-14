from config import FUNCTION_CONFIG
import torch
import torch.nn as nn
import torch.nn.functional as F

class EVNetwork(nn.Module):
	def __init__(self, hidden_dim=256):
		super(EVNetwork, self).__init__()
		total_dim = FUNCTION_CONFIG["obs_dim"] + FUNCTION_CONFIG["n_discrete"]
		
		self.fc = nn.Sequential(
			nn.Linear(total_dim, hidden_dim),
			nn.ReLU(),
			nn.Linear(hidden_dim, hidden_dim),
			nn.ReLU(),
			nn.Linear(hidden_dim, hidden_dim),
			nn.ReLU(),
			nn.Linear(hidden_dim, hidden_dim),
			nn.ReLU(),
			nn.Linear(hidden_dim, hidden_dim),
			nn.ReLU(),
			nn.Linear(hidden_dim, hidden_dim),
			nn.ReLU(),
			nn.Linear(hidden_dim, 1)  # Output a single scalar EV.
		)
	
	def forward(self, obs):
		return F.tanh(self.fc(obs))
	
class PolicyNetwork(nn.Module):
	def __init__(self, hidden_dim=256):
		super(PolicyNetwork, self).__init__()
		input_dim = FUNCTION_CONFIG["obs_dim"]
		output_dim = FUNCTION_CONFIG["n_discrete"]
		
		self.fc = nn.Sequential(
			nn.Linear(input_dim, hidden_dim),
			nn.ReLU(),
			nn.Linear(hidden_dim, hidden_dim),
			nn.ReLU(),
			nn.Linear(hidden_dim, hidden_dim),
			nn.ReLU(),
			nn.Linear(hidden_dim, hidden_dim),
			nn.ReLU(),
			nn.Linear(hidden_dim, hidden_dim),
			nn.ReLU(),
			nn.Linear(hidden_dim, hidden_dim),
			nn.ReLU(),
			nn.Linear(hidden_dim, output_dim)  # Output a single scalar EV.
		)
	
	def forward(self, obs):
		logits = self.fc(obs)
		return torch.log_softmax(logits, dim=-1)
		


'''
class EVNetwork(nn.Module):
	def __init__(self, non_card_input_dim, num_cards=7, rank_embed_dim=8, suit_embed_dim=4, 
				 card_embed_dim=16, hidden_dim=256, action_embed_dim=8):
		"""
		EV network that takes in the non-card state, card information, and an action index.
		It outputs a scalar expected value.
		
		Instead of summing over the card embeddings, we concatenate (flatten) them,
		so the model knows which card comes from which position.
		"""
		super(EVNetwork, self).__init__()
		self.num_cards = num_cards
		
		# Embedding layers for card features.
		self.rank_embed = nn.Embedding(num_embeddings=14, embedding_dim=rank_embed_dim, padding_idx=0)
		self.suit_embed = nn.Embedding(num_embeddings=5, embedding_dim=suit_embed_dim, padding_idx=0)
		self.card_embed = nn.Embedding(num_embeddings=53, embedding_dim=card_embed_dim, padding_idx=0)
		
		# Action embedding for 7 abstracted actions.
		self.action_embed = nn.Embedding(num_embeddings=7, embedding_dim=action_embed_dim)
		
		# Process non-card features.
		self.non_card_fc = nn.Sequential(
			nn.Linear(non_card_input_dim, hidden_dim),
			nn.ReLU(),
			nn.Linear(hidden_dim, hidden_dim),
			nn.ReLU(),
			nn.Linear(hidden_dim, hidden_dim),
			nn.ReLU()
		)
		
		# The dimension for a single card's embedding.
		card_embedding_dim = rank_embed_dim + suit_embed_dim + card_embed_dim
		# When concatenated for all cards:
		total_card_dim = num_cards * card_embedding_dim
		
		# Total dimension when combining non-card state, flattened card embeddings, and action embedding.
		total_dim = hidden_dim + total_card_dim + action_embed_dim
		
		self.fc = nn.Sequential(
			nn.Linear(total_dim, hidden_dim),
			nn.ReLU(),
			nn.Linear(hidden_dim, hidden_dim),
			nn.ReLU(),
			nn.Linear(hidden_dim, hidden_dim),
			nn.ReLU(),
			nn.Linear(hidden_dim, hidden_dim),
			nn.ReLU(),
			nn.Linear(hidden_dim, 1)  # Output a single scalar EV.
		)
	
	def forward(self, non_card_state, rank_data, suit_data, card_data, action_idx):
		"""
		Args:
			non_card_state: Tensor of shape (batch_size, non_card_input_dim)
			rank_data: Tensor of shape (batch_size, num_cards) with values in 0-13 (0 indicates non-existent)
			suit_data: Tensor of shape (batch_size, num_cards) with values in 0-4 (0 indicates non-existent)
			card_data: Tensor of shape (batch_size, num_cards) with values in 0-52 (0 indicates non-existent)
			action_idx: Tensor of shape (batch_size,) with values in 0-6 corresponding to actions
		Returns:
			ev: Tensor of shape (batch_size, 1) with the expected value.
		"""
		non_card_out = self.non_card_fc(non_card_state)
		
		# Get card embeddings.
		rank_embeds = self.rank_embed(rank_data.long())
		suit_embeds = self.suit_embed(suit_data.long())
		card_embeds = self.card_embed(card_data.long())
		
		# Concatenate embeddings for each card along the last dimension.
		# Resulting shape: (batch_size, num_cards, card_embedding_dim)
		card_combined = torch.cat([rank_embeds, suit_embeds, card_embeds], dim=-1)
		
		# Instead of summing, we flatten the card embeddings to retain position information.
		card_flat = card_combined.view(card_combined.size(0), -1)  # Shape: (batch_size, num_cards * card_embedding_dim)
		
		# Embed the action.
		action_emb = self.action_embed(action_idx.long()).squeeze(1)
		
		# Concatenate all features.

		combined = torch.cat([non_card_out, card_flat, action_emb], dim=-1)
		ev = self.fc(combined)
		return ev

class PolicyNetwork(nn.Module):
	def __init__(self, non_card_input_dim, num_cards=7, rank_embed_dim=8, suit_embed_dim=4, 
				 card_embed_dim=16, hidden_dim=64, num_actions=7):
		"""
		Policy network that takes in the non-card state and card information,
		and outputs a probability distribution over the 7 actions.
		
		We flatten the concatenated card embeddings to preserve card positions.
		"""
		super(PolicyNetwork, self).__init__()
		self.num_cards = num_cards
		
		# Embedding layers for card features.
		self.rank_embed = nn.Embedding(num_embeddings=14, embedding_dim=rank_embed_dim, padding_idx=0)
		self.suit_embed = nn.Embedding(num_embeddings=5, embedding_dim=suit_embed_dim, padding_idx=0)
		self.card_embed = nn.Embedding(num_embeddings=53, embedding_dim=card_embed_dim, padding_idx=0)
		
		# Process non-card state.
		self.non_card_fc = nn.Sequential(
			nn.Linear(non_card_input_dim, hidden_dim),
			nn.ReLU(),
			nn.Linear(hidden_dim, hidden_dim),
			nn.ReLU(),
			nn.Linear(hidden_dim, hidden_dim),
			nn.ReLU()
		)
		
		#card_embedding_dim = rank_embed_dim + suit_embed_dim + card_embed_dim
		#total_card_dim = num_cards * card_embedding_dim
		#total_dim = hidden_dim + total_card_dim

		total_dim = hidden_dim + num_cards * (rank_embed_dim + suit_embed_dim + card_embed_dim)

		self.fc = nn.Sequential(
			nn.Linear(total_dim, hidden_dim),
			nn.ReLU(),
			nn.Linear(hidden_dim, hidden_dim),
			nn.ReLU(),
			nn.Linear(hidden_dim, hidden_dim),
			nn.ReLU(),
			nn.Linear(hidden_dim, hidden_dim),
			nn.ReLU(),
			nn.Linear(hidden_dim, num_actions)  # Output a single scalar EV.
		)
	
	def forward(self, non_card_state, rank_data, suit_data, card_data):
		"""
		Args:
			non_card_state: Tensor of shape (batch_size, non_card_input_dim)
			rank_data: Tensor of shape (batch_size, num_cards) with values in 0-13
			suit_data: Tensor of shape (batch_size, num_cards) with values in 0-4
			card_data: Tensor of shape (batch_size, num_cards) with values in 0-52
		Returns:
			probabilities: Tensor of shape (batch_size, num_actions) representing action probabilities.
		"""
		non_card_out = self.non_card_fc(non_card_state)
		
		# Get card embeddings.
		rank_embeds = self.rank_embed(rank_data.long())
		suit_embeds = self.suit_embed(suit_data.long())
		card_embeds = self.card_embed(card_data.long())
		# Concatenate embeddings for each card.
		card_combined = torch.cat([rank_embeds, suit_embeds, card_embeds], dim=-1)
		# Flatten the embeddings so the network knows the positions.
		card_flat = card_combined.view(card_combined.size(0), -1)
		
		combined = torch.cat([non_card_out, card_flat], dim=-1)
		logits = self.fc(combined)
		
		# Apply softmax to obtain probability distribution over actions
		log_probs = torch.log_softmax(logits, dim=-1)
		
		return log_probs'
'''