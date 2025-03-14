from game_env import GameEnv

def get_blind_values():
	while True:
		try:
			sb = int(input("Enter the small blind (sb) value: "))
			bb = int(input("Enter the big blind (bb) value: "))
			if bb > sb:
				return sb, bb
			else:
				print("Big blind (bb) must be larger than small blind (sb). Please try again.")
		except ValueError:
			print("Invalid input. Please enter integer values.")

if __name__ == "__main__":
	sb, bb = get_blind_values()
	game = GameEnv(sb=sb, bb=bb)

	while True:
		game.play_round()