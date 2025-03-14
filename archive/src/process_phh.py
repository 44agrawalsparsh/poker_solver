import argparse

def parse_hand_history(filename):
    """
    Reads a hand history file that contains Python-style assignments.
    Returns a dictionary of the variables.
    """
    with open(filename, 'r') as f:
        data = f.read()
    data = data.replace('false', 'False').replace('true', 'True')
    # Use a safe dictionary to execute the file content.
    history = {}
    exec(data, {}, history)
    return history

def split_cards(card_str):
    """
    Splits a string of concatenated card codes into a list.
    Assumes each card is represented by two characters (e.g. Qh, Jd).
    """
    return [card_str[i:i+2] for i in range(0, len(card_str), 2)]

def interpret_action(action, players_map):
    """
    Converts an action string into a human readable sentence.
    """
    tokens = action.split()
    
    if tokens[0] == 'd':
        # Dealer actions start with 'd'
        if tokens[1] == 'dh':  # deal hole cards
            # tokens: d dh p1 QhJd
            player_code = tokens[2]
            player = players_map.get(player_code, player_code)
            cards_list = split_cards(tokens[3])
            return f"Dealer deals hole cards to {player}: {', '.join(cards_list)}"
        elif tokens[1] == 'db':  # deal board cards
            # tokens: d db 2h9s9c (or similar)
            cards_list = split_cards(tokens[2])
            return f"Dealer deals board cards: {', '.join(cards_list)}"
        else:
            return f"Unknown dealer action: {' '.join(tokens)}"
    
    else:
        # Player actions start with a player code (p1, p2, etc.)
        player_code = tokens[0]
        player = players_map.get(player_code, player_code)
        action_code = tokens[1]
        
        if action_code == 'cbr':
            # Player bets/raises/calls with an amount
            amount = tokens[2]
            return f"{player} bets {amount}"
        elif action_code == 'cc':
            # Player calls or checks
            return f"{player} calls or checks"
        elif action_code == 'sm':
            # Player shows their cards
            cards_list = split_cards(tokens[2])
            return f"{player} shows cards: {', '.join(cards_list)}"
        elif action_code == 'f':
            return f"{player} folds"
        else:
            return f"Unknown player action: {' '.join(tokens)}"

def interpret_hand_history_file(filename):
    """
    Loads a hand history file and returns a human interpretable summary.
    """
    history = parse_hand_history(filename)
    
    # Map player tokens (p1, p2, etc.) to the actual player names.
    players = history.get("players", [])
    players_map = {}
    if len(players) >= 1:
        players_map["p1"] = players[0]
    if len(players) >= 2:
        players_map["p2"] = players[1]
    
    # Build a summary output.
    output_lines = []
    output_lines.append("Game Setup:")
    output_lines.append(f"  Variant: {history.get('variant', 'Unknown')}")
    output_lines.append(f"  Ante Trimming Status: {history.get('ante_trimming_status', 'Unknown')}")
    output_lines.append(f"  Antes: {history.get('antes', [])}")
    output_lines.append(f"  Blinds/Straddles: {history.get('blinds_or_straddles', [])}")
    output_lines.append(f"  Minimum Bet: {history.get('min_bet', 'Unknown')}")
    output_lines.append(f"  Starting Stacks: {history.get('starting_stacks', [])}")
    output_lines.append("")
    output_lines.append("Actions:")
    
    actions = history.get("actions", [])
    for act in actions:
        output_lines.append("  " + interpret_action(act, players_map))
    
    return "\n".join(output_lines)

def main():
    # Setup command-line argument parsing
    parser = argparse.ArgumentParser(description="Parse and summarize a poker hand history file.")
    parser.add_argument("filename", help="The path to the hand history file.")
    
    # Parse arguments
    args = parser.parse_args()
    
    # Interpret the hand history file
    summary = interpret_hand_history_file(args.filename)
    
    # Print the result
    print(summary)

if __name__ == "__main__":
    main()
