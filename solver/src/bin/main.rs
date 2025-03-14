use postflop_solver::*;
use std::fs::File;
use std::io::{self, BufRead};
use std::path::Path;
//use rand::Rng;
use std::collections::HashSet;
use std::collections::HashMap;
use serde::{Serialize, Deserialize};
use std::io::Write;

fn read_lines<P>(filename: P) -> io::Result<Vec<String>>
where
    P: AsRef<Path>,
{
    let file = File::open(filename)?;
    let reader = io::BufReader::new(file);
    reader.lines().collect()
}

fn read_file_content<P>(filename: P) -> io::Result<String>
where
    P: AsRef<Path>,
{
    std::fs::read_to_string(filename).map(|s| s.trim().to_string())
}

#[derive(Serialize, Deserialize)]
struct OutputNode {
    history_human: String,  // Represents the sequence of actions leading to this node
    history_internal: Vec<usize>,
    ip_range: HashMap<String, f32>,
    oop_range: HashMap<String, f32>,
    strategy: HashMap<String, HashMap<String, f32>>, // Nested HashMap for strategies
    children: HashMap<String, OutputNode>, // Maps action strings to child nodes
    in_position: bool, // Whether this is for an in-position player or not
}

// Assuming PostFlopGame is a defined struct and other functions are present
fn dfs_helper(game: &mut PostFlopGame, node: &mut OutputNode, seen: &mut HashSet<String>) {
    //println!("Node History: {:?}", node.history_human);
    

    if game.is_chance_node() || game.is_terminal_node() {
        //println!("Current History: {:?}", node.history_human);
        return;
    }
    seen.insert(node.history_human.clone());

    let initial_history = node.history_internal.clone();
    //println!("Current History: {:?}", node.history_human);

    let cur_actions = game.available_actions();
    let mut cur_actions_str: Vec<String> = Vec::new();

    // Formatting action strings
    for a in &cur_actions {
        cur_actions_str.push(format!("{:?}", a));
    }

    for a in 0..cur_actions.len() {
        // Apply history with a mutable reference to the game
        game.apply_history(&initial_history);

        let action_str = cur_actions_str[a].clone();
        let new_history_human = node.history_human.clone() + "->" + &action_str;
        let mut new_history = node.history_internal.clone();
        new_history.push(a);

        // Apply history again with the new history

        //println!("Current history saved: {:?}", game.history());
        //println!("theo new history: {:?}, {:?}", new_history, new_history_human);
        game.apply_history(&new_history);
        //println!("real new history: {:?}", game.history());
        if game.is_chance_node() || game.is_terminal_node() {
            continue;
        }

        let child_node = gen_output(game, &new_history_human);
        //println!("Child Node History: {:?}", child_node.history_human);
        node.children.insert(action_str.clone(), child_node);

        // Pass a mutable reference of child_node to dfs_helper
        dfs_helper(game, &mut node.children.get_mut(&action_str).unwrap(), seen);
    }
}

fn dfs(game: &mut PostFlopGame) -> OutputNode {
    let initial_history = "".to_string();
    let mut output = gen_output(game, &initial_history);
    let mut seen: HashSet<String> = HashSet::new();
    dfs_helper(game, &mut output, &mut seen);
    output
}

fn gen_output(game: &PostFlopGame, human_history: &String) -> OutputNode {
    //println!("In gen output: {:?}", human_history);
    // Assuming game.weights() and game.private_cards() return the correct types
    let current_ip_weights = game.weights(1);
    let current_oop_weights = game.weights(0);

    // Assuming holes_to_strings can return a Result<String, SomeError> or similar
    let private_ip = match holes_to_strings(game.private_cards(1)) {
        Ok(cards) => cards,
        Err(_err) => {
            //println!("Error converting private IP cards: {}", err);
            return OutputNode {
                history_human: "".to_string(),
                history_internal: Vec::new(),
                ip_range: HashMap::new(),
                oop_range: HashMap::new(),
                strategy: HashMap::new(),
                in_position: false,
                children: HashMap::new(),
            };
        }
    };

    let private_oop = match holes_to_strings(game.private_cards(0)) {
        Ok(cards) => cards,
        Err(_err) => {
            //println!("Error converting private OOP cards: {}", err);
            return OutputNode {
                history_human: "".to_string(),
                history_internal: Vec::new(),
                ip_range: HashMap::new(),
                oop_range: HashMap::new(),
                strategy: HashMap::new(),
                in_position: false,
                children: HashMap::new(),
            };
        }
    };

    let mut cur_ip_range: HashMap<String, f32> = HashMap::new();
    let mut cur_oop_range: HashMap<String, f32> = HashMap::new();

    // Populating IP range map
    for (i, card) in private_ip.iter().enumerate() {
        cur_ip_range.insert(card.clone(), current_ip_weights[i]);
    }

    // Populating OOP range map
    for (i, card) in private_oop.iter().enumerate() {
        cur_oop_range.insert(card.clone(), current_oop_weights[i]);
    }

    // Getting the available actions
    let cur_actions = game.available_actions();
    let mut cur_actions_str: Vec<String> = Vec::new();

    // Formatting action strings
    for a in &cur_actions {
        cur_actions_str.push(format!("{:?}", a));
    }

    // Assuming strategy is a vector with probabilities, for each action and hand combination
    let cur_strategy = game.strategy();
    let mut formatted_strategy: HashMap<String, HashMap<String, f32>> = HashMap::new();

    let num_hands = game.private_cards(game.current_player()).len();

    // Select the relevant range based on the current player
    let rel_range = if game.current_player() == 0 {
        private_oop
    } else {
        private_ip
    };

    // Populating strategy for each hand
    for j in 0..num_hands {
        let mut hand_map = HashMap::new();
        for i in 0..cur_actions.len() {
            let idx = num_hands * i + j;
            let prob = cur_strategy[idx];  // Ensure `cur_strategy` returns a vector of f64
            hand_map.insert(cur_actions_str[i].clone(), prob);
        }
        formatted_strategy.insert(rel_range[j].clone(), hand_map);
    }

    // Creating the OutputNode
    OutputNode {
        history_human: human_history.clone(),
        history_internal: game.history().to_vec(),
        ip_range: cur_ip_range,
        oop_range: cur_oop_range,
        strategy: formatted_strategy,
        in_position: game.current_player() == 1,
        children: HashMap::new(),
    }
}

fn write_output_json(node: &OutputNode, filename: &str) -> std::io::Result<()> {
    // Convert the node to a pretty JSON string.
    let json_str = serde_json::to_string_pretty(node)
        .expect("Failed to serialize OutputNode to JSON");

    // Create or truncate the file.
    let mut file = File::create(filename)?;
    // Write the JSON string into the file.
    file.write_all(json_str.as_bytes())?;
    Ok(())
}

fn main() -> io::Result<()> {
    let lines = read_lines("../solver_input_files/input.txt")?;
    if lines.len() < 4 {
        //println!("Error: input.txt must contain at least 4 lines (OOP file, IP file, flop, starting pot)");
        return Ok(());
    }

    let oop_range = read_file_content(&lines[0])?;
    let ip_range = read_file_content(&lines[1])?;
    let board = &lines[2];
    let starting_pot: i32 = lines[3].parse().expect("Invalid starting pot value");
    let effective_stack: i32 = lines[4].parse().expect("Invalid effective stack");
    let output_path = &lines[5];
    let bet_sizes_str = &lines[6];
    let raise_sizes_str = &lines[7];
    //limited bet sizing for later streets to simplify actions for flop solves
    let turn_bet_sizes_str = &lines[8]; 
    let turn_raise_sizes_str = &lines[9]; 
    let river_bet_sizes_str = &lines[10];
    let river_raise_sizes_str = &lines[11]; 
    // Parsing the flop to check how many cards are dealt
    let board_cards: Vec<&str> = board.split_whitespace().collect();

    let bet_sizes = BetSizeCandidates::try_from((bet_sizes_str.as_str(), raise_sizes_str.as_str())).unwrap();
    let bet_sizes_turn = BetSizeCandidates::try_from((turn_bet_sizes_str.as_str(), turn_raise_sizes_str.as_str())).unwrap();
    let bet_sizes_river = BetSizeCandidates::try_from((river_bet_sizes_str.as_str(), river_raise_sizes_str.as_str())).unwrap();

    // Set up the card config based on the number of cards in the flop
    let card_config = if board_cards.len() == 3 {
        // If there are exactly 3 cards in the flop
        CardConfig {
            range: [oop_range.parse().unwrap(), ip_range.parse().unwrap()],
            flop: flop_from_str(&board_cards[..3].join("")).unwrap(),
            turn: NOT_DEALT,   // No turn yet
            river: NOT_DEALT,  // No river yet
        }
    } else if board_cards.len() == 4 {
        // If there are exactly 4 cards, set the turn
        CardConfig {
            range: [oop_range.parse().unwrap(), ip_range.parse().unwrap()],
            flop: flop_from_str(&board_cards[..3].join("")).unwrap(),
            turn: card_from_str(board_cards[3]).unwrap(),
            river: NOT_DEALT,
        }
    } else if board_cards.len() == 5 {
        // If there are exactly 5 cards, set the river
        CardConfig {
            range: [oop_range.parse().unwrap(), ip_range.parse().unwrap()],
            flop: flop_from_str(&board_cards[..3].join(" ")).unwrap(),
            turn: card_from_str(board_cards[3]).unwrap(),
            river: card_from_str(board_cards[4]).unwrap(),
        }
    } else {
        // Handle unexpected flop lengths
        //println!("Error: Invalid number of cards in the flop.");
        return Ok(());
    };

    let tree_config = if board_cards.len() == 3 {
        TreeConfig {
            initial_state: BoardState::Flop,
            starting_pot,
            effective_stack,
            rake_rate: 0.0,
            rake_cap: 0.0,
            flop_bet_sizes: [bet_sizes.clone(), bet_sizes.clone()],
            turn_bet_sizes: [bet_sizes_turn.clone(), bet_sizes_turn.clone()],
            river_bet_sizes: [bet_sizes_river.clone(), bet_sizes_river],
            turn_donk_sizes: None,
            river_donk_sizes: Some(DonkSizeCandidates::try_from("50%").unwrap()),
            add_allin_threshold: 1.5,
            force_allin_threshold: 0.15,
            merging_threshold: 0.1,
        }
    } else if board_cards.len() == 4 {
        TreeConfig {
            initial_state: BoardState::Turn,
            starting_pot,
            effective_stack,
            rake_rate: 0.0,
            rake_cap: 0.0,
            flop_bet_sizes: [bet_sizes.clone(), bet_sizes.clone()],
            turn_bet_sizes: [bet_sizes.clone(), bet_sizes.clone()],
            river_bet_sizes: [bet_sizes.clone(), bet_sizes],
            turn_donk_sizes: None,
            river_donk_sizes: Some(DonkSizeCandidates::try_from("50%").unwrap()),
            add_allin_threshold: 1.5,
            force_allin_threshold: 0.15,
            merging_threshold: 0.1,
        }
    } else if board_cards.len() == 5 {
        TreeConfig {
            initial_state: BoardState::River,
            starting_pot,
            effective_stack,
            rake_rate: 0.0,
            rake_cap: 0.0,
            flop_bet_sizes: [bet_sizes.clone(), bet_sizes.clone()],
            turn_bet_sizes: [bet_sizes.clone(), bet_sizes.clone()],
            river_bet_sizes: [bet_sizes.clone(), bet_sizes],
            turn_donk_sizes: None,
            river_donk_sizes: Some(DonkSizeCandidates::try_from("50%").unwrap()),
            add_allin_threshold: 1.5,
            force_allin_threshold: 0.15,
            merging_threshold: 0.1,
        }
    } else {
        // Handle unexpected flop lengths
        //println!("Error: Invalid number of cards in the flop.");
        return Ok(());
    };

    
    

    // build the game tree
    let action_tree = ActionTree::new(tree_config).unwrap();
    let mut game = PostFlopGame::with_config(card_config, action_tree).unwrap();

    //allocate with compression
    game.allocate_memory(true);

    // solve the game
    let max_num_iterations = 1000;
    let target_exploitability = game.tree_config().starting_pot as f32 * 0.025; // 2.5% of the pot
    solve(&mut game, max_num_iterations, target_exploitability, true);
    //let _exploitability = solve(&mut game, max_num_iterations, target_exploitability, true);

    let processed_game:OutputNode = dfs(&mut game);
    let _ = write_output_json(&processed_game, output_path);
    Ok(())
}
