import numpy as np
import os
from utils import Position
import pdb
import json

def get_all_hands():
    ''' 
    helper to get all the possible hands
    '''
    ranks = 'AKQJT98765432'
    output = []
    for i in range(len(ranks)):
        for j in range(i, len(ranks)):
            base = f"{ranks[i]}{ranks[j]}"
            if i == j:
                output.append(base)
            else:
                output.append(base + 'o')
                output.append(base + 's')
    return output

HANDS = get_all_hands()


def process_range_file(filepath):
    with open(filepath, 'r') as f:
        data = f.read()
        data = data.split(',')
        output = {card:float(prob_str) for card,prob_str in [val.split(':') for val in data]}
        return output
    raise Exception('smth wrong')



class DecisionNode:
    def __init__(self, parent, decision):
        self.parent = parent #a strategy node
        self.decision = decision

        self.cur_position = Position(2)
        if self.parent != None:
            self.cur_position = self.parent.position

        self.child = None

    def get_player_order(self):
        output = [(self.cur_position.value + i) % 6 for i in range(6)]
        output = [Position(o).name for o in output]
        return output
    
    def set_child(self, child):
        self.child = child
    

    
    def process_range(self, position, fp):
        self.parent.process_range(position, fp, self.decision)

    def assemble_dict(self):
        output = {}
        output["Decision"] = self.decision
        if self.child:
            output["next_action"] = self.child.assemble_dict()
        return output

class StrategyNode:

    def __init__(self, parent, position):
        self.parent = parent
        self.position = position

    def set_up(self, options):
        self.options = options

        self.processed = {action : False for action in options}

        self.strategies = {}
        for option in options:
            strat = {hand : 0 for hand in HANDS}
            self.strategies[option] = strat
        fold_strat = {hand : 1 for hand in HANDS}
        self.strategies["fold"] = fold_strat

        self.children = {decision : DecisionNode(self, decision) for decision in self.strategies}

    def process_range(self, position, fp, decision):
        if self.position == position:
            assert decision != "fold", "seeing a fold decision have a range"
            if self.processed[decision] == True:
                return
            try:
                data = process_range_file(fp)
            except Exception:
                return
            for hand, prob in data.items():
                #make sure we update our strategy info
                self.strategies[decision][hand] = prob
                self.strategies["fold"][hand] -= prob
            self.processed[decision] = True

        else:
            #keep back tracking until we find the responsible decision
            self.parent.process_range(position, fp)

    def formulate_strategy(self):
        strategy = {}
        for hand in HANDS:
            strategy[hand] = {}
            for action in self.strategies:
                strategy[hand][action] = min(max(0,self.strategies[action][hand]),1)
            strategy_sum = sum(strategy[hand].values())
            strategy[hand] = {a : val/strategy_sum for a,val in strategy[hand].items()}
        return strategy

    def assemble_dict(self):
        output = {}
        output["position"] = self.position.name
        output["strategy"] = self.formulate_strategy()
        output["child_nodes"] = {decision:child.assemble_dict() for decision,child in self.children.items()}
        return output


def reformat_directory(cur_path, order_to_act):
    #pdb.set_trace()
    order_to_act = order_to_act[::-1]
    sub_folders = {entry for entry in os.listdir(cur_path)
                   if os.path.isdir(os.path.join(cur_path, entry)) and '.txt' not in entry and entry[0] != '.'}
    
    # Filter order_to_act to include only folders that exist in cur_path
    order_to_act = [folder for folder in order_to_act if folder in sub_folders]
    
    # Iterate over consecutive pairs of folders in the order
    for i in range(len(order_to_act) - 1):
        current_folder = order_to_act[i]
        next_folder = order_to_act[i + 1]
        
        src = os.path.join(cur_path, current_folder)
        # Define the destination: move current_folder to next_folder/fold/current_folder
        dest_fold_dir = os.path.join(cur_path, next_folder, "fold")
        dest = os.path.join(dest_fold_dir, current_folder)
        
        # Create the 'fold' directory if it doesn't exist
        os.makedirs(dest_fold_dir, exist_ok=True)
        
        # Move current_folder to the newly created 'fold' directory in next_folder
        os.rename(src, dest)
        print(f"Moved {src} -> {dest}")


    
def build_tree(cur_path, cur_node):
    try:
        files_in_dir = os.listdir(cur_path)
    except FileNotFoundError:
        #print(f"{cur_path} doesn't exist")
        return

    range_files = [x for x in files_in_dir if '.txt' in x and 'info' not in x and x[0] != '.']
    sub_dirs = [x for x in files_in_dir if '.txt' not in x and x[0] != '.']

    if isinstance(cur_node, DecisionNode):
        #pass up the range files
        for file in range_files:
            position = Position[file.split('.')[0]]
            fp = os.path.join(cur_path, file)
            cur_node.process_range(position, fp)
        #process the subdirectories
        ''' 
        First is moving around folders for this to be logical

        the file structure has a lot of implicit steps not present. This starts from the get go where we jump to position based off who opens.
        A more logical tree would be:

        UTG 
        -> action1
        -> action2
        -> fold
        ---> HJ
        -------> action1
        -------> action2
        -------> fold
        -------------> CO

        etc

        Right now we have

        UTG 
        -> action1
        -> action2
        -> fold
        HJ
        -> action1
        -> action2
        -> fold
        CO
        -> action1
        -> action2
        -> fold

        We can generate the former by doing the following:

        In the second last to act create a fold subdirectory, underwhich we move the last to act
        keep repeating this until we just have the first to act left
        
        
        '''
        order = cur_node.get_player_order()
        reformat_directory(cur_path, order)

        files_in_dir = os.listdir(cur_path)
        sub_dirs = [x for x in files_in_dir if '.txt' not in x and x[0] != '.']
        if len(sub_dirs):
            new_node = StrategyNode(cur_node, Position[sub_dirs[0]])
            new_path = os.path.join(cur_path, sub_dirs[0])
            cur_node.set_child(new_node)
            build_tree(new_path, new_node)


    elif isinstance(cur_node, StrategyNode):
        #identify all the decisions, set up and move forward
        options = sub_dirs
        cur_node.set_up(options)
        for decision in cur_node.children:
            new_path = os.path.join(cur_path, decision)
            new_node = cur_node.children[decision]
            build_tree(new_path, new_node)

        



def generate_strategy(root_path, out_path):
    '''
    Description:
        Generates strategy from preflop ranges (must be complete)
    Parameters:
        root_path(string) - file path to root of ranges directory
        out_path(string) - where we would like to store our end strategy (will be a JSON file)
    '''

    root_node = DecisionNode(None, "Game Start")
    build_tree(root_path, root_node)
    strategy = root_node.assemble_dict()

    with open(out_path, "w") as f:
        json.dump(strategy, f, indent=4)

if __name__ == "__main__":
    generate_strategy('ranges/preflop_ranges', 'ranges/processed_preflop.json')