#####################################
# CREDITS
#####################################


#####################################
# UI INIT
#####################################

# Clearscreen
import os

def clearscreen():
    if os.name == 'nt':
        os.system('cls') 
    else:
        os.system('clear')

# UI
from rich.table import Table
from rich.console import Console
from rich.panel import Panel
from rich.status import Status
from rich.text import Text
from rich.style import Style
from rich.rule import Rule

# Console
CONSOLE = Console()

# Splash
def splash_screen():
    title_style = Style(**{ 'bold': True, 'color': 'deep_sky_blue1' })
    subtitle_style = Style(**{ 'italic': True })

    panel_text = Text(text = 'AI Powered Tic-Tac-Toe', justify='center')
    title_text = Text(text = 'Debunking the "AI-Powered" Fuzz', style=title_style)
    subtitle_text = Text(text = 'by Renaud Bidou', style=subtitle_style)

    panel = Panel(
        panel_text, 
        title = title_text, 
        subtitle = subtitle_text,
        padding=1
    )

    CONSOLE.print(panel)

clearscreen()
splash_screen()

#####################################
# IMPORTS
#####################################

import math
from pprint import pprint
from copy import deepcopy
import argparse
from time import time
import random
import pickle
import json

#####################################
# GLOBALS
#####################################

#
# ANN
#

# ANN STRUCTURE
NN_INPUT_SIZE = 18
NN_HIDDEN_SIZE = 10
NN_OUTPUT_SIZE = 9

INIT_ANN = {
    'weights_ih': [],
    'weights_ho': [],
    'biases_h': [],
    'biases_o': [],

    'inputs': [],
    'hidden': [],
    'raw_logits': [],
    'outputs': []
}

# LOADED ANN

LOADED_ANNS = [ None, None ]

LOADED_ANN_TEMPLATE = {
    'filename': None,
    'type': None, # 0: ANN, 1: Deterministic ANN
    'name': '',
    'stats': {
        'wins': 0,
        'losses': 0,
        'ties': 0
    },
    'params': {}
}

ANN_PARAMS = {
    'hidden_layers': [ '# Hidden Layers', 'int', 10 ],
    'reward_draw': [ 'Reward for Draw', 'float', 0.3 ],
    'reward_win': [ 'Reward for Win', 'float', 1 ],
    'reward_lost': [ 'Reward for Loss', 'float', -2 ],
    'learning_rate': [ 'Learning Rate', 'float', 0.1 ],
    'move_rank_factor': [ 'Last Moves Higher Weight', 'bool', True ],
    'learn_from_winner': [ 'Learn fom Winner', 'bool', False ]

}

ANN_TYPES = [
    'Neural Network',
    'Deterministic AI'
]

# RUNNING ANNS
ANNS = [ None, None ]

# TRAIN PARAMS
DRAW_REWARD = 0.3
WIN_REWARD = 1
LOST_REWARD = -2
LEARNING_RATE = 0.1

#
# GAME PARAMS
#

INIT_GAMESTATE = {
    'board': [ '.' for i in range(9) ],
    'current_player': 0,
    'winner': None
}

GAMESTATE = None

DRAW_WINNER = 2

PLAYERS = [ None, None ] # 0: Human / 1: ANN / 2: Random / 3: Trainer / 4: Deterministic AI
PLAYERS_TYPES = {
    0: 'Human',
    1: 'Neural Network',
    2: 'Random',
    3: 'Trainer',
    4: 'Deterministic AI'
}
SYMBOLS = [ 'X', 'O' ]

MOVE_HISTORY = []

#
# PROGRAM PARAMS
#

# DEBUG
DEBUG = False

# PLAYING / TRAINING
STAGE = 0
GAME_MODE = None # 0: Training / 1: Game / 2: View AI / 9: Exit
GAMES_NUMBER = 150000
DEFAULT_TRAINING_GAMES = 150000

# FILES
MODELS_FILE = 'models.json'
MODELS_DIR = 'models/'

# PRNG
RANDOM_SEED = int(time())

#####################################
# GAME PARAMS
#####################################

# CLI arguments

def parse_args():

    global DEBUG

    parser = argparse.ArgumentParser()

    parser.add_argument('--debug', action='store_true', help='Display internal ANN data')

    args = parser.parse_args()

    if args.debug:
        DEBUG = True

# Game mode

def select_game_mode():

    global GAME_MODE, STAGE

    STAGE = 0
    GAME_MODE = None
    
    display_stage('Select Game Mode')

    table = Table()

    table.add_column('#')
    table.add_column('Game Mode')

    table.add_row('0', 'Training')
    table.add_row('1', 'Just play games')
    table.add_row('2', 'View AIs details')
    table.add_row('9', 'Exit')


    CONSOLE.print(table)
    CONSOLE.print()

    while GAME_MODE is None:
        mode_input = CONSOLE.input('Enter game mode: ')
        mode_input = mode_input.strip()
        if mode_input.isdigit():
            if int(mode_input) in [0, 1, 2, 9]:
                GAME_MODE = int(mode_input)

# Players

def select_players():

    global PLAYERS

    PLAYERS = [ None, None ]

    for player_index in range(len(PLAYERS)):

        allowed_players = [ i for i in PLAYERS_TYPES ]
        if GAME_MODE == 0:
            allowed_players.remove(0)

            # Enforce having an AI player to train
            if player_index == 1 and not PLAYERS[0] in [1,4]:
                allowed_players = [1, 4]
                

        display_stage(f'Select Player {player_index+1}')

        table = Table()
        table.add_column('#')
        table.add_column('Player Type')
        for player_type, player_label in PLAYERS_TYPES.items():
            if not player_type in allowed_players:
                continue
            table.add_row(str(player_type), player_label)
            
        CONSOLE.print(table)

        while PLAYERS[player_index] is None:

            CONSOLE.print()
            player_input = CONSOLE.input('Enter player type: ')
            player_input = player_input.strip()
            if player_input.isdigit():
                if int(player_input) in allowed_players:
                    PLAYERS[player_index] = int(player_input)

        if is_ai(player_index):
            select_ann(player_index)

    CONSOLE.print()
    CONSOLE.print(f'[>>>] Player 1: {PLAYERS_TYPES[PLAYERS[0]]}')
    CONSOLE.print(f'[>>>] Player 2: {PLAYERS_TYPES[PLAYERS[1]]}')

# ANNs

def select_ann(player_index):

    player_type = PLAYERS[player_index]

    CONSOLE.print()
    CONSOLE.print('[*] Select AI')
    CONSOLE.print()

    if player_type == 1:
        ann_type = 0
    elif player_type == 4:
        ann_type = 1

    ANNS[player_index] = load_ann(player_index, ann_type)

# Game number

def set_games_number():

    global GAMES_NUMBER

    games_number = None

    if 0 in PLAYERS:
        games_number = 1

    else:
        display_stage('Set Number of Training Games')

        while games_number is None:
            games_input = CONSOLE.input(f'[>>>] Games played for training ({DEFAULT_TRAINING_GAMES}): ')
            games_input = games_input.strip()
            if len(games_input) == 0:
                games_number = DEFAULT_TRAINING_GAMES
            elif games_input.isdigit():
                games_number = int(games_input)

    GAMES_NUMBER = games_number
        
#####################################
# INITIALIZATION
#####################################

def check_files_and_dir():

    if not os.path.exists(MODELS_FILE):
        with open(MODELS_FILE, 'w') as f:
            json.dump([], f)

    if not os.path.exists(MODELS_DIR):
        os.mkdir(MODELS_DIR)

def init_game():

    return deepcopy(INIT_GAMESTATE)

#####################################################################
# ANN
#
# Code below is 99% Python port 
# of Tic Tac Toe with Reinforcement Learning
# by antirez (https://github.com/antirez/ttt-rl/blob/main/ttt.c)
#
#####################################################################

# Utils

def relu(x):
    result = x if x > 0 else 0
    return float(result)

def relu_derivative(x):
    result = 1 if x > 0 else 0
    return float(result)

def softmax(inputs):

    max_val = max(inputs)

    sum_inputs = 0

    outputs = []
    exp_inputs = []

    for i in range(len(inputs)):
        input_value = inputs[i]
        exp_input = math.exp(input_value - max_val)
        exp_inputs.append(exp_input)
        sum_inputs += exp_input

    if sum_inputs > 0:
        outputs = [ exp_inputs[i] / sum_inputs for i in range(len(inputs)) ]
    else:
        outputs = [ 1/len(inputs) for i in range(len(inputs))]

    return outputs

def random_weight():
    return random.random() - 0.5

# Init
 
def init_neural_networks(ann_index):

    random.seed(int(time()+ann_index))

    ann = deepcopy(INIT_ANN)
    ann_data = LOADED_ANNS[ann_index]
    nn_hidden_size = ann_data['params']['hidden_layers']

    ann['weights_ih'] = [ random_weight() for i in range( NN_INPUT_SIZE * nn_hidden_size) ]
    ann['weights_ho'] = [ random_weight() for i in range( nn_hidden_size * NN_OUTPUT_SIZE) ]
    ann['biases_h'] = [ random_weight() for i in range(nn_hidden_size) ]
    ann['biases_o'] = [ random_weight() for i in range(NN_OUTPUT_SIZE) ]

    ann['inputs'] = [ 0 for i in range(NN_INPUT_SIZE) ]
    ann['hidden'] = [ 0 for i in range(nn_hidden_size)]
    ann['raw_logits'] = [ 0 for i in range(NN_OUTPUT_SIZE) ]
    ann['outputs'] = [ 0 for i in range(NN_OUTPUT_SIZE) ]

    if DEBUG:
        print('### DEBUG: ANN Initialization')
        pprint(ann)
        input()

    return ann

# Core mechanisms

def forward_pass(inputs, player = None):

    global ANNS

    current_player = GAMESTATE['current_player'] if player is None else player
    ann = ANNS[current_player]
    ann_params = LOADED_ANNS[current_player]['params']
    nn_hidden_size = ann_params['hidden_layers']

    ann['inputs'] = inputs

    # Input to hidden layer
    for i in range(nn_hidden_size):
        hidden_sum = ann['biases_h'][i]
        for j in range(NN_INPUT_SIZE):
            hidden_sum += inputs[j] * ann['weights_ih'][ j * nn_hidden_size + i ] 
        ann['hidden'][i] = relu(hidden_sum)

    # Hidden layer to output
    for i in range(NN_OUTPUT_SIZE):
        ann['raw_logits'][i] = ann['biases_o'][i]
        for j in range(nn_hidden_size):
            ann['raw_logits'][i] +=  ann['hidden'][j] * ann['weights_ho'][ j * NN_OUTPUT_SIZE +i ]

    ann['outputs'] = softmax(ann['raw_logits'])

    if DEBUG:
        print(f'### DEBUG: ANN output - Player {current_player}')
        pprint(ann['outputs'])
        input()

    ANNS[current_player] = ann

def board_inputs(game):

    inputs = [ 0 for i in range(len(game['board']) * 2) ]

    for i in range(len(game['board'])):

        box_value = game['board']

        # Empty box: 00
        if box_value == '.':
            inputs[i * 2] = 0
            inputs[i * 2 + 1] = 0

        # X : 10
        elif box_value == 'X':
            inputs[i * 2] = 1
            inputs[i * 2 + 1] = 0

        # O : 01
        elif box_value == 'O':
            inputs[i * 2] = 0
            inputs[i * 2 + 1] = 1

    return inputs

def back_propagation(ann_index, target_probs, learning_rate, reward_scaling):

    global ANNS

    ann = ANNS[ann_index]
    ann_params = LOADED_ANNS[ann_index]['params']
    nn_hidden_size = ann_params['hidden_layers']

    output_deltas = [ 0 for i in range(NN_OUTPUT_SIZE) ]
    hidden_deltas = [ 0 for i in range(nn_hidden_size) ]

    #
    # Calculate delta
    #

    for i in range(NN_OUTPUT_SIZE):
        output_deltas[i] = ann['outputs'][i] - target_probs[i] * abs(reward_scaling)

    # Backpropagation to hidden layer
    for i in range(nn_hidden_size):
        error = 0
        for j in range(NN_OUTPUT_SIZE):
            error += output_deltas[j] * ann['weights_ho'][ i * NN_OUTPUT_SIZE +j ]
        hidden_deltas[i] = error * relu_derivative(ann['hidden'][i])

    #
    # Update weights
    #

    # Output layer weight and biases
    for i in range(nn_hidden_size):
        for j in range(NN_OUTPUT_SIZE):
            ann['weights_ho'][i] -= learning_rate * output_deltas[j] * ann['hidden'][i]

    for i in range(NN_OUTPUT_SIZE):
        ann['biases_o'][i] -= learning_rate * output_deltas[i]

    # Hidden layers weight and biases
    for i in range(NN_INPUT_SIZE):
        for j in range(nn_hidden_size):
            ann['weights_ih'][i] -= learning_rate * hidden_deltas[j] * ann['inputs'][i]

    for i in range(nn_hidden_size):
        ann['biases_h'][i] -= learning_rate * hidden_deltas[i]

def learn_from_game():

    for ann_index in range(len(ANNS)):

        ann = ANNS[ann_index]

        if ann is None:
            continue

        ann_params = LOADED_ANNS[ann_index]['params']

        num_moves = len(MOVE_HISTORY)

        player_index = ann_index

        if GAMESTATE['winner'] == DRAW_WINNER:
            if LOADED_ANNS[ann_index]['params']['learn_from_winner']:
                continue
            reward = DRAW_REWARD
        elif GAMESTATE['winner'] == player_index:
            reward = WIN_REWARD
        else:
            reward = LOST_REWARD

        
        learn_index = player_index

        # Learning from opponent if it was the winner
        if LOADED_ANNS[ann_index]['params']['learn_from_winner']:
            opponent_index = 0 if ann_index == 1 else 1
            if GAMESTATE['winner'] == opponent_index:
                learn_index = opponent_index
                reward = WIN_REWARD

        if DEBUG:

            if GAMESTATE['winner'] == DRAW_WINNER:
                print(f'### DEBUG: Player {ann_index} learning from draw')
            elif GAMESTATE['winner'] == ann_index:
                print(f'### DEBUG: Player {ann_index} learning from win')
            else:
                print(f'### DEBUG: Player {ann_index} learning from loss')

        if DEBUG:
            print(f'Learning from {'Player' if learn_index == player_index else 'Opponent'}')

        for move_index in range(num_moves):

            if not move_index %2 == learn_index:
                
                if DEBUG:
                    print(f'Skipping move {move_index} - Other player move')

                continue

            # Recreate game  
            game = init_game()

            for i in range(move_index):
                symbol = 'X' if i % 2 == 0 else 'O'
                game['board'][MOVE_HISTORY[i]] = symbol

            # onvert board to inputs
            inputs = board_inputs(game)
            forward_pass(inputs, ann_index)

            # Move to reward
            move = MOVE_HISTORY[move_index]

            # Move score is impacted by how late in the game it took place
            move_importance = 0.5 + 0.5 * (move_index/num_moves) if ann_params['move_rank_factor'] else 1
            scaled_reward = reward * move_importance

            # Target probability distribution
            target_probs = [ 0 for i in range(NN_OUTPUT_SIZE) ]

            # Positive reward: the move is the only valid one
            if scaled_reward > 0:
                target_probs[move] = 1

            # Negative reward: distribute probabilities to all other valid moves
            else:
                valid_moves_left = 9 - move_index - 1
                other_probs = 1 / valid_moves_left
                for i in range(9):
                    if game['board'][i] == '.':
                        target_probs[i] = other_probs

            if DEBUG:
                print('### DEBUG: target probabilities')
                pprint(target_probs)
                input()

            # Call back propagation
            back_propagation(ann_index, target_probs, ann_params['learning_rate'], scaled_reward)

#####################################
# GAME 
#####################################

def check_game_status(board = None):

    global GAMESTATE

    winner = None

    if board is None:
        board = GAMESTATE['board']

    check_lists = []
        
    # rows
    for i in range(3):
        check_lists.append([ board[ i * 3 + j] for j in range(3)])

    # columns
    for i in range(3):
        check_lists.append([ board[ i + j * 3] for j in range(3)])

    # diagonals
    check_lists.append([ board[ i * 4 ] for i in range(3)])
    check_lists.append([ board[ (i+1) * 2 ] for i in range(3)])

    for check in check_lists:
        if not '.' in check and len(set(check)) == 1:
            winner_symbol = check[0]
            winner = SYMBOLS.index(winner_symbol)
            break

    if winner is None and board.count('.') == 0:
        winner = DRAW_WINNER

    return winner

def play_game():

    global GAMESTATE, MOVE_HISTORY

    if 2 in PLAYERS:
        random.seed(int(time()))

    num_moves = 0
    winner = None
    MOVE_HISTORY = []
    GAMESTATE = init_game()

    while winner is None:

        if DEBUG or 0 in PLAYERS:
            display_board()

        current_player = GAMESTATE['current_player']

        current_player_type = PLAYERS[current_player]

        if current_player_type == 1:
            move = get_ann_move()
        elif current_player_type == 2:
            move = get_random_move()    
        elif current_player_type == 3:
            move = get_trainer_move()
        elif current_player_type == 4:
            move = get_deterministic_ai_move()
        else:
            move = get_human_move(current_player)

        num_moves += 1

        GAMESTATE['board'][move] = SYMBOLS[current_player]
        MOVE_HISTORY.append(move)

        # Check if we have a winner
        winner = check_game_status()

        if winner is None:
            GAMESTATE['current_player'] = 1 if current_player == 0 else 0
        else:
            GAMESTATE['winner'] = winner
        
    if DEBUG or 0 in PLAYERS or GAMES_NUMBER == 1:
        display_board()

        CONSOLE.print()

        if GAMESTATE['winner'] in [0, 1]:
            winner_index = GAMESTATE['winner'] 
            winner_index_label = str(winner_index + 1)
            winner_type_label = PLAYERS_TYPES[PLAYERS[winner_index]]
            CONSOLE.print(f'[>>>] Player {winner_index_label} ({winner_type_label}) won !')
        else:
            CONSOLE.print('[>>>] Draw !!!')

    learn_from_game()

    return GAMESTATE['winner']

def train_session():

    global LOADED_ANNS, PLAYERS_STATS

    PLAYERS_STATS = [ [ 0,0,0 ], [ 0,0,0 ] ]

    display_stage(f'Training Neural Netwoks for {GAMES_NUMBER} games')

    status = Status('Starting training')
    status.start()

    for i in range(GAMES_NUMBER):

        status.update(f'Playing game {i+1} / {GAMES_NUMBER}' )
        result = play_game()
        update_stats(result)

    status.stop()

def games_session():

    global PLAYERS_STATS

    PLAYERS_STATS = [ [ 0,0,0 ], [ 0,0,0 ] ]

    display_stage('Play games')

    play_again = True

    if GAMES_NUMBER == 1:

        while play_again:

            result = play_game()
            update_stats(result)

            CONSOLE.print()
            again_input = CONSOLE.input('Play another game (y/n) ? ')
            again_input = again_input.strip()
            if again_input.lower().startswith('n'):
                play_again = False

    else:

        status = Status('Starting training')
        status.start()

        for i in range(GAMES_NUMBER):

            status.update(f'Playing game {i+1} / {GAMES_NUMBER}' )
            result = play_game()
            update_stats(result)

        status.stop()

def update_stats(result):

    global PLAYERS_STATS

    if result == DRAW_WINNER:

        for player_index in range(len(PLAYERS)):
            if PLAYERS[player_index] in [1,4]:
                LOADED_ANNS[player_index]['stats']['ties'] += 1

        PLAYERS_STATS[0][2] += 1
        PLAYERS_STATS[1][2] += 1

    else:
        winner = result
        loser = 1 if winner == 0 else 0

        if PLAYERS[winner] in [1,4]:
                LOADED_ANNS[winner]['stats']['wins'] += 1
        if PLAYERS[loser] in [1,4]:
            LOADED_ANNS[loser]['stats']['losses'] += 1

        PLAYERS_STATS[winner][0] += 1
        PLAYERS_STATS[loser][1] += 1

#####################################
# PLAYERS MOVES
#####################################

def get_ann_move():

    current_player = GAMESTATE['current_player']
    ann = ANNS[current_player]

    inputs = board_inputs(GAMESTATE)
    forward_pass(inputs)

    max_prob = 0
    best_move = None

    for i in range(len(ann['outputs'])):
        ann_output = ann['outputs'][i]
        if ann_output >= max_prob and GAMESTATE['board'][i] == '.':
            max_prob = ann_output
            best_move = i

    return best_move

def get_human_move(player_index):

    human_move = None

    CONSOLE.print()

    while human_move is None:
        human_input = CONSOLE.input(f'Player {player_index+1} move: ')
        human_input = human_input.strip()
        if len(human_input) == 1 and human_input.isdigit():
            test_move = int(human_input)
            if GAMESTATE['board'][test_move] == '.':
                human_move = test_move

    return human_move

def get_random_move():

    random_move = None

    while random_move is None:

        random_test = random.randint(0, 8)
        if GAMESTATE['board'][random_test] == '.':
            random_move = random_test        

    return random_move

def get_trainer_move():

    move = get_deterministic_move()
    if move is None:
        move = get_random_move()

    return move

def get_deterministic_move():

    play_board = deepcopy(GAMESTATE['board'])
    player = GAMESTATE['current_player']
    adversary = 1 if player == 0 else 0

    move_found = False
    move = None
    
    # Moves structures
    selected_moves = []
    available_moves = []

    # Get available moves
    for box_index in range(len(play_board)):
        if play_board[box_index] == '.':
            available_moves.append(box_index)

    # Check possibility to win
    if not move_found:
        for test_index in available_moves:
            test_board = deepcopy(play_board)
            test_board[test_index] = SYMBOLS[player]
            if check_game_status(test_board) == player:
                selected_moves.append(test_index)
                move_found = True

    # Check possibility to lose
    if not move_found:
        for test_index in available_moves:
            test_board = deepcopy(play_board)
            test_board[test_index] = SYMBOLS[adversary]
            if check_game_status(test_board) == adversary:
                selected_moves.append(test_index)
                move_found = True

    # Play the center box
    if not move_found:
        if 4 in available_moves:
            selected_moves.append(4)
            move_found = True

    #
    # More rules to add
    #
        

    # Get move from selected moves
    if move_found:
        move_index = random.randint(0, len(selected_moves)-1)
        move = selected_moves[move_index]

    return move

def get_deterministic_ai_move():

    move = get_deterministic_move()
    if move is None:
        move = get_ann_move()

    return move

#####################################
# UI
#####################################

def display_stage(stage_text, keep_stage = False):

    global STAGE

    if not keep_stage:
        STAGE += 1

    clearscreen()
    splash_screen()

    CONSOLE.print()
    CONSOLE.print(Rule(f'{STAGE}. {stage_text}', align='left'))
    CONSOLE.print()

def display_board():

    display_stage('Play games', True)

    table = Table(show_header=False, show_lines=True, padding=(1,2) )

    table.add_column()
    table.add_column()
    table.add_column()
    
    for row in range(3):
        table.add_row(GAMESTATE['board'][row*3], GAMESTATE['board'][row*3 + 1], GAMESTATE['board'][row*3 + 2])

    CONSOLE.print(table)

def display_games_stats():
    
    display_stage('Results')

    total_games = sum(PLAYERS_STATS[0])

    CONSOLE.print()
    CONSOLE.print(f'[>>>] Games Played: {total_games}')
    CONSOLE.print()

    table = Table()

    table.add_column('Player')
    table.add_column('Type')
    table.add_column('Wins', justify='right')
    table.add_column('Losses', justify='right')
    table.add_column('Ties', justify='right')

    for player_index in range(len(PLAYERS_STATS)):

        player_type_index = PLAYERS[player_index]
        player_type = PLAYERS_TYPES[player_type_index]
        if is_ai(player_index) and len(LOADED_ANNS[player_index]['name']) > 0: 
            player_type += f' ({LOADED_ANNS[player_index]['name']})'
        player_index_label = str(player_index+1)
        player_stats = PLAYERS_STATS[player_index]
        table.add_row(player_index_label, player_type, str(player_stats[0]), str(player_stats[1]), str(player_stats[2]))

    CONSOLE.print(table)

    CONSOLE.print()
    CONSOLE.input('[>>>] Press any key to continue')
    
def display_anns():

    display_stage('AI Details')

    with open(MODELS_FILE) as f:
        models = json.load(f)


    table = Table(show_lines=True)
    table.add_column('Type')
    table.add_column('Name')
    table.add_column('Games', justify='right')
    table.add_column('Hidden Layers', justify='right')
    table.add_column('Win Reward', justify='right')
    table.add_column('Loss Reward', justify='right')
    table.add_column('Draw Reward', justify='right')
    table.add_column('Learning Rate', justify='right')
    table.add_column('Move Rank Factor')
    table.add_column('Learn from Winner')

    for ann in models:

        type = ANN_TYPES[ann['type']]
        name = ann['name']
        stats = ann['stats']
        games = str(stats['wins']+stats['losses']+stats['ties'])
        params = ann['params']
        hidden_layers = str(params['hidden_layers'])
        reward_win = str(params['reward_win'])
        reward_lost = str(params['reward_lost'])
        reward_draw = str(params['reward_draw'])
        learning_rate = str(params['learning_rate'])
        move_rank_factor = str(params['move_rank_factor'])
        learn_from_winner = str(params['learn_from_winner'])

        table.add_row(
            type,
            name,
            games,
            hidden_layers,
            reward_win,
            reward_lost,
            reward_draw,
            learning_rate,
            move_rank_factor,
            learn_from_winner
        )

    CONSOLE.print(table)
    CONSOLE.print()
    CONSOLE.input('[>>>] Press any key to continue')

#####################################
# SAVE / LOAD
#####################################

def save_anns():

    with open(MODELS_FILE) as f:
        models = json.load(f)

    for ann_index in range(len(ANNS)):

        ann = ANNS[ann_index]

        if ann is None:
            continue

        ann_data = LOADED_ANNS[ann_index]
        ann_filename = ann_data['filename']

        ann_model_index = None

        for model_index in range(len(models)):

            model_data = models[model_index]
            if model_data['filename'] == ann_filename:
                ann_model_index = model_index
                break

        if ann_model_index is None:
            models.append(ann_data)
        else:
            models[ann_model_index] = ann_data

        with open(MODELS_DIR+ann_filename, 'bw') as f:
            pickle.dump(ann, f)

    with open(MODELS_FILE, 'w') as f:
        json.dump(models, f, indent=3)

def load_ann(player_index, ann_type):

    global LOADED_ANNS

    loaded_ann = None
    valid_models = []

    with open(MODELS_FILE) as f:
        models = json.load(f)

    table = Table(show_lines=True)
    table.add_column('Model #')
    table.add_column('Name')
    table.add_column('Games', justify='right')
    table.add_column('Win %', justify='right')
    table.add_column('Loss %', justify='right')
    table.add_column('Tie %', justify='right')

    table.add_row('0', 'New model', '0', '0', '0', '0')

    for model_index in range(len(models)):

        model = models[model_index]

        if not model['type'] == ann_type: 
            continue

        valid_models.append(model_index)

        name = model['name']
        wins = model['stats']['wins']
        losses = model['stats']['losses']
        ties = model['stats']['ties']
        total = wins + losses + ties
        win_ratio = int(100 * wins / total)
        loss_ratio = int(100 * losses / total)
        tie_ratio = int(100 * ties / total)

        table.add_row(str(len(valid_models)), name, str(total), f'{win_ratio}%', f'{loss_ratio}%', f'{tie_ratio}%')

    CONSOLE.print(table)
    print()
    
    selected_model = None

    while selected_model is None:
        model_string = input('Select model: ')
        model_string = model_string.strip()
        if all([
            model_string.isdigit(),
            int(model_string) >= 0,
            int(model_string) <= len(valid_models)
        ]):
            selected_model = int(model_string)

    if selected_model == 0:

        loaded_ann_data = deepcopy(LOADED_ANN_TEMPLATE)
        loaded_ann_data['type'] = ann_type

        loaded_ann_data['filename'] = f'model-{int(time())}' 

        (loaded_ann_name, loaded_ann_params) = setup_ann()

        loaded_ann_data['name'] = loaded_ann_name
        loaded_ann_data['params'] = loaded_ann_params

        LOADED_ANNS[player_index] = loaded_ann_data
        loaded_ann = init_neural_networks(player_index)

    else:

        valid_model_index = selected_model - 1
        model_index = valid_models[valid_model_index]

        model_filename = models[model_index]['filename']

        LOADED_ANNS[player_index] = models[model_index]

        with open(MODELS_DIR+model_filename, 'rb') as f:
            loaded_ann = pickle.load(f)

    return loaded_ann

def setup_ann():

    ann_params = {}
    ann_name = 'My AI'

    CONSOLE.print()
    CONSOLE.print('[*] Setup AI param (<enter> to keep default value)')
    CONSOLE.print()

    input_name = CONSOLE.input('[?] Enter your AI name: ')
    ann_name = input_name.strip()

    for param_key, param_data in ANN_PARAMS.items():

        (param_label, param_type, param_default) = param_data

        param_value = None

        param_default_label = str(param_default)

        if param_type == 'bool':
            param_default_label = 1 if param_default else 0

        while param_value is None:

            input_param = CONSOLE.input(f'[?] {param_label} ({param_default_label}): ')
            input_param = input_param.strip()
            if len(input_param) == 0:
                param_value = param_default
            else:
                param_value = check_param(input_param, param_type)

        ann_params[param_key] = param_value

    return (ann_name, ann_params)

#####################################
# MISC
#####################################

def is_ai(player_index):

    return True if PLAYERS[player_index] in [1, 4] else False 

def check_param(param_value, param_type):

    result = None

    if param_type == 'int' and param_value.isdigit():
        result = int(param_value)
    elif param_type == 'unint' and param_value.replace('-', '', 1).isdigit():
        result = float(param_value)
    elif param_type == 'float' and param_value.replace('-', '', 1).replace('.', '', 1).isdigit():
        result = float(param_value)
    elif param_type == 'bool' and param_value.isdigit() and int(param_value) in [0, 1]:
        result = True if int(param_value) == 1 else False
        
    return result

if __name__ == '__main__':

    parse_args()
    check_files_and_dir()

    exit = False

    while True:

        select_game_mode()

        if GAME_MODE == 9:
            break

        if GAME_MODE == 2:
            display_anns()
            continue

        select_players()
        set_games_number()
        
        if GAME_MODE == 0:
            train_session()
        else:
            games_session()

        display_games_stats()
        save_anns()

    CONSOLE.print()
    CONSOLE.print('[>>>] Exiting')
    CONSOLE.print()


