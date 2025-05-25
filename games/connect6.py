import datetime
import math
import pathlib

import numpy
import torch

from .abstract_game import AbstractGame


class MuZeroConfig:
    def __init__(self):
        # fmt: off
        # More information is available here: https://github.com/werner-duvaud/muzero-general/wiki/Hyperparameter-Optimization

        self.seed = 0  # Seed for numpy, torch and the game
        self.max_num_gpus = None  # Fix the maximum number of GPUs to use. It's usually faster to use a single GPU (set it to 1) if it has enough memory. None will use every GPUs available



        ### Game
        self.observation_shape = (3, 19, 19)  # Dimensions of the game observation, must be 3 (channel, height, width). For a 1D array, please reshape it to (1, 1, length of array)
        self.action_space = list(range(19 * 19))  # Fixed list of all possible actions. You should only edit the length
        self.players = list(range(2))  # List of players. You should only edit the length
        self.stacked_observations = 0  # Number of previous observations and previous actions to add to the current observation

        # Evaluate
        self.muzero_player = 0  # Turn Muzero begins to play (0: MuZero plays first, 1: MuZero plays second)
        self.opponent = "random"  # Hard coded agent that MuZero faces to assess his progress in multiplayer games. It doesn't influence training. None, "random" or "expert" if implemented in the Game class



        ### Self-Play
        self.num_workers = 2  # Number of simultaneous threads/workers self-playing to feed the replay buffer
        self.selfplay_on_gpu = False
        self.max_moves = 361  # Maximum number of moves if game is not finished before
        self.num_simulations = 200  # Number of future moves self-simulated
        self.discount = 1  # Chronological discount of the reward
        self.temperature_threshold = None  # Number of moves before dropping the temperature given by visit_softmax_temperature_fn to 0 (ie selecting the best action). If None, visit_softmax_temperature_fn is used every time

        # Root prior exploration noise
        self.root_dirichlet_alpha = 0.3
        self.root_exploration_fraction = 0.25

        # UCB formula
        self.pb_c_base = 19652
        self.pb_c_init = 1.25



        ### Network
        self.network = "resnet"  # "resnet" / "fullyconnected"
        self.support_size = 10  # Value and reward are scaled (with almost sqrt) and encoded on a vector with a range of -support_size to support_size. Choose it so that support_size <= sqrt(max(abs(discounted reward)))
        
        # Residual Network
        self.downsample = False  # Downsample observations before representation network, False / "CNN" (lighter) / "resnet" (See paper appendix Network Architecture)
        self.blocks = 6  # Number of blocks in the ResNet
        self.channels = 128  # Number of channels in the ResNet
        self.reduced_channels_reward = 2  # Number of channels in reward head
        self.reduced_channels_value = 2  # Number of channels in value head
        self.reduced_channels_policy = 4  # Number of channels in policy head
        self.resnet_fc_reward_layers = [64]  # Define the hidden layers in the reward head of the dynamic network
        self.resnet_fc_value_layers = [64]  # Define the hidden layers in the value head of the prediction network
        self.resnet_fc_policy_layers = [64]  # Define the hidden layers in the policy head of the prediction network
        
        # Fully Connected Network
        self.encoding_size = 32
        self.fc_representation_layers = []  # Define the hidden layers in the representation network
        self.fc_dynamics_layers = [64]  # Define the hidden layers in the dynamics network
        self.fc_reward_layers = [64]  # Define the hidden layers in the reward network
        self.fc_value_layers = []  # Define the hidden layers in the value network
        self.fc_policy_layers = []  # Define the hidden layers in the policy network



        ### Training
        self.results_path = pathlib.Path(__file__).resolve().parents[1] / "results" / pathlib.Path(__file__).stem / datetime.datetime.now().strftime("%Y-%m-%d--%H-%M-%S")  # Path to store the model weights and TensorBoard logs
        self.save_model = True  # Save the checkpoint in results_path as model.checkpoint
        self.training_steps = 100000  # Total number of training steps (ie weights update according to a batch)
        self.batch_size = 64  # Number of parts of games to train on at each training step
        self.checkpoint_interval = 100 # Number of training steps before using the model for self-playing
        self.value_loss_weight = 1  # Scale the value loss to avoid overfitting of the value function, paper recommends 0.25 (See paper appendix Reanalyze)
        self.train_on_gpu = torch.cuda.is_available()  # Train on GPU if available

        self.optimizer = "Adam"  # "Adam" or "SGD". Paper uses SGD
        self.weight_decay = 1e-4  # L2 weights regularization
        self.momentum = 0.9  # Used only if optimizer is SGD

        # Exponential learning rate schedule
        self.lr_init = 0.002  # Initial learning rate
        self.lr_decay_rate = 0.9  # Set it to 1 to use a constant learning rate
        self.lr_decay_steps = 10000



        ### Replay Buffer
        self.replay_buffer_size = 10000  # Number of self-play games to keep in the replay buffer
        self.num_unroll_steps = 10  # Number of game moves to keep for every batch element
        self.td_steps = 20  # Number of steps in the future to take into account for calculating the target value
        self.PER = True  # Prioritized Replay (See paper appendix Training), select in priority the elements in the replay buffer which are unexpected for the network
        self.PER_alpha = 0.5  # How much prioritization is used, 0 corresponding to the uniform case, paper suggests 1

        # Reanalyze (See paper appendix Reanalyse)
        self.use_last_model_value = True  # Use the last model to provide a fresher, stable n-step value (See paper appendix Reanalyze)
        self.reanalyse_on_gpu = False



        ### Adjust the self play / training ratio to avoid over/underfitting
        self.self_play_delay = 0  # Number of seconds to wait after each played game
        self.training_delay = 0  # Number of seconds to wait after each training step
        self.ratio = 1  # Desired training steps per self played step ratio. Equivalent to a synchronous version, training can take much longer. Set it to None to disable it
        # fmt: on

    def visit_softmax_temperature_fn(self, trained_steps):
        """
        Parameter to alter the visit count distribution to ensure that the action selection becomes greedier as training progresses.
        The smaller it is, the more likely the best action (ie with the highest visit count) is chosen.

        Returns:
            Positive float.
        """
        if trained_steps < 0.5 * self.training_steps:
            return 1.0
        elif trained_steps < 0.75 * self.training_steps:
            return 0.5
        else:
            return 0.25


class Game(AbstractGame):
    """
    Game wrapper.
    """
 
    def __init__(self, seed=None):
        self.env = GomokuSix()

    def step(self, action):
        """
        Apply action to the game.

        Args:
            action : action of the action_space to take.

        Returns:
            The new observation, the reward and a boolean if the game has ended.
        """
        observation, reward, done = self.env.step(action)
        return observation, reward, done

    def to_play(self):
        """
        Return the current player.

        Returns:
            The current player, it should be an element of the players list in the config.
        """
        return self.env.to_play()

    def legal_actions(self):
        """
        Should return the legal actions at each turn, if it is not available, it can return
        the whole action space. At each turn, the game have to be able to handle one of returned actions.

        For complex game where calculating legal moves is too long, the idea is to define the legal actions
        equal to the action space but to return a negative reward if the action is illegal.

        Returns:
            An array of integers, subset of the action space.
        """
        return self.env.legal_actions()

    def reset(self):
        """
        Reset the game for a new game.

        Returns:
            Initial observation of the game.
        """
        return self.env.reset()

    def close(self):
        """
        Properly close the game.
        """
        pass

    def render(self):
        """
        Display the game observation.
        """
        self.env.render()
        input("Press enter to take a step ")

    def human_to_action(self):
        """
        For multiplayer games, ask the user for a legal action
        and return the corresponding action number.

        Returns:
            An integer from the action space.
        """
        valid = False
        while not valid:
            valid, action = self.env.human_input_to_action()
        return action

    def action_to_string(self, action):
        """
        Convert an action number to a string representing the action.
        Args:
            action_number: an integer from the action space.
        Returns:
            String representing the action.
        """
        return self.env.action_to_human_input(action)
import math
import numpy

class GomokuSix:
    def __init__(self):
        self.board_size = 19
        self.board = numpy.zeros((self.board_size, self.board_size), dtype="int32")
        self.current_player_id = 1  # Player 1 starts (represented by 1)
        self.game_move_count = 0  # Total stones placed in the game
        self.stones_placed_this_turn = 0  # Stones placed by the current player in their current logical turn
        self.winning_length = 6

        self.board_markers = [
            chr(x) for x in range(ord("A"), ord("A") + self.board_size)
        ]

    def to_play(self):
        """Returns the player index (0 or 1) for MuZero."""
        return 0 if self.current_player_id == 1 else 1

    def reset(self):
        self.board = numpy.zeros((self.board_size, self.board_size), dtype="int32")
        self.current_player_id = 1
        self.game_move_count = 0
        self.stones_placed_this_turn = 0
        return self.get_observation()

    def step(self, action):
        if action is None or not isinstance(action, (int, numpy.integer)):
            raise ValueError("Invalid action received in GomokuSix.step")

        x = math.floor(action / self.board_size)
        y = action % self.board_size

        if self.board[x][y] != 0:
            print(f"Warning: Illegal move attempted at ({x},{y}), already occupied. Agent should use legal_actions.")
            pass # Or raise error

        self.board[x][y] = self.current_player_id
        self.game_move_count += 1
        self.stones_placed_this_turn += 1

        won, _ = self._check_win_condition(x, y, self.current_player_id)
        
        reward = 0
        done = False

        if won:
            reward = 1  # Current player made a winning move
            done = True
        elif not self.legal_actions(): # No more legal moves possible (board full)
            reward = 0  # Draw
            done = True
        
        # If the game is not over, decide if the turn switches
        if not done:
            switch_to_next_player = False
            if self.game_move_count == 1:  # First player's first move (plays 1 stone)
                switch_to_next_player = True
            else:
                # For subsequent turns, players play 2 stones
                if self.stones_placed_this_turn == 2:
                    switch_to_next_player = True
            
            if switch_to_next_player:
                self.current_player_id *= -1  # Switch player
                self.stones_placed_this_turn = 0 # Reset for new player's turn
            # Else, current player continues (they've placed 1 of 2 stones)

        return self.get_observation(), reward, done

    def _check_win_condition(self, r, c, player):
        """Checks if the player who just moved to (r,c) won."""
        if player == 0: # Should not happen, player is 1 or -1
            return False, []
            
        directions = [(0, 1), (1, 0), (1, 1), (1, -1)] # Horizontal, Vertical, Diagonal /, Diagonal \
        for dr, dc in directions:
            count = 1 # Count the stone just placed
            # Check in one direction
            for i in range(1, self.winning_length):
                nr, nc = r + dr * i, c + dc * i
                if 0 <= nr < self.board_size and 0 <= nc < self.board_size and self.board[nr][nc] == player:
                    count += 1
                else:
                    break
            # Check in the opposite direction
            for i in range(1, self.winning_length):
                nr, nc = r - dr * i, c - dc * i
                if 0 <= nr < self.board_size and 0 <= nc < self.board_size and self.board[nr][nc] == player:
                    count += 1
                else:
                    break
            
            if count >= self.winning_length:
                return True, [] # Winning positions not strictly needed for MuZero, but could be implemented
        return False, []

    def get_observation(self):
        board_player1 = numpy.where(self.board == 1, 1.0, 0.0)
        board_player2 = numpy.where(self.board == -1, 1.0, 0.0)
        board_to_play = numpy.full((self.board_size, self.board_size), self.current_player_id, dtype="float32") 
        return numpy.array([board_player1, board_player2, board_to_play])

    def legal_actions(self):
        legal = []
        for i in range(self.board_size):
            for j in range(self.board_size):
                if self.board[i][j] == 0:
                    legal.append(i * self.board_size + j)
        return legal

    def render(self):
        marker = "   " # Adjusted for row numbers
        for i in range(self.board_size):
            marker = marker + self.board_markers[i] + " "
        print(marker)
        for row in range(self.board_size):
            # print(f"{self.board_markers[row]:<2}", end=" ") # Row labels
            print(f"{row+1:<2}", end=" ") # Row numbers 1-11
            for col in range(self.board_size):
                ch = self.board[row][col]
                if ch == 0:
                    print(".", end=" ")
                elif ch == 1:
                    print("X", end=" ") # Player 1
                elif ch == -1:
                    print("O", end=" ") # Player 2
            print()
        
        current_player_symbol = "X" if self.current_player_id == 1 else "O"
        stones_to_play_now = 1 if self.game_move_count == 0 or (self.game_move_count > 0 and self.stones_placed_this_turn == 0 and self.game_move_count !=1) else (2-self.stones_placed_this_turn if self.game_move_count >0 else 1)
        if self.game_move_count == 0 : stones_to_play_now = 1
        elif self.game_move_count > 0 and self.stones_placed_this_turn ==1 and self.game_move_count !=1 : stones_to_play_now =1
        elif self.game_move_count > 0 and self.stones_placed_this_turn ==0 and self.game_move_count !=1 : stones_to_play_now =2

        print(f"Player to move: {current_player_symbol} (Player ID: {self.current_player_id}, MuZero Index: {self.to_play()})")
        print(f"Total stones on board: {self.game_move_count}")
        print(f"Stones placed by {current_player_symbol} this logical turn: {self.stones_placed_this_turn}")
        if self.game_move_count == 0:
             print(f"{current_player_symbol} plays 1 stone this turn.")
        elif self.stones_placed_this_turn <2 and self.game_move_count !=1 :
             print(f"{current_player_symbol} plays {2-self.stones_placed_this_turn} more stone(s) this turn.")


    def human_input_to_action(self):
        human_input = input(f"Enter action (e.g., A1, K11) for player {('X' if self.current_player_id == 1 else 'O')} : ").strip().upper()
        if not (2 <= len(human_input) <= 3):
            print("Invalid input length. Format: ColumnRow, e.g., 'CA' for col C, row A or 'C1' for Col C, Row 1.")
            return False, -1

        col_char = human_input[0]
        row_str = human_input[1:]

        if col_char not in self.board_markers:
            print(f"Invalid column character. Use A-{self.board_markers[-1]}.")
            return False, -1
        
        try:
            # Assuming row_str is like "A" or "1"
            if row_str in self.board_markers: # e.g. AA, AB
                 y = ord(col_char) - ord("A") # Col
                 x = ord(row_str) - ord("A") # Row
            elif row_str.isdigit(): # e.g. A1, K11
                 y = ord(col_char) - ord("A") # Col
                 x = int(row_str) - 1       # Row (0-indexed)
            else:
                print(f"Invalid row input. Use A-{self.board_markers[-1]} or 1-{self.board_size}.")
                return False,-1

            if not (0 <= x < self.board_size and 0 <= y < self.board_size):
                 print("Coordinates out of bounds.")
                 return False, -1
                 
            if self.board[x][y] == 0:
                # The action is (row * board_size) + col
                # Input is typically (col, row) like "C5" -> col C, row 5
                # Here, x is row_idx, y is col_idx
                return True, x * self.board_size + y
            else:
                print("Cell already occupied.")
                return False, -1
        except ValueError:
            print(f"Invalid row number. Use 1-{self.board_size}.")
            return False, -1
        
    def action_to_human_input(self, action):
        if action is None: return "None"
        x = math.floor(action / self.board_size) # row index
        y = action % self.board_size            # col index
        # Convert to 1-based row and char-based col for human readability
        return f"{self.board_markers[y]}{x+1}"
