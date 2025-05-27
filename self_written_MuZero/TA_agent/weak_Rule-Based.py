import sys
import numpy as np
import random

class Connect6Game:
    def __init__(self, size=19):
        self.size = size
        self.board = np.zeros((size, size), dtype=int)  # 0: Empty, 1: Black, 2: White
        self.turn = 1  # 1: Black, 2: White
        self.game_over = False
        self.last_opponent_move = None

    def reset_board(self):
        """Resets the board and game state."""
        self.board.fill(0)
        self.turn = 1
        self.game_over = False
        print("= ", flush=True)
        return
    def set_board_size(self, size):
        """Changes board size and resets the game."""
        self.size = size
        self.board = np.zeros((size, size), dtype=int)
        self.turn = 1
        self.game_over = False
        print("= ", flush=True)

    def check_win(self):
        """Checks if a player has won. Returns 1 (Black wins), 2 (White wins), or 0 (no winner)."""
        directions = [(0, 1), (1, 0), (1, 1), (1, -1)]
        for r in range(self.size):
            for c in range(self.size):
                if self.board[r, c] != 0:
                    current_color = self.board[r, c]
                    for dr, dc in directions:
                        prev_r, prev_c = r - dr, c - dc
                        if 0 <= prev_r < self.size and 0 <= prev_c < self.size and self.board[prev_r, prev_c] == current_color:
                            continue
                        count = 0
                        rr, cc = r, c
                        while 0 <= rr < self.size and 0 <= cc < self.size and self.board[rr, cc] == current_color:
                            count += 1
                            rr += dr
                            cc += dc
                        if count >= 6:
                            return current_color
        return 0

    def index_to_label(self, col):
        """Converts a column index to a letter (skipping 'I')."""
        return chr(ord('A') + col + (1 if col >= 8 else 0))

    def label_to_index(self, col_char):
        """Converts a column letter to an index (handling the missing 'I')."""
        col_char = col_char.upper()
        if col_char >= 'J':  # 'I' is skipped
            return ord(col_char) - ord('A') - 1
        else:
            return ord(col_char) - ord('A')

    def play_move(self, color, move):
        """Processes a move and updates the board."""
        if self.game_over:
            print("? Game over")
            return

        stones = move.split(',')
        positions = []

        for stone in stones:
            stone = stone.strip()
            col_char = stone[0].upper()
            col = self.label_to_index(col_char)
            row = int(stone[1:]) - 1
            if not (0 <= row < self.size and 0 <= col < self.size) or self.board[row, col] != 0:
                print("? Invalid move")
                return
            positions.append((row, col))

        for row, col in positions:
            self.board[row, col] = 1 if color.upper() == 'B' else 2

        self.last_opponent_move = positions[-1]  # Track the opponent's last move
        self.turn = 3 - self.turn
        print('= ', end='', flush=True)

    def generate_move(self, color):
        """Generates the best move based on predefined rules and ensures output."""
        if self.game_over:
            print("? Game over", flush=True)
            return
            
        if np.count_nonzero(self.board) == 0:
            middle = self.size // 2
            move_str = f"{self.index_to_label(middle)}{middle+1}"
            self.play_move(color, move_str)
            
            print(move_str, flush=True)
            
            print(move_str, file=sys.stderr)

            return
        my_color = 1 if color.upper() == 'B' else 2
        opponent_color = 3 - my_color
        empty_positions = [(r, c) for r in range(self.size) for c in range(self.size) if self.board[r, c] == 0]
        
        if random.uniform(0, 1) < 0.2:
            selected = random.choice(empty_positions)
            move_str = f"{self.index_to_label(selected[1])}{selected[0]+1}"
            self.play_move(color, move_str)
            print(move_str, flush=True)
            return
        
        # 1. Winning move
        for r, c in empty_positions:
            self.board[r, c] = my_color
            if self.check_win() == my_color:
                self.board[r, c] = 0
                move_str = f"{self.index_to_label(c)}{r+1}"
                self.play_move(color, move_str)
                print(move_str, flush=True)
                return
            self.board[r, c] = 0

        # 2. Block opponent's winning move
        for r, c in empty_positions:
            self.board[r, c] = opponent_color
            if self.check_win() == opponent_color:
                self.board[r, c] = 0
                move_str = f"{self.index_to_label(c)}{r+1}"
                self.play_move(color, move_str)
                print(move_str, flush=True)
                return
            self.board[r, c] = 0
        
        # 3. Attack: prioritize strong formations
        best_move = None
        best_score = 0
        for r, c in empty_positions:
            score = self.evaluate_position(r, c, my_color)
            if score > best_score:
                best_score = score
                best_move = (r, c)

        # 4. Defense: prevent opponent from forming strong positions
        for r, c in empty_positions:
            opponent_score = self.evaluate_position(r, c, opponent_color)
            if opponent_score > best_score:
                best_score = opponent_score
                best_move = (r, c)

        # 5. Execute best move
        if best_move:
            r, c = best_move
            move_str = f"{self.index_to_label(c)}{r+1}"
            self.play_move(color, move_str)
            print(move_str, flush=True)
            return

        # 6. Default move: play near last opponent move
        if self.last_opponent_move:
            
            last_r, last_c = self.last_opponent_move
            potential_moves = [(r, c) for r in range(max(0, last_r - 2), min(self.size, last_r + 3))
                                           for c in range(max(0, last_c - 2), min(self.size, last_c + 3))
                                           if self.board[r, c] == 0]
            if potential_moves:
                selected = random.choice(potential_moves)
                move_str = f"{self.index_to_label(selected[1])}{selected[0]+1}"
                self.play_move(color, move_str)
                print(move_str, flush=True)
                return

        # 7. Random move as fallback
        selected = random.choice(empty_positions)
        move_str = f"{self.index_to_label(selected[1])}{selected[0]+1}"
        self.play_move(color, move_str)
        print(move_str, flush=True)
        return
    def evaluate_position(self, r, c, color):
        """Evaluates the strength of a position based on alignment potential."""
        directions = [(0, 1), (1, 0), (1, 1), (1, -1)]
        score = 0

        for dr, dc in directions:
            count = 1
            rr, cc = r + dr, c + dc
            while 0 <= rr < self.size and 0 <= cc < self.size and self.board[rr, cc] == color:
                count += 1
                rr += dr
                cc += dc
            rr, cc = r - dr, c - dc
            while 0 <= rr < self.size and 0 <= cc < self.size and self.board[rr, cc] == color:
                count += 1
                rr -= dr
                cc -= dc

            if count >= 5:
                score += 10000
            elif count == 4:
                score += 5000
            #elif count == 3:
            #    score += 1000
            #elif count == 2:
            #    score += 100
            
    
        return score
    def show_board(self):
        """Displays the board in text format."""
        print("= ")
        for row in range(self.size - 1, -1, -1):
            line = f"{row+1:2} " + " ".join("X" if self.board[row, col] == 1 else "O" if self.board[row, col] == 2 else "." for col in range(self.size))
            print(line)
        col_labels = "   " + " ".join(self.index_to_label(i) for i in range(self.size))
        print(col_labels)
        print(flush=True)

    def list_commands(self):
        """Lists all available commands."""
        print("= ", flush=True)  

    def process_command(self, command):
        """Parses and executes GTP commands."""
        command = command.strip()
        if command == "get_conf_str env_board_size:":
            print("env_board_size=19", flush=True)

        if not command:
            return
        
        parts = command.split()
        cmd = parts[0].lower()

        if cmd == "boardsize":
            try:
                size = int(parts[1])
                self.set_board_size(size)
            except ValueError:
                print("? Invalid board size")
        elif cmd == "clear_board":
            self.reset_board()
        elif cmd == "play":
            if len(parts) < 3:
                print("? Invalid play command format")
            else:
                self.play_move(parts[1], parts[2])
                print('', flush=True)
        elif cmd == "genmove":
            if len(parts) < 2:
                print("? Invalid genmove command format")
            else:
                self.generate_move(parts[1])
        elif cmd == "showboard":
            self.show_board()
        elif cmd == "list_commands":
            self.list_commands()
        elif cmd == "quit":
            print("= ", flush=True)
            sys.exit(0)
        else:
            print("? Unsupported command")

    def run(self):
        """Main loop that reads GTP commands from standard input."""
        while True:
            try:
                line = sys.stdin.readline()
                if not line:
                    break
                self.process_command(line)
            except KeyboardInterrupt:
                break
            except Exception as e:
                print(f"? Error: {str(e)}")

if __name__ == "__main__":
    game = Connect6Game()
    game.run()
