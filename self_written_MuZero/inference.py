#!/usr/bin/env python
"""GTP engine for Connect‑6 driven by a **trained MuZero model** (single‑stone
encoding). The script wraps the lightweight GTP loop you provided and replaces
random move generation with MuZero inference + MCTS. It is compatible with
Ludii's GTP bridge.

Run example
-----------
python gtp_muzero_connect6.py \
    --model checkpoints/muzero_connect6_10000.pt \
    --board_size 19 \
    --num_simulations 100  # MCTS playouts per stone
"""

import argparse
import sys
import logging
from pathlib import Path

import numpy as np
import torch
import random 

# --------------------------- logging setup ---------------------------------
Path("./logs").mkdir(exist_ok=True)
logging.basicConfig(
    filename="./logs/inference.log",
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
#  >>> Import MuZero implementation <<<
# ---------------------------------------------------------------------------
try:
    from train import MuZeroNet, MCTS, Connect6Env, TurnInfo
except ImportError as e:
    raise ImportError("Cannot import MuZero components; ensure train.py is in PYTHONPATH") from e

# -----------------------------  MuZero Agent  ------------------------------

class MuZeroAgent:
    def __init__(self, model_path: str, board_size: int, device: str = "cpu", num_sim: int = 100):
        self.device = torch.device(device)
        self.board_size = board_size
        self.num_sim = num_sim
        self.net = MuZeroNet(action_size=board_size * board_size, board_size=board_size, device=device)
        self.net.load_state_dict(torch.load(model_path, map_location=self.device))
        self.net.eval()

    # ------------------------------------------------------------------
    def _to_env(self, board: np.ndarray, turn: int, stones_left: int, last_move):
        env = Connect6Env(self.board_size)
        # 適配 1/2 或 1/-1 編碼
        env.board[:] = 0
        env.board[board == 1] = 1
        if (board == 2).any():
            env.board[board == 2] = -1
        else:  # 若外部已用 -1
            env.board[board == -1] = -1
        env.moves = int((board != 0).sum())
        env.turn = TurnInfo(player=turn - 1, stones_left=stones_left)
        env.last_move = last_move
        return env

    # ------------------------------------------------------------------
    def select_stone(self, board: np.ndarray, turn: int, stones_left: int, last_move):
        env = self._to_env(board, turn, stones_left, last_move)
        legal = np.array(env.legal_actions())
        if legal.size == 0:
            logger.error("No legal moves (board full?) turn=%d", turn)
            r, c = np.random.randint(self.board_size, size=2)
            return int(r), int(c)

        obs = env._observation()
        policy = MCTS(self.net, env, self.num_sim).run(obs)
        # 再次遮非法 action
        mask = np.zeros_like(policy, dtype=bool)
        mask[legal] = True
        policy = np.where(mask, policy, 0)
        if policy.sum() == 0:
            logger.warning("Policy all zero after masking; choosing uniformly")
            action = int(np.random.choice(legal))
        else:
            policy /= policy.sum()
            action = int(np.random.choice(env.action_size, p=policy))
        if action not in legal:
            logger.error("Sampled illegal action %d; falling back to random legal", action)
            action = int(np.random.choice(legal))
        return divmod(action, self.board_size)

# -----------------------------  GTP Engine  --------------------------------

class Connect6Game:
    def __init__(self, agent: MuZeroAgent, size: int = 19):
        self.size = size
        self.board = np.zeros((size, size), dtype=int)  # 0 empty, 1 black, 2 white
        self.turn = 1
        self.stones_left = 1
        self.last_move = None
        self.game_over = False
        self.agent = agent

    # ---------- coord helpers (skip 'I') ----------
    @staticmethod
    def index_to_label(col):
        return chr(ord('A') + col + (1 if col >= 8 else 0))

    @staticmethod
    def label_to_index(ch):
        ch = ch.upper()
        return ord(ch) - ord('A') - (1 if ch >= 'J' else 0)

    # ---------------------------------------------
    def _stone_list_to_str(self, stones):
        return ",".join(f"{self.index_to_label(c)}{r+1}" for r, c in stones)

    def _place(self, r, c, colour):
        if self.board[r, c] != 0:
            raise ValueError("occupied")
        self.board[r, c] = 1 if colour == 'B' else 2
        self.last_move = (r, c)

    # ---------------- GTP commands ----------------
    def generate_move(self, colour):
        if self.game_over:
            print("? Game over", flush=True); return
        stones = []
        for _ in range(self.stones_left):
            r, c = self.agent.select_stone(self.board, self.turn, self.stones_left, self.last_move)
            if self.board[r, c] != 0:  # 最後防線
                logger.error("MuZero chose occupied (%d,%d) — picking random", r, c)
                empties = np.argwhere(self.board == 0)
                r, c = map(int, random.choice(empties))
            self._place(r, c, colour.upper())
            stones.append((r, c))
            self.stones_left -= 1
        self.turn, self.stones_left = 3 - self.turn, 2
        move_str = self._stone_list_to_str(stones)
        print(f"= {move_str}\n", flush=True)
        print(move_str, file=sys.stderr)

    # -------- other minimal GTP handlers (boardsize, play, etc.) --------
    def play_move(self, colour, coord):
        col = self.label_to_index(coord[0])
        row = int(coord[1:]) - 1
        self._place(row, col, colour.upper())
        self.turn, self.stones_left = 3 - self.turn, 2
        print("=", flush=True)

    def reset_board(self):
        self.__init__(self.agent, self.size)
        print("=", flush=True)

    def process_command(self, line):
        parts = line.strip().split()
        if not parts: return
        cmd = parts[0].lower()
        if cmd == "boardsize":
            self.__init__(self.agent, int(parts[1])); print("=", flush=True)
        elif cmd == "clear_board":
            self.reset_board()
        elif cmd == "play":
            self.play_move(parts[1], parts[2])
        elif cmd == "genmove":
            self.generate_move(parts[1])
        elif cmd == "quit":
            print("=", flush=True); sys.exit(0)
        else:
            print("?", flush=True)

    def run(self):
        while (line := sys.stdin.readline()):
            self.process_command(line)


# ------------------------------  CLI  ---------------------------------------

def parse_args():
    ap = argparse.ArgumentParser(description="Run MuZero‑powered Connect‑6 GTP engine")
    ap.add_argument("--model", default="./checkpoints/test/muzero_connect6_1500.pt", help="Path to trained MuZero checkpoint (.pt)")
    ap.add_argument("--board_size", type=int, default=19)
    ap.add_argument("--device", choices=["cpu", "cuda"], default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--num_simulations", type=int, default=400)
    return ap.parse_args()

# -------------------------------- main --------------------------------------

if __name__ == "__main__":
    args = parse_args()
    agent = MuZeroAgent(args.model, args.board_size, device=args.device, num_sim=args.num_simulations)
    game = Connect6Game(agent, size=args.board_size)
    game.run()
