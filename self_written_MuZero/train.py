#!/usr/bin/env python
"""MuZero training script for Connect‑6 (single‑stone action encoding).

Only **bug‑fixes / correctness patches** applied; architecture & CLI remain
unchanged so your previous experiments stay compatible.
"""

import argparse
import math
import random
import os
from collections import deque, namedtuple
from copy import deepcopy
from pathlib import Path
import matplotlib.pyplot as plt

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

# -----------------------------  ENVIRONMENT  ----------------------------------

TurnInfo = namedtuple("TurnInfo", ["player", "stones_left"])


class Connect6Env:
    """Connect‑6 with single‑stone action interface."""

    def __init__(self, board_size: int = 19):
        self.N = board_size
        self.action_size = board_size * board_size
        self.max_moves = self.action_size  # draw when board is full
        self.reset()

    # ----------------------- internal utilities ----------------------------
    def _check_win(self, r: int, c: int, player: int) -> bool:
        """Return True if *player* has >=6 connected stones including (r,c)."""
        b = self.board
        for dr, dc in ((1, 0), (0, 1), (1, 1), (1, -1)):
            count = 1
            for sgn in (1, -1):
                nr, nc = r + sgn * dr, c + sgn * dc
                while 0 <= nr < self.N and 0 <= nc < self.N and b[nr, nc] == player:
                    count += 1
                    nr += sgn * dr
                    nc += sgn * dc
            if count >= 6:
                return True
        return False

    def _observation(self):
        """4‑plane observation stack (C×H×W)."""
        p = self.turn.player
        my_plane = (self.board == (1 if p == 0 else -1)).astype(np.float32)
        opp_plane = (self.board == (-1 if p == 0 else 1)).astype(np.float32)
        last_plane = np.zeros_like(self.board, dtype=np.float32)
        if self.last_move is not None:
            last_plane[self.last_move] = 1.0
        left_plane = np.full_like(self.board, self.turn.stones_left / 2, dtype=np.float32)
        return np.stack([my_plane, opp_plane, last_plane, left_plane], 0)

    # --------------------------- public API --------------------------------
    def reset(self):
        self.board = np.zeros((self.N, self.N), np.int8)
        self.moves = 0
        self.last_move = None
        self.turn = TurnInfo(0, 1)  # Black, 1 stone first turn
        return self._observation()

    def legal_actions(self):
        return [i for i in range(self.action_size) if self.board[i // self.N, i % self.N] == 0]

    def step(self, action: int):
        r, c = divmod(action, self.N)
        assert self.board[r, c] == 0, "Illegal move"
        player = self.turn.player
        self.board[r, c] = 1 if player == 0 else -1
        self.moves += 1
        self.last_move = (r, c)

        done = False
        reward = 0.0
        if self._check_win(r, c, self.board[r, c]):
            done, reward = True, 1.0
        elif self.moves >= self.max_moves:
            done = True  # draw
        # next turn bookkeeping
        stones_left = self.turn.stones_left - 1
        if stones_left == 0 and not done:
            self.turn = TurnInfo(1 - player, 2)
        else:
            self.turn = TurnInfo(player, stones_left)
        # canonical reward (+1 black win, -1 white win)
        if done and reward:
            reward = reward if player == 0 else -reward
        return self._observation(), reward, done, {}

    @property
    def current_player(self):
        return self.turn.player

# -----------------------------  NETWORK -------------------------------------

def _init_weights(m):
    if isinstance(m, (nn.Conv2d, nn.Linear)):
        nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
        if m.bias is not None:
            nn.init.zeros_(m.bias)


class ResidualBlock(nn.Module):
    def __init__(self, ch: int):
        super().__init__()
        self.conv1 = nn.Conv2d(ch, ch, 3, 1, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(ch)
        self.conv2 = nn.Conv2d(ch, ch, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(ch)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return F.relu(out + x)


class RepresentationNet(nn.Module):
    def __init__(self, in_ch: int, ch: int, n_blocks: int):
        super().__init__()
        self.stem = nn.Conv2d(in_ch, ch, 3, 1, 1, bias=False)
        self.bn = nn.BatchNorm2d(ch)
        self.res = nn.Sequential(*[ResidualBlock(ch) for _ in range(n_blocks)])

    def forward(self, x):
        x = F.relu(self.bn(self.stem(x)))
        return self.res(x)


class DynamicsNet(nn.Module):
    def __init__(self, ch: int, n_blocks: int, action_size: int, board_size: int):
        super().__init__()
        self.act_embed = nn.Embedding(action_size, ch)
        self.conv = nn.Conv2d(ch * 2, ch, 1)
        self.res = nn.Sequential(*[ResidualBlock(ch) for _ in range(n_blocks)])
        flat = ch * board_size * board_size
        self.reward_head = nn.Sequential(
            nn.Conv2d(ch, ch, 3, 1, 1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(flat, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )

    def forward(self, s, a):
        b, _, h, w = s.shape
        a_emb = self.act_embed(a.long()).view(b, -1, 1, 1).expand(-1, -1, h, w)
        x = torch.cat([s, a_emb], 1)
        x = F.relu(self.conv(x))
        x = self.res(x)
        r = self.reward_head(x).squeeze(-1)
        return x, r


class PredictionNet(nn.Module):
    def __init__(self, ch: int, action_size: int, board_size: int):
        super().__init__()
        flat = board_size * board_size
        self.policy_head = nn.Sequential(
            nn.Conv2d(ch, 2, 1),
            nn.Flatten(),
            nn.Linear(2 * flat, action_size),
        )
        self.value_head = nn.Sequential(
            nn.Conv2d(ch, 1, 1),
            nn.Flatten(),
            nn.Linear(flat, 256),
            nn.ReLU(),
            nn.Linear(256, 1),
            nn.Tanh(),
        )

    def forward(self, s):
        return self.policy_head(s), self.value_head(s).squeeze(-1)


class MuZeroNet(nn.Module):
    def __init__(self, action_size: int, board_size: int, ch: int = 128, device: str = "cpu"):
        super().__init__()
        self.device = torch.device(device)
        self.repr = RepresentationNet(4, ch, 8)
        self.dyn = DynamicsNet(ch, 8, action_size, board_size)
        self.pred = PredictionNet(ch, action_size, board_size)
        self.apply(_init_weights)
        self.to(self.device)

    # -------- public inference wrappers (input / output on self.device) -----
    def initial_inference(self, obs):
        obs = obs.to(self.device).float()
        s = self.repr(obs)
        p_logits, v = self.pred(s)
        return s, v, p_logits

    def recurrent_inference(self, s, a):
        a = a.to(self.device)
        next_s, r = self.dyn(s, a)
        p_logits, v = self.pred(next_s)
        return next_s, r, v, p_logits

# ------------------------------  MCTS  ---------------------------------------

class Node:
    def __init__(self, prior, player):
        self.prior = prior
        self.player = player
        self.visit_count = 0
        self.value_sum = 0.0
        self.children = {}

    @property
    def value(self):
        return 0.0 if self.visit_count == 0 else self.value_sum / self.visit_count

    def expanded(self):
        return bool(self.children)


class MCTS:
    def __init__(self, net: MuZeroNet, env: Connect6Env, sims: int, c1=1.25, c2=19652):
        self.net, self.env, self.sims = net, env, sims
        self.c1, self.c2 = c1, c2

    def run(self, obs):
        obs_t = torch.from_numpy(obs).unsqueeze(0).float().to(self.net.device)
        with torch.no_grad():
            s0, v0, logits0 = self.net.initial_inference(obs_t)
        root = Node(0.0, self.env.current_player)
        prior = F.softmax(logits0.squeeze(0), -1).cpu().numpy()
        legal = self.env.legal_actions()
        prior_masked = np.zeros_like(prior)
        prior_masked[legal] = prior[legal]
        if prior_masked.sum() == 0:
            prior_masked[legal] = 1 / len(legal)
        else:
            prior_masked /= prior_masked.sum()
        for a in legal:
            root.children[a] = Node(prior_masked[a], 1 - self.env.current_player)
        root.value_sum = v0.item()

        for _ in range(self.sims):
            node, scratch_env, state = root, deepcopy(self.env), s0.clone()
            search_path, done = [node], False
            while node.expanded():
                action, node = self._select_child(node)
                _, rew, done, _ = scratch_env.step(action)
                if done:
                    break
                a_t = torch.tensor([action], dtype=torch.long, device=self.net.device)
                with torch.no_grad():
                    state, _, v_pred, logits_pred = self.net.recurrent_inference(state, a_t)
                search_path.append(node)

            # Expansion (if not terminal)
            if not done:
                legal = scratch_env.legal_actions()
                prior = F.softmax(logits_pred.squeeze(0), -1).cpu().numpy()
                pm = np.zeros_like(prior)
                pm[legal] = prior[legal]
                if pm.sum() == 0:
                    pm[legal] = 1.0 / len(legal)
                else:
                    pm /= pm.sum()
                for a in legal:
                    node.children[a] = Node(pm[a], 1 - scratch_env.current_player)
                value = v_pred.item()
            else:
                value = rew

            # Backup along the path
            for n in reversed(search_path):
                n.visit_count += 1
                n.value_sum += value if n.player == scratch_env.current_player else -value

        # Policy target = visit count distribution
        visits = np.zeros(self.env.action_size, np.float32)
        for a, child in root.children.items():
            visits[a] = child.visit_count
        total = visits.sum()
        if total == 0:                       
            policy_target = np.zeros_like(visits, dtype=np.float32)
            legal = self.env.legal_actions()
            policy_target[legal] = 1.0 / len(legal)
        else:
            policy_target = visits / total
        return policy_target

    # ---------------- internal helpers ----------------
    def _ucb_score(self, parent, child):
        pb_c = math.log((parent.visit_count + self.c2 + 1) / self.c2) + self.c1
        pb_c *= math.sqrt(parent.visit_count) / (child.visit_count + 1)
        return child.value + pb_c * child.prior

    def _select_child(self, node):
        return max(node.children.items(), key=lambda item: self._ucb_score(node, item[1]))

# ---------------------------  REPLAY & GAME HISTORY --------------------------

class GameHistory:
    def __init__(self):
        self.observations, self.policies, self.rewards = [], [], []

    def add(self, obs, policy, reward):
        self.observations.append(obs)
        self.policies.append(policy)
        self.rewards.append(reward)


class ReplayBuffer:
    def __init__(self, cap):
        self.buffer = deque(maxlen=cap)

    def add(self, g):
        self.buffer.append(g)

    def sample(self, n):
        return random.sample(self.buffer, n)

# ------------------------------ SELF‑PLAY ------------------------------------

def self_play(net, env, sims):
    g = GameHistory()
    obs, done = env.reset(), False
    while not done:
        policy = MCTS(net, env, sims).run(obs)
        action = np.random.choice(env.action_size, p=policy)
        obs, reward, done, _ = env.step(action)
        g.add(obs, policy, reward)
    return g

# ------------------------------ TRAINING -------------------------------------

def make_target(g: GameHistory, idx: int, td_steps: int, discount: float = 1.0):
    value = 0.0
    for i in range(td_steps):
        if idx + i < len(g.rewards):
            value += (discount ** i) * g.rewards[idx + i]
        else:
            break
    return value


def train_step(net, buffer, opt, batch_size, td_steps):
    if len(buffer.buffer) < batch_size:
        return None
    batch = buffer.sample(batch_size)
    obs, policy, value = [], [], []
    for g in batch:
        k = random.randrange(len(g.observations))
        obs.append(g.observations[k])
        policy.append(g.policies[k])
        value.append(make_target(g, k, td_steps))
    obs_t = torch.from_numpy(np.stack(obs)).float().to(net.device)
    policy_t = torch.from_numpy(np.stack(policy)).float().to(net.device)
    value_t = torch.tensor(value, dtype=torch.float32, device=net.device)

    _, pred_v, pred_logits = net.initial_inference(obs_t)
    policy_loss = F.cross_entropy(pred_logits, policy_t.argmax(1).long())
    value_loss = F.mse_loss(pred_v, value_t)
    # print('policy loss: ',policy_loss.item(), 'value loss: ',value_loss.item())
    loss = policy_loss + value_loss

    opt.zero_grad()
    loss.backward()
    opt.step()
    return loss.item(), policy_loss.item(), value_loss.item()

# ------------------------------  MAIN  ---------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="MuZero Connect‑6 trainer")
    p.add_argument("--board_size", type=int, default=19)
    p.add_argument("--total_steps", type=int, default=10000)
    p.add_argument("--num_simulations", type=int, default=400) # 50, 200
    p.add_argument("--buffer_size", type=int, default=10000)
    p.add_argument("--batch_size", type=int, default=32) # 16, 256
    p.add_argument("--learning_rate", type=float, default=1e-3)
    p.add_argument("--td_steps", type=int, default=10)
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--save_path", type=str, default="./checkpoints/MuZero/muzero_connect6.pt")
    return p.parse_args()


def main():
    args = parse_args()
    env = Connect6Env(board_size=args.board_size)
    net = MuZeroNet(action_size=env.action_size, board_size=args.board_size, device=args.device)
    optimizer = optim.Adam(net.parameters(), lr=args.learning_rate)
    replay = ReplayBuffer(args.buffer_size)

    losses = []
    p_losses = []
    v_losses = []
    for step in range(1, args.total_steps + 1):
        game = self_play(net, deepcopy(env), args.num_simulations)
        replay.add(game)
        results = train_step(net, replay, optimizer, args.batch_size, args.td_steps)
        if results is not None:
            loss, p_loss, v_loss = results
            losses.append(loss)
            p_losses.append(p_loss)
            v_losses.append(v_loss)

        if step % 100 == 0:
            print(f"Step {step}/{args.total_steps}  Loss: {loss}")

            os.makedirs(Path(args.save_path).parent, exist_ok=True)
            torch.save(net.state_dict(), args.save_path.replace(".pt", f"_{step}.pt"))

            # plot losses
            plt.figure(figsize=(8, 12))
            plt.subplot(3, 1, 1)
            plt.plot(losses, label="Total Loss")
            plt.title("Total Loss")
            plt.subplot(3, 1, 2)
            plt.plot(p_losses, label="Policy Loss", color='orange')
            plt.title("Policy Loss")
            plt.subplot(3, 1, 3)
            plt.plot(v_losses, label="Value Loss", color='green')
            plt.title("Value Loss")
            plt.tight_layout()
            plt.savefig(args.save_path.replace(".pt", f"_{step}.png"))
            plt.close()


    print("Training finished.")


if __name__ == "__main__":
    main()
