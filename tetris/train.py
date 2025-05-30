"""
@author: Viet Nguyen <nhviet1009@gmail.com>
"""
import argparse
import os
from random import random, randint, sample
from collections import deque

import numpy as np
import torch
import torch.nn as nn
import wandb

from src.deep_q_network import DeepQNetwork
from src.tetris3d2 import Tetris3D


def get_args():
    parser = argparse.ArgumentParser("DQN for Tetris with wandb logging")
    parser.add_argument("--width", type=int, default=10)
    parser.add_argument("--height", type=int, default=20)
    parser.add_argument("--block_size", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--initial_epsilon", type=float, default=1.0)
    parser.add_argument("--final_epsilon", type=float, default=1e-3)
    parser.add_argument("--num_decay_epochs", type=float, default=2000)
    parser.add_argument("--num_epochs", type=int, default=10000)
    parser.add_argument("--save_interval", type=int, default=2500)
    parser.add_argument("--replay_memory_size", type=int, default=30000)
    parser.add_argument("--saved_path", type=str, default="trained_3d_tetris_models")
    return parser.parse_args()


def train(opt):
    wandb.init(project="tetris-dqn-3d", config=vars(opt))

    if torch.cuda.is_available():
        torch.cuda.manual_seed(123)
    else:
        torch.manual_seed(123)

    os.makedirs(opt.saved_path, exist_ok=True)

    env = Tetris3D()
    model = DeepQNetwork()
    optimizer = torch.optim.Adam(model.parameters(), lr=opt.lr)
    criterion = nn.MSELoss()

    start_epoch = 0
    best_score = float('-inf')

    checkpoint_path = f"{opt.saved_path}/last_checkpoint.pth"
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        best_score = checkpoint['best_score']
        print(f"Resumed from checkpoint at epoch {start_epoch}, best score: {best_score}")

    state = env.reset()
    if torch.cuda.is_available():
        model.cuda()
        state = state.cuda()

    replay_memory = deque(maxlen=opt.replay_memory_size)
    epoch = start_epoch

    while epoch < opt.num_epochs:
        next_steps = env.get_next_states()

        epsilon = opt.final_epsilon + (max(opt.num_decay_epochs - epoch, 0) *
                                       (opt.initial_epsilon - opt.final_epsilon) / opt.num_decay_epochs)
        u = random()
        random_action = u <= epsilon
        next_actions, next_states = zip(*next_steps.items())
        next_states = torch.stack(next_states)

        if torch.cuda.is_available():
            next_states = next_states.cuda()

        model.eval()
        with torch.no_grad():
            predictions = model(next_states)[:, 0]
        model.train()

        if random_action:
            index = randint(0, len(next_steps) - 1)
        else:
            index = torch.argmax(predictions).item()

        next_state = next_states[index]
        action = next_actions[index]

        reward, done = env.step(action)
        if torch.cuda.is_available():
            next_state = next_state.cuda()

        replay_memory.append([state, reward, next_state, done])

        if done:
            final_score = env.score
   
            final_cleared_planes = env.cleared_planes
            state = env.reset()
            if torch.cuda.is_available():
                state = state.cuda()
        else:
            state = next_state
            continue

        if len(replay_memory) < opt.replay_memory_size / 10:
            continue

        epoch += 1
        batch = sample(replay_memory, min(len(replay_memory), opt.batch_size))
        state_batch, reward_batch, next_state_batch, done_batch = zip(*batch)

        state_batch = torch.stack(state_batch)
        reward_batch = torch.tensor(np.array(reward_batch, dtype=np.float32)[:, None])
        next_state_batch = torch.stack(next_state_batch)

        if torch.cuda.is_available():
            state_batch = state_batch.cuda()
            reward_batch = reward_batch.cuda()
            next_state_batch = next_state_batch.cuda()

        q_values = model(state_batch)
        model.eval()
        with torch.no_grad():
            next_prediction_batch = model(next_state_batch)
        model.train()

        y_batch = torch.cat([
            reward if done else reward + opt.gamma * prediction
            for reward, done, prediction in zip(reward_batch, done_batch, next_prediction_batch)
        ])[:, None]

        optimizer.zero_grad()
        loss = criterion(q_values, y_batch)
        loss.backward()
        optimizer.step()

        print("Epoch: {}/{}, Action: {}, Score: {},  Cleared planes: {}, Epsilon: {:.4f}".format(
            epoch,
            opt.num_epochs,
            action,
            final_score,
            final_cleared_planes,
            epsilon
        ))

        wandb.log({
            "Score": final_score,
            "Cleared planes": final_cleared_planes,
            "Epsilon": epsilon,
            "Loss": loss.item(),
            "Epoch": epoch
        })

        if epoch > 0 and epoch % opt.save_interval == 0:
            torch.save(model, f"{opt.saved_path}/tetris_{epoch}.pth")

        if final_score > best_score:
            best_score = final_score
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_score': best_score
            }
            torch.save(checkpoint, f"{opt.saved_path}/best_model.pth")


if __name__ == "__main__":
    opt = get_args()
    train(opt)
