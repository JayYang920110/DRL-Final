"""
@author: Viet Nguyen <nhviet1009@gmail.com>
"""
import argparse
import torch
import cv2
from src.tetris import Tetris
from src.deep_q_network import DeepQNetwork
from train_ppo import Agent

def get_args():
    parser = argparse.ArgumentParser(
        """Implementation of Deep Q Network to play Tetris""")

    parser.add_argument("--width", type=int, default=10, help="The common width for all images")
    parser.add_argument("--height", type=int, default=20, help="The common height for all images")
    parser.add_argument("--block_size", type=int, default=30, help="Size of a block")
    parser.add_argument("--fps", type=int, default=300, help="frames per second")
    parser.add_argument("--saved_path", type=str, default="trained_models")
    parser.add_argument("--output", type=str, default="output.mp4")

    args = parser.parse_args()
    return args


def test(opt,PPO=False):
    device=torch.device("cuda" if torch.cuda.is_available()  else "cpu")
    if torch.cuda.is_available():
        torch.cuda.manual_seed(123)
    else:
        torch.manual_seed(123)
    if torch.cuda.is_available():
        if PPO:
            model=Agent()
            model.load_state_dict(torch.load("./trained/PPO/tetris_3/stage3.pth"))
        else:
            model= DeepQNetwork()
            model.load_state_dict(torch.load("trained_models/tetris_3000.pth"))
    else:
        raise RuntimeError("No, GPU!!")
    model.eval()
    env = Tetris(width=opt.width, height=opt.height, block_size=opt.block_size)
    env.reset()
    if torch.cuda.is_available():
        model.to(device)
    out = cv2.VideoWriter(opt.output, cv2.VideoWriter_fourcc(*"MJPG"), opt.fps,
                          (int(1.5*opt.width*opt.block_size), opt.height*opt.block_size))
    while True:
        if PPO:
            next_steps = env.get_next_states()
            next_actions, next_states = zip(*next_steps.items())
            obs = env.get_ppo_state()
            obs = torch.tensor(obs,dtype=torch.float32).to(device)
            action_idx,_,_,_ =model.get_action_and_value(obs.unsqueeze(0))
            if action_idx>=len(next_actions):
                action = next_actions[0]
            else:
                action = next_actions[action_idx]
            _, done = env.step(action, render=True, video=out)
        else:
            next_steps = env.get_next_states()
            next_actions, next_states = zip(*next_steps.items())
            next_states = torch.stack(next_states)
            if torch.cuda.is_available():
                next_states = next_states.cuda()
            predictions = model(next_states)[:, 0]
            index = torch.argmax(predictions).item()
            action = next_actions[index]
            _, done = env.step(action, render=True, video=out)

        if done:
            out.release()
            break
            


if __name__ == "__main__":
    opt = get_args()
    test(opt,True)
