import argparse
import torch
from src.tetris3d2 import Tetris3D, render_voxel_video_matplotlib
from src.deep_q_network import DeepQNetwork


def get_args():
    parser = argparse.ArgumentParser("""Implementation of Deep Q Network to play Tetris""")
    parser.add_argument("--width", type=int, default=4, help="Board width")
    parser.add_argument("--height", type=int, default=20, help="Board height")
    parser.add_argument("--depth", type=int, default=4, help="Board depth")
    parser.add_argument("--fps", type=int, default=5, help="frames per second")
    parser.add_argument("--saved_path", type=str, default="/home/jay/DRL/FinalProject/3dTetris/trained_3d_tetris_models/best_model.pth")
    parser.add_argument("--output", type=str, default="output.mp4")
    return parser.parse_args()


def test(opt, video_index=None):
    model = DeepQNetwork()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(123)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(123)

    # Load model
    checkpoint = torch.load(opt.saved_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device).eval()

    env = Tetris3D(height=opt.height, width=opt.width, depth=opt.depth, device=device)
    env.reset()
    frames = []

    total_score = 0
    done = False
    while not done:
        # if total_score > 50:
        #     break
        next_steps = env.get_next_states()
        next_actions, next_states = zip(*next_steps.items())
        next_states = torch.stack(next_states).to(device)

        with torch.no_grad():
            predictions = model(next_states)[:, 0]

        epsilon = 0.001
        if torch.rand(1).item() < epsilon:
            index = torch.randint(len(next_actions), (1,)).item()
        else:
            index = torch.argmax(predictions).item()

        action = next_actions[index]
        # reward, done = env.step(action, render=True, frames=frames)  # 會有落下的畫面，但時間很久，建議一個分數就停下來
        reward, done = env.step(action) # ✅ 只記錄「落地後」的樣子
        frames.append(env.board.copy())  
        total_score += reward

    # Save voxel video
    video_name = opt.output if video_index is None else opt.output.replace(".mp4", f"_{video_index}.mp4")
    render_voxel_video_matplotlib(frames, save_path=video_name, fps=opt.fps)

    return total_score


if __name__ == "__main__":
    opt = get_args()
    scores = []
    for i in range(10):
        print(f"Running test game {i + 1}...")
        score = test(opt, video_index=i + 1)
        scores.append(score)
        print(f"Score: {score:.2f}")

    avg_score = sum(scores) / len(scores)
    print(f"\n🎮 Average Score over 10 runs: {avg_score:.2f}")
