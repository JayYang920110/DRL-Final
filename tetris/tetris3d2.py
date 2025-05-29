import numpy as np
import random
import torch
from typing import Optional
import pyvista as pv
import numpy as np
import imageio
class Tetris3D:
    """Minimal 3‑D Tetris with one (1×2×2) block and only horizontal translations.

    Board tensor indices: board[z, x, y]
        z : height axis (0 at top, increases downward)
        x : left → right
        y : front → back
    The only block = shape (1, 2, 2) so rotation is unnecessary.
    """

    # ------------------------------------------------------------------
    # Initialise
    # ------------------------------------------------------------------
    def __init__(self, height: int = 20, width: int = 4, depth: int = 4, device: str = "cpu"):
        self.height = height   # z axis size
        self.width = width     # x axis size
        self.depth = depth     # y axis size
        self.device = device

        self.pieces = [np.ones((1, 2, 2), dtype=np.int8)]  # single O‑block
        self.reset()

    # ------------------------------------------------------------------
    # Public RL API
    # ------------------------------------------------------------------
    def reset(self):
        self.board = np.zeros((self.height, self.width, self.depth), dtype=np.int8)
        self.score = 0
        self.cleared_planes = 0
        self.gameover = False
        self.current_piece = self.pieces[0]
        self._spawn_piece()
        return self.state_features()

    def step(self, action):
        """Place the block at horizontal coords (x, y). Return (reward, done)."""
        if self.gameover:
            raise RuntimeError("Episode finished – call reset() first.")

        x, y = action
        px, py = 2, 2  # block footprint

        # bounds check
        if x < 0 or x + px > self.width or y < 0 or y + py > self.depth:
            self.gameover = True
            return -10.0, True

        pos = {"x": x, "y": y, "z": 0}
        while not self._collision(self.current_piece, pos):
            pos["z"] += 1
        pos["z"] -= 1
        # while not self._collision(self.current_piece, pos):
        #     frames.append(self.board.copy())  # Optional: for video
        #     pos["z"] += 1
        # pos["z"] -= 1
        self._store(self.current_piece, pos)
        planes = self._clear_planes()
        reward = 1 + (planes ** 2) * (self.width * self.depth)
        self.score += reward
        self.cleared_planes += planes

        self._spawn_piece()
        if self._collision(self.current_piece, self.current_pos):
            self.gameover = True
            reward -= 2

        return float(reward), self.gameover

    # ------------------------------------------------------------------
    # Helper to enumerate all next states externally --------------------
    # ------------------------------------------------------------------
    def get_next_states(self, as_feature: bool = True):
        """Return a dict {(x, y): board / feature} for every legal placement."""
        px, py = 2, 2
        states = {}
        for x in range(self.width - px + 1):
            for y in range(self.depth - py + 1):
                pos = {"x": x, "y": y, "z": 0}
                while not self._collision(self.current_piece, pos):
                    pos["z"] += 1
                pos["z"] -= 1
                board_copy = self.board.copy()
                for dz in range(1):
                    for dx in range(px):
                        for dy in range(py):
                            board_copy[pos["z"] + dz, x + dx, y + dy] = 1
                states[(x, y)] = self.state_features(board_copy) if as_feature else board_copy
        return states

    def state_features(self, board: Optional[np.ndarray] = None):
        """4‑D feature vector for given board (or current)."""
        if board is None:
            board = self.board
        planes, _ = self._count_planes(board)
        holes = self._count_holes(board)
        bump, h_sum = self._bumpiness_height(board)
        return torch.tensor([planes, holes, bump, h_sum], dtype=torch.float32, device=self.device)

    # ------------------------------------------------------------------
    # Board mechanics
    # ------------------------------------------------------------------
    def _collision(self, piece, pos):
        for dx in range(2):
            for dy in range(2):
                z = pos["z"]
                x = pos["x"] + dx
                y = pos["y"] + dy
                if z >= self.height or self.board[z, x, y]:
                    return True
        return False

    def _store(self, piece, pos):
        for dx in range(2):
            for dy in range(2):
                self.board[pos["z"], pos["x"] + dx, pos["y"] + dy] = 1

    def _count_planes(self, board):
        filled = [z for z in range(self.height) if np.all(board[z])]
        return len(filled), filled

    def _clear_planes(self):
        n, filled = self._count_planes(self.board)
        for z in filled:
            self.board = np.delete(self.board, z, axis=0)
            self.board = np.vstack([np.zeros((1, self.width, self.depth), dtype=np.int8), self.board])
        return n

    def _count_holes(self, board):
        holes = 0
        for x in range(self.width):
            for y in range(self.depth):
                col = board[:, x, y]
                top = np.where(col)[0]
                if top.size:
                    holes += np.sum(col[top[0] + 1:] == 0)
        return float(holes)

    def _bumpiness_height(self, board):
        heights = np.zeros((self.width, self.depth), int)
        for x in range(self.width):
            for y in range(self.depth):
                col = board[:, x, y]
                filled = np.where(col)[0]
                heights[x, y] = 0 if filled.size == 0 else self.height - filled[0]
        total_h = heights.sum()
        bump = 0
        for x in range(self.width):
            for y in range(self.depth):
                h = heights[x, y]
                for dx, dy in ((1, 0), (0, 1)):
                    nx, ny = x + dx, y + dy
                    if nx < self.width and ny < self.depth:
                        bump += abs(h - heights[nx, ny])
        return float(bump), float(total_h)

    # ------------------------------------------------------------------
    # Actions -----------------------------------------------------------
    # ------------------------------------------------------------------
    def legal_actions(self):
        px, py = 2, 2
        return [(x, y) for x in range(self.width - px + 1) for y in range(self.depth - py + 1)]

    def _spawn_piece(self):
        self.current_pos = {"x": self.width // 2 - 1, "y": self.depth // 2 - 1, "z": 0}



def render_voxel_video(board_sequence, save_path="tetris3d.mp4", fps=5):
    images = []
    h = board_sequence[0].shape[0]   # board 高度

    for board in board_sequence:
        plotter = pv.Plotter(off_screen=True, window_size=[400, 400])

        z, x, y = np.where(board > 0)
        for xi, yi, zi in zip(x, y, z):
            world_z = h - 1 - zi       # ### FIX: 反轉 z 軸
            cube = pv.Cube(
                center=(xi + 0.5, yi + 0.5, world_z + 0.5),  # 置中
                x_length=1, y_length=1, z_length=1
            )
            plotter.add_mesh(cube, color="cyan", show_edges=True)

        plotter.set_background("white")
        plotter.view_isometric()       # 這時看起來就像「往下掉」
        plotter.show(screenshot="frame.png", auto_close=True)
        images.append(imageio.imread("frame.png"))

    imageio.mimsave(save_path, images, fps=fps)
    print(f"Saved to {save_path}")
# ----------------------------------------------------------------------
# Quick self‑test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    frames = []
    env = Tetris3D(height=20, width=4, depth=4)
    s = env.reset()
    done = False
    while not done:
        a = random.choice(env.legal_actions())
        r, done = env.step(a)
        frames.append(env.board.copy())
    print("Total score:", env.score, "planes cleared:", env.cleared_planes)
    render_voxel_video(frames)

# 檢查重力及高度
# def test_gravity_and_height():
#     frames = []
#     env = Tetris3D(height=20, width=4, depth=4)
#     env.reset()

#     actions = [(0,0), (2,0), (0,2), (0,0), (2,2)]  # 5 步，如上表
#     for a in actions:
#         reward, done = env.step(a)
#         frames.append(env.board.copy())
#         assert not done, "不應該提前 Game-Over"
#     render_voxel_video(frames, save_path="gravity_check.mp4", fps=1)
#     # 1️⃣ 應該只清除 1 層
#     assert env.cleared_planes == 1, f"預期 cleared_planes=1，實際={env.cleared_planes}"

#     # 2️⃣ 檢查 z=19（最底層）是否有原本那塊 2×2
#     bottom = env.board[-1]            # shape = (width, depth) = (4,4)
#     assert np.all(bottom[0:2, 0:2] == 1), "上層方塊沒有掉到底層！"

#     # 3️⃣ 棋盤高度仍為 20
#     assert env.board.shape[0] == 20, f"高度異常：{env.board.shape[0]} ≠ 20"

#     print("✅ 重力 & 高度測試通過！")

# if __name__ == "__main__":
#     test_gravity_and_height()