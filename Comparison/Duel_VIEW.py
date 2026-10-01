#==========导入库
import csv
import random
import time
from collections import deque
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import sys
# torch.nn.functional 不需要，已删除

#=============设备配置
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"当前使用的设备：{device}")

#=================环境参数设置
GRID_SIZE = 30
START = (0, 0)
GOAL = (GRID_SIZE - 1, GRID_SIZE - 1)

#==========定义静态障碍生成函数
def generate_static_obstacles(num_obstacles=160, seed=25):
    np.random.seed(seed)
    obstacles = set()
    while len(obstacles) < num_obstacles:
        x = np.random.randint(0, GRID_SIZE)
        y = np.random.randint(0, GRID_SIZE)
        if (x, y) != START and (x, y) != GOAL:
            obstacles.add((x, y))
    return list(obstacles)

STATIC_OBSTACLES = generate_static_obstacles(num_obstacles=160)

#================定义动态障碍生成函数
def generate_dynamic_obstacles(num_dynamic=6):
    dy_obs = set()
    attempts = 0
    while len(dy_obs) < num_dynamic and attempts < 500:
        x = np.random.randint(0, GRID_SIZE)
        y = np.random.randint(0, GRID_SIZE)
        if (x, y) != START and (x, y) != GOAL and (x, y) not in STATIC_OBSTACLES:
            dy_obs.add((x, y))
        attempts += 1
    return list(dy_obs)

NUM_DYNAMIC = 6
DYNAMIC_OBSTACLES = generate_dynamic_obstacles(NUM_DYNAMIC)

#==========定义动态障碍巡逻路径
def generate_patrol_paths(num_paths=4, path_length=10, grid_size=GRID_SIZE):
    """生成若干条随机巡逻路径，避开静态障碍、起点和终点"""
    paths = []
    for _ in range(num_paths):
        path = []
        attempts = 0
        while len(path) < path_length and attempts < 500:
            if not path:
                x = np.random.randint(0, grid_size)
                y = np.random.randint(0, grid_size)
                if (x, y) not in STATIC_OBSTACLES and (x, y) != START and (x, y) != GOAL:
                    path.append((x, y))
            else:
                last_x, last_y = path[-1]
                neighbors = [(last_x - 1, last_y), (last_x + 1, last_y),
                             (last_x, last_y - 1), (last_x, last_y + 1)]
                np.random.shuffle(neighbors)
                for nx, ny in neighbors:
                    if 0 <= nx < grid_size and 0 <= ny < grid_size:
                        if (nx, ny) not in STATIC_OBSTACLES and (nx, ny) not in path \
                                and (nx, ny) != START and (nx, ny) != GOAL:
                            path.append((nx, ny))
                            break
            attempts += 1
        if len(path) >= 4:
            paths.append(path)
    return paths

PATROL_PATHS = generate_patrol_paths(num_paths=NUM_DYNAMIC, path_length=10)
while len(PATROL_PATHS) < NUM_DYNAMIC:
    dummy_path = [(1, 1), (2, 1), (2, 2), (1, 2)]
    if all((x, y) not in STATIC_OBSTACLES and (x, y) != START and (x, y) != GOAL for x, y in dummy_path):
        PATROL_PATHS.append(dummy_path)
    else:
        PATROL_PATHS.append([(1, 1), (1, 2), (2, 2), (2, 1)])

#===============定义动态障碍移动函数
def move_dynamic_obstacles_patrol(dyn_obs, path_indices, directions, agent_pos=None):
    new_dyn = []
    new_indices = []
    new_dirs = []
    for i, (x, y) in enumerate(dyn_obs):
        path = PATROL_PATHS[i]
        idx = path_indices[i]
        dir = directions[i]
        next_idx = idx + dir
        if next_idx >= len(path) or next_idx < 0:
            dir *= -1
            next_idx = idx + dir
            directions[i] = dir
        next_pos = path[next_idx]
        if agent_pos is not None and next_pos == agent_pos:
            new_dyn.append((x, y))
            new_indices.append(idx)
            new_dirs.append(dir)
        else:
            new_dyn.append(next_pos)
            new_indices.append(next_idx)
            new_dirs.append(dir)
    return new_dyn, new_indices, new_dirs

#================== Dueling DQN 神经网络定义
class DuelingDQN(nn.Module):
    def __init__(self, input_size, output_size):
        super(DuelingDQN, self).__init__()
        self.fc1 = nn.Linear(input_size, 256)
        self.fc2 = nn.Linear(256, 256)
        self.value_fc = nn.Linear(256, 128)
        self.value = nn.Linear(128, 1)
        self.advantage_fc = nn.Linear(256, 128)
        self.advantage = nn.Linear(128, output_size)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        v = torch.relu(self.value_fc(x))
        v = self.value(v)
        a = torch.relu(self.advantage_fc(x))
        a = self.advantage(a)
        q = v + (a - a.mean(dim=1, keepdim=True))
        return q

#==============DQN智能体类（加入动作历史和多尺度状态）
class DQNAgent(nn.Module):
    def __init__(self):
        super(DQNAgent, self).__init__()
        self.action_size = 8
        self.history_length = 4
        self.action_history = deque(maxlen=self.history_length)

        # 各特征维度
        pos_dim = 4
        local_5x5_dim = (2 * 2 + 1) ** 2  # 25
        local_3x3_dim = 9
        obs_info_dim = 3
        velocity_dim = 3
        action_history_dim = self.history_length * self.action_size  # 32
        self.state_size = (pos_dim + local_5x5_dim + local_3x3_dim +
                           obs_info_dim + velocity_dim + action_history_dim)  # 76

        self.memory = deque(maxlen=50000)
        self.gamma = 0.99
        self.epsilon = 1.0
        self.epsilon_min = 0.10
        self.epsilon_decay = 0.995
        self.learning_rate = 0.0005

        self.model = DuelingDQN(self.state_size, self.action_size).to(device)
        self.target_model = DuelingDQN(self.state_size, self.action_size).to(device)
        self.target_model.load_state_dict(self.model.state_dict())
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        self.criterion = nn.MSELoss()

        self.train_step_count = 0
        self.target_model_update = 500

    def get_action_history_vector(self):
        vec = np.zeros(self.history_length * self.action_size, dtype=np.float32)
        for i, act in enumerate(self.action_history):
            if act >= 0:
                start = i * self.action_size
                vec[start + act] = 1.0
        return vec

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def action(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
        with torch.no_grad():
            q_val = self.model(state_tensor)
        return q_val.squeeze(0).argmax().item()

    def replay(self, batch_size):
        if len(self.memory) < batch_size:
            return
        minibatch = random.sample(self.memory, batch_size)
        states = np.array([t[0] for t in minibatch], dtype=np.float32)
        actions = np.array([t[1] for t in minibatch], dtype=np.int64)
        rewards = np.array([t[2] for t in minibatch], dtype=np.float32)
        next_states = np.array([t[3] for t in minibatch], dtype=np.float32)
        dones = np.array([t[4] for t in minibatch], dtype=np.float32)

        states = torch.from_numpy(states).to(device)
        actions = torch.from_numpy(actions).to(device)
        rewards = torch.from_numpy(rewards).to(device)
        next_states = torch.from_numpy(next_states).to(device)
        dones = torch.from_numpy(dones).to(device)

        current_q = self.model(states).gather(1, actions.unsqueeze(1)).squeeze(1)

        with torch.no_grad():
            next_action = self.model(next_states).argmax(1).unsqueeze(1)
            next_q = self.target_model(next_states).gather(1, next_action).squeeze(1)
            target_q = rewards + self.gamma * next_q * (1 - dones)

        loss = self.criterion(current_q, target_q)
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        self.optimizer.step()

        self.train_step_count += 1
        if self.train_step_count % self.target_model_update == 0:
            self.update_target_model()

    def update_target_model(self):
        self.target_model.load_state_dict(self.model.state_dict())

#===================环境辅助函数（增强状态）
def get_local_view(x, y, radius=2):
    view = np.zeros((2 * radius + 1, 2 * radius + 1), dtype=np.float32)
    for i in range(-radius, radius + 1):
        for j in range(-radius, radius + 1):
            nx, ny = x + i, y + j
            if 0 <= nx < GRID_SIZE and 0 <= ny < GRID_SIZE:
                if (nx, ny) in STATIC_OBSTACLES:
                    view[i + radius, j + radius] = 1.0
                elif (nx, ny) in DYNAMIC_OBSTACLES:
                    view[i + radius, j + radius] = 2.0
                elif (nx, ny) == GOAL:
                    view[i + radius, j + radius] = 3.0
    return view.flatten()

def get_local_view_3x3(x, y):
    view = np.zeros((3, 3), dtype=np.float32)
    for i in range(-1, 2):
        for j in range(-1, 2):
            nx, ny = x + i, y + j
            if 0 <= nx < GRID_SIZE and 0 <= ny < GRID_SIZE:
                if (nx, ny) in STATIC_OBSTACLES:
                    view[i + 1, j + 1] = 1.0
                elif (nx, ny) in DYNAMIC_OBSTACLES:
                    view[i + 1, j + 1] = 2.0
                elif (nx, ny) == GOAL:
                    view[i + 1, j + 1] = 3.0
    return view.flatten()

def get_nearest_obstacle_info(x, y):
    min_dist = float('inf')
    nearest_dx, nearest_dy = 0.0, 0.0
    for ox, oy in STATIC_OBSTACLES + DYNAMIC_OBSTACLES:
        dist = abs(x - ox) + abs(y - oy)
        if dist < min_dist:
            min_dist = dist
            nearest_dx = ox - x
            nearest_dy = oy - y
    if min_dist == float('inf'):
        return 1.0, 0.0, 0.0
    norm_dist = min(1.0, min_dist / (GRID_SIZE * 2))
    if min_dist > 0:
        nearest_dx /= min_dist
        nearest_dy /= min_dist
    else:
        nearest_dx, nearest_dy = 0.0, 0.0
    return norm_dist, nearest_dx, nearest_dy

def get_velocity_info(agent_x, agent_y, dyn_obs, path_indices, directions):
    if not dyn_obs:
        return (0.0, 0.0, 0.0)
    min_dist = float('inf')
    nearest_idx = 0
    for i, (ox, oy) in enumerate(dyn_obs):
        d = abs(agent_x - ox) + abs(agent_y - oy)
        if d < min_dist:
            min_dist = d
            nearest_idx = i
    path = PATROL_PATHS[nearest_idx]
    idx = path_indices[nearest_idx]
    dir = directions[nearest_idx]
    next_idx = idx + dir
    if next_idx >= len(path) or next_idx < 0:
        dir *= -1
        next_idx = idx + dir
    next_pos = path[next_idx]
    dx = next_pos[0] - dyn_obs[nearest_idx][0]
    dy = next_pos[1] - dyn_obs[nearest_idx][1]
    speed = np.sqrt(dx * dx + dy * dy)
    if speed > 0:
        dx_norm = dx / speed
        dy_norm = dy / speed
    else:
        dx_norm, dy_norm = 0.0, 0.0
    return (dx_norm, dy_norm, speed)

def get_state(x, y, gx, gy, action_history_vector=None, velocity_info=None):
    max_val = GRID_SIZE - 1
    pos = np.array([x / max_val, y / max_val, gx / max_val, gy / max_val], dtype=np.float32)
    local_5x5 = get_local_view(x, y, radius=2)
    local_3x3 = get_local_view_3x3(x, y)
    obs_info = np.array(get_nearest_obstacle_info(x, y), dtype=np.float32)
    parts = [pos, local_5x5, local_3x3, obs_info]
    if velocity_info is not None:
        parts.append(np.array(velocity_info, dtype=np.float32))
    if action_history_vector is not None:
        parts.append(action_history_vector)
    return np.concatenate(parts)

#==============2）奖励函数（原版）
def get_reward(x, y, prev_dist=None):
    if (x, y) == GOAL:
        return 2000.0
    if (x, y) in STATIC_OBSTACLES or (x, y) in DYNAMIC_OBSTACLES:
        return -100.0
    cur_dist = abs(x - GOAL[0]) + abs(y - GOAL[1])
    if prev_dist is not None:
        return (prev_dist - cur_dist) * 1.0 - 0.5
    return -0.5

#===============绘图函数（保持不变）
def plot_static_map():
    plt.figure(figsize=(8, 8))
    first_static = True
    for ox, oy in STATIC_OBSTACLES:
        label = 'Static Obstacle' if first_static else ""
        plt.gca().add_patch(plt.Rectangle((oy - 0.5, ox - 0.5), 1, 1,
                                          fill=True, color='darkred', alpha=0.8, label=label))
        first_static = False
    first_dyn = True
    for ox, oy in DYNAMIC_OBSTACLES:
        label = 'Dynamic Obstacle' if first_dyn else ""
        plt.gca().add_patch(plt.Rectangle((oy - 0.5, ox - 0.5), 1, 1,
                                          fill=True, color='orange', alpha=0.8, label=label))
        first_dyn = False
    plt.scatter(START[1], START[0], c='lime', marker='o', s=120, edgecolors='black', label='Start')
    plt.scatter(GOAL[1], GOAL[0], c='gold', marker='*', s=200, edgecolors='black', label='Goal')
    plt.xticks(range(GRID_SIZE))
    plt.yticks(range(GRID_SIZE))
    plt.xlim(-0.5, GRID_SIZE - 0.5)
    plt.ylim(-0.5, GRID_SIZE - 0.5)
    plt.grid(True, linestyle='--', alpha=0.3)
    plt.legend(fontsize=12, loc='upper left')
    plt.title(f"Static Obstacle Map ({GRID_SIZE}x{GRID_SIZE}, {len(STATIC_OBSTACLES)} obstacles)")
    plt.tight_layout()
    plt.show()

def count_turns(path):
    if len(path) < 3: return 0
    turns = 0
    for i in range(1, len(path) - 1):
        prev = path[i - 1]
        curr = path[i]
        nxt = path[i + 1]
        if (prev[0] - curr[0], prev[1] - curr[1]) != (curr[0] - nxt[0], curr[1] - nxt[1]):
            turns += 1
    return turns

def plot_path(path, save_path=None):
    grid = np.zeros((GRID_SIZE, GRID_SIZE))
    for ox, oy in STATIC_OBSTACLES: grid[ox, oy] = 1
    plt.figure(figsize=(10, 10))
    first_static = True
    for ox, oy in STATIC_OBSTACLES:
        label = 'Static Obstacle' if first_static else ""
        plt.gca().add_patch(plt.Rectangle((oy - 0.5, ox - 0.5), 1, 1,
                                          fill=True, color='darkred', alpha=0.8, label=label))
        first_static = False
    for ox, oy in STATIC_OBSTACLES:
        plt.gca().add_patch(plt.Rectangle((oy - 0.5, ox - 0.5), 1, 1,
                                          fill=True, color='darkred', alpha=0.7, linewidth=0))
    first_dyn = True
    for ox, oy in DYNAMIC_OBSTACLES:
        label = 'Dynamic Obstacle' if first_dyn else ""
        plt.gca().add_patch(plt.Rectangle((oy - 0.5, ox - 0.5), 1, 1,
                                          fill=True, color='orange', alpha=0.8, label=label))
        first_dyn = False
    plt.scatter(START[1], START[0], c='lime', marker='o', s=200, edgecolors='black', linewidth=2, label='Start', zorder=5)
    plt.scatter(GOAL[1], GOAL[0], c='gold', marker='*', s=300, edgecolors='black', linewidth=2, label='Goal', zorder=5)
    if path:
        py, px = zip(*path)
        plt.scatter(px, py, c='cyan', s=120, alpha=0.6, edgecolors='none', label='Path Cell', zorder=3)
        plt.plot(px, py, 'b-', linewidth=3, alpha=0.9, label='Path Line', zorder=4)
    plt.xticks(range(GRID_SIZE))
    plt.yticks(range(GRID_SIZE))
    plt.xlim(-0.5, GRID_SIZE - 0.5)
    plt.ylim(-0.5, GRID_SIZE - 0.5)
    plt.grid(True, linestyle='-', linewidth=0.5, alpha=0.5, color='gray')
    plt.legend(fontsize=12, loc='upper left')
    plt.title(f"Path Planning (GRID {GRID_SIZE}x{GRID_SIZE})", fontsize=14)
    if save_path:
        plt.savefig(save_path, dpi=200, bbox_inches='tight')
        print(f"轨迹图已保存至: {save_path}")
    plt.show()

def moving_average(data, window=20):
    if len(data) < window: return data
    return np.convolve(data, np.ones(window) / window, mode='valid')

def print_group_progress(completed, total, bar_length=20):
    percent = completed / total
    filled_len = int(bar_length * percent)
    bar = '█' * filled_len + '░' * (bar_length - filled_len)
    print(f'\r组内进度: |{bar}| {percent:.0%} ({completed}/{total})', end='', flush=True)

def save_training_data(filename, reward_history, cumulative_reward_history,
                       success_history, trajectory_length_history,
                       path_length_history, turns_history,
                       episode_time_history, epsilon_history):
    n = len(reward_history)
    with open(filename, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Episode', 'Reward', 'CumulativeReward', 'Success',
                         'TrajectoryLength', 'PathLength', 'Turns',
                         'TimeCost', 'Epsilon'])
        for i in range(n):
            pl = path_length_history[i] if not np.isnan(path_length_history[i]) else ''
            tr = turns_history[i] if not np.isnan(turns_history[i]) else ''
            writer.writerow([i, reward_history[i], cumulative_reward_history[i],
                             int(success_history[i]), trajectory_length_history[i],
                             pl, tr, episode_time_history[i], epsilon_history[i]])
    print(f"训练数据已保存至 {filename}")

#===============测试函数（增强状态）
def test_agent(agent, render=True):
    global DYNAMIC_OBSTACLES
    agent.model.eval()
    dyn_obs = [PATROL_PATHS[i][0] for i in range(NUM_DYNAMIC)]
    DYNAMIC_OBSTACLES = dyn_obs
    path_indices = [0] * NUM_DYNAMIC
    directions = [1] * NUM_DYNAMIC

    x, y = START
    agent.action_history.clear()
    path = [(x, y)]

    dyn_obs, path_indices, directions = move_dynamic_obstacles_patrol(
        dyn_obs, path_indices, directions, agent_pos=(x, y))
    DYNAMIC_OBSTACLES = dyn_obs
    vel_info = get_velocity_info(x, y, dyn_obs, path_indices, directions)
    state = get_state(x, y, GOAL[0], GOAL[1], agent.get_action_history_vector(), vel_info)

    done = False
    steps = 0
    max_steps = 500
    collision = False

    start_time = time.time()
    while not done and steps < max_steps:
        action = agent.action(state)
        agent.action_history.append(action)

        dyn_obs, path_indices, directions = move_dynamic_obstacles_patrol(
            dyn_obs, path_indices, directions, agent_pos=None
        )
        DYNAMIC_OBSTACLES = dyn_obs

        dir_map = {0: (-1, 0), 1: (1, 0), 2: (0, -1), 3: (0, 1),
                   4: (-1, -1), 5: (-1, 1), 6: (1, -1), 7: (1, 1)}
        dx, dy = dir_map[action]
        target_nx, target_ny = x + dx, y + dy

        if not (0 <= target_nx < GRID_SIZE and 0 <= target_ny < GRID_SIZE):
            steps += 1
            continue

        cross_obstacle = False
        if action >= 4:
            adj1 = (x, target_ny)
            adj2 = (target_nx, y)
            if (adj1 in STATIC_OBSTACLES or adj1 in DYNAMIC_OBSTACLES) or \
               (adj2 in STATIC_OBSTACLES or adj2 in DYNAMIC_OBSTACLES):
                cross_obstacle = True

        target_obstacle = (target_nx, target_ny) in STATIC_OBSTACLES or \
                          (target_nx, target_ny) in DYNAMIC_OBSTACLES

        if target_obstacle or cross_obstacle:
            steps += 1
            continue

        x, y = target_nx, target_ny
        path.append((x, y))
        if (x, y) == GOAL:
            done = True

        vel_info = get_velocity_info(x, y, dyn_obs, path_indices, directions)
        state = get_state(x, y, GOAL[0], GOAL[1], agent.get_action_history_vector(), vel_info)
        steps += 1

    elapsed = time.time() - start_time
    path_has_obstacle = any((px, py) in STATIC_OBSTACLES or (px, py) in DYNAMIC_OBSTACLES
                            for (px, py) in path)
    success = ((x, y) == GOAL) and (not path_has_obstacle) and (not collision)
    length = len(path) if success else None
    turns = count_turns(path) if success else None
    if render: plot_path(path)
    return success, path, length, turns, elapsed

#=================评估函数（增强状态）
def evaluate_agent(agent, num_trials=100, max_steps=500):
    agent.model.eval()
    success_count = 0
    total_length = 0
    total_turns = 0

    for trial in range(num_trials):
        dyn_obs = [PATROL_PATHS[i][0] for i in range(NUM_DYNAMIC)]
        global DYNAMIC_OBSTACLES
        DYNAMIC_OBSTACLES = dyn_obs
        path_indices = [0] * NUM_DYNAMIC
        directions = [1] * NUM_DYNAMIC

        x, y = START
        agent.action_history.clear()
        path = [(x, y)]

        dyn_obs, path_indices, directions = move_dynamic_obstacles_patrol(
            dyn_obs, path_indices, directions, agent_pos=None)
        DYNAMIC_OBSTACLES = dyn_obs
        vel_info = get_velocity_info(x, y, dyn_obs, path_indices, directions)
        state = get_state(x, y, GOAL[0], GOAL[1], agent.get_action_history_vector(), vel_info)

        done = False
        steps = 0
        collision = False

        while not done and steps < max_steps:
            action = agent.action(state)
            agent.action_history.append(action)

            dyn_obs, path_indices, directions = move_dynamic_obstacles_patrol(
                dyn_obs, path_indices, directions, agent_pos=None)
            DYNAMIC_OBSTACLES = dyn_obs

            dir_map = {0: (-1, 0), 1: (1, 0), 2: (0, -1), 3: (0, 1),
                       4: (-1, -1), 5: (-1, 1), 6: (1, -1), 7: (1, 1)}
            dx, dy = dir_map[action]
            target_nx, target_ny = x + dx, y + dy

            if not (0 <= target_nx < GRID_SIZE and 0 <= target_ny < GRID_SIZE):
                steps += 1
                continue

            cross_obstacle = False
            if action >= 4:
                adj1 = (x, target_ny)
                adj2 = (target_nx, y)
                if (adj1 in STATIC_OBSTACLES or adj1 in DYNAMIC_OBSTACLES) or \
                   (adj2 in STATIC_OBSTACLES or adj2 in DYNAMIC_OBSTACLES):
                    cross_obstacle = True

            target_obstacle = (target_nx, target_ny) in STATIC_OBSTACLES or \
                              (target_nx, target_ny) in DYNAMIC_OBSTACLES

            if target_obstacle or cross_obstacle:
                steps += 1
                continue

            x, y = target_nx, target_ny
            path.append((x, y))
            if (x, y) == GOAL:
                done = True

            vel_info = get_velocity_info(x, y, dyn_obs, path_indices, directions)
            state = get_state(x, y, GOAL[0], GOAL[1], agent.get_action_history_vector(), vel_info)
            steps += 1

        path_has_obstacle = any((px, py) in STATIC_OBSTACLES or (px, py) in DYNAMIC_OBSTACLES
                                for (px, py) in path)
        success = ((x, y) == GOAL) and (not path_has_obstacle) and (not collision)

        if success:
            success_count += 1
            total_length += len(path)
            total_turns += count_turns(path)

    avg_success = success_count / num_trials * 100
    avg_len = total_length / success_count if success_count > 0 else None
    avg_turn = total_turns / success_count if success_count > 0 else None

    print(f"\n评估结果（{num_trials} 次独立测试）:")
    print(f"  成功率: {avg_success:.2f}%")
    if avg_len is not None:
        print(f"  平均路径长度: {avg_len:.2f}")
        print(f"  平均拐点数量: {avg_turn:.2f}")
    else:
        print("  无成功案例，无法计算路径统计")
    return avg_success, avg_len, avg_turn

#==================训练主程序
if __name__ == "__main__":
    agent = DQNAgent()
    TRAIN_MODE = True

    if TRAIN_MODE:
        episodes = 2000
        batch_size = 64

        reward_history = []
        success_history = []
        steps_history = []
        epsilon_history = []

        cumulative_reward_history = []
        episode_time_history = []
        path_length_history = []
        trajectory_length_history = []
        turns_history = []

        total_success = 0
        cumulative_reward = 0.0

        print("开始训练...")
        print(f"静态障碍数量: {len(STATIC_OBSTACLES)}，动态障碍数量: {len(DYNAMIC_OBSTACLES)}")
        train_start = time.time()

        group_size = 50
        ep_in_group = 0

        path_indices = [0] * 4
        directions = [1] * 4
        for e in range(episodes):
            episode_start = time.time()
            dyn_obs = [PATROL_PATHS[i][0] for i in range(NUM_DYNAMIC)]
            path_indices = [0] * NUM_DYNAMIC
            directions = [1] * NUM_DYNAMIC
            x, y = START
            agent.action_history.clear()
            prev_dist = abs(x - GOAL[0]) + abs(y - GOAL[1])

            # 初始状态（包含速度）
            dyn_obs, path_indices, directions = move_dynamic_obstacles_patrol(
                dyn_obs, path_indices, directions, agent_pos=(x, y))
            DYNAMIC_OBSTACLES = dyn_obs
            vel_info = get_velocity_info(x, y, dyn_obs, path_indices, directions)
            state = get_state(x, y, GOAL[0], GOAL[1], agent.get_action_history_vector(), vel_info)

            total_reward = 0
            done = False
            steps = 0
            max_episode_steps = 2000

            episode_collision = False
            episode_success = False
            episode_path = [(x, y)]

            while not done and steps < max_episode_steps:
                dyn_obs, path_indices, directions = move_dynamic_obstacles_patrol(
                    dyn_obs, path_indices, directions, agent_pos=(x, y))
                DYNAMIC_OBSTACLES = dyn_obs

                vel_info = get_velocity_info(x, y, dyn_obs, path_indices, directions)
                state = get_state(x, y, GOAL[0], GOAL[1], agent.get_action_history_vector(), vel_info)

                action = agent.action(state)
                agent.action_history.append(action)
                nx, ny = x, y
                dir_map = {0: (-1, 0), 1: (1, 0), 2: (0, -1), 3: (0, 1),
                           4: (-1, -1), 5: (-1, 1), 6: (1, -1), 7: (1, 1)}
                dx, dy = dir_map[action]
                target_nx, target_ny = x + dx, y + dy

                if not (0 <= target_nx < GRID_SIZE and 0 <= target_ny < GRID_SIZE):
                    reward = -5.0
                    next_state = state
                    done = False
                    agent.remember(state, action, reward, next_state, done)
                    state = next_state
                    total_reward += reward
                    steps += 1
                    if len(agent.memory) >= batch_size:
                        agent.replay(batch_size)
                    continue

                cross_obstacle = False
                if action >= 4:
                    adj1 = (x, target_ny)
                    adj2 = (target_nx, y)
                    if (adj1 in STATIC_OBSTACLES or adj1 in DYNAMIC_OBSTACLES) or \
                            (adj2 in STATIC_OBSTACLES or adj2 in DYNAMIC_OBSTACLES):
                        cross_obstacle = True

                target_obstacle = (target_nx, target_ny) in STATIC_OBSTACLES or \
                                  (target_nx, target_ny) in DYNAMIC_OBSTACLES

                if target_obstacle or cross_obstacle:
                    reward = -5.0
                    next_state = state
                    done = False
                    agent.remember(state, action, reward, next_state, done)
                    state = next_state
                    total_reward += reward
                    steps += 1
                    if len(agent.memory) >= batch_size:
                        agent.replay(batch_size)
                else:
                    x, y = target_nx, target_ny
                    episode_path.append((x, y))
                    cur_dist = abs(x - GOAL[0]) + abs(y - GOAL[1])
                    reward = get_reward(x, y, prev_dist)
                    prev_dist = cur_dist

                    # 下一状态（包含速度）
                    vel_info = get_velocity_info(x, y, dyn_obs, path_indices, directions)
                    next_state = get_state(x, y, GOAL[0], GOAL[1], agent.get_action_history_vector(), vel_info)

                    if (x, y) == GOAL:
                        episode_success = True
                        done = True
                    elif (x, y) in STATIC_OBSTACLES or (x, y) in DYNAMIC_OBSTACLES:
                        episode_collision = True
                        done = True

                agent.remember(state, action, reward, next_state, done)
                state = next_state
                total_reward += reward
                steps += 1

            if agent.epsilon > agent.epsilon_min:
                agent.epsilon *= agent.epsilon_decay

            reward_history.append(total_reward)
            cumulative_reward += total_reward
            cumulative_reward_history.append(cumulative_reward)
            success_history.append(1 if episode_success else 0)
            steps_history.append(steps)
            epsilon_history.append(agent.epsilon)
            trajectory_length_history.append(steps)
            episode_time_history.append(time.time() - episode_start)

            if episode_success:
                total_success += 1
                path_length_history.append(steps)
                turns = count_turns(episode_path)
                turns_history.append(turns)
            else:
                path_length_history.append(np.nan)
                turns_history.append(np.nan)

            ep_in_group += 1
            print_group_progress(ep_in_group, group_size)

            if ep_in_group == group_size:
                print()
                print(f"Episode {e + 1:4d} | Reward: {total_reward:7.2f} | Steps: {steps:3d} | "
                      f"Epsilon: {agent.epsilon:.3f} | Success: {episode_success}")
                sys.stdout.flush()
                ep_in_group = 0

        train_end = time.time()
        training_time = train_end - train_start

        torch.save(agent.model.state_dict(), "trained_model.pth")
        print("模型已保存为 trained_model.pth")

        print("\n" + "=" * 50)
        print("训练统计结果")
        print("=" * 50)
        print(f"总训练回合数: {episodes}")
        print(f"累计总奖励: {cumulative_reward:.2f}")
        print(f"平均每回合奖励: {cumulative_reward / episodes:.2f}")
        print(f"总成功次数: {total_success}  | 成功率: {total_success / episodes * 100:.2f}%")
        print(f"训练总耗时: {training_time:.2f} 秒")
        print("=" * 50)

        # 显示地图和单次测试
        print("\n正在显示静态障碍地图...")
        plot_static_map()
        print("\n开始测试规划效果...")
        agent.epsilon = 0.0
        success, path, length, turns, test_time = test_agent(agent, render=False)
        if success:
            print(f"成功到达目标")
            print(f"   - 路径长度 (步数): {length}")
            print(f"   - 拐点数量: {turns}")
            print(f"   - 测试规划耗时: {test_time:.4f} 秒")
            plot_path(path, save_path="successful_path.png")
            with open("optimal_path.csv", 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(['X', 'Y'])
                for (px, py) in path: writer.writerow([px, py])
            print("最优路径已保存至 optimal_path.csv")
        else:
            print("未能到达目标")
            plot_path(path, save_path="failed_path.png")

        save_training_data("training_results.csv", reward_history, cumulative_reward_history,
                           success_history, trajectory_length_history, path_length_history,
                           turns_history, episode_time_history, epsilon_history)
        # ========== 独立图表显示 ==========
        # 1. 总奖励曲线
        plt.figure(figsize=(10, 5))
        plt.plot(reward_history, alpha=0.4, color='blue', label='Episode Reward')
        if len(reward_history) >= 20:
            ma = moving_average(reward_history, 20)
            plt.plot(range(19, len(reward_history)), ma, 'b-', linewidth=2, label='20-ep Moving Avg')
        plt.title('Reward per Episode')
        plt.xlabel('Episode')
        plt.ylabel('Reward')
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.show()

        # 2. 累计回报曲线
        plt.figure(figsize=(10, 5))
        plt.plot(cumulative_reward_history, 'g-', linewidth=2)
        plt.title('Cumulative Reward')
        plt.xlabel('Episode')
        plt.ylabel('Cumulative Reward')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()

        # 3. 每回合耗时
        plt.figure(figsize=(10, 5))
        plt.plot(episode_time_history, 'c-', linewidth=1.5, alpha=0.8)
        plt.title('Time Cost per Episode')
        plt.xlabel('Episode')
        plt.ylabel('Time (seconds)')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()

        # 5. 轨迹长度（每回合步数）
        plt.figure(figsize=(10, 5))
        plt.plot(trajectory_length_history, 'm-', alpha=0.6, label='Steps per Episode')
        if len(trajectory_length_history) >= 20:
            ma_steps = moving_average(trajectory_length_history, 20)
            plt.plot(range(19, len(trajectory_length_history)), ma_steps, 'm-', linewidth=2, label='20-ep Moving Avg')
        plt.axhline(y=29, color='gray', linestyle='--', label='Chebyshev Min Steps (29)')
        plt.title('Trajectory Length per Episode')
        plt.xlabel('Episode')
        plt.ylabel('Steps')
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.show()

        # 6. 路径长度（仅成功回合）
        plt.figure(figsize=(10, 5))
        success_indices = [i for i, v in enumerate(success_history) if v == 1]
        if success_indices:
            success_lengths = [path_length_history[i] for i in success_indices]
            plt.scatter(success_indices, success_lengths, c='blue', alpha=0.5, s=20, label='Path Length (Success)')
            if len(success_lengths) >= 5:
                ma_len = moving_average(success_lengths, 5)
                plt.plot(success_indices[4:], ma_len, 'b-', linewidth=2, label='5-success Moving Avg')
            plt.axhline(y=29, color='gray', linestyle='--', label='Chebyshev Min Steps (29)')
            plt.legend()
        else:
            plt.text(0.5, 0.5, 'No successful episodes yet', ha='center', va='center', transform=plt.gca().transAxes)
        plt.title('Path Length (Only Successful Episodes)')
        plt.xlabel('Episode')
        plt.ylabel('Length (Steps)')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()

        # 7. 拐点数量（仅成功回合）
        plt.figure(figsize=(10, 5))
        if success_indices:
            success_turns = [turns_history[i] for i in success_indices]
            plt.scatter(success_indices, success_turns, c='orange', alpha=0.5, s=20, label='Turns (Success)')
            if len(success_turns) >= 5:
                ma_turns = moving_average(success_turns, 5)
                plt.plot(success_indices[4:], ma_turns, 'orange', linewidth=2, label='5-success Moving Avg')
            plt.legend()
        else:
            plt.text(0.5, 0.5, 'No successful episodes yet', ha='center', va='center', transform=plt.gca().transAxes)
        plt.title('Number of Turns (Only Successful Episodes)')
        plt.xlabel('Episode')
        plt.ylabel('Turns')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()

        # 4. 成功率（滑动窗口）
        plt.figure(figsize=(10, 5))
        window = 20
        if len(success_history) >= window:
            success_rate = moving_average(success_history, window) * 100
            plt.plot(range(window - 1, len(success_history)), success_rate, 'g-', linewidth=2)
        else:
            plt.plot(np.cumsum(success_history) / np.arange(1, len(success_history) + 1) * 100, 'g-')
        plt.title(f'Success Rate (Moving Avg, window={window})')
        plt.xlabel('Episode')
        plt.ylabel('Success Rate (%)')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()

        # 8. 探索率 (Epsilon) 衰减曲线
        plt.figure(figsize=(10, 5))
        plt.plot(epsilon_history, 'm-', linewidth=2)
        plt.title('Exploration Rate Decay')
        plt.xlabel('Episode')
        plt.ylabel('Epsilon')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()

        # ---------- 图表组 2：最终路径指标对比
        # 独立测试评估（无探索，动态障碍按固定路径移动）
        # 100次独立评估
        print("\n正在进行 100 次独立测试评估...")
        agent.epsilon = 0.0
        avg_sr, avg_len, avg_turns = evaluate_agent(agent, num_trials=100)

    else:
        agent.model.load_state_dict(torch.load("trained_model.pth", map_location=device))
        agent.epsilon = 0
        print("已加载预训练模型 trained_model.pth")