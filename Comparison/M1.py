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

#=============设备配置
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"当前使用的设备：{device}")

#=================环境参数设置
GRID_SIZE = 30
START = (0,0)
GOAL = (GRID_SIZE-1,GRID_SIZE-1)

#==========静态障碍生成函数
def generate_static_obstacles(num_obstacles=140,seed=100):
    np.random.seed(seed)
    obstacles = set()
    while len(obstacles) < num_obstacles:
        x = np.random.randint(0,GRID_SIZE)
        y = np.random.randint(0,GRID_SIZE)
        if (x,y)!= START and (x,y) != GOAL:
            obstacles.add((x,y))
    return list(obstacles)
STATIC_OBSTACLES = generate_static_obstacles(num_obstacles=140)

#================动态障碍生成函数
def generate_dynamic_obstacles(num_dynamic=20):
    dy_obs = set()
    attempts = 0
    while len(dy_obs) < num_dynamic and attempts < 500:
        x = np.random.randint(0, GRID_SIZE)
        y = np.random.randint(0, GRID_SIZE)
        if (x, y) != START and (x, y) != GOAL and (x, y) not in STATIC_OBSTACLES:
            dy_obs.add((x, y))
        attempts += 1
    return list(dy_obs)
NUM_DYNAMIC = 20
DYNAMIC_OBSTACLES = generate_dynamic_obstacles(NUM_DYNAMIC)

#================动态障碍巡逻路径
def generate_patrol_paths(num_paths=4, path_length=10, grid_size=GRID_SIZE):
    paths = []
    for _ in range(num_paths):
        path = []
        attempts = 0
        while len(path) < path_length and attempts < 500:
            if not path:
                x = np.random.randint(0, grid_size)
                y = np.random.randint(0, grid_size)
                if (x,y) not in STATIC_OBSTACLES and (x,y)!=START and (x,y)!=GOAL:
                    path.append((x,y))
            else:
                last_x, last_y = path[-1]
                neighbors = [(last_x-1, last_y), (last_x+1, last_y),
                             (last_x, last_y-1), (last_x, last_y+1)]
                np.random.shuffle(neighbors)
                for nx, ny in neighbors:
                    if 0<=nx<grid_size and 0<=ny<grid_size:
                        if (nx,ny) not in STATIC_OBSTACLES and (nx,ny) not in path \
                           and (nx,ny)!=START and (nx,ny)!=GOAL:
                            path.append((nx,ny))
                            break
            attempts += 1
        if len(path)>=4: paths.append(path)
    return paths

PATROL_PATHS = generate_patrol_paths(num_paths=NUM_DYNAMIC, path_length=10)
while len(PATROL_PATHS) < NUM_DYNAMIC:
    dummy_path = [(1,1),(2,1),(2,2),(1,2)]
    if all((x,y) not in STATIC_OBSTACLES and (x,y)!=START and (x,y)!=GOAL for x,y in dummy_path):
        PATROL_PATHS.append(dummy_path)
    else:
        PATROL_PATHS.append([(1,1),(1,2),(2,2),(2,1)])

#================动态障碍移动函数
def move_dynamic_obstacles_patrol(dyn_obs, path_indices, directions, agent_pos=None):
    new_dyn = []
    new_indices = []
    new_dirs = []
    for i, (x,y) in enumerate(dyn_obs):
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
            new_dyn.append((x,y))
            new_indices.append(idx)
            new_dirs.append(dir)
        else:
            new_dyn.append(next_pos)
            new_indices.append(next_idx)
            new_dirs.append(dir)
    return new_dyn, new_indices, new_dirs

#================== Dueling DQN 网络（不含BN，但保留与M5相同的结构以公平对比）
class DuelingDQN(nn.Module):
    def __init__(self, input_size, output_size):
        super(DuelingDQN, self).__init__()
        self.fc1 = nn.Linear(input_size, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 128)        # 与M5保持一致
        # 价值流
        self.value_fc = nn.Linear(128, 128)
        self.value = nn.Linear(128, 1)
        # 优势流
        self.advantage_fc = nn.Linear(128, 128)
        self.advantage = nn.Linear(128, output_size)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        v = torch.relu(self.value_fc(x))
        v = self.value(v)
        a = torch.relu(self.advantage_fc(x))
        a = self.advantage(a)
        q = v + (a - a.mean(dim=1, keepdim=True))
        return q

#==============DQN智能体类 (M1)
class DQNAgent(nn.Module):
    def __init__(self):
        super(DQNAgent,self).__init__()
        # 状态：4坐标 + 25局部视野 = 29
        self.state_size = 29
        self.action_size = 8
        # 普通经验回放
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
        self.target_model_update = 500   # 硬更新间隔

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

#===================环境辅助函数
def get_local_view(x, y, radius=2):
    view = np.zeros((2*radius+1, 2*radius+1), dtype=np.float32)
    for i in range(-radius, radius+1):
        for j in range(-radius, radius+1):
            nx, ny = x + i, y + j
            if 0 <= nx < GRID_SIZE and 0 <= ny < GRID_SIZE:
                if (nx, ny) in STATIC_OBSTACLES:
                    view[i+radius, j+radius] = 1.0
                elif (nx, ny) in DYNAMIC_OBSTACLES:
                    view[i+radius, j+radius] = 2.0
                elif (nx, ny) == GOAL:
                    view[i+radius, j+radius] = 3.0
    return view.flatten()

def get_state(x, y, gx, gy):
    max_val = GRID_SIZE - 1
    pos = np.array([x/max_val, y/max_val, gx/max_val, gy/max_val], dtype=np.float32)
    local_5x5 = get_local_view(x, y, radius=2)
    return np.concatenate([pos, local_5x5])

def get_reward(x, y, prev_dist=None):
    if (x, y) == GOAL:
        return 2000.0
    if (x, y) in STATIC_OBSTACLES or (x, y) in DYNAMIC_OBSTACLES:
        return -100.0
    cur_dist = abs(x - GOAL[0]) + abs(y - GOAL[1])
    if prev_dist is not None:
        return (prev_dist - cur_dist) * 1.0 - 0.5
    return -0.5

# 以下绘图、评估函数与M5相同，仅需将状态构造改为只用get_state(x,y,GOAL[0],GOAL[1])
# 此处省略重复代码，实际使用时应保留 test_agent、evaluate_agent 等函数，并去除其中与velocity/history相关的部分
# 训练循环中奖励直接使用 get_reward，无势场、无区域奖励、无平滑惩罚
# 删除 compute_potential 及相关调用
                        #计算当前到终点的曼哈顿距离cur_dist,提供了上一步距离prev_dist，返回 ------*0.5
#===============静态障碍地图绘制函数
def plot_static_map():
    plt.figure(figsize=(8, 8))
    # 绘制深红色障碍矩形（与 plot_path 风格一致）
    first_static = True
    for ox, oy in STATIC_OBSTACLES:
        label = 'Static Obstacle' if first_static else ""
        plt.gca().add_patch(plt.Rectangle((oy - 0.5, ox - 0.5), 1, 1,
                                          fill=True, color='darkred', alpha=0.8, label=label))
        first_static = False
    # 动态障碍（橙色）
    first_dyn = True
    for ox, oy in DYNAMIC_OBSTACLES:
        label = 'Dynamic Obstacle' if first_dyn else ""
        plt.gca().add_patch(plt.Rectangle((oy - 0.5, ox - 0.5), 1, 1,
                                          fill=True, color='orange', alpha=0.8, label=label))
        first_dyn = False
    # 起点和终点（保持不变）
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
#===============路径评估与绘图函数
def count_turns(path):
    if len(path) < 3:
        return 0
    turns = 0
    for i in range(1, len(path) - 1):
        prev = path[i-1]
        curr = path[i]
        nxt = path[i+1]
        if (prev[0] - curr[0], prev[1] - curr[1]) != (curr[0] - nxt[0], curr[1] - nxt[1]):
            turns += 1
    return turns
#===============测试函数
def test_agent(agent, render=True):
    global DYNAMIC_OBSTACLES
    agent.model.eval()
    dyn_obs = [PATROL_PATHS[i][0] for i in range(NUM_DYNAMIC)]
    DYNAMIC_OBSTACLES = dyn_obs
    path_indices = [0] * NUM_DYNAMIC
    directions = [1] * NUM_DYNAMIC
    x, y = START
    path = [(x, y)]
    state = get_state(x, y, GOAL[0], GOAL[1])
    done = False
    steps = 0
    max_steps = 2000
    collision = False

    start_time = time.time()
    while not done and steps < max_steps:
        # 1. 移动动态障碍
        dyn_obs, path_indices, directions = move_dynamic_obstacles_patrol(
            dyn_obs, path_indices, directions, agent_pos=None)
        DYNAMIC_OBSTACLES = dyn_obs

        # 2. 构造状态
        state = get_state(x, y, GOAL[0], GOAL[1])

        # 3. 动作选择
        action = agent.action(state)
        nx, ny = x, y
        dir_map = {0: (-1,0), 1: (1,0), 2: (0,-1), 3: (0,1),
                   4: (-1,-1), 5: (-1,1), 6: (1,-1), 7: (1,1)}
        dx, dy = dir_map[action]
        target_nx, target_ny = x + dx, y + dy

        # 边界检查
        if not (0 <= target_nx < GRID_SIZE and 0 <= target_ny < GRID_SIZE):
            state = get_state(x, y, GOAL[0], GOAL[1])
            steps += 1
            continue

        # 穿角/目标障碍检测
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
            state = get_state(x, y, GOAL[0], GOAL[1])
            steps += 1
            continue

        # 合法移动
        x, y = target_nx, target_ny
        path.append((x, y))

        # 关键：立即构造下一状态（与训练完全一致）
        next_state = get_state(x, y, GOAL[0], GOAL[1])
        state = next_state
        steps += 1

        if (x, y) == GOAL:
            done = True
        elif (x, y) in STATIC_OBSTACLES or (x, y) in DYNAMIC_OBSTACLES:
            collision = True
            done = True

    elapsed = time.time() - start_time
    path_has_obstacle = any((px, py) in STATIC_OBSTACLES for (px, py) in path)
    success = ((x, y) == GOAL) and (not path_has_obstacle) and (not collision)
    length = len(path) if success else None
    turns = count_turns(path) if success else None
    if render:
        plot_path(path)
    return success, path, length, turns, elapsed

def evaluate_agent(agent, num_trials=100, max_steps=2000):
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
        path = [(x, y)]
        state = get_state(x, y, GOAL[0], GOAL[1])
        done = False
        steps = 0
        collision = False

        while not done and steps < max_steps:
            dyn_obs, path_indices, directions = move_dynamic_obstacles_patrol(
                dyn_obs, path_indices, directions, agent_pos=None)
            DYNAMIC_OBSTACLES = dyn_obs

            state = get_state(x, y, GOAL[0], GOAL[1])

            action = agent.action(state)
            dir_map = {0: (-1,0), 1: (1,0), 2: (0,-1), 3: (0,1),
                       4: (-1,-1), 5: (-1,1), 6: (1,-1), 7: (1,1)}
            dx, dy = dir_map[action]
            target_nx, target_ny = x + dx, y + dy

            if not (0 <= target_nx < GRID_SIZE and 0 <= target_ny < GRID_SIZE):
                state = get_state(x, y, GOAL[0], GOAL[1])
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
                state = get_state(x, y, GOAL[0], GOAL[1])
                steps += 1
                continue

            x, y = target_nx, target_ny
            path.append((x, y))
            next_state = get_state(x, y, GOAL[0], GOAL[1])
            state = next_state
            steps += 1

            if (x, y) == GOAL:
                done = True
            elif (x, y) in STATIC_OBSTACLES or (x, y) in DYNAMIC_OBSTACLES:
                collision = True
                done = True

        path_has_obstacle = any((px, py) in STATIC_OBSTACLES for (px, py) in path)
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
#=============路径绘图函数
def plot_path(path, save_path=None):
    grid = np.zeros((GRID_SIZE, GRID_SIZE))
    for ox, oy in STATIC_OBSTACLES:
        grid[ox, oy] = 1

    plt.figure(figsize=(8, 8))
    # 第一次绘制静态障碍（带标签）
    first_static = True
    for ox, oy in STATIC_OBSTACLES:
        label = 'Static Obstacle' if first_static else ""
        plt.gca().add_patch(plt.Rectangle((oy - 0.5, ox - 0.5), 1, 1,
                                          fill=True, color='darkred', alpha=0.8, label=label))
        first_static = False
    # 第二次绘制静态障碍（无标签，仅增强立体感）
    for ox, oy in STATIC_OBSTACLES:
        plt.gca().add_patch(plt.Rectangle((oy-0.5, ox-0.5), 1, 1,
                                          fill=True, color='darkred', alpha=0.7, linewidth=0))
    # 动态障碍（橙色）
    first_dyn = True
    for ox, oy in DYNAMIC_OBSTACLES:
        label = 'Dynamic Obstacle' if first_dyn else ""
        plt.gca().add_patch(plt.Rectangle((oy - 0.5, ox - 0.5), 1, 1,
                                          fill=True, color='orange', alpha=0.8, label=label))
        first_dyn = False
    # 绘制起点和终点
    plt.scatter(START[1], START[0], c='lime', marker='o', s=200,
                edgecolors='black', linewidth=2, label='Start', zorder=5)
    plt.scatter(GOAL[1], GOAL[0], c='gold', marker='*', s=300,
                edgecolors='black', linewidth=2, label='Goal', zorder=5)

    # 绘制路径：先画背景圆点（显示占据的格子），再画连接线
    if path:
        py, px = zip(*path)
        # 在路径格子上绘制半透明青色圆点
        plt.scatter(px, py, c='cyan', s=120, alpha=0.6, edgecolors='none', label='Path Cell', zorder=3)
        # 绘制蓝色连接线
        plt.plot(px, py, 'b-', linewidth=3, alpha=0.9, label='Path Line', zorder=4)

    # 设置坐标轴
    plt.xticks(range(GRID_SIZE))
    plt.yticks(range(GRID_SIZE))
    plt.tick_params(axis='both', labelsize=14)
    plt.xlim(-0.5, GRID_SIZE-0.5)
    plt.ylim(-0.5, GRID_SIZE-0.5)
    plt.grid(True, linestyle='-', linewidth=0.5, alpha=0.5, color='gray')
    plt.legend(fontsize=18, loc='upper left')
   # plt.title(f"Path Planning (GRID {GRID_SIZE}x{GRID_SIZE})", fontsize=14)
    plt.tight_layout(pad=0.5)
    if save_path:
        plt.savefig(save_path, dpi=200, bbox_inches='tight')
        print(f"轨迹图已保存至: {save_path}")
    plt.show()
#=================辅助函数——滑动平均
def moving_average(data, window=20):
    if len(data) < window:
        return data
    return np.convolve(data, np.ones(window)/window, mode='valid')
#=================进度函数
def print_group_progress(completed, total, bar_length=20):
    """
    在控制台用 \r 覆盖打印组内进度条
    例如: 组内进度: |████████░░░░░░░░░░░░| 50% (25/50)
    """
    percent = completed / total
    filled_len = int(bar_length * percent)
    bar = '█' * filled_len + '░' * (bar_length - filled_len)
    print(f'\r组内进度: |{bar}| {percent:.0%} ({completed}/{total})', end='', flush=True)
#==========数据保存函数
def save_training_data(filename,
                       reward_history, cumulative_reward_history,
                       success_history, trajectory_length_history,
                       path_length_history, turns_history,
                       episode_time_history, epsilon_history):
    """将训练统计数据保存为 CSV 文件，便于导入 Origin"""

    # 确定最大长度（所有列表长度相同）
    n = len(reward_history)

    with open(filename, 'w', newline='') as f:
        writer = csv.writer(f)
        # 写入表头
        writer.writerow([
            'Episode', 'Reward', 'CumulativeReward', 'Success',
            'TrajectoryLength', 'PathLength', 'Turns',
            'TimeCost', 'Epsilon'
        ])
        for i in range(n):
            # 路径长度和拐点仅在成功时有效，否则留空
            pl = path_length_history[i] if not np.isnan(path_length_history[i]) else ''
            tr = turns_history[i] if not np.isnan(turns_history[i]) else ''
            writer.writerow([
                i,
                reward_history[i],
                cumulative_reward_history[i],
                int(success_history[i]),
                trajectory_length_history[i],
                pl,
                tr,
                episode_time_history[i],
                epsilon_history[i]
            ])
    print(f"训练数据已保存至 {filename}")

#==================训练主程序
if __name__ == "__main__":
    agent = DQNAgent()
    #设置模式：True=重新训练，False=直接加载已有的模型测试

    TRAIN_MODE=False #改为False则跳过训练，直接加载模型测试

    if TRAIN_MODE:
        episodes = 2000
        batch_size = 64

        # 基础统计
        reward_history = []
        success_history = []
        steps_history = []
        epsilon_history = []

        # 新增统计
        cumulative_reward_history = []   # 累计回报
        episode_time_history = []        # 每回合耗时
        path_length_history = []         # 成功时的路径长度
        trajectory_length_history = []   # 每回合实际步数
        turns_history = []               # 成功时的拐点数

        total_success = 0
        cumulative_reward = 0.0

        print("开始训练...")
        print(f"静态障碍数量: {len(STATIC_OBSTACLES)}，动态障碍数量: {len(DYNAMIC_OBSTACLES)}")
        train_start = time.time()

        group_size = 50  # 每组包含的 episode 数
        ep_in_group = 0  # 当前组内已完成的 episode 数

        # 巡逻路径已在全局生成，此处为每个动态障碍绑定初始索引和方向
        path_indices = [0] * 4       # 都从路径起点开始
        directions = [1] * 4        # 初始前进方向
        for e in range(episodes):
            episode_start = time.time()
            # 重置动态障碍到各自路径的起点
            dyn_obs = [PATROL_PATHS[i][0] for i in range(NUM_DYNAMIC)]
            path_indices = [0] * NUM_DYNAMIC
            directions = [1] * NUM_DYNAMIC
            x, y = START
            prev_dist = abs(x - GOAL[0]) + abs(y - GOAL[1])
            state = get_state(x, y, GOAL[0], GOAL[1])
            total_reward = 0
            done = False
            steps = 0
            max_episode_steps = 2000

            episode_collision = False
            episode_success = False
            episode_path = [(x, y)]   # 记录完整路径用于计算拐点

            while not done and steps < max_episode_steps:
                # --- 移动动态障碍（巡逻模式，避开智能体当前位置）---
                dyn_obs, path_indices, directions = move_dynamic_obstacles_patrol(
                    dyn_obs, path_indices, directions, agent_pos=None
                )
                DYNAMIC_OBSTACLES = dyn_obs

                action = agent.action(state)
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
                    # 刷新状态
                    state = get_state(x, y, GOAL[0], GOAL[1])
                    if len(agent.memory) >= batch_size:
                        agent.replay(batch_size)
                    continue

                # 穿角 / 目标为障碍
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
                    # → 补充结束
                    if len(agent.memory) >= batch_size:
                        agent.replay(batch_size)
                else:
                    # 合法移动
                    x, y = target_nx, target_ny
                    episode_path.append((x, y))
                    cur_dist = abs(x - GOAL[0]) + abs(y - GOAL[1])
                    reward = get_reward(x, y, prev_dist)
                    prev_dist = cur_dist
                    next_state = get_state(x, y, GOAL[0], GOAL[1])

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
            # 每回合结束更新 epsilon
            if agent.epsilon > agent.epsilon_min:
                agent.epsilon *= agent.epsilon_decay

            # 记录数据
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

            # -------- 组内进度条更新 --------
            ep_in_group += 1
            print_group_progress(ep_in_group, group_size)

            # 若该组已完成（满50个 episode）
            if ep_in_group == group_size:
                print()  # 换行，结束当前进度条行
                print(f"Episode {e + 1:4d} | Reward: {total_reward:7.2f} | Steps: {steps:3d} | "
                      f"Epsilon: {agent.epsilon:.3f} | Success: {episode_success}")
                sys.stdout.flush()   # 强制刷新，确保 IDE 控制台及时显示
                ep_in_group = 0  # 重置，准备下一组

        train_end = time.time()
        training_time = train_end - train_start

        torch.save(agent.model.state_dict(), "M1_trained_model.pth")
        print("模型已保存为 M1_trained_model.pth")

        # ---------- 输出累计统计结果 ----------
        print("\n" + "="*50)
        print("训练统计结果")
        print("="*50)
        print(f"总训练回合数: {episodes}")
        print(f"累计总奖励: {cumulative_reward:.2f}")
        print(f"平均每回合奖励: {cumulative_reward/episodes:.2f}")
        print(f"总成功次数: {total_success}  | 成功率: {total_success/episodes*100:.2f}%")
        print(f"训练总耗时: {training_time:.2f} 秒")
        print("="*50)

        # ---------- 显示静态障碍地图 ----------
        print("\n正在显示静态障碍地图...")
        plot_static_map()

        # ---------- 测试最终规划效果 ----------
        print("\n开始测试规划效果...")
        agent.epsilon = 0.0  # 关闭探索
        success, path, length, turns, test_time = test_agent(agent, render=False)

        if success:
            # 打印成功信息
            print(f"成功到达目标")
            print(f"   - 路径长度 (步数): {length}")
            print(f"   - 拐点数量: {turns}")
            print(f"   - 测试规划耗时: {test_time:.4f} 秒")
            # 保存轨迹图
            plot_path(path, save_path="M1_successful_path.png")
            # 保存最优路径 CSV
            with open("M1_optimal_path.csv", 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(['X', 'Y'])
                for (px, py) in path:
                    writer.writerow([px, py])
            print("最优路径已保存至 M1_optimal_path.csv")
        else:
            print("未能到达目标")
            plot_path(path, save_path="M1_failed_path.png")
        # ---------- 保存训练数据至 CSV（便于导入 Origin）
        save_training_data(
            "M1_training_results.csv",
            reward_history, cumulative_reward_history,
            success_history, trajectory_length_history,
            path_length_history, turns_history,
            episode_time_history, epsilon_history
        )
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
            plt.plot(range(window-1, len(success_history)), success_rate, 'g-', linewidth=2)
        else:
            plt.plot(np.cumsum(success_history)/np.arange(1, len(success_history)+1)*100, 'g-')
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
        print("\n正在进行 100 次独立测试评估...")
        agent.epsilon = 0.0
        avg_sr, avg_len, avg_turns = evaluate_agent(agent, num_trials=100)

    else:  # 加载已有模型
        agent.model.load_state_dict(torch.load("M1_trained_model.pth", map_location=device))
        agent.epsilon = 0
        print("已加载预训练模型 M1_trained_model.pth")
        # 测试最终规划效果
        print("\n开始测试规划效果...")
        success, path, length, turns, test_time = test_agent(agent, render=False)
        if success:
            print(f"成功到达目标")
            print(f"   - 路径长度 (步数): {length}")
            print(f"   - 拐点数量: {turns}")
            print(f"   - 测试规划耗时: {test_time:.4f} 秒")
            plot_path(path, save_path="M1_successful_path.png")
            with open("M1_optimal_path.csv", 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(['X', 'Y'])
                for (px, py) in path:
                    writer.writerow([px, py])
            print("最优路径已保存至 M1_optimal_path.csv")
        else:
            print("未能到达目标")
            plot_path(path, save_path="M1_failed_path.png")
        # 100次独立评估
        print("\n正在进行 100 次独立测试评估...")
        avg_sr, avg_len, avg_turns = evaluate_agent(agent, num_trials=100)