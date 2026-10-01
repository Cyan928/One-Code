#==========导入库
import time
import numpy as np                                 #提供矩阵运算和数学函数
import matplotlib.pyplot as plt                    #绘制所有的可视化内容
import random                                      #随机数生成库
import torch                                       #构建和训练神经网络
import torch.nn as nn                              #提供神经网络的层
import torch.optim as optim                        #提供算法优化，更新神经网络的权重
from collections import deque                      #记忆库

#=============设备配置
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"当前使用的设备：{device}")                    #device后续会用于将张量移动到指定设备上

#=================环境参数设置
GRID_SIZE = 30
START=(0,0)
GOAL=(GRID_SIZE-1,GRID_SIZE-1)

def generate_static_obstacles(num_obstacles=160,seed=25):
    np.random.seed(seed)
    obstacles = set()
    while len(obstacles) < num_obstacles:
        x = np.random.randint(0,GRID_SIZE)
        y = np.random.randint(0,GRID_SIZE)
        if (x,y)!= START and (x,y) != GOAL:
            obstacles.add((x,y))
    return list(obstacles)
STATIC_OBSTACLES = generate_static_obstacles(num_obstacles=160)
#调用函数生成80个障碍，存入全局变量

#==================DQN神经网络定义
# class DQN(nn.Module):
#     def __init__(self,input_size,output_size):
#         super(DQN,self).__init__()
#         self.fc1 = nn.Linear(input_size,256)
#         self.fc2 = nn.Linear(256,256)
#         self.fc3 = nn.Linear(256,256)
#         self.fc4 = nn.Linear(256,output_size) #定义三个全连接层
#     def forward(self,x):                                #定义数据在神经网络中的前向传播路径
#         x = torch.relu(self.fc1(x))
#         x = torch.relu(self.fc2(x))
#         x = torch.relu(self.fc3(x))
#         return self.fc4(x)                              #定义数据在神经网络中的前向传播路径
#===============Dueling DQN神经网络定义
#================== Dueling DQN 神经网络定义 ==================
class DuelingDQN(nn.Module):
    def __init__(self, input_size, output_size):
        super(DuelingDQN, self).__init__()
        # 共享特征层
        self.fc1 = nn.Linear(input_size, 256)
        self.fc2 = nn.Linear(256, 256)
        # 价值流 V(s)
        self.value_fc = nn.Linear(256, 128)
        self.value = nn.Linear(128, 1)
        # 优势流 A(s,a)
        self.advantage_fc = nn.Linear(256, 128)
        self.advantage = nn.Linear(128, output_size)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        # 价值分支
        v = torch.relu(self.value_fc(x))
        v = self.value(v)                     # shape: (batch, 1)
        # 优势分支
        a = torch.relu(self.advantage_fc(x))
        a = self.advantage(a)                 # shape: (batch, action_size)
        # 组合 Q = V + (A - mean(A))
        q = v + (a - a.mean(dim=1, keepdim=True))
        return q
#==============DQN智能体类
#======1）初始化函数init
class DQNAgent(nn.Module):
    def __init__(self):
        super(DQNAgent,self).__init__()
        self.state_size = 4        #4个状态
        self.action_size = 8       #动作
        self.memory = deque(maxlen=20000)                             #经验回放池容量
        self.gamma = 0.99                     # 原 0.90 → 0.99
        self.epsilon = 1.0
        self.epsilon_min = 0.10                # 原 0.15 → 0.05
        self.epsilon_decay = 0.995             # 原 0.998 → 0.995（适当加快衰减）
        self.learning_rate = 0.0005             # 原 0.005 → 0.001（稳定训练） #优化器的学习率，控制权重更新的步长

        self.model = DuelingDQN(self.state_size, self.action_size).to(device)
        self.target_model = DuelingDQN(self.state_size, self.action_size).to(device)
        # self.model = DQN(self.state_size,self.action_size).to(device) #当前网络，实时决策实时更新
        # self.target_model = DQN(self.state_size,self.action_size).to(device) #目标网络，参数更新慢，计算Q值，稳定训练
        self.target_model.load_state_dict(self.model.state_dict()) #当前网络的参数复制给目标网络，初始一致
        self.optimizer = optim.Adam(self.model.parameters(),lr=self.learning_rate) #优化器优化
        self.criterion = nn.MSELoss()                            #均方误差作为损失函数，衡量Q值

        self.train_step_count = 0
        self.target_model_update = 500                            #调用计数器记录训练次数 累计500更新一次目标网络

#========2）记忆存储函数
    def remember(self,state,action,reward,next_state,done):
        self.memory.append((state,action,reward,next_state,done)) #将一条经验5个元素打包成一个元组，添加到memory，满挤出旧

#=========3）动作选择函数
    def action(self,state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)      #生成一个0~1的随机数，小于epsilon则随机选择一个动作，否则状态转化为张量，输入
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)  # 增加 batch 维度
        with torch.no_grad():
            q_val = self.model(state_tensor)                              # 输出形状 (1, action_size)
        return q_val.squeeze(0).argmax().item()                           # 去除 batch 维后再取动作

#=======4）学习函数
    def replay(self,batch_size):
        if len(self.memory) < batch_size:
            return                                         #从经验池中随机抽取batch_size条经验，池中经验不足则直接返回不学习
        minibatch = random.sample(self.memory,batch_size)

        states = np.array([t[0] for t in minibatch],dtype=np.float32)
        actions = np.array([t[1] for t in minibatch],dtype=np.int64)
        rewards = np.array([t[2] for t in minibatch],dtype=np.float32)
        next_states = np.array([t[3] for t in minibatch],dtype=np.float32)
        dones = np.array([t[4] for t in minibatch],dtype=np.float32)

        states = torch.from_numpy(states).to(device)
        actions = torch.from_numpy(actions).to(device)
        rewards = torch.from_numpy(rewards).to(device)
        next_states = torch.from_numpy(next_states).to(device)
        dones = torch.from_numpy(dones).to(device)                       #将抽取的minibatch先转numpy再转化为张量

        current_q = self.model(states).gather(1,actions.unsqueeze(1)).squeeze(1)
        #计算当前网络对这批状态在每个实际执行动作上的Q值，self.model(states)输出形状（batch，4），每个状态对应4个动作的Q值
        #.gather(1,actions.unsqueeze(1))根据actions索引，取出4个动作的Q值
        #.squeeze(1)将形状从（batch,1)变为(batch, )
        # 经典 DQN 目标 Q 值
        # with torch.no_grad():
        #     next_q = self.target_model(next_states).max(1)[0]   # 目标网络直接取最大 Q 值
        #     target_q = rewards + self.gamma * next_q * (1 - dones)
        #Double DQN目标Q值
        with torch.no_grad():
            next_action = self.model(next_states).argmax(1).unsqueeze(1) #用当前网络选出下一个状态的最大Q值对应的动作索引
            next_q = self.target_model(next_states).gather(1,next_action).squeeze(1) #用目标网络评估这个动作的Q值
            target_q = rewards + self.gamma * next_q * (1 - dones)

        loss = self.criterion(current_q,target_q)
        self.optimizer.zero_grad()                                  #清空优化器中的梯度缓存
        loss.backward()                                             #反向传播，自动计算每个参数的梯度
        torch.nn.utils.clip_grad_norm_(self.model.parameters(),1.0) #对梯度进行裁剪，防止梯度爆炸
        self.optimizer.step()                                       #根据梯度更新网络权重

        self.train_step_count += 1
        if self.train_step_count % self.target_model_update == 0:
            self.update_target_model()                              #训练一次计数器加1，500的倍数将当前网络复制给目标网络

#============5）目标网络更新函数
    def update_target_model(self):
        self.target_model.load_state_dict(self.model.state_dict()) #复制网络

#===================环境辅助函数
#===============1)状态归一化函数

def get_state(x,y,gx,gy):
    """归一化状态0~1"""
    max_val = GRID_SIZE-1
    return np.array([x/max_val,y/max_val,gx/max_val,gy/max_val],dtype=np.float32)
#==============2）奖励函数
def get_reward(x, y, prev_dist=None):
    if (x, y) == GOAL:
        return 2000.0
    if (x, y) in STATIC_OBSTACLES:
        return -100.0
    cur_dist = abs(x - GOAL[0]) + abs(y - GOAL[1])
    if prev_dist is not None:
        return (prev_dist - cur_dist) * 1.0 - 0.5
    return -0.5
                        #计算当前到终点的曼哈顿距离cur_dist,提供了上一步距离prev_dist，返回 ------*0.5
#===============静态障碍地图绘制函数
def plot_static_map():
    plt.figure(figsize=(8, 8))
    # 绘制深红色障碍矩形（与 plot_path 风格一致）
    for ox, oy in STATIC_OBSTACLES:
        plt.gca().add_patch(plt.Rectangle((oy - 0.5, ox - 0.5), 1, 1,
                                          fill=True, color='darkred', alpha=0.8))
    # 绘制起点和终点
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

def test_agent(agent, render=True):
    x, y = START
    path = [(x, y)]
    state = get_state(x, y, GOAL[0], GOAL[1])
    done = False
    steps = 0
    max_steps = 500
    collision = False

    start_time = time.time()
    while not done and steps < max_steps:
        action = agent.action(state)
        nx, ny = x, y

        dir_map = {0: (-1, 0), 1: (1, 0), 2: (0, -1), 3: (0, 1),
                   4: (-1, -1), 5: (-1, 1), 6: (1, -1), 7: (1, 1)}
        dx, dy = dir_map[action]
        target_nx, target_ny = x + dx, y + dy

        if not (0 <= target_nx < GRID_SIZE and 0 <= target_ny < GRID_SIZE):
            # 直接跳过本次动作，不移动，但继续尝试（步数照常累计）
            steps += 1
            continue

        cross_obstacle = False
        if action >= 4:
            adj1 = (x, target_ny)
            adj2 = (target_nx, y)
            if (adj1 in STATIC_OBSTACLES) or (adj2 in STATIC_OBSTACLES):
                cross_obstacle = True

        if (target_nx, target_ny) in STATIC_OBSTACLES or cross_obstacle:
            steps += 1
            continue

        # 合法移动
        fail_count = 0
        x, y = target_nx, target_ny
        path.append((x, y))
        if (x, y) == GOAL:
            done = True
        state = get_state(x, y, GOAL[0], GOAL[1])
        steps += 1

    elapsed = time.time() - start_time
    path_has_obstacle = any((px, py) in STATIC_OBSTACLES for (px, py) in path)
    success = ((x, y) == GOAL) and (not path_has_obstacle) and (not collision)

    length = len(path) if success else None
    turns = count_turns(path) if success else None

    if render:
        plot_path(path)

    return success, path, length, turns, elapsed
def plot_path(path, save_path=None):
    # 创建网格，障碍物标记为 1
    grid = np.zeros((GRID_SIZE, GRID_SIZE))
    for ox, oy in STATIC_OBSTACLES:
        grid[ox, oy] = 1

    plt.figure(figsize=(10, 10))
    # 绘制障碍物底色（浅灰）和障碍物本身（深红）
    # 不调用 imshow，直接画矩形
    for ox, oy in STATIC_OBSTACLES:
        plt.gca().add_patch(plt.Rectangle((oy - 0.5, ox - 0.5), 1, 1,
                                          fill=True, color='darkred', alpha=0.8))
    # 再次绘制障碍物边界，增强立体感
    for ox, oy in STATIC_OBSTACLES:
        plt.gca().add_patch(plt.Rectangle((oy-0.5, ox-0.5), 1, 1,
                                          fill=True, color='darkred', alpha=0.7, linewidth=0))

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
    plt.xlim(-0.5, GRID_SIZE-0.5)
    plt.ylim(-0.5, GRID_SIZE-0.5)
    plt.grid(True, linestyle='-', linewidth=0.5, alpha=0.5, color='gray')
    plt.legend(fontsize=12, loc='upper left')
    plt.title(f"Path Planning (GRID {GRID_SIZE}x{GRID_SIZE})", fontsize=14)

    if save_path:
        plt.savefig(save_path, dpi=200, bbox_inches='tight')
        print(f"轨迹图已保存至: {save_path}")
    plt.show()
#=================辅助函数——滑动平均
def moving_average(data, window=20):
    if len(data) < window:
        return data
    return np.convolve(data, np.ones(window)/window, mode='valid')
#==================训练主程序
if __name__ == "__main__":
    agent = DQNAgent()
    #设置模式：True=重新训练，False=直接加载已有的模型测试


    TRAIN_MODE=True #改为False则跳过训练，直接加载模型测试


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
        print(f"当前静态障碍数量: {len(STATIC_OBSTACLES)}")
        train_start = time.time()

        for e in range(episodes):
            episode_start = time.time()
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
                action = agent.action(state)
                nx, ny = x, y
                dir_map = {0: (-1, 0), 1: (1, 0), 2: (0, -1), 3: (0, 1),
                           4: (-1, -1), 5: (-1, 1), 6: (1, -1), 7: (1, 1)}
                dx, dy = dir_map[action]
                target_nx, target_ny = x + dx, y + dy

                # 边界检查
                if not (0 <= target_nx < GRID_SIZE and 0 <= target_ny < GRID_SIZE):
                    reward = -5.0
                    next_state = state
                    done = False
                    agent.remember(state, action, reward, next_state, done)
                    state = next_state
                    total_reward += reward
                    steps += 1
                    continue

                # 穿角检测
                cross_obstacle = False
                if action >= 4:
                    adj1 = (x, target_ny)
                    adj2 = (target_nx, y)
                    if (adj1 in STATIC_OBSTACLES) or (adj2 in STATIC_OBSTACLES):
                        cross_obstacle = True

                # 目标格子是否为障碍
                target_obstacle = (target_nx, target_ny) in STATIC_OBSTACLES

                if target_obstacle or cross_obstacle:
                    reward = -5.0
                    next_state = state
                    done = False
                else:
                    x, y = target_nx, target_ny
                    episode_path.append((x, y))
                    cur_dist = abs(x - GOAL[0]) + abs(y - GOAL[1])
                    reward = get_reward(x, y, prev_dist)
                    prev_dist = cur_dist
                    next_state = get_state(x, y, GOAL[0], GOAL[1])

                    if (x, y) == GOAL or (x, y) in STATIC_OBSTACLES:
                        episode_collision = ((x, y) in STATIC_OBSTACLES)
                        episode_success = ((x, y) == GOAL)
                        done = True

                agent.remember(state, action, reward, next_state, done)
                state = next_state
                total_reward += reward
                steps += 1

                if len(agent.memory) >= batch_size:
                    agent.replay(batch_size)
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

            if e % 50 == 0:
                print(f"Episode {e:4d} | Reward: {total_reward:7.2f} | Steps: {steps:3d} | "
                      f"Epsilon: {agent.epsilon:.3f} | Success: {episode_success} ")

        train_end = time.time()
        training_time = train_end - train_start

        torch.save(agent.model.state_dict(), "trained_model.pth")
        print("模型已保存为 trained_model.pth")


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
            print(f"成功到达目标")
            print(f"   - 路径长度 (步数): {length}")
            print(f"   - 拐点数量: {turns}")
            print(f"   - 测试规划耗时: {test_time:.4f} 秒")
            plot_path(path, save_path="successful_path.png")
        else:
            print("未能到达目标")
            plot_path(path, save_path="failed_path.png")  # 失败保存，便于分析
        # ========== 独立图表显示 ==========
        # 1. 总奖励曲线
        plt.figure(figsize=(10, 5))
        plt.plot(reward_history, alpha=0.4, color='blue', label='Episode Reward')
        if len(reward_history) >= 20:
            ma = moving_average(reward_history, 20)
            plt.plot(range(19, len(reward_history)), ma, 'b-', linewidth=2, label='20-ep Moving Avg')
        plt.title('Reward per Episode')
        plt.ylim(-9500,2500)
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
        plt.ylim(-0.2,7.2)
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
        plt.ylim(-50,2100)
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

        plt.ylim(-50, 2100)

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

        plt.ylim(-50,1500)

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
        plt.ylim([0, 105])
        plt.tight_layout()
        plt.show()

        # 8. 探索率 (Epsilon) 衰减曲线
        plt.figure(figsize=(10, 5))
        plt.plot(epsilon_history, 'm-', linewidth=2)
        plt.title('Exploration Rate Decay')
        plt.ylim(0, 1.2)
        plt.xlabel('Episode')
        plt.ylabel('Epsilon')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()

          # ---------- 图表组 2：最终路径指标对比
        if success_indices:
            last_success_len = np.mean(success_lengths[-10:]) if len(success_lengths) >= 10 else np.mean(success_lengths)
            last_success_turns = np.mean(success_turns[-10:]) if len(success_turns) >= 10 else np.mean(success_turns)
            fig3, ax3 = plt.subplots(figsize=(8, 6))
            categories = ['Path Length', 'Turns', 'Min Possible Steps']
            values = [last_success_len, last_success_turns, 29]
            colors = ['steelblue', 'darkorange', 'lightgray']
            bars = ax3.bar(categories, values, color=colors, edgecolor='black')
            ax3.set_ylabel('Count / Steps')
            ax3.set_title('Final Policy Performance (Average of Last 10 Successes)')
            for bar, val in zip(bars, values):
                ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5, f'{val:.1f}',
                        ha='center', va='bottom', fontweight='bold')
            ax3.grid(axis='y', alpha=0.3)
            plt.ylim(0, 55)
            plt.tight_layout()
            plt.show()

    else: #加载已有模型
        agent.model.load_state_dict(torch.load("trained_model.pth",map_location = device))
        agent.epsilon = 0 #测试时关闭探索
        print("已加载预训练模型 trained_model.pth")