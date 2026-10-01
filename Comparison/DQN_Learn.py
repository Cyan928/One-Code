#导入块
import numpy as np
import matplotlib.pyplot as plt #绘图
import random                   #python自带的随机工具箱
import torch                    #提供张量计算
import torch.nn as nn           #提供神经网络的各种“积木块”
import torch.optim as optim     #提供优化器，更新神经网络的参数
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"当前使用设备: {device}")   #检测到cuda就使用GPU计算
from collections import deque   #从python模块C当中拿出工具d（固定长度的记忆库）
#地图障碍物设置
GRID_SIZE = 12                  #定义变量，值为12，12*12的方格世界，横纵11。
START=(0,0)                     #定义起点坐标
GOAL=(GRID_SIZE-1,GRID_SIZE-1)  #定义终点坐标（11，11）
STATIC_OBSTACLES=[(3,3),(3,4),(4,3),(5,5),(6,6),(7,8),(8,7)] #定义固定障碍物坐标
DYNAMIC_OBSTACLES=[(2,6),(6,2),(7,3),(9,5)] #定义动态障碍物初始位置
#定义神经网络
class DQN(nn.Module):           #定义类名DQN，类继承自（nn.module）
    def __init__(self,input_dim,output_dim):               #输入给神经网络的特征个数，神经网络输出动作个数
        super(DQN,self).__init__()                         #调用父类nn.module的初始化函数
        self.fc1 = nn.Linear(input_dim,64)      #输入层，接收input_dim，输出64个数字
        self.fc2 = nn.Linear(64,64)   #隐藏层
        self.fc3 = nn.Linear(64,64)   #输出层，接收64个数字，输出output_dim
    def forward(self,x):                                   #定义神经网络中的数据流动路径
        x = torch.relu(self.fc1(x))                        #输入x喂给第一层fc1，经过ReLU函数
        x = torch.relu(self.fc2(x))                        #经过第二层和ReLU
        return self.fc3(x)                                 #经过第三层，不用激活函数，直接返回结果。
#定义智能体Agent
class DQNAgent:                 #类包含了智能体所有的“零部件”和行为，神经网络、记忆库、决策方法、学习方法
    def __init__(self):         #初始化智能体
        self.state_size=4
        self.action_size=4      #状态包含4个数字，动作有4种
        self.memory = deque(maxlen=10000)                #创建一个记忆库，最多能存一万条经验，经验回放机制
        self.gamma = 0.95                                #定义折扣因子
        self.epsilon = 1.0                               #三个变量平衡探索与利用的平衡，初始1.0，100%随机探索
        self.epsilon_min = 0.01                          #探索率最低降到0.01，1%随机走，99%策略走
        self.epsilon_decay = 0.995                       #每次学习后epsilon*0.995,逐步减少探索
        self.model=DQN(self.state_size,self.action_size).to(device)#用定义的类创建具体的神经网络实例
        self.optimizer = optim.Adam(self.model.parameters(),lr=0.001)

        self.target_model = DQN(self.state_size, self.action_size).to(device)
        self.target_model.load_state_dict(self.model.state_dict())  # 初始同步参数

        #创建一个优化器，根据误差调整神经网络权重,Adam是一种常用的优化算法，lr(学习率)每次调整的步长
    def remember(self,state,action,reward,next_state,done): self.memory.append((state,action,reward,next_state,done))
    #将经验存入记忆库，一条经验包含：当前状态，执行的动作、得到的奖励、下一个状态、是否结束
    def act(self,state):                                 #智能体根据当前的状态决定采取那个动作
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)    #生成0~1随机数小于等于epsilon，随机选择一个动作，探索
        state=torch.FloatTensor(state).to(device)        #状态数组变为张量格式
        act_values = self.model(state)                   #状态喂给神经网络，得到四个动作的Q值
        return np.argmax(act_values.cpu().detach().numpy())    #返回Q值最大的动作的索引，利用
    def replay(self,batch_size):                         #从记忆库当中抽取一小批经验，用来训练神经网络
        minibatch = random.sample(self.memory,batch_size)      #随机抽取经验
        for state, action,reward,next_state,done in minibatch: #循环处理抽到的每一条经验
            state=torch.FloatTensor(state).to(device)
            next_state=torch.FloatTensor(next_state).to(device)           #将状态数据转化为pytorch张量

            target=reward
            if not done:
                target=reward + self.gamma*torch.max(self.target_model(next_state)).item() #贝尔曼方程，计算目标Q值
                target_f=self.model(state).cpu().detach().numpy()
                target_f[action]=target                              #获取当前状态下四个动作的Q值，实际采取动作的Q值，替换成刚才的目标值

                self.optimizer.zero_grad()                           #pytorch神经训练四部曲，把上一轮的梯度清零
                loss=nn.MSELoss()(self.model(state),torch.FloatTensor(target_f).to(device)) #计算损失
                loss.backward()                                      #反向传播，计算每个权重对损失的贡献（梯度）
                self.optimizer.step()                                #根据梯度，用优化器更新权重，大脑变聪明
            if self.epsilon > self.epsilon_min:
                self.epsilon *= self.epsilon_decay                   #每次学完后，探索率大于最小值，就乘以衰减系数0.995，往后越少探索
#动态障碍移动函数
def move_dynamic_obstacles(dyn_obs):                             #定义一个函数，输入当前所有的动态障碍物的坐标，返回移动一步后的新坐标列表
    new_dyn=[]                                                   #创建一个空列表，装移动后的新坐标
    for (x,y) in dyn_obs:                                        #遍历列表中的每一个障碍物坐标
        dx,dy=random.choice([(-1,0),(1,0),(0,-1),(0,1)])         #随机选择一个方向
        nx=x+dx
        ny=y+dy                                                  #计算移动后的新坐标

        if 0<=nx<GRID_SIZE and 0<=ny<GRID_SIZE:
            new_dyn.append((nx,ny))
        else:
            new_dyn.append((dx,dy))                              #新坐标没出边界采用新坐标，否则原地不动
    return new_dyn                                               #移动后的新坐标返回给调用者
#环境交互函数
def get_state(x,y,gx,gy):
    return np.array([x,y,gx,gy])                                 #把智能体当前位置和目标位置打包成数组
def get_reward(x,y,done):                                        #根据当前位置和是否结束计算并返回一个奖励值
    if (x,y)==GOAL:
        return 100
    if (x,y) in STATIC_OBSTACLES or (x,y) in DYNAMIC_OBSTACLES:
        return -50
    dist_to_goal = abs(x-GOAL[0])+abs(y-GOAL[1])                 #计算当前位置到重点的曼哈顿距离，基础步数惩罚-0.1，靠近奖励越近越大
    return -0.1+0.3/(dist_to_goal+1)
#训练主程序
agent = DQNAgent() #创建一个DQN智能体实例
episodes = 900                                    #训练总回合数
batch_size = 32                                   #每次从记忆库抽取32条经验来学习
reward_history = []                               #准备一个空的列表，记录每回合的总奖励数，最后绘图
TARGET_UPDATE_FREQ = 10
for e in range(episodes):                         #开始一个循环，循环300次，每次代表一个episode
    x,y=START
    state = get_state(x,y,GOAL[0],GOAL[1])
    total_reward = 0
    done = False
    steps = 0                                     #每局开始，初始化位置，状态，总奖励，结束标志，步数计数器
    #global DYNAMIC_OBSTACLES  # 修改全局变量D的值，让动态障碍物在每回合结束后能保留位置变化到下一回合
    new_obs = move_dynamic_obstacles(DYNAMIC_OBSTACLES)
    DYNAMIC_OBSTACLES.clear()
    DYNAMIC_OBSTACLES.extend(new_obs)

    while not done and steps < 200:               #一局内的循环，只要没结束且步数小于200继续走
        action = agent.act(state)                 #根据当前状态决定动作
        if action==0 and x>0: x-=1
        elif action==1 and x<GRID_SIZE-1: x+=1
        elif action==2 and y>0: y-=1
        elif action==3 and y<GRID_SIZE-1: y+=1                       #根据动作编号更新智能体的坐标

        DYNAMIC_OBSTACLES=move_dynamic_obstacles(DYNAMIC_OBSTACLES)  #调用移动函数，更新动态障碍物位置
        reward=get_reward(x,y,done)
        next_state = get_state(x,y,GOAL[0],GOAL[1])                  #根据新位置计算奖励，构造下一个状态
        if (x,y)==GOAL or (x,y) in STATIC_OBSTACLES or (x,y) in DYNAMIC_OBSTACLES:
            done=True                                                #判断游戏是否结束，到终点，撞到静态，撞到动态
        agent.remember(state,action,reward,next_state,done)          #经验存入智能体记忆库
        state = next_state
        total_reward += reward
        steps += 1                                                   #状态更新为下一状态，累计总奖励，步数加1

        if len(agent.memory)>batch_size:
            agent.replay(batch_size)                                 #记忆库的数据超过32条开始学习（训练一次神经网络）
    if e % TARGET_UPDATE_FREQ == 0:
        agent.target_model.load_state_dict(agent.model.state_dict())
        print(f"Episode {e}: 目标网络已同步")
    reward_history.append(total_reward)
    print(f"Episode{e:3d}|Total Reward:{total_reward:6.2f}|Steps:{steps:3d}")  #把本局的总奖励记录到列表里，
plt.plot(reward_history)
plt.title("Reward Convergence")
plt.xlabel("Episode")
plt.ylabel("Total Reward")
plt.show()