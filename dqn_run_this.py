

from global_dqn import DeepQNetwork
from global_env import *

n_actions = 30
n_features = 63
RL = DeepQNetwork(n_actions, n_features,
                  learning_rate=0.01,
                  reward_decay=0.9,
                  e_greedy=0.9,
                  replace_target_iter=200,
                  memory_size=2000,
                  output_graph=True
                  )


JOB_LENGTH = 100
# 初始化当前状态。
# 调用函数，初始化服务器状态，以及生成随机job
cluster, job, cluster_index_list = init(JOB_LENGTH)
print(cluster, job, cluster_index_list)


    # 选择动作。


    # 执行动作，得到：下一个状态，执行动作的得分，是否结束。


    # 保存记忆。


    # 先观察一段时间累积足够的记忆在进行训练。


