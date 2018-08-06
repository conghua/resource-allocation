import tensorflow as tf
import numpy as np
from collections import deque
import random
from global_env import init


class DeepQNetwork:

    #执行步数
    step_index = 0

    # 训练之前观察多少步。
    OBSERVE = 1000.

    # 选取的小批量训练样本数。
    BATCH = 20

    # epsilon 的最小值，当 epsilon 小于该值时，将不在随机选择行为。
    FINAL_EPSILON = 0.0001

    # epsilon 的初始值，epsilon 逐渐减小。
    INITIAL_EPSILON = 0.1

    # epsilon 衰减的总步数。
    EXPLORE = 3000000.

    # 探索模式计数。
    epsilon = 0

    # 训练步数统计。
    learn_step_counter = 0

    # 学习率。
    learning_rate = 0.001

    # γ经验折损率。
    gamma = 0.9

    # 记忆上限。
    memory_size = 5000

    # 当前记忆数。
    memory_counter = 0

    # 保存观察到的执行过的行动的存储器，即：曾经经历过的记忆。
    replay_memory_store = deque()

    # q_eval 网络。
    q_eval_input = None
    action_input = None
    q_target = None
    q_eval = None
    predict = None
    loss = None
    train_op = None
    cost_his = None
    reward_action = None

    # tensorflow 会话。
    session = None

    def __init__(self, learning_rate=0.001, gamma=0.9, memory_size=5000):
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.memory_size = memory_size

        # # 初始化成一个 6 X 6 的状态矩阵。
        # self.state_list = np.identity(self.state_num)
        #
        # # 初始化成一个 6 X 6 的动作矩阵。
        # self.action_list = np.identity(self.action_num)
        self.cluster, self.job, self.cluster_index_list = init(1000)

        # 创建神经网络。
        self.create_network()

        # 初始化 tensorflow 会话。
        self.session = tf.InteractiveSession()

        # 初始化 tensorflow 参数。
        self.session.run(tf.global_variables_initializer())

        # 记录所有 loss 变化。
        self.cost_his = []

    def create_network(self):
        """
        创建神经网络。
        :return:
        """
        self.q_eval_input = tf.placeholder(shape=[None, self.state_num], dtype=tf.float32)
        self.action_input = tf.placeholder(shape=[None, self.action_num], dtype=tf.float32)
        self.q_target = tf.placeholder(shape=[None], dtype=tf.float32)

        neuro_layer_1 = 3
        w1 = tf.Variable(tf.random_normal([self.state_num, neuro_layer_1]))
        b1 = tf.Variable(tf.zeros([1, neuro_layer_1]) + 0.1)
        l1 = tf.nn.relu(tf.matmul(self.q_eval_input, w1) + b1)

        w2 = tf.Variable(tf.random_normal([neuro_layer_1, self.action_num]))
        b2 = tf.Variable(tf.zeros([1, self.action_num]) + 0.1)
        self.q_eval = tf.matmul(l1, w2) + b2

        # 取出当前动作的得分。
        self.reward_action = tf.reduce_sum(tf.multiply(self.q_eval, self.action_input), reduction_indices=1)
        self.loss = tf.reduce_mean(tf.square((self.q_target - self.reward_action)))
        self.train_op = tf.train.GradientDescentOptimizer(self.learning_rate).minimize(self.loss)

        self.predict = tf.argmax(self.q_eval, 1)






























# from global_env import init, get_jobs_this_time, update_cluster_state
# import random
#
#
# def allocate_job_to_cluster(job_broker, cluster_index_list, cluster, time_i, cluster_running_job):
#     # 分配任务到各个服务器
#     # 对于每个服务器，更新他们的任务列表，资源信息
#     while not job_broker == []:
#         single_job = job_broker[0]
#         cluster_index = random.choice(cluster_index_list)
#         if single_job[3] <= cluster[cluster_index][0] and single_job[4] <= cluster[cluster_index][1]:
#             cluster[cluster_index][0] -= single_job[3]
#             cluster[cluster_index][1] -= single_job[4]
#             # ready_job [到达时间，ID，结束时间，需要CPU，需要的memory]：把第三项改成结束时间
#             ready_job = single_job
#             ready_job[2] = time_i + single_job[2]
#             # 注意：其实这里列表，single_job和ready_job指针指向内容一样了，single_job内容也变了
#             temp = cluster_running_job.get(cluster_index, None)
#             if temp is None:
#                 cluster_running_job[cluster_index] = []
#                 cluster_running_job[cluster_index].append(ready_job)
#             else:
#                 cluster_running_job[cluster_index].append(ready_job)
#             job_broker.pop(0)
#         else:
#             break
#     return job_broker, cluster, cluster_running_job
#
#
#
#
# JOB_LENGTH = 100
# sum_job_latency = 0
#
# # 调用函数，初始化服务器状态，以及生成随机job
# cluster, job, cluster_index_list = init(JOB_LENGTH)
#
# time_i = 0
# job_broker = []
# # 服务器上面运行的任务，key是每个服务器的index，value是在此服务器上执行的job
# cluster_running_job = {}
#
#
# while time_i < 2500:
#     # 调用函数getJobsThisTime，返回当前时间到达的任务，加入任务等待队列
#     job_broker = get_jobs_this_time(time_i, job_broker, job)
#
#     # 调用函数，更新服务器状态
#     cluster_running_job, cluster, sum_job_latency = \
#         update_cluster_state(cluster_index_list, cluster_running_job, time_i, cluster, sum_job_latency)
#
#     # 调用函数，分配任务到各个服务器
#     # 对于每个服务器，更新他们的任务列表，资源信息
#     job_broker, cluster, cluster_running_job = \
#         allocate_job_to_cluster(job_broker, cluster_index_list, cluster, time_i, cluster_running_job)
#
#     time_i += 1
#
# print(sum_job_latency)


