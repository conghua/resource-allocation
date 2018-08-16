

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



step = 0
for episode in range(30):
    sum_job_latency = 0
    # 调用函数，初始化服务器状态，以及生成随机job
    cluster, job, cluster_index_list = init(JOB_LENGTH)
    # 初始化
    time_i = 0
    job_broker = []
    # 服务器上面运行的任务，key是每个服务器的index，value是在此服务器上执行的job
    cluster_running_job = {}

    while time_i < 250:

        # 调用函数getJobsThisTime，返回当前时间到达的任务，加入任务等待队列
        job_broker = get_jobs_this_time(time_i, job_broker, job)
        # print("time_i is :", time_i, " ====== job_broker is:", job_broker)

        # 调用函数，更新服务器状态
        cluster_running_job, cluster, sum_job_latency = \
            update_cluster_state(cluster_index_list, cluster_running_job, time_i, cluster, sum_job_latency)
        # print("time_i is :", time_i, " ====== cluster_running_job is:", cluster_running_job)

        # 调用函数，分配任务到各个服务器
        # 对于每个服务器，更新他们的任务列表，资源信息
        job_broker, cluster, cluster_running_job = \
            allocate_job_to_cluster(job_broker, cluster_index_list, cluster, time_i, cluster_running_job)

        time_i += 1








