

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
for episode in range(1):
    sum_job_latency = 0
    # 调用函数，初始化服务器状态，以及生成随机job
    cluster, job, cluster_index_list = init(JOB_LENGTH)
    # 初始化
    time_i = 0
    job_broker = []
    # 服务器上面运行的任务，key是每个服务器的index，value是在此服务器上执行的job
    cluster_running_job = {}
    temp_transition = []

    while True:

        # 调用函数getJobsThisTime，返回当前时间到达的任务，加入任务等待队列
        job_broker = get_jobs_this_time(time_i, job_broker, job)
        # print("time_i is :", time_i, " ====== job_broker is:", job_broker)

        # 调用函数，更新服务器状态
        cluster_running_job, cluster, sum_job_latency = \
            update_cluster_state(cluster_index_list, cluster_running_job, time_i, cluster, sum_job_latency)
        # print("time_i is :", time_i, " ====== cluster_running_job is:", cluster_running_job)

        # 分配任务到各个服务器
        # 对于每个服务器，更新他们的任务列表，资源信息
        if not job_broker == []:
            job_broker_job_index = 0
            LEN = len(job_broker)
            for i in range(LEN):
                # 决策节点decision epoch
                single_job = job_broker[job_broker_job_index]
                # 抽象出state
                state = []
                for cluster_index in cluster_index_list:
                    state.append(cluster[cluster_index][0])
                    state.append(cluster[cluster_index][1])
                state.append(single_job[3])
                state.append(single_job[4])
                state.append(single_job[2])

                # 选择行为
                cluster_index = RL.choose_action(np.array(state))

                # 存储记忆
                # TODO:reward定义还未完成
                reward = -len(job_broker)
                print(reward)
                # 这里temp_transition不为空就是True
                if temp_transition:
                    temp_transition.append(reward)
                    temp_transition.append(np.array(state))
                    observation, action, reward, observation_ = temp_transition
                    RL.store_transition(observation, action, reward, observation_)
                    temp_transition = []
                temp_transition.append(np.array(state))
                temp_transition.append(cluster_index)



                # TODO: 执行动作，得到：下一个状态，执行动作的得分，是否结束。
                if single_job[3] <= cluster[cluster_index][0] and single_job[4] <= cluster[cluster_index][1]:
                    cluster[cluster_index][0] -= single_job[3]
                    cluster[cluster_index][1] -= single_job[4]
                    # ready_job [到达时间，ID，结束时间，需要CPU，需要的memory]：把第三项改成结束时间
                    ready_job = single_job
                    ready_job[2] = time_i + single_job[2]
                    # 注意：其实这里列表，single_job和ready_job指针指向内容一样了，single_job内容也变了
                    temp = cluster_running_job.get(cluster_index, None)
                    if temp is None:
                        cluster_running_job[cluster_index] = []
                        cluster_running_job[cluster_index].append(ready_job)
                    else:
                        cluster_running_job[cluster_index].append(ready_job)
                    job_broker.pop(job_broker_job_index)
                else:
                    job_broker_job_index += 1

        time_i += 1

        if time_i > JOB_LENGTH + 2 and job_broker == []:
            break









