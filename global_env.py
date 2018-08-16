import numpy as np
import random

# env setting
P_ONE_HUNDRED_PERCENT = 145
P_ZERO_PERCENT = 87
TIME_ON = 30
TIME_OFF = 30


def take_arrival_time(elem):
    return elem[0]


def init(JOB_LENGTH):

    # cluster create
    # 0 ~ 29 meaning index of servers
    M = 30
    # 0 , 1 index , meaning two kinds of resource
    D = 2
    cluster = []
    cluster_index_list = []
    for i in range(M):
        cluster_index_list.append(i)
        cluster.append([1, 1])

    print("cluster create:", cluster)

    # job create：job arrival time，job ID，job duration time，required CPU，required memory
    job = []
    for i in range(JOB_LENGTH):
        job.append([random.choice([i, i+1, i+2]), i, random.randint(1, 100), round(random.random(), 4), round(random.random(), 4)])

    # job按照到达时间进行排序
    job.sort(key=take_arrival_time)

    print("job create:", job)

    return cluster, job, cluster_index_list


def get_jobs_this_time(time, job_queue, job):
    for row in job:
        if row[0] == time:
            job_queue.append(row)
    return job_queue


def update_cluster_state(cluster_index_list, cluster_running_job, time_i, cluster, sum_job_latency):
    # TODO:获取此时刻服务器的状态，对于duration完成的任务释放相应的资源
    for cluster_index in cluster_index_list:
        temp = cluster_running_job.get(cluster_index, None)
        if temp is not None:
            temp_row_index = 0
            LEN = len(temp)
            for i in range(LEN):
                if temp[temp_row_index][2] == time_i:
                    cluster[cluster_index][0] += temp[temp_row_index][3]
                    cluster[cluster_index][1] += temp[temp_row_index][4]
                    sum_job_latency += temp[temp_row_index][2] - temp[temp_row_index][0]
                    del temp[temp_row_index]
                else:
                    temp_row_index += 1
    return cluster_running_job, cluster, sum_job_latency


def allocate_job_to_cluster(job_broker, cluster_index_list, cluster, time_i, cluster_running_job):
    # 分配任务到各个服务器
    # 对于每个服务器，更新他们的任务列表，资源信息
    if not job_broker == []:
        job_broker_job_index = 0
        LEN = len(job_broker)
        for i in range(LEN):
            single_job = job_broker[job_broker_job_index]
            cluster_index = random.choice(cluster_index_list)
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
    return job_broker, cluster, cluster_running_job


# 主函数
def get_figure_data_random_method(JOB_LENGTH):

    sum_job_latency = 0

    # 调用函数，初始化服务器状态，以及生成随机job
    cluster, job, cluster_index_list = init(JOB_LENGTH)

    time_i = 0
    job_broker = []
    # 服务器上面运行的任务，key是每个服务器的index，value是在此服务器上执行的job
    cluster_running_job = {}

    while True:

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

        if time_i > JOB_LENGTH + 2 and job_broker == []:
            break

    return sum_job_latency






