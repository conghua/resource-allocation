import csv


data_dict = {}
for row in csv.reader(open('google-cluster-data-1.csv', encoding='utf-8'), delimiter=' '):
    temp = data_dict.get(row[2], None)
    if temp is None:
        data_dict[row[2]] = []
        data_dict[row[2]].append(row)
    else:
        data_dict[row[2]].append(row)

data_list = []
for x in data_dict:
    #  taskID ,arrival time, duration time, peak cpu, peak memory
    #  string, int ,  int,  float, float
    pass
    one_data_bar = [0, 0, 0, 0, 0]
    one_data_bar[0] = x
    temp_same_task_time_list = []
    temp_same_task_cpu_list = []
    temp_same_task_memory_list = []
    for row in data_dict[x]:
        temp_same_task_time_list.append(int(row[0]))
        temp_same_task_cpu_list.append(float(row[4]))
        temp_same_task_memory_list.append(float(row[5]))
    one_data_bar[1] = min(temp_same_task_time_list)
    one_data_bar[2] = max(temp_same_task_time_list) - min(temp_same_task_time_list) + 300
    one_data_bar[3] = max(temp_same_task_cpu_list)
    one_data_bar[4] = max(temp_same_task_memory_list)
    if one_data_bar[3] != 0.0 or one_data_bar[4] != 0.0:
        data_list.append(one_data_bar)

csvFile = open('processed-data-2.csv', 'w', newline='')  # 设置newline，否则两行之间会空一行
writer = csv.writer(csvFile)
m = len(data_list)
for i in range(m):
    writer.writerow(data_list[i])
csvFile.close()


