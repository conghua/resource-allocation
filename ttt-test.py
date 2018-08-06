import numpy as np
import random


class Network:
    r = 1
    a = 2
    b = 5
    list = None
    state_num = 6

    def __init__(self):
        self.list = np.identity(self.state_num)

network = Network()
print(network.list)
state = network.list[0:2]
print(state)

actions_value = [[3 ,5 ,7],
                 [9, 15, 2]]
action = np.argmax(actions_value, axis= 0)
print(action)

list = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
slice = random.sample(actions_value, 2)  #从list中随机获取5个元素，作为一个片断返回
print(slice)
print(list) #原有序列并没有改变。

qiguai = None
for i in range(2):
    if qiguai is None:
        qiguai = 4
    elif qiguai is not None:
        qiguai = np.vstack((qiguai, 5))
print(qiguai)