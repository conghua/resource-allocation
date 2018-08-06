
from global_env import get_figure_data_random_method


import pylab as pl


pl.rcParams['font.sans-serif'] = ['FangSong'] # 指定默认字体
pl.rcParams['axes.unicode_minus'] = False # 解决保存图像是负号'-'显示为方块的问题

NUM_OF_POINT = 10
x = [(i+1)*100 for i in range(NUM_OF_POINT)]
y = [0 for i in range(NUM_OF_POINT)]
for i in range(NUM_OF_POINT):
    y[i] = get_figure_data_random_method(x[i])

pl.xlabel("job数量")
pl.ylabel("sum of jobs latency")
pl.plot(x, y, color="red", marker="s", ls="--", label='这是随机方法')
pl.legend(loc='upper right')


pl.show()