import package.main as main
from datetime import timedelta

# settings
token = 'panda'
T = timedelta(days = 90)# 间隔为T的时间保持影响力，单位为天
s = timedelta(days = 5)# 起始时间间隔，单位为天
Tw = timedelta(days = 15) # time-dependent中的时间窗口
Tsg = 30 # 滤波窗口
beta = 1e-4# 考虑无权图 因此我们选取value>max*beta的数据
percent = [0.020,0.040,0.060,0.080]
degree = 2 # 滤波的阶数
a = 0.9 # alpha = a/rho ,a=0.9来自文献总结

df = main.screen(token, beta)
times, xrrate, xbrate = main.process(df, T, s, Tw, a, percent)
fig = main.figplot(token, percent, times, xrrate, xbrate, T, degree, beta, Tsg, s, Tw, a)