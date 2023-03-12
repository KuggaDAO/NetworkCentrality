# centime and individual
import package.main as main
from datetime import timedelta
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# settings
token = 'SAI'
beta = 1e-4# 考虑无权图 因此我们选取value>max*beta的数据
T = timedelta(days = 90)# 间隔为T的时间保持影响力，单位为天
s = timedelta(days = 5)# 起始时间间隔，单位为天
Tw = timedelta(days = 15) # time-dependent中的时间窗口
a = 0.9 # alpha = a/rho ,a=0.9来自文献总结
lbound = 0.5
percent = [0.020,0.040,0.060,0.080]
degree = 2
Tsg = 30 # Tsg代表几个s


# screen data
df = main.screen(token, beta)
'''
# just for DAI
df = main.screen('DAI_0_c',beta)
df2 = main.screen('DAI_c_d',beta)
df3 = main.screen('DAI_d_e',beta)
df4 = main.screen('DAI_e_f',beta)
df5 = main.screen('DAI_f_l',beta)
df = pd.concat([df,df2],ignore_index=True) 
df = pd.concat([df,df3],ignore_index=True)
df = pd.concat([df,df4],ignore_index=True)
df = pd.concat([df,df5],ignore_index=True)
'''

# top individual:time dictionary based on katz centrality
def katz_individual(df, T, s, Tw, a, lbound, percent):
    # time
    start_time = df.iloc[0,-1]
    end_time = df.iloc[-1,-1]
    t = start_time
    d = {}
    times = []
    xrrate = []
    xbrate = []
    for p in percent:
        xrrate.append([])
        xbrate.append([])

    # time window
    while t >= start_time and t+T <=end_time:
        # 不考虑尾观察时长未到T的情况
        con1 = df["time"] >= t
        con2 = df["time"] <= t+T
        edge = df[con1 & con2].values
        NG, nodes = main.total_nodes(edge)
        xr,xb = main.time_dependent(edge, t, t + T, Tw, a)
        for p in percent:
            xrrate[percent.index(p)].append(main.meanr(xr, p, NG))
            xbrate[percent.index(p)].append(main.meanr(xb, p, NG))
        con1 = xr>lbound
        con2 = xb>lbound
        index = np.where(con1 & con2)
        index = np.array(index).flatten()
        nodes = np.array(nodes)

        # dictionary
        for i in index:
            if nodes[i] in d:
                d[nodes[i]].append(t)
            else:
                d[nodes[i]] = [t]
        
        print(t)
        times.append(t)
        t = t + s
    return d,xrrate,xbrate,times

# plot
def dicplot(d, T, s, Tw, a, lbound):
    fig = plt.figure(figsize=(10,6))
    ax = fig.add_subplot(111)
    center = []
    for key in d:
        d[key] = pd.to_datetime(d[key])
        center.append(key)
        ax.plot(d[key], [key[:10]]*len(d[key]), 'o')
    plt.title(token + 'Individuals with katz centrality > '+str(lbound)+' T'+str(T.days)+' s'+str(s.days)+' Tw'+str(Tw)+' a'+str(a))
    plt.xlabel('time')
    plt.ylabel('individual')
    ax.set_yticks([])# 关闭y轴刻度，实际上从下到上就是按csv文件中的顺序排列
    plt.show()
    return fig,center

d,xrrate,xbrate,times = katz_individual(df, T, s, Tw, a, lbound, percent)
fig,center = dicplot(d, T, s, Tw, a, lbound)
figr = main.figplot(token, percent, times, xrrate, xbrate, T, degree, beta, Tsg, s, Tw, a)
center = pd.DataFrame(center)
center.to_csv(token+'_'+str(lbound)+'_'+str(T)+'_'+str(s)+'_'+str(Tw)+'.csv')
d = pd.DataFrame.from_dict(d)
d.to_csv('dictionary'+token+'_'+str(lbound)+'_'+str(T)+'_'+str(s)+'_'+str(Tw)+'.csv')