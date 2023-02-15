import pandas as pd
import networkx as nx
from datetime import timedelta
import numpy as np
import scipy.sparse.linalg as spla
from scipy.sparse import csr_matrix
import time
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import heapq # 数组最大n个值
import os

# settings
token = 'panda'
T = timedelta(days=10)# 间隔为T的时间保持影响力，单位为天
s = timedelta(days=5)# 起始时间间隔，单位为天
# percent = [0.010,0.020,0.030,0.040,0.050,0.060,0.070,0.080,0.090]
percent = [0.001,0.002,0.003,0.004,0.005,0.006,0.007,0.008,0.009]

def scale(n):
    # 创建叶子结点中心性与网络规模的联系
    return n

def meanr(token,T,s,percent):

    df = pd.read_csv('./data/'+token+'.csv',low_memory= False,usecols=["from", "to", "value", "metadata.blockTimestamp"])
    df.columns = ["from", "to", "value", "time"]
    df["time"] = df["time"].str.slice(0,10)
    df["time"] = pd.to_datetime(df["time"])

    # graph nodes
    edge = df.values
    G = nx.DiGraph()
    for iter in edge:
        u = iter[0]
        v = iter[1]
        G.add_nodes_from([u, v])

    NG = G.number_of_nodes()# 参与总人数

    start_time = df.iloc[0,-1]
    end_time = df.iloc[-1,-1]
    t = start_time
    cputime_gmres = []
    cputime_s = []
    times = []
    xr = []
    xsr = []
    for i in range(len(percent)):
        xr.append([])
        xsr.append([])

    # 不考虑尾观察时长未到T的情况
    while t >= start_time and t+T <=end_time:

        # graph
        con1 = df["time"] >= t
        con2 = df["time"] < t+T
        edge = df[con1 & con2].values
        G = nx.DiGraph()
        for iter in edge:
            u = iter[0]
            v = iter[1]
            w = iter[2]
            G.add_nodes_from([u, v])
            G.add_edge(u, v, weight = w)

        # prepare
        n = G.number_of_nodes()
        b = scale(n) * np.ones(n)
        A = nx.to_scipy_sparse_array(G) # 按照G.nodes()顺序,csr按照行的稀疏矩阵
        I = np.diag(np.ones(n))

        # alpha--spectrum
        rho, vec = spla.eigs(A, k=1) # vec用不到
        a = 0.9/abs(rho) # 文献设置

        # GMRES
        cputime = time.process_time()
        x, exitCode = spla.gmres(I-a*A.T, b) # 默认restart=20,x即为完整的kazt中心性向量,exitCode=0表示success
        cputime_gmres.append(time.process_time() - cputime)

        # three degrees
        cputime = time.process_time()
        # x * y不再执行矩阵乘法，而是逐元素乘法（就像 NumPy 数组一样）
        xs = b + a*A@b
        xs = b + a*A@xs
        xs = b + a*A@xs
        cputime_s.append(time.process_time() - cputime)

        # top n/total
        for i in range(len(percent)):
            N = int(NG * percent[i])
            xtop = heapq.nlargest(N,x)
            xstop = heapq.nlargest(N,xs)
            xr[i].append(np.mean(xtop)/np.mean(x))
            xsr[i].append(np.mean(xstop)/np.mean(xs))

        times.append(t)
        t = t + s

    # 画图
    for i in range(len(percent)):
        fig = plt.figure(figsize=(10,5))
        ax1 = fig.add_subplot(1,1,1)
        ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        ax1.xaxis.set_major_locator(mdates.MonthLocator())
        ax1.plot(times, xr[i], 'r', label = 'katz')
        ax1.plot(times, xsr[i], 'b', label = 'three')
        plt.legend()
        plt.xlabel('time')
        plt.ylabel('meanr')
        plt.gcf().autofmt_xdate()  # 自动旋转日期标记
        figure_save_path = './cen/centop_w/'+token
        if not os.path.exists(figure_save_path):
            os.makedirs(figure_save_path)
        plt.savefig(figure_save_path+'/'+token+'_T'+str(T)+'_top'+str(percent[i])+'.png')
    return 

meanr(token,T,s,percent)