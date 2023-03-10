import pandas as pd
import networkx as nx
from datetime import timedelta
import numpy as np
import scipy.sparse.linalg as spla
from scipy.sparse import csr_matrix
import time
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import os

token = 'panda'
T = timedelta(days=10)# 间隔为T的时间
beta = 0.0001

def scale(n):
    # 创建叶子结点中心性与网络规模的联系
    return n

def centime_nw(token,T,beta):

    df = pd.read_csv('./data/'+token+'.csv',low_memory= False,usecols=["from", "to", "value", "metadata.blockTimestamp"])
    df.columns = ["from", "to", "value", "time"]
    df["time"] = df["time"].str.slice(0,10)
    df["time"] = pd.to_datetime(df["time"])
    con = df["value"]>df["value"].max()*beta
    df = df[con]

    start_time = df.iloc[0,-1]
    end_time = df.iloc[-1,-1]
    t = end_time - T
    cputime_gmres = []
    cputime_s = []

    # 不考虑尾观察时长未到T的情况
    # 注意逆序
    while t >= start_time and t+T <=end_time:

        # graph nodes完整
        G = nx.DiGraph()
        for iter in df.values:
            u = iter[0]
            v = iter[1]
            G.add_nodes_from([u, v])

        con1 = df["time"] >= t
        con2 = df["time"] < t+T
        edge = df[con1 & con2].values
        for iter in edge:
            u = iter[0]
            v = iter[1]
            G.add_edge(u, v)

        # prepare
        n = G.number_of_nodes()
        A = nx.to_scipy_sparse_array(G, dtype='f') # 按照G.nodes()顺序,csr按照行的稀疏矩阵
        I = np.diag(np.ones(n))
        b = scale(n) * np.ones(n)

        if t == end_time - T:
            x = b
            xs = b

        # alpha--spectrum
        rho, vec = spla.eigs(A, k=1) # vec用不到
        a = 0.9/abs(rho) # 文献设置

        # GMRES
        cputime = time.process_time()
        x, exitCode = spla.gmres(I-a*A.T, x) # 默认restart=20,x即为完整的kazt中心性向量,exitCode=0表示success
        x = x/max(abs(x)) # 归一化
        cputime_gmres.append(time.process_time() - cputime)

        # three degrees
        cputime = time.process_time()
        # x * y不再执行矩阵乘法，而是逐元素乘法（就像 NumPy 数组一样）
        x0 = xs
        xs = x0 + a*A@xs
        xs = x0 + a*A@xs
        xs = x0 + a*A@xs
        xs = xs/max(abs(xs))
        cputime_s.append(time.process_time() - cputime)

        t = t - T

    fig = plt.figure(figsize=(10,5))
    ax2 = fig.add_subplot(1,1,1)
    ax2.scatter(range(len(x)), x, c='r', marker='o', label = 'kazt')
    ax2.scatter(range(len(xs)), xs, c='b',  marker='*', alpha = 0.5, label = 'three')
    plt.legend()
    plt.xlabel('number_of_node')
    plt.ylabel('centrality')
    plt.gcf().autofmt_xdate()  # 自动旋转日期标记
    figure_save_path = './cen/centime_nw/'+token
    if not os.path.exists(figure_save_path):
        os.makedirs(figure_save_path)
    plt.savefig(figure_save_path+'/'+token+'_T'+str(T)+'_b'+str(beta)+'.png')

    return

centime_nw(token,T,beta)