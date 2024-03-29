# window + no weights + time-dependent
import networkx as nx
import numpy as np
import scipy.sparse.linalg as spla
from scipy.sparse import csr_matrix
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import heapq # 数组最大n个值
from scipy.signal import savgol_filter #滤波trend
import pandas as pd

# 创建叶子结点中心性与网络规模的联系
def scale(n):
    return n

# screen data
def screen(token, beta):
    df = pd.read_csv('./data/'+token+'.csv',low_memory= False,usecols=["from", "to", "value", "metadata.blockTimestamp"])
    df.columns = ["from", "to", "value", "time"]
    df["time"] = df["time"].str.slice(0,10)
    df["time"] = pd.to_datetime(df["time"])
    con = df["value"]>df["value"].max()*beta
    df = df[con]
    return df

def total_nodes(edge):
    # graph nodes
    G = nx.DiGraph()
    for iter in edge:
        u = iter[0]
        v = iter[1]
        G.add_nodes_from([u, v])
    return G.number_of_nodes(), list(G.nodes())

# time-dependent Katz receive and broadcast
def time_dependent(edge, start_time, end_time, Tw, a):
    t = end_time - Tw
    while t >= start_time and t+Tw <=end_time:
        # graph nodes
        G = nx.DiGraph()
        for iter in edge:
            u = iter[0]
            v = iter[1]
            G.add_nodes_from([u, v])

        # graph edges
        con = (edge[:,3]>=t) & (edge[:,3]<t+Tw)
        G.add_edges_from(edge[con][:,[0,1]])

        # prepare
        n = G.number_of_nodes()
        A = nx.to_scipy_sparse_array(G, dtype='f') # 按照G.nodes()顺序,csr按照行的稀疏矩阵
        I = np.diag(np.ones(n))
        b = scale(n) * np.ones(n)

        # initialize
        if t == end_time - Tw:
            xr = b
            xb = b

        # alpha--spectrum
        rho = spla.eigs(A, k=1,return_eigenvectors=False) # 关掉求特征值！
        alpha = a/abs(rho) # 文献设置

        # GMRES
        xr, exitCode = spla.gmres(I-alpha*A.T, xr) # 默认restart=20,x即为完整的kazt中心性向量,exitCode=0表示success
        xr = xr/max(abs(xr)) # 归一化
        xb, exitCode = spla.gmres(I-alpha*A, xb) # 默认restart=20,x即为完整的kazt中心性向量,exitCode=0表示success
        xb = xb/max(abs(xb)) # 归一化

        t -= Tw
    return xr,xb

def meanr(x, p, NG):
    # topn/total
    N = int(NG*p)
    xtop = heapq.nlargest(N, x)
    xr = np.mean(xtop)/np.mean(x)
    return xr

def process(df, T, s, Tw, a, percent):
    # time
    start_time = df.iloc[0,-1]
    end_time = df.iloc[-1,-1]
    t = start_time
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
        NG, nodes = total_nodes(edge)
        xr,xb = time_dependent(edge, t, t + T, Tw, a)
        for p in percent:
            xrrate[percent.index(p)].append(meanr(xr, p, NG))
            xbrate[percent.index(p)].append(meanr(xb, p, NG))
        times.append(t)
        t += s
    
    return times, xrrate, xbrate

def figplot(token, percent, times, xrrate, xbrate, T, degree, beta, Tsg, s, Tw, a):
    times = pd.to_datetime(times)

    fig = plt.figure(figsize=(12,6))
    ax1 = fig.add_subplot(211)
    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    for p in percent:
        # data
        ax1.plot(times, xrrate[percent.index(p)], label='katz receive'+str(int(p*100))+'%')
        # trend
        xr_smooth = savgol_filter(xrrate[percent.index(p)], Tsg, degree)
        ax1.plot(times, xr_smooth, '--',linewidth = 2, label='trend'+str(int(p*100))+'%')
    ax1.legend(fontsize = 8)
    plt.ylabel('top mean receive/mean receive')
    plt.title('Katz broadcast '+str(token)+' beta'+str(beta)+' T'+str(T.days)+' s'+str(s.days)+' Tw'+str(Tw)+' a'+str(a))

    ax2 = fig.add_subplot(212)
    ax2.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    for p in percent:
        ax2.plot(times, xbrate[percent.index(p)], label='katz broadcast'+str(int(p*100))+'%')
        xb_smooth = savgol_filter(xbrate[percent.index(p)], Tsg, degree)
        ax2.plot(times, xb_smooth, '--',linewidth = 2, label='trend'+str(int(p*100))+'%')
    ax2.legend(fontsize = 8)
    plt.ylabel('top mean broadcast/mean broadcast')
    plt.gcf().autofmt_xdate()  # 自动旋转日期标记
    plt.title('Katz receive '+str(token)+' beta'+str(beta)+' T'+str(T.days)+' s'+str(s.days)+' Tw'+str(Tw)+' a'+str(a))
    plt.xlabel('time') 
    plt.show()
    return fig