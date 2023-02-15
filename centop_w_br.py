import pandas as pd
import networkx as nx
import numpy as np
import scipy.sparse.linalg as spla
import matplotlib.pyplot as plt
import os

token = 'panda'

def scale(n):
    # 创建叶子结点中心性与网络规模的联系
    return n

def centop_w_br(token):

    df = pd.read_csv('./data/'+token+'.csv',low_memory= False,usecols=["from", "to", "value", "metadata.blockTimestamp"])
    df.columns = ["from", "to", "value", "time"]
    df["time"] = df["time"].str.slice(0,10)
    df["time"] = pd.to_datetime(df["time"])

    # graph nodes完整
    G = nx.DiGraph()
    for iter in df.values:
        u = iter[0]
        v = iter[1]
        w = iter[2]
        G.add_nodes_from([u, v])
        G.add_edge(u, v, weight = w)

    # prepare
    n = G.number_of_nodes()
    A = nx.to_scipy_sparse_array(G, dtype='f') # 按照G.nodes()顺序,csr按照行的稀疏矩阵
    AT = A.T
    I = np.diag(np.ones(n))
    b = scale(n) * np.ones(n)

    # alpha--spectrum
    rho, vec = spla.eigs(A, k=1) # vec用不到
    a = 0.9/abs(rho) # 文献设置

    # GMRES
    x, exitCode = spla.gmres(I-a*A, b) # 默认restart=20,x即为完整的kazt中心性向量,exitCode=0表示success
    x = x/max(abs(x)) # 归一化
    xT, exitCode = spla.gmres(I-a*AT, b) 
    xT = xT/max(abs(xT)) # 归一化

    # three degrees
    xs = b + a*A@b
    xs = b + a*A@xs
    xs = b + a*A@xs
    xs = xs/max(abs(xs))
    xsT = b + a*A@b
    xsT = b + a*A@xsT
    xsT = b + a*A@xsT
    xsT = xsT/max(abs(xsT))


    fig = plt.figure()
    ax1 = fig.add_subplot(1,1,1)
    ax1.scatter(x, xT, c='r', marker='+', label = 'kazt')
    ax1.scatter(xs, xsT, c='b',  marker='*', alpha = 0.5, label = 'three')
    ax1.set_aspect('equal', adjustable='box')
    plt.legend()
    plt.xlabel('broadcast')
    plt.ylabel('receive')
    figure_save_path = './cen/centop_w_br'
    if not os.path.exists(figure_save_path):
        os.makedirs(figure_save_path)
    plt.savefig(figure_save_path+'/'+token+'_n'+str(n)+'.png')

    return

centop_w_br(token)