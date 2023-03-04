# sort values to determine beta
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

token = 'panda'
#'DAI_c_d'好像有点问题，value不能转化到float

def plot_sorted(token):
    df = pd.read_csv('./data/'+token+'.csv',low_memory= False,usecols=["value"])
    percent = [1e-1,1e-2,1e-3,1e-4,1e-5,1e-6]
    num = []
    for p in percent:
        con = df["value"]>df["value"].max()*p
        num.append(len(df[con])/len(df))
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(percent, num, 'o-')
    ax.set_xscale('log')
    ax.set_xlabel('beta')
    ax.set_ylabel('percentage')
    ax.set_title('token: '+token)
    plt.show()
    return fig

fig = plot_sorted(token)
