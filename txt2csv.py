import numpy as np
import pandas as pd

filename = './cen/individual/panda_0.5_30_5_5.txt'
data = np.loadtxt(filename, delimiter=', ', dtype=str)
data = pd.DataFrame(data)
data.to_csv('./cen/individual/panda_0.5_30_5_5.csv')