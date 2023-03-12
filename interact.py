import matplotlib.pyplot as plt
import pandas as pd
import package.main as main
import networkx

# settings
token = 'MKR'
filename = './cen/individual/MKR_0.5_90_5_15.csv'
beta = 1e-4

# read data
df = main.screen(token,beta)
nodes = pd.read_csv(filename)

# create network
G = networkx.DiGraph()
for node in nodes:
    G.add_node(node, label=node)

for edge in df:
    if G.has_node(edge[0]) and G.has_node(edge[1]):
        G.add_edge(edge[0],edge[1])

# plot
networkx.draw(G, with_labels=True, node_size=1000, node_color='skyblue', edge_color='grey', font_size=8, font_color='white')
plt.show()



