import matplotlib.pyplot as plt
import networkx as nx
import pickle

emptygraph = nx.Graph()
file = open("q_samples_caveman.pkl", 'rb')
samples = pickle.load(file)
file.close()
fig, axes = plt.subplots(5,11, figsize=(25,13))
plt.subplots_adjust(hspace=0.5, wspace=0.1)
colors = ['#e31a1c', '#1f78b4','#33a02c', '#fb9a99', '#a6cee3', '#ff7f00', '#b2df8a', '#fdbf6f', 'blue','#e31a1c', '#1f78b4','#33a02c']



for i in range(2,13):
    items = samples['subgraphs'][i].items()
    items = sorted(items, key=lambda x:samples['subgraph_frequency'][x[0]], reverse=True)
    items = items[:5]

    for idx, (hash_id, subgraph) in enumerate(items):
        if idx > 5:break
        nx.draw(subgraph, ax=axes[idx][i-2], node_size=80, node_color=colors[i-2])#, with_labels=True)
        ax = axes[idx][i-2]
        ax.title.set_text(str(samples['subgraph_frequency'][hash_id]))
    if idx!=4:
        for idx_ct in range(idx, 5):
            nx.draw(emptygraph, ax=axes[idx_ct][i - 2], node_size=80)
plt.savefig("demo.pdf")
plt.show()

