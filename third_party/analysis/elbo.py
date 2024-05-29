import pandas as pd
import matplotlib.pyplot as plt

unif_elbos = "/home/golf/code/graph_generation/output/GraphRNN_path_unif_nobfs_2020_12_20_18_41_40/logging/epoch_history.csv"
gat_elbos = "/home/golf/code/graph_generation/output/GraphRNN_path_gat_nobfs_2020_12_20_18_42_30/logging/epoch_history.csv"

unif_elbos = pd.read_csv(unif_elbos)["valid_elbo"]
gat_elbos = pd.read_csv(gat_elbos)["valid_elbo"]

plt.plot(range(len(unif_elbos))[::4], unif_elbos[::4], label="uniform")

plt.plot(range(len(gat_elbos))[::4], gat_elbos[::4], label="gat")
plt.xlabel("epochs")
plt.ylabel("elbo")
plt.ylim(-220, -70)
plt.legend()
plt.show()