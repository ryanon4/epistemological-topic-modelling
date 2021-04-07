import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

def scatter_plot(data, num_topics, clusterer, name, noise, model):
    cluster_labels = clusterer.labels_
    doc_labels = model.doc_top

    if noise == True:
        color_palette = sns.color_palette('bright', num_topics)
        # Colour points for each label, if the label is -1 (deemed noise) then colour it grey.
        cluster_colors = [color_palette[x] if x >= 0
                          else (0.5, 0.5, 0.5)
                          # else (1, 1, 1)
                          for x in cluster_labels]
        cluster_member_colors = [sns.desaturate(x, p) for x, p in
                                 zip(cluster_colors, clusterer.probabilities_)]

        # Create Scatter plot of 2d vectors
        fig, ax = plt.subplots()

        scatter = ax.scatter(*data.T, linewidth=0, c=cluster_member_colors, alpha=1, s=4.5)
        plt.savefig("graphs/" + str(name) + "_noise.svg", format="svg")

    elif noise == False:
        color_palette = sns.color_palette('bright', num_topics)
        # Colour points for each label, if the label is -1 (deemed noise) then colour it grey.
        cluster_colors = [color_palette[x] if x >= 0
                          #else (0.5, 0.5, 0.5)
                          else (1, 1, 1)
                          for x in cluster_labels]
        cluster_member_colors = [sns.desaturate(x, p) for x, p in
                                 zip(cluster_colors, clusterer.probabilities_)]

        # Create Scatter plot of 2d vectors
        fig, ax = plt.subplots()

        labels = np.float32(cluster_labels)

        data_df = pd.DataFrame(data)
        data_df[3] = labels
        data_df[4] = cluster_member_colors
        data_df[5] = doc_labels
        data_df.columns = ["x", "y", "c_label", "colors", "d_label"]



        # Drop Noise
        index = data_df[data_df["c_label"] == -1].index
        data_df.drop(index, inplace=True)

        print(data_df["d_label"].value_counts())

        __data = np.array(data_df[["x", "y"]])

        scatter = ax.scatter(*__data.T, linewidth=0, c=data_df["colors"].values.tolist(), alpha=1, s=4.5)

        # Plot Labels in center of each cluster
        for i, label in tqdm(enumerate(data_df["d_label"].unique())):
            ax.annotate(int(label),
                        data_df.loc[data_df['d_label'] == label, ['x', 'y']].mean(),
                        horizontalalignment='center',
                        verticalalignment='center',
                        size=10, #weight='bold',
                        )#color=color_palette[int(label)]
        plt.savefig("graphs/" + str(name) + "_noise_removed.svg", format="svg")



    # Generate legend from all document labels.
    # As -1 is used to denote noisy docs, increase everything by 1.
    """legend_range = list(range(labels.min()+1, labels.max()+1))
    legend1 = ax.legend(*scatter.legend_elements(),
                        loc="lower left", title="Topic ID")
    ax.add_artist(legend1)"""
    #plt.show()