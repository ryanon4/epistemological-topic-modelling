import pandas as pd
from top2vec import Top2Vec
import seaborn as sns
import matplotlib.pyplot as plt
import umap
import timeit
import numpy as np
from tqdm import tqdm

# PROVIDE YOUR TEXT DATA HERE
data = pd.read_csv("../../../all_abstracts.tsv", sep="\t")

data["pubdate"] = data["pubdate"].astype(str).str[0:4]
data = data[pd.to_numeric(data["pubdate"], errors = "coerce").notnull()]
data = data.dropna(subset=["pubdate"])
data["pubdate"] = data["pubdate"].astype(float)
data = data.dropna(subset=["pubdate"])
#data = data.loc[(data.pubdate >= 2010) & (data.pubdate <=2020)]
data = data.sort_values("pubdate")
data = data.reset_index()
print("TOTAL RECORDS:" + str(len(data)))

documents = data["combined"].tolist()
import time

start = time.perf_counter()

model = Top2Vec(documents,speed="deep-learn", workers=4)
stop = time.perf_counter()
print(f"Runtime {start - stop:0.4f} seconds")
model.save("models/top2vec_d2v")

model = Top2Vec.load("models/top2vec_d2v")

print("Number of Topics:" + str(model.get_num_topics()))
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


#scatter_plot(model.data, model.num_topics, model.cluster, "d2v_master", noise=False, model=model)
topic_sizes, topic_nums = model.get_topic_sizes()

topic_words, word_scores, topic_nums = model.get_topics()
for i, words  in enumerate(topic_words):
    print("Topic ID: " + str(topic_nums[i]))
    print(words)
    print("\n")


# Perform topic modelling of the smaller sub-topic
documents = pd.DataFrame(documents)
documents["topic_id"] = model.doc_top
for topic_id in topic_nums:
    print("TopicID: " + str(topic_id))
    sub_documents = documents.loc[documents["topic_id"]==topic_id]
    print("Topic Length: "+str(len(sub_documents)))
    #print(topic_words)


vecs = model._get_document_vectors(norm=False)
vecs_reduced = model.data
labels = model.doc_top
c_labels = model.cluster.labels_

sil_data = pd.DataFrame(list(zip(vecs, vecs_reduced, labels, c_labels)))
sil_data.columns = ["vecs", "vecs_reduced", "labels", "c_labels"]
sil_data["topic_id"] = documents["topic_id"]
# Drop Noise
index = sil_data[sil_data["c_labels"] == -1].index
sil_data.drop(index, inplace=True)

tlist = [0,5,7,8,12, 13,14,17,19,28]
sil_data = sil_data[sil_data["topic_id"].isin(tlist)]

from sklearn.metrics import silhouette_score
score = silhouette_score(sil_data["vecs_reduced"].values.tolist(), sil_data["labels"].values.tolist(), metric='euclidean')
print("SCORE: " + str(score))
