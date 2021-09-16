import pandas as pd
from top2vec import Top2Vec
import umap
import hdbscan
from functions import scatter_plot

# PROVIDE YOUR TEXT DATA HERE
data = pd.read_csv("all_abstracts.tsv", sep="\t")

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

model = Top2Vec(documents,speed="deep-learn", workers=8)
stop = time.perf_counter()
print(f"Runtime {start - stop:0.4f} seconds")
model.save("models/top2vec_d2v")
model = Top2Vec.load("models/top2vec_d2v")

print("Number of Topics Identified:" + str(model.get_num_topics()))
model.model.init_sims()
data = model.model.docvecs.vectors_docs

umap_args = {'n_neighbors': 15,
             'n_components': 5,
             'metric': 'cosine'}

umap_model = umap.UMAP(**umap_args).fit(model.model.docvecs.vectors_docs)

# find dense areas of document vectors

hdbscan_args = {'min_cluster_size': 15,
                'metric': 'euclidean',
                'cluster_selection_method': 'eom'}

cluster = hdbscan.HDBSCAN(**hdbscan_args).fit(umap_model.embedding_)
scatter_plot(data, model.get_num_topics(), cluster, "d2v_master", noise=False, model=model)
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
