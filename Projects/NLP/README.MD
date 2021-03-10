# Clusterization of news articles
```py
import numpy as np
import matplotlib.pyplot as plt

from sklearn import metrics
from sklearn.cluster import KMeans
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import CountVectorizer

# Choose 4 categories
categories = [
    'rec.sport.hockey', # hockey
    'talk.politics.mideast', # near east news
    'comp.graphics', # comp vision
    'sci.crypt' # cryptography
]

# download dataset
dataset = fetch_20newsgroups(subset='all', categories=categories,
                             shuffle=True, random_state=42)

print("%d documents" % len(dataset.data))
print("%d categories" % len(dataset.target_names))

# save labels
labels = dataset.target

"""## Preprocessing"""

# first 10 examples
for i in range(10):
    print(dataset.data[i], end='\n' + '*' * 50 + '\n\n')

# Tokenizer
analyzer = CountVectorizer(stop_words='english').build_analyzer()

# Tokenizing
docs = []
for document in dataset.data:
    docs.append(analyzer(document.replace('_', '')))

# 10 examples again
for i in range(10):
    print(docs[i], end='\n\n')

# check length
len(docs)

"""## Vectorization"""


#!pip install -U gensim

from gensim.models import Word2Vec

# train vectorizer
# output: feature vector
model = Word2Vec(docs, min_count=20, size=50)

# put together the vectors for each doc
def doc_vectorizer(doc, model):
    doc_vector = []
    num_words = 0
    for word in doc:
        try:
            if num_words == 0:
                doc_vector = model[word]
            else:
                doc_vector = np.add(doc_vector, model[word])
            num_words += 1
        except:
            pass
     
    return np.asarray(doc_vector) / num_words

# create embeddings
X = []
for doc in docs:
    X.append(doc_vectorizer(doc, model))

# first 10 docs
X[:10]

# size of the document
print(np.asarray(X).shape)

# t-SNE – dimensionality reduction
from sklearn.manifold import TSNE

# t-SNE instance
tsne = TSNE(n_components=2, random_state=0)

# size 2
X = tsne.fit_transform(X)

print(np.asarray(X).shape)

"""## Clustering"""

# Kmeans
kmeans = KMeans(n_clusters=4)

# Train
kmeans.fit(X)

# Predict
y_pred = kmeans.labels_.astype(np.int)

# Coordinates of centers
print ("Centers:\n", kmeans.cluster_centers_)

# Metrics
print ("silhouette: %0.3f" % metrics.silhouette_score(X, y_pred, metric='euclidean'))
print("homegeneity: %0.3f" % metrics.homogeneity_score(labels, y_pred))
print("completeness: %0.3f" % metrics.completeness_score(labels, y_pred))
print("V-measure: %0.3f" % metrics.v_measure_score(labels, y_pred))

# Viz
plt.rcParams['figure.figsize'] = 10, 10
plt.scatter(X[:, 0], X[:, 1], c=y_pred, s=200, alpha=.5)
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], c='red', s=200, marker='+')
plt.show()

# True values viz
colors = ("red", "green", "blue", "yellow")

for i in range(4):
    plt.scatter(X[labels==i][:, 0], X[labels==i][:, 1], \
                s=200, alpha=.5, c=colors[i], label=dataset.target_names[i])
    plt.legend(loc=2)

plt.show()
