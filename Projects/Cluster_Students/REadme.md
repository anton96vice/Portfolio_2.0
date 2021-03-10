# Cluster Students by their Social Network Tags
```py
import pandas as pd
import matplotlib.pyplot as plt

#Data Load
df = pd.read_csv('/content/snsdata.csv')
df = df.iloc[:,3:]

# Normalization
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X = sc.fit_transform(df)

# DBSCAN
from sklearn.cluster import DBSCAN

dbscan = DBSCAN(eps=0.5,
                min_samples=5)
y = dbscan.fit_predict(X)

#Dimensionality Reduction
from sklearn.manifold import TSNE
X_embedded = TSNE(n_components=2).fit_transform(X)
df["x_component"]=X_embedded[:,0]
df["y_component"]=X_embedded[:,1]

# Print out related features
for i in range(len(np.unique(dbscan.labels_))):
    print(i, df[km.labels_==i][df==1].dropna(axis=1, how='all').columns)
    
#Print top words
for k, group in df.groupby(km.labels_):
    print(k)
    top_words = group.iloc[:,:-1].mean()\
                 .sort_values(ascending=False)\
                 .head(10)
    print(top_words)

#Viz
import plotly.express as px
    
fig = px.scatter(df, x="x_component", y="y_component", hover_name=dbscan.labels_, color = dbscan.labels_, size_max=60)
fig.update_layout(
     height=800)
fig.show()
