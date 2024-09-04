

### **Introduction**

This script demonstrates how to perform text clustering on movie descriptions using TF-IDF (Term Frequency-Inverse Document Frequency) and K-Means clustering. We will process a dataset, extract features, cluster the data, and visualize results to provide recommendations.

---

### **Imports and Configuration**

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
from sklearn.cluster import MiniBatchKMeans
import time
import math

plt.style.use('ggplot')  # Changed to 'ggplot' as 'seaborn' style might not be available
plt.rcParams['figure.figsize'] = [14, 14]
```

- **Imports**: Libraries such as `pandas`, `numpy`, `matplotlib`, `networkx`, and `sklearn` are imported for data manipulation, visualization, and machine learning.
- **Configuration**: The plotting style is set to `'ggplot'` and figure size is configured to ensure clear visualizations.

---

### **Load Data**

```python
def load_data(filepath):
    """Load and preprocess the dataset."""
    try:
        df = pd.read_csv(filepath)
    except FileNotFoundError:
        raise FileNotFoundError(f"The file at {filepath} was not found.")
    
    df['date_added'] = df['date_added'].str.strip()
    df['date_added'] = pd.to_datetime(df['date_added'], format="%B %d, %Y", errors='coerce')
    
    df['year'] = df['date_added'].dt.year
    df['month'] = df['date_added'].dt.month
    df['day'] = df['date_added'].dt.day
    
    df['directors'] = df['director'].apply(lambda l: [] if pd.isna(l) else [i.strip() for i in l.split(",")])
    df['categories'] = df['listed_in'].apply(lambda l: [] if pd.isna(l) else [i.strip() for i in l.split(",")])
    df['actors'] = df['cast'].apply(lambda l: [] if pd.isna(l) else [i.strip() for i in l.split(",")])
    df['countries'] = df['country'].apply(lambda l: [] if pd.isna(l) else [i.strip() for i in l.split(",")])
    
    return df
```

- **Function `load_data`**: Loads and preprocesses the dataset.
  - Reads the CSV file into a DataFrame.
  - Converts the `date_added` column to datetime format and extracts year, month, and day.
  - Splits and processes textual columns (`director`, `cast`, `listed_in`, `country`) into lists.

---

### **TF-IDF Vectorization**

```python
def vectorize_text(text_content):
    """Convert text content into TF-IDF matrix."""
    vectorizer = TfidfVectorizer(
        max_df=0.4,
        min_df=1,
        stop_words='english',
        lowercase=True,
        use_idf=True,
        norm='l2',
        smooth_idf=True
    )
    return vectorizer.fit_transform(text_content), vectorizer
```

- **Function `vectorize_text`**: Converts text content into a TF-IDF matrix.
  - `TfidfVectorizer` is used to transform text into numerical vectors that represent the importance of terms in each document.
  - Parameters such as `max_df`, `min_df`, and `stop_words` help filter out common or rare words.

---

### **K-Means Clustering**

```python
def perform_clustering(tfidf_matrix, k):
    """Cluster the TF-IDF matrix using K-Means."""
    kmeans = MiniBatchKMeans(n_clusters=k, n_init=3)
    kmeans.fit(tfidf_matrix)
    return kmeans
```

- **Function `perform_clustering`**: Applies K-Means clustering on the TF-IDF matrix.
  - `MiniBatchKMeans` is used for efficient clustering, particularly useful for large datasets.
  - **Choosing \( k = 200 \)**: The number of clusters \( k \) is chosen based on experimentation and domain knowledge. In practice, you can determine \( k \) using methods like the Elbow Method or Silhouette Score. Here, 200 clusters are used to balance granularity and manageability.

---

### **Find Similar Items**

```python
def find_similar(tfidf_matrix, index, top_n=5):
    """Find top_n similar items to the item at index."""
    cosine_similarities = linear_kernel(tfidf_matrix[index:index+1], tfidf_matrix).flatten()
    related_docs_indices = [i for i in cosine_similarities.argsort()[::-1] if i != index]
    return related_docs_indices[:top_n]
```

- **Function `find_similar`**: Finds similar items based on TF-IDF similarity.
  - Calculates cosine similarity between the TF-IDF vector of the item at a given index and all other items.
  - Returns the indices of the top \( n \) similar items, excluding the item itself.

---

### **Create and Visualize Graph**

```python
def create_graph(df, tfidf_matrix, kmeans):
    """Create and visualize the network graph."""
    G = nx.Graph(label="MOVIE")
    start_time = time.time()
    
    for i, row in df.iterrows():
        if i % 1000 == 0:
            print(f"Processing {i} -- {time.time() - start_time:.2f} seconds --")
        
        G.add_node(row['title'], key=row['show_id'], label="MOVIE", mtype=row['type'], rating=row['rating'])
        
        for actor in row['actors']:
            G.add_node(actor, label="PERSON")
            G.add_edge(row['title'], actor, label="ACTED_IN")
        for category in row['categories']:
            G.add_node(category, label="CAT")
            G.add_edge(row['title'], category, label="CAT_IN")
        for director in row['directors']:
            G.add_node(director, label="PERSON")
            G.add_edge(row['title'], director, label="DIRECTED")
        for country in row['countries']:
            G.add_node(country, label="COU")
            G.add_edge(row['title'], country, label="COU_IN")
        
        indices = find_similar(tfidf_matrix, i, top_n=5)
        snode = f"Sim({row['title'][:15].strip()})"
        G.add_node(snode, label="SIMILAR")
        G.add_edge(row['title'], snode, label="SIMILARITY")
        for index in indices:
            G.add_edge(snode, df['title'].iloc[index], label="SIMILARITY")
    
    print(f"Finished processing -- {time.time() - start_time:.2f} seconds --")
    return G
```

- **Function `create_graph`**: Constructs a network graph of movies and their attributes.
  - Adds nodes and edges representing movies, actors, categories, directors, and countries.
  - Connects movies to similar items based on the TF-IDF matrix.

---

### **Subgraph Visualization**

```python
def get_all_adj_nodes(G, list_in):
    """Get all adjacent nodes of the given nodes."""
    sub_graph = set()
    for node in list_in:
        sub_graph.add(node)
        sub_graph.update(G.neighbors(node))
    return list(sub_graph)

def draw_sub_graph(G, sub_graph):
    """Draw the subgraph with colored nodes."""
    subgraph = G.subgraph(sub_graph)
    colors = []
    label_colors = {
        "MOVIE": 'blue',
        "PERSON": 'red',
        "CAT": 'green',
        "COU": 'yellow',
        "SIMILAR": 'orange',
        "CLUSTER": 'orange'
    }
    
    for node in subgraph.nodes():
        label = G.nodes[node].get('label', 'UNKNOWN')
        colors.append(label_colors.get(label, 'grey'))
    
    nx.draw(subgraph, with_labels=True, font_weight='bold', node_color=colors)
    plt.show()
```

- **Functions `get_all_adj_nodes` and `draw_sub_graph`**: Extract and visualize subgraphs.
  - `get_all_adj_nodes` retrieves all adjacent nodes of the given nodes to form a subgraph.
  - `draw_sub_graph` visualizes the subgraph using NetworkX, with different colors for different node types.

---

### **Recommendation System**

```python
def get_recommendation(G, root):
    """Get movie recommendations based on similar nodes."""
    commons_dict = {}
    for neighbor in G.neighbors(root):
        for neighbor_of_neighbor in G.neighbors(neighbor):
            if neighbor_of_neighbor == root:
                continue
            if G.nodes[neighbor_of_neighbor]['label'] == "MOVIE":
                commons_dict.setdefault(neighbor_of_neighbor, []).append(neighbor)
    
    movies = []
    weights = []
    for movie, neighbors in commons_dict.items():
        weight = sum(1 / math.log(G.degree(neighbor)) for neighbor in neighbors)
        movies.append(movie)
        weights.append(weight)
    
    result = pd.Series(data=np.array(weights), index=movies)
    result.sort_values(inplace=True, ascending=False)
    return result
```

- **Function `get_recommendation`**: Generates movie recommendations based on network graph similarities

.
  - Computes weights based on common neighbors and their degrees.
  - Returns a sorted list of recommended movies.

---

This setup ensures a thorough analysis and clustering of the dataset, enabling effective recommendations based on movie descriptions and their relationships in the graph. The choice of \( k = 200 \) in K-Means is a balance between too few and too many clusters, aiming for detailed but manageable clusters for diverse movie content.
