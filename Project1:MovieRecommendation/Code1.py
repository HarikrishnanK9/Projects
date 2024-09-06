import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
from sklearn.cluster import MiniBatchKMeans
import time
import math

# Set plotting style
plt.style.use('ggplot')  # Changed to 'ggplot' as 'seaborn' style might not be available
plt.rcParams['figure.figsize'] = [14, 14]

# Load the dataset
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

# Vectorize text using TF-IDF
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

# Perform K-Means clustering
def perform_clustering(tfidf_matrix, k):
    """Cluster the TF-IDF matrix using K-Means."""
    kmeans = MiniBatchKMeans(n_clusters=k, n_init=3)
    kmeans.fit(tfidf_matrix)
    return kmeans

# Find similar items based on TF-IDF similarity
def find_similar(tfidf_matrix, index, top_n=5):
    """Find top_n similar items to the item at index."""
    cosine_similarities = linear_kernel(tfidf_matrix[index:index+1], tfidf_matrix).flatten()
    related_docs_indices = [i for i in cosine_similarities.argsort()[::-1] if i != index]
    return related_docs_indices[:top_n]

# Create and visualize the network graph
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

def main(filepath, k):
    df = load_data(filepath)
    tfidf_matrix, vectorizer = vectorize_text(df['description'])
    kmeans = perform_clustering(tfidf_matrix, k)
    df['cluster'] = kmeans.predict(tfidf_matrix)
    
    G = create_graph(df, tfidf_matrix, kmeans)
    
    # Example usage
    list_in = ["Ocean's Twelve", "Ocean's Thirteen"]
    sub_graph = get_all_adj_nodes(G, list_in)
    
    # Plot 1: Subgraph for Ocean's Twelve and Ocean's Thirteen
    plt.title("Subgraph: Ocean's Twelve and Ocean's Thirteen")
    draw_sub_graph(G, sub_graph)
    
    recommendations = {
        "Ocean's Twelve": get_recommendation(G, "Ocean's Twelve"),
        "Ocean's Thirteen": get_recommendation(G, "Ocean's Thirteen"),
        "The Devil Inside": get_recommendation(G, "The Devil Inside"),
        "Stranger Things": get_recommendation(G, "Stranger Things")
    }
    
    for title, recs in recommendations.items():
        print(f"\n{'*' * 40}\nRecommendation for '{title}'\n{'*' * 40}")
        print(recs.head())

    # Plot 2: Recommendation subgraph for Ocean's Twelve
    reco = list(recommendations["Ocean's Twelve"].index[:4].values)
    reco.extend(["Ocean's Twelve"])
    sub_graph = get_all_adj_nodes(G, reco)
    plt.title("Recommendation Subgraph: Ocean's Twelve")
    draw_sub_graph(G, sub_graph)

    # Plot 3: Recommendation subgraph for Stranger Things
    reco = list(recommendations["Stranger Things"].index[:4].values)
    reco.extend(["Stranger Things"])
    sub_graph = get_all_adj_nodes(G, reco)
    plt.title("Recommendation Subgraph: Stranger Things")
    draw_sub_graph(G, sub_graph)

if __name__ == "__main__":
    FILEPATH = '/home/harikrishnan/Projects/Personal Project Datas/netflix_titles.csv'  # Update this path to your dataset
    K = 200
    main(FILEPATH, K)
