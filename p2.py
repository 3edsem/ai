!pip install gensim scikit-learn matplotlib

import gensim.downloader as api 
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA 
from sklearn.manifold import TSNE
 
print("Loading pre-trained word vectors...")
word_vectors = api.load("word2vec-google-news-300") 

domain_words = ["computer", "software", "hardware", "algorithm", "data", "network", "programming", "machine", "learning", "artificial"]


domain_vectors = np.array([word_vectors[word] for word in domain_words])

def visualize_word_embeddings(words, vectors, method='pca', perplexity=5):
        reducer = PCA(n_components=2) 
    elif method == 'tsne':
        reducer = TSNE(n_components=2, random_state=42, perplexity=perplexity) 
    else:
        raise ValueError("Method must be 'pca' or 'tsne'.")

    reduced_vectors = reducer.fit_transform(vectors)

    plt.figure(figsize=(10, 8))
    for i, word in enumerate(words):
      plt.scatter(reduced_vectors[i, 0], reduced_vectors[i, 1], marker='o', color='blue')
      plt.text(reduced_vectors[i, 0] + 0.02, reduced_vectors[i, 1] + 0.02, word, fontsize=12)

    plt.title(f"Word Embeddings Visualization using {method.upper()}") 
    plt.xlabel("Component 1")
    plt.ylabel("Component 2") 
    plt.grid(True)
    plt.show()

visualize_word_embeddings(domain_words, domain_vectors, method='pca')


visualize_word_embeddings(domain_words, domain_vectors, method='tsne', perplexity=3)


def generate_similar_words(word):
 
    try:
        similar_words = word_vectors.most_similar(word, topn=5) 
        print(f"\nTop 5 semantically similar words to '{word}':") 
        for similar_word, similarity in similar_words:
            print(f"{similar_word}: {similarity:.4f}") 
    except KeyError as e:
        print(f"Error: {e} not found in the vocabulary.")

generate_similar_words("computer") 
generate_similar_words("learning")


'''
=======================
Explanation of the Code
Loading Pre-trained Word Vectors:
•	The gensim.downloader module loads the word2vec-google-news-300 model, which contains 300-dimensional word vectors.
•	Selecting Domain-Specific Words:
•	A list of 10 words from the technology domain is selected for visualization and analysis.
Dimensionality Reduction:
•	PCA and t-SNE are used to reduce the 300-dimensional word vectors to 2D for visualization.
•	Visualization:
•	The reduced vectors are plotted in a 2D space using matplotlib. Words with similar meanings appear closer together.
Semantic Similarity:
•	The generate_similar_words function finds the top 5 semantically similar words for a given input word using the most_similar method.
Step 4: Output and Analysis Visualization Output
•	PCA Plot: Words like "computer," "software," and "hardware" will appear close to
each other, indicating their semantic similarity.
•	t-SNE Plot: Words like "machine," "learning," and "artificial" will form a tight cluster, reflecting their contextual relationships.
This program demonstrates how to:
Visualize word embeddings using PCA and t-SNE. Analyze clusters and relationships between words.
 
Generate semantically similar words using pre-trained embeddings.

Output:
2 (a) Loading pre-trained word vectors...


Word Relationship: king - man + woman
Most similar words to the result (excluding input words): queen: 0.7301
monarch: 0.6455
princess: 0.6156
crown_prince: 0.5819
prince: 0.5777


 
Output 2 (b)
Loading pre-trained word vectors…




Top 5 semantically similar words to 'computer': computers: 0.7979
laptop: 0.6640
laptop_computer: 0.6549
Computer: 0.6473
com_puter: 0.6082


Top 5 semantically similar words to 'learning':
 
teaching: 0.6602
learn: 0.6365
Learning: 0.6208
reteaching: 0.5810
learner_centered: 0.5739
 '''

