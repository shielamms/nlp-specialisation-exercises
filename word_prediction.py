import numpy as np

from modules.distance import cosine_similarity, euclidean_distance
from modules.pca import PCA

def get_closest_word(feature1,
                     label1,
                     feature2,
                     embeddings,
                     distance_function=cosine_similarity):
    """
    Parameters:
        feature1: a string
        label1: a string (the label associated to feature1)
        feature2: a string
        embeddings: a dictionary where the keys are words and values are
                    word vectors
        distance_function: the function to calculate the distance between words
    Returns:
        labels: a dictionary with the most likely label and its similarity score
    """

    group = set([feature1, label1, feature2])
    feature1_emb = embeddings[feature1]
    label1_emb = embeddings[label1]
    feature2_emb = embeddings[feature2]
    label2 = ''

    # label2 is the word that is closest to the vector that is parallel
    # to the vector label1 - feature1
    label2_vector = label1_emb - feature1_emb + feature2_emb

    similarity = -1

    for word in embeddings.keys():
        if word not in group:
            word_emb = embeddings[word]
            cur_similarity = distance_function(label2_vector, word_emb)

            if cur_similarity > similarity:
                similarity = cur_similarity
                label2 = (word, similarity)

    return label2


def test_pca():
    np.random.seed(1)
    X = np.random.rand(3, 10)
    pca = PCA(n_components=2)
    pca.fit(X)
    X_reduced = pca.transform(X)
    expected = np.array([[0.43437323, 0.49820384],
                         [0.42077249, -0.50351448],
                         [-0.85514571, 0.00531064]])
    print("Your original matrix was " + str(X.shape) + " and it became:")
    print(X_reduced)
    print("The expected matrix is " + str(expected.shape))
    print(expected)

if __name__ == '__main__':
    test_pca()