import numpy as np


def euclidean_distance(A, B):
    """
    Parameters:
        A: a numpy array representing a word vector
        B: a numpy array representing a word vector
    Returns:
        The Euclidean Distance between A and B.
        d(A,B) is the square root of the sum of the square differences
        between corresponding elements of A and B:
            d(A,B) = sqrt(sum(A-B)**2)
    """

    return np.sqrt(np.sum(A-B)**2)


def cosine_similarity(A, B):
    """
    Parameters:
        A: a numpy array representing a word vector
        B: a numpy array representing a word vector
    Returns:
        cos: the Cosine Distance between A and B.
        cos(A,B) is the dot product of A and B over the product of the
        norm of A and the norm of B
            cos(A,B) = A.B / (norm(A) * norm(B))
            where, norm(X) = sqrt(sum(X**2))
    """

    dot_prod = np.dot(A, B)
    norm_a = np.sqrt(np.sum(A**2))
    norm_b = np.sqrt(np.sum(B**2))
    cos = dot_prod / (norm_a * norm_b)
