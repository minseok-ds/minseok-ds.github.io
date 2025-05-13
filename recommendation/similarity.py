from sklearn.metrics.pairwise import cosine_similarity
import numpy as np


def compute_user_similarity(user_item_matrix):
    """
    Compute the cosine similarity between users based on their item ratings.

    Parameters:
    user_item_matrix (numpy.ndarray): A 2D array where rows represent users and columns represent items.

    Returns:
    numpy.ndarray: A 2D array representing the cosine similarity between users.
    """
    # Compute cosine similarity
    user_similarity = cosine_similarity(user_item_matrix)

    return user_similarity
