import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from .base import BaseRecommender


class ItemBasedCF(BaseRecommender):
    """
    Item-Based Collaborative Filtering Recommender System.

    This class implements an item-based collaborative filtering recommender system using cosine similarity to find similar items.

    Attributes:
        user_item_matrix (numpy.ndarray): A 2D array where rows represent users and columns represent items.
        item_similarity (numpy.ndarray): A 2D array representing the cosine similarity between items.
    """

    def __init__(self):
        """
        Initialize the ItemBasedCF recommender
        """
        self.user_item_matrix = None
        self.item_similarity = None
        self.user_ids = None
        self.item_ids = None

    def fit(self, user_item_matrix, user_ids=None, item_ids=None):
        """
        Fit the recommender model to the user-item interaction matrix.

        Parameters:
            user_item_matrix (numpy.ndarray): A 2D array where rows represent users and columns represent items.
            user_ids (list, optional): A list of user IDs.
            item_ids (list, optional): A list of item IDs.
        """
        self.user_item_matrix = user_item_matrix
        self.user_ids = user_ids
        self.item_ids = item_ids
        self.item_similarity = cosine_similarity(user_item_matrix.T)

    def recommend(self, user_id, n_recommendations=5):
        """
        Recommend items for a given user based on item-based collaborative filtering.

        Parameters:
            user_id (int): The ID of the user for whom to recommend items.
            n_recommendations (int): The number of recommendations to make.

        Returns:
            list: A list of recommended item IDs.
        """
        if self.item_similarity is None:
            raise ValueError(
                "The item similarity matrix has not been computed. Please call the 'fit' method first."
            )
        if self.user_item_matrix is None:
            raise ValueError(
                "The user-item matrix has not been set. Please call the 'fit' method first."
            )
        user_index = np.where(self.user_ids == user_id)[0][0]
        user_ratings = self.user_item_matrix[user_index].toarray().flatten()

        scores = self.item_similarity.dot(user_ratings)
        scores[user_ratings > 0] = 0  # Exclude already rated items

        top_indices = np.argsort(scores)[::-1][:n_recommendations]
        if self.item_ids is None:
            raise ValueError(
                "Item IDs are not set. Please provide item_ids when calling the 'fit' method."
            )
        return [self.item_ids[i] for i in top_indices]
