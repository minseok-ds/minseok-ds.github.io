import numpy as np
from .base import BaseRecommender
from .similarity import compute_user_similarity


class UserBasedCF(BaseRecommender):
    """
    User-Based Collaborative Filtering Recommender System.

    This class implements a user-based collaborative filtering recommender system using cosine similarity to find similar users.

    Attributes:
        user_item_matrix (numpy.ndarray): A 2D array where rows represent users and columns represent items.
        user_similarity (numpy.ndarray): A 2D array representing the cosine similarity between users.
    """

    def __init__(self):
        """
        Initialize the UserBasedCF recommender
        """
        self.user_similarity = None
        self.user_item_matrix = None
        self.user_ids = None
        self.item_ids = None

    def fit(self, user_item_matrix, user_ids=None, item_ids=None):
        """
        Fit the recommender to the user-item interaction matrix.

        Parameters:
            user_item_matrix (numpy.ndarray): A 2D array where rows represent users and columns represent items.
            user_ids (list, optional): List of user IDs. Defaults to None.
            item_ids (list, optional): List of item IDs. Defaults to None.
        """
        self.user_item_matrix = user_item_matrix
        if user_item_matrix is None:
            raise ValueError("user_item_matrix cannot be None.")
        self.user_similarity = compute_user_similarity(user_item_matrix)
        self.user_ids = user_ids
        self.item_ids = item_ids

    def recommend(self, user_id, n_recommendations=5):
        """
        Recommend items for a given user based on user-based collaborative filtering.

        Parameters:
            user_id (int): The ID of the user for whom to recommend items.
            n_recommendations (int): The number of recommendations to make.

        Returns:
            list: A list of recommended item IDs.
        """
        if self.user_similarity is None:
            raise ValueError(
                "The user similarity matrix has not been computed. Please call the 'fit' method first."
            )
        user_index = np.where(self.user_ids == user_id)[0][0]

        user_index = np.where(self.user_ids == user_id)[0][0]
        if self.user_item_matrix is None:
            raise ValueError(
                "user_item_matrix is not set. Please call the 'fit' method with a valid matrix."
            )

        if hasattr(self.user_item_matrix, "toarray"):
            user_scores = self.user_similarity[user_index].dot(
                self.user_item_matrix.toarray()
            )
        else:
            user_scores = self.user_similarity[user_index].dot(self.user_item_matrix)

        user_scores[user_index] = -np.inf
        item_indices = np.argsort(user_scores)[::-1][:n_recommendations]
        if self.item_ids is None:
            raise ValueError(
                "item_ids is not set. Please provide item_ids when calling the 'fit' method."
            )
        return self.item_ids[item_indices].tolist()
