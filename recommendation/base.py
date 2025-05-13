from abc import ABC, abstractmethod


class BaseRecommender(ABC):
    @abstractmethod
    def fit(self, user_item_matrix):
        """
        Fit the recommender model to the user-item interaction matrix.

        Parameters:
        user_item_matrix (pd.DataFrame): A DataFrame where rows are users, columns are items, and values are interaction scores.
        """
        pass

    @abstractmethod
    def recommend(self, user_id, n=10):
        """
        Recommend items for a given user.

        Parameters:
        user_id (int): The ID of the user to recommend items for.
        n (int): The number of items to recommend.

        Returns:
        list: A list of recommended item IDs.
        """
        pass
