import pandas as pd
from scipy.sparse import csr_matrix


def build_user_item_matrix(data):
    """
    Build a user-item interaction matrix from the given data.

    Parameters:
    data (pd.DataFrame): A DataFrame containing user-item interactions with columns ['user_id', 'item_id', 'interaction'].

    Returns:
    pd.DataFrame: A user-item interaction matrix.
    """
    user_item_matrix = data.pivot(
        index="user_id", columns="item_id", values="interaction"
    ).fillna(0)
    return user_item_matrix
