import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from scipy.sparse import csr_matrix
from recommendation import ItemBasedCF, build_user_item_matrix

# 1. Generate synthetic data
np.random.seed(42)
num_users = 10000
num_items = 1000
num_interactions = 500000

user_ids = np.random.choice(range(num_users), size=num_interactions, replace=True)
item_ids = np.random.choice(range(num_items), size=num_interactions, replace=True)
ratings = np.random.randint(1, 6, size=num_interactions)

data = pd.DataFrame(
    {
        "user_id": user_ids,
        "item_id": item_ids,
        "interaction": ratings,
    }
)

# 2. Build user-item matrix
pivot_table = data.pivot_table(
    index="user_id", columns="item_id", values="interaction", fill_value=0
)

# 3. Convert to sparse matrix
user_item_matrix = csr_matrix(pivot_table.values)

# 4. Initialize and fit the Item-Based CF model
item_cf = ItemBasedCF()
item_cf.fit(user_item_matrix, user_ids=pivot_table.index, item_ids=pivot_table.columns)

# 5. Recommend items for a specific user
user_id = 42
n_recommendations = 10
recommended_items = item_cf.recommend(user_id, n_recommendations)

print(f"Recommended items for user {user_id}: {recommended_items}")
