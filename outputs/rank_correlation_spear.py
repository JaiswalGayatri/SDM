import pandas as pd
from scipy.stats import spearmanr

# Read the CSV file
df = pd.read_csv("malabar_similarities.csv")

# Compute Spearman rank correlation for each pair
rank_corr_euclidean, _ = spearmanr(df['Euclidean_Similarity'], df['Average_Probability'])
rank_corr_cosine, _ = spearmanr(df['Cosine_Similarity'], df['Average_Probability'])

# Add the correlation values as new columns
df['Rank_Correlation_Euclidean'] = rank_corr_euclidean
df['Rank_Correlation_Cosine'] = rank_corr_cosine

# Display the updated DataFrame
df.to_csv("malabar_similarities_rank_correlation.csv", index=False)
