import pandas as pd
from scipy.stats import spearmanr

# Read the CSV file
df = pd.read_csv("india_similarities.csv")

# Add ranks for columns used in rank correlation
df['Rank_Euclidean_Similarity'] = df['Euclidean_Similarity'].rank()
df['Rank_Average_Probability_Euclidean'] = df['Average_Probability'].rank()

df['Rank_Cosine_Similarity'] = df['Cosine_Similarity'].rank()
df['Rank_Average_Probability_Cosine'] = df['Average_Probability'].rank()

# Compute Spearman rank correlation for each pair
rank_corr_euclidean, _ = spearmanr(df['Euclidean_Similarity'], df['Average_Probability'])
rank_corr_cosine, _ = spearmanr(df['Cosine_Similarity'], df['Average_Probability'])

# Add the correlation values as new columns
df['Rank_Correlation_Euclidean'] = rank_corr_euclidean
df['Rank_Correlation_Cosine'] = rank_corr_cosine

# Save the updated DataFrame to a new CSV file
df.to_csv("india_similarities_rank_correlation.csv", index=False)


