import pandas as pd
import numpy as np

def read_similarity_matrix(file_path):
    """Read similarity matrix from text file."""
    try:
        # Read the similarity matrix with proper handling of whitespace and numerical precision
        df = pd.read_csv(file_path, delimiter='\t', index_col=0, skipinitialspace=True)
        
        # Clean up column names and index by stripping any whitespace
        df.columns = df.columns.str.strip()
        df.index = df.index.str.strip()
        
        # Convert to float64 to maintain precision
        df = df.astype(np.float64)
        
        return df
    except Exception as e:
        print(f"Error reading similarity file {file_path}: {e}")
        return None

def read_probability_file(file_path):
    """Read probability file."""
    try:
        df = pd.read_csv(file_path)
        # Create a dictionary mapping ecoregion to probability
        prob_dict = dict(zip(df['Ecoregion'], df['Average_Probability']))
        return prob_dict
    except Exception as e:
        print(f"Error reading probability file {file_path}: {e}")
        return None

def process_similarities(euclidean_file, cosine_file, malabar_prob_file, india_prob_file, target_ecoregion):
    """Process similarity files and create output CSV files."""
    # Read input files
    euclidean_df = read_similarity_matrix(euclidean_file)
    cosine_df = read_similarity_matrix(cosine_file)
    malabar_probs = read_probability_file(malabar_prob_file)
    india_probs = read_probability_file(india_prob_file)
    
    if not all([euclidean_df is not None, cosine_df is not None, 
                malabar_probs is not None, india_probs is not None]):
        print("Error: Failed to read one or more input files")
        return
    
    try:
        # Get the similarities from respective files
        euclidean_similarities = euclidean_df[target_ecoregion]
        cosine_similarities = cosine_df[target_ecoregion]
        
        # Create base dataframe with ecoregion names
        result_df = pd.DataFrame({
            'Ecoregion': euclidean_similarities.index.str.replace('.wkt', ''),
            'Euclidean_Similarity': euclidean_similarities.values,
            'Cosine_Similarity': cosine_similarities.values
        })
        
        # Create two copies for different probability sources
        malabar_result = result_df.copy()
        india_result = result_df.copy()
        
        # Add probabilities
        malabar_result['Average_Probability'] = [malabar_probs.get(eco, None) 
                                               for eco in malabar_result['Ecoregion']]
        india_result['Average_Probability'] = [india_probs.get(eco, None) 
                                             for eco in india_result['Ecoregion']]
        
        # Round numerical values to 4 decimal places for consistency
        float_columns = ['Euclidean_Similarity', 'Cosine_Similarity', 'Average_Probability']
        for col in float_columns:
            malabar_result[col] = malabar_result[col].round(4)
            india_result[col] = india_result[col].round(4)
        
        # Sort by Average_Probability in descending order
        malabar_result = malabar_result.sort_values('Average_Probability', ascending=False)
        india_result = india_result.sort_values('Average_Probability', ascending=False)
        
        # Save to CSV files
        malabar_result.to_csv('malabar_similarities.csv', index=False)
        india_result.to_csv('india_similarities.csv', index=False)
        
        print(f"Successfully processed similarities for {target_ecoregion}")
        # Print first few rows of sorted results
        print("\nTop 5 ecoregions by probability (Malabar):")
        print(malabar_result[['Ecoregion', 'Average_Probability']].head())
        print("\nTop 5 ecoregions by probability (India):")
        print(india_result[['Ecoregion', 'Average_Probability']].head())
        
    except Exception as e:
        print(f"Error processing similarities: {e}")

# Usage
if __name__ == "__main__":
    target = "Malabar_Coast_moist_forests.wkt"
    process_similarities(
        euclidean_file="euclidean_similarity_matrix.txt",
        cosine_file="cosine_similarity_matrix.txt",
        malabar_prob_file="malabar_trained_matrix.txt",
        india_prob_file="india_trained_matrix.txt",
        target_ecoregion=target
    )