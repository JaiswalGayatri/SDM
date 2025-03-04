from . import features_extractor 
import geopandas as gpd
from shapely.geometry import Point, Polygon, box
from shapely.wkt import loads
import random
import pandas as pd 
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances
import pandas as pd
import numpy as np
import Levenshtein
import os 


def divide_polygon_to_grids(polygon, grid_size=10, points_per_cell=5):
    
    polygon = loads(polygon)
    min_x, min_y, max_x, max_y = polygon.bounds
    step_x = (max_x - min_x) / grid_size
    step_y = (max_y - min_y) / grid_size
    sampled_points = []
    total= 0
    for i in range(grid_size):
        for j in range(grid_size):
            # Define the current grid cell as a polygon
            cell_min_x = min_x + i * step_x
            cell_max_x = min_x + (i + 1) * step_x
            cell_min_y = min_y + j * step_y
            cell_max_y = min_y + (j + 1) * step_y
            grid_cell = box(cell_min_x, cell_min_y, cell_max_x, cell_max_y)

            # Get the intersection of the grid cell with the ecoregion
            intersection = polygon.intersection(grid_cell)

            if not intersection.is_empty:
                # Sample points within the intersection
                points_in_cell = []
                while len(points_in_cell) < points_per_cell:
                    # Generate a random point within the grid cell bounds
                    random_x = random.uniform(cell_min_x, cell_max_x)
                    random_y = random.uniform(cell_min_y, cell_max_y)
                    point = Point(random_x, random_y)
                    # Check if the point lies within the intersection
                    if intersection.contains(point):
                        points_in_cell.append(point)
                total+=len(points_in_cell)
                # print(total)
                sampled_points.extend(points_in_cell)

    # Convert points to a list of [longitude, latitude] pairs
    # print('points sampled',total)
    sampled_points = [[point.x, point.y] for point in sampled_points]

    # Now create the DataFrame with correct shape (longitude, latitude)
    sampled_points = pd.DataFrame(sampled_points, columns=["longitude", "latitude"])

    return sampled_points


def representative_feature_vector_for_polygon(sampled_points, ee):

    feature_Extractor = features_extractor.Feature_Extractor(ee)
    

    features_df = feature_Extractor.add_features(sampled_points)
    

    feature_vector = features_df.mean(axis=0, skipna=True).tolist()
    feature_vector=feature_vector[2:]
    return feature_vector

def find_representive_vectors_from_files(input_folder, ee):
    feature_vectors = []
    file_names = []
    
    # Iterate over all WKT files in the given folder
    for filename in os.listdir(input_folder):
        if filename.endswith('.wkt'):
            print('processing',filename)
            with open(os.path.join(input_folder, filename), 'r') as file:
                polygon_wkt = file.read().strip()
                
            # Ensure the polygon_wkt is in string format (WKT format), not a Polygon object
            if isinstance(polygon_wkt, str):
                # Now, correctly load the WKT string into a Polygon object
                polygon = polygon_wkt
                
                # Generate sampled points from the polygon
                sampled_points = divide_polygon_to_grids(polygon, grid_size=10, points_per_cell=10)
                
                # Generate the representative feature vector for the polygon
                feature_vector = representative_feature_vector_for_polygon(sampled_points, ee)
                
                # Store the vector and corresponding file name (for identification)
                feature_vectors.append(feature_vector)
                file_names.append(filename)
            else:
                print(f"Skipping {filename} because it is not a valid WKT string.")
    
    # Convert feature vectors into a DataFrame
    feature_vectors_df = pd.DataFrame(feature_vectors)
    feature_vectors_df.index = file_names  # Use file names as the index
    feature_vectors_df.to_csv('data/representative_vectors_eco_region_wise.csv')
    
    return feature_vectors_df


def calculate_cosine_similarity_matrix(feature_vectors_df):
    similarity_matrix = cosine_similarity(feature_vectors_df)
    return similarity_matrix

# Function to calculate Euclidean Distance similarity matrix
def calculate_euclidean_similarity_matrix(feature_vectors_df):
    # Euclidean distance is often interpreted as the inverse of the distance (the smaller the distance, the higher the similarity)
    distance_matrix = euclidean_distances(feature_vectors_df)
    similarity_matrix = 1 / (1 + distance_matrix)  # Add 1 to avoid division by zero and make it a similarity measure
    return similarity_matrix



def jaccard_similarity(input_file, similarity_threshold=0.8):
    # Read the CSV file containing eco-regions and genera
    df = pd.read_csv(input_file)
    print(df.columns)
    
    # Function to check if two genus names are similar based on Levenshtein distance
    def are_genus_similar(genus1, genus2, threshold):
        # Convert genus names to lowercase before comparing
        genus1 = genus1.lower() if isinstance(genus1, str) else ""
        genus2 = genus2.lower() if isinstance(genus2, str) else ""
        
        # Calculate Levenshtein distance and convert it to a similarity score
        if pd.isna(genus1) or pd.isna(genus2):  # Handle missing values
            return False
        
        lev_distance = Levenshtein.distance(genus1, genus2)
        max_len = max(len(genus1), len(genus2))
        similarity = 1 - lev_distance / max_len
        return similarity >= threshold
    
    # Create a dictionary to store the genus sets for each eco-region
    eco_region_genus = {}

    # Populate the dictionary with eco-region and associated genera
    for _, row in df.iterrows():
        eco_region = row["Eco-region"]
        genus_list = row["Genus List"].split(", ")
        
        # Merge similar genus names
        merged_genus_list = []
        for genus in genus_list:
            # Add genus to the list if it is not a close match with any already present genus
            to_add = True
            for existing_genus in merged_genus_list:
                if are_genus_similar(genus, existing_genus, similarity_threshold):
                    to_add = False
                    break
            if to_add:
                merged_genus_list.append(genus)
        
        eco_region_genus[eco_region] = set(merged_genus_list)

    # Extract the eco-regions and initialize an empty matrix for similarities
    eco_regions = list(eco_region_genus.keys())
    similarity_matrix = np.zeros((len(eco_regions), len(eco_regions)))

    # Create a dictionary to map indices to eco-region names
    eco_region_index_map = {i: eco_region for i, eco_region in enumerate(eco_regions)}

    # Calculate the Jaccard similarity for each pair of eco-regions
    for i in range(len(eco_regions)):
        for j in range(i, len(eco_regions)):  # We only need to calculate once for each pair
            eco_region_i = eco_regions[i]
            eco_region_j = eco_regions[j]
            
            # Calculate the intersection and union of genus sets
            intersection = len(eco_region_genus[eco_region_i].intersection(eco_region_genus[eco_region_j]))
            union = len(eco_region_genus[eco_region_i].union(eco_region_genus[eco_region_j]))
            
            # Jaccard similarity
            similarity = intersection / union if union != 0 else 0
            
            # Fill the matrix with the calculated similarity
            similarity_matrix[i, j] = similarity
            similarity_matrix[j, i] = similarity  # Symmetric matrix

  
    
    # Save the resulting similarity matrix to a CSV file
    save_matrix_to_text(similarity_matrix, "outputs/jaccard_similarity_matrix.txt", eco_regions)

    print("Jaccard Similarity Matrix has been calculated and saved to 'jaccard_similarity_matrix.csv'.")
   









def save_matrix_to_text(matrix, filename, labels):
    """
    Save matrix to a human-readable text file with row and column labels.
    
    Parameters:
    matrix (np.ndarray): Similarity matrix to save
    filename (str): Output filename
    labels (list): Row and column labels
    """
    with open(filename, 'w') as f:
        # Write column headers
        f.write(' ' * 50)  # Indent for row labels
        f.write('\t'.join(labels) + '\n')
        
        # Write matrix with row labels
        for i, row_label in enumerate(labels):
            # Format row with row label and values
            row_values = [f"{val:.4f}" for val in matrix[i]]
            formatted_row = f"{row_label:<50}\t" + '\t'.join(row_values) + '\n'
            f.write(formatted_row)





