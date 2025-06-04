# Import necessary libraries for geospatial analysis, machine learning, and data processing
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
    """
    Divides a polygon into a grid and samples random points from each grid cell that intersects with the polygon.
    This creates a spatially distributed sampling strategy to represent the entire polygon area.
    
    Args:
        polygon: WKT string representation of the polygon to be sampled
        grid_size: Number of grid divisions along each axis (creates grid_size x grid_size grid)
        points_per_cell: Number of random points to sample from each non-empty grid cell
    
    Returns:
        pandas.DataFrame: DataFrame with columns ['longitude', 'latitude'] containing sampled points
    """
    
    # Convert WKT string to Shapely Polygon object for geometric operations
    polygon = loads(polygon)
    
    # Get the bounding box coordinates of the polygon to define grid boundaries
    min_x, min_y, max_x, max_y = polygon.bounds
    
    # Calculate the step size for grid divisions in both x and y directions
    step_x = (max_x - min_x) / grid_size
    step_y = (max_y - min_y) / grid_size
    
    # Initialize list to store all sampled points and counter for total points
    sampled_points = []
    total= 0
    
    # Iterate through each grid cell in the grid_size x grid_size grid
    for i in range(grid_size):
        for j in range(grid_size):
            # Calculate the boundaries of the current grid cell
            cell_min_x = min_x + i * step_x
            cell_max_x = min_x + (i + 1) * step_x
            cell_min_y = min_y + j * step_y
            cell_max_y = min_y + (j + 1) * step_y
            
            # Create a rectangular polygon representing the current grid cell
            grid_cell = box(cell_min_x, cell_min_y, cell_max_x, cell_max_y)

            # Find the intersection between the grid cell and the original polygon
            # This handles cases where the polygon doesn't fill the entire grid cell
            intersection = polygon.intersection(grid_cell)

            # Only proceed if there's an actual intersection (non-empty area)
            if not intersection.is_empty:
                # Initialize list to store points sampled from this specific cell
                points_in_cell = []
                
                # Continue sampling until we have the desired number of points per cell
                while len(points_in_cell) < points_per_cell:
                    # Generate random coordinates within the grid cell boundaries
                    random_x = random.uniform(cell_min_x, cell_max_x)
                    random_y = random.uniform(cell_min_y, cell_max_y)
                    point = Point(random_x, random_y)
                    
                    # Only keep the point if it actually lies within the polygon intersection
                    # This ensures we don't sample points outside the original polygon
                    if intersection.contains(point):
                        points_in_cell.append(point)
                
                # Update total counter and add points from this cell to the overall collection
                total+=len(points_in_cell)
                # print(total)  # Debug line to track sampling progress
                sampled_points.extend(points_in_cell)

    # Convert Shapely Point objects to coordinate pairs [longitude, latitude]
    # print('points sampled',total)  # Debug line to show total points sampled
    sampled_points = [[point.x, point.y] for point in sampled_points]

    # Create a pandas DataFrame with proper column names for the coordinate data
    sampled_points = pd.DataFrame(sampled_points, columns=["longitude", "latitude"])

    return sampled_points


def representative_feature_vector_for_polygon(sampled_points, ee):
    """
    Generates a representative feature vector for a polygon by extracting features from sampled points
    and computing their mean values. This creates a single vector that characterizes the entire polygon.
    
    Args:
        sampled_points: DataFrame containing longitude/latitude coordinates of sampled points
        ee: Earth Engine object or similar service for feature extraction
    
    Returns:
        list: Mean feature vector representing the polygon's characteristics
    """

    # Initialize the feature extractor with the Earth Engine service
    feature_Extractor = features_extractor.Feature_Extractor(ee)
    
    # Extract features for all sampled points (likely environmental/remote sensing data)
    # This could include things like elevation, temperature, precipitation, vegetation indices, etc.
    features_df = feature_Extractor.add_features(sampled_points)
    
    # Calculate the mean of each feature across all sampled points
    # This creates a representative "average" feature vector for the entire polygon
    # skipna=True handles any missing values in the feature data
    feature_vector = features_df.mean(axis=0, skipna=True).tolist()
    
    # Remove the first two columns (likely longitude/latitude) to keep only the extracted features
    feature_vector=feature_vector[2:]
    
    return feature_vector

def find_representive_vectors_from_files(input_folder, ee):
    """
    Processes multiple WKT polygon files to generate representative feature vectors for each.
    This function handles batch processing of multiple ecoregions or geographic areas.
    
    Args:
        input_folder: Path to folder containing .wkt files with polygon definitions
        ee: Earth Engine object for feature extraction
    
    Returns:
        pandas.DataFrame: DataFrame where each row is a feature vector for one polygon file
    """
    # Initialize lists to store results
    feature_vectors = []  # Will store the computed feature vectors
    file_names = []       # Will store corresponding file names for identification
    
    # Iterate through all files in the input folder
    for filename in os.listdir(input_folder):
        # Process only files with .wkt extension (Well-Known Text format for geometries)
        if filename.endswith('.wkt'):
            print('processing',filename)  # Progress indicator
            
            # Read the WKT polygon definition from the file
            with open(os.path.join(input_folder, filename), 'r') as file:
                polygon_wkt = file.read().strip()
                
            # Verify that we have a valid WKT string (not a Polygon object)
            if isinstance(polygon_wkt, str):
                # Use the WKT string directly (it's already in the correct format)
                polygon = polygon_wkt
                
                # Generate spatially distributed sample points within the polygon
                # Using larger grid and more points per cell for better representation
                sampled_points = divide_polygon_to_grids(polygon, grid_size=10, points_per_cell=10)
                
                # Extract and compute the representative feature vector for this polygon
                feature_vector = representative_feature_vector_for_polygon(sampled_points, ee)
                
                # Store the results with corresponding file name for later identification
                feature_vectors.append(feature_vector)
                file_names.append(filename)
            else:
                print(f"Skipping {filename} because it is not a valid WKT string.")
    
    # Convert the list of feature vectors into a structured DataFrame
    feature_vectors_df = pd.DataFrame(feature_vectors)
    # Use file names as row indices for easy identification of each ecoregion
    feature_vectors_df.index = file_names
    
    # Save the results to CSV for future use without reprocessing
    feature_vectors_df.to_csv('data/representative_vectors_eco_region_wise.csv')
    
    return feature_vectors_df


def calculate_cosine_similarity_matrix(feature_vectors_df):
    """
    Calculates cosine similarity between all pairs of feature vectors.
    Cosine similarity measures the cosine of the angle between vectors,
    focusing on orientation rather than magnitude. Values range from -1 to 1,
    where 1 indicates identical orientation.
    
    Args:
        feature_vectors_df: DataFrame where each row is a feature vector
    
    Returns:
        numpy.ndarray: Square matrix of cosine similarities between all vector pairs
    """
    similarity_matrix = cosine_similarity(feature_vectors_df)
    return similarity_matrix

def calculate_euclidean_similarity_matrix(feature_vectors_df):
    """
    Calculates similarity based on Euclidean distance between feature vectors.
    Euclidean distance measures straight-line distance in multi-dimensional space.
    The function converts distances to similarities using inverse transformation.
    
    Args:
        feature_vectors_df: DataFrame where each row is a feature vector
    
    Returns:
        numpy.ndarray: Square matrix of similarity values based on Euclidean distances
    """
    # Calculate pairwise Euclidean distances between all feature vectors
    distance_matrix = euclidean_distances(feature_vectors_df)
    
    # Convert distances to similarities: closer points (smaller distances) get higher similarity
    # Adding 1 prevents division by zero and ensures all values are positive
    similarity_matrix = 1 / (1 + distance_matrix)
    
    return similarity_matrix



def jaccard_similarity(input_file, similarity_threshold=0.8):
    """
    Calculates Jaccard similarity between ecoregions based on their genus composition.
    Jaccard similarity measures overlap between sets: |A ∩ B| / |A ∪ B|
    Also incorporates fuzzy string matching to handle similar genus names.
    
    Args:
        input_file: CSV file containing ecoregions and their associated genus lists
        similarity_threshold: Threshold for considering genus names as similar (0-1)
    """
    # Load the CSV file containing ecoregion data and genus information
    df = pd.read_csv(input_file)
    print(df.columns)  # Display available columns for debugging
    
    def are_genus_similar(genus1, genus2, threshold):
        """
        Determines if two genus names are similar using Levenshtein distance.
        This handles cases where genus names might have slight spelling variations.
        
        Args:
            genus1, genus2: Genus names to compare
            threshold: Similarity threshold (0=completely different, 1=identical)
        
        Returns:
            bool: True if genera are considered similar enough
        """
        # Convert to lowercase for case-insensitive comparison, handle missing values
        genus1 = genus1.lower() if isinstance(genus1, str) else ""
        genus2 = genus2.lower() if isinstance(genus2, str) else ""
        
        # Skip comparison if either genus name is missing
        if pd.isna(genus1) or pd.isna(genus2):
            return False
        
        # Calculate edit distance (number of character changes needed)
        lev_distance = Levenshtein.distance(genus1, genus2)
        # Normalize by the length of the longer string to get similarity ratio
        max_len = max(len(genus1), len(genus2))
        similarity = 1 - lev_distance / max_len
        
        # Return True if similarity exceeds the threshold
        return similarity >= threshold
    
    # Dictionary to store the set of genera for each ecoregion
    eco_region_genus = {}

    # Process each row in the dataset to build genus sets for each ecoregion
    for _, row in df.iterrows():
        eco_region = row["Eco-region"]
        # Split the comma-separated genus list into individual genus names
        genus_list = row["Genus List"].split(", ")
        
        # Merge similar genus names to avoid double-counting near-duplicates
        merged_genus_list = []
        for genus in genus_list:
            # Check if this genus is similar to any already processed genus
            to_add = True
            for existing_genus in merged_genus_list:
                if are_genus_similar(genus, existing_genus, similarity_threshold):
                    # If similar genus already exists, don't add this one
                    to_add = False
                    break
            # Add genus only if it's not similar to existing ones
            if to_add:
                merged_genus_list.append(genus)
        
        # Store the merged genus set for this ecoregion
        eco_region_genus[eco_region] = set(merged_genus_list)

    # Extract list of all ecoregions and initialize similarity matrix
    eco_regions = list(eco_region_genus.keys())
    similarity_matrix = np.zeros((len(eco_regions), len(eco_regions)))

    # Create mapping from matrix indices to ecoregion names (for reference)
    eco_region_index_map = {i: eco_region for i, eco_region in enumerate(eco_regions)}

    # Calculate Jaccard similarity for each pair of ecoregions
    for i in range(len(eco_regions)):
        for j in range(i, len(eco_regions)):  # Only calculate upper triangle (symmetric matrix)
            eco_region_i = eco_regions[i]
            eco_region_j = eco_regions[j]
            
            # Calculate set intersection (genera present in both ecoregions)
            intersection = len(eco_region_genus[eco_region_i].intersection(eco_region_genus[eco_region_j]))
            # Calculate set union (all unique genera across both ecoregions)
            union = len(eco_region_genus[eco_region_i].union(eco_region_genus[eco_region_j]))
            
            # Jaccard similarity = intersection size / union size
            # Handles edge case where union is empty (though unlikely with real data)
            similarity = intersection / union if union != 0 else 0
            
            # Fill both symmetric positions in the matrix
            similarity_matrix[i, j] = similarity
            similarity_matrix[j, i] = similarity

    # Save the resulting similarity matrix in human-readable format
    save_matrix_to_text(similarity_matrix, "outputs/jaccard_similarity_matrix.txt", eco_regions)

    print("Jaccard Similarity Matrix has been calculated and saved to 'jaccard_similarity_matrix.csv'.")


def save_matrix_to_text(matrix, filename, labels):
    """
    Saves a similarity matrix to a human-readable text file with proper formatting.
    The output includes row and column labels for easy interpretation.
    
    Parameters:
    matrix (np.ndarray): Square similarity matrix to save
    filename (str): Path and filename for the output text file
    labels (list): Labels for rows and columns (typically ecoregion names)
    """
    with open(filename, 'w') as f:
        # Write column headers with proper spacing
        # First column is left empty to align with row labels
        f.write(' ' * 50)  # Indent to align with row label width
        f.write('\t'.join(labels) + '\n')
        
        # Write each row of the matrix with its corresponding label
        for i, row_label in enumerate(labels):
            # Format each similarity value to 4 decimal places for readability
            row_values = [f"{val:.4f}" for val in matrix[i]]
            # Create formatted row: label (left-aligned, 50 chars) + tab + values
            formatted_row = f"{row_label:<50}\t" + '\t'.join(row_values) + '\n'
            f.write(formatted_row)