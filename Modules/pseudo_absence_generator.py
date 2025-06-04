# Import necessary libraries for numerical computing, geospatial analysis, and parallel processing
import numpy as np 
import pandas as pd 
from shapely.geometry import Point, shape, Polygon
from tqdm.notebook import tqdm  # Progress bar for Jupyter notebooks
import concurrent.futures  # For parallel processing
from . import features_extractor  # Custom module for extracting environmental features
from . import LULC_filter  # Custom module for land use/land cover filtering
from scipy.spatial.distance import cdist  # For calculating distances between points in feature space

# Define the list of environmental feature columns that will be used for analysis
# These represent various bioclimatic and environmental variables
feature_cols = [
    'annual_mean_temperature',           # Average temperature throughout the year
    'mean_diurnal_range',               # Average daily temperature range
    'isothermality',                    # Temperature evenness (day-night vs seasonal variation)
    'temperature_seasonality',          # Temperature variation coefficient
    'max_temperature_warmest_month',    # Highest temperature in warmest month
    'min_temperature_coldest_month',    # Lowest temperature in coldest month
    'temperature_annual_range',         # Difference between max and min monthly temperatures
    'mean_temperature_wettest_quarter', # Average temperature during wettest 3 months
    'mean_temperature_driest_quarter',  # Average temperature during driest 3 months
    'mean_temperature_warmest_quarter', # Average temperature during warmest 3 months
    'mean_temperature_coldest_quarter', # Average temperature during coldest 3 months
    'annual_precipitation',             # Total yearly precipitation
    'precipitation_wettest_month',      # Precipitation in wettest month
    'precipitation_driest_month',       # Precipitation in driest month
    'precipitation_seasonality',        # Precipitation variation coefficient
    'precipitation_wettest_quarter',    # Precipitation during wettest 3 months
    'precipitation_driest_quarter',     # Precipitation during driest 3 months
    'precipitation_warmest_quarter',    # Precipitation during warmest 3 months
    'precipitation_coldest_quarter',    # Precipitation during coldest 3 months
    'aridity_index',                   # Measure of dryness/wetness
    'topsoil_ph',                      # pH level of surface soil
    'subsoil_ph',                      # pH level of deeper soil
    'topsoil_texture',                 # Physical characteristics of surface soil
    'subsoil_texture',                 # Physical characteristics of deeper soil
    'elevation'                        # Height above sea level
]

class PseudoAbsences:
    """
    A class for generating pseudo-absence points for species distribution modeling.
    Pseudo-absences are artificially created absence points used when true absence data
    is not available, which is common in ecological studies.
    
    The class generates points that:
    1. Fall within suitable habitat (based on land use/land cover)
    2. Are environmentally dissimilar to known presence points
    3. Are within the study region (ecoregion boundaries)
    """
    
    def __init__(self, ee):
        """
        Initialize the PseudoAbsences class with Earth Engine and required datasets.
        
        Args:
            ee: Google Earth Engine object for accessing geospatial datasets
        """
        self.ee = ee 
        
        # Initialize feature extractor for getting environmental data at specific locations
        self.feature_extractor = features_extractor.Feature_Extractor(self.ee)
        
        # Load land use/land cover filter to identify suitable habitat areas
        self.modeLULC = LULC_filter.LULC_Filter(self.ee).load_modeLULC()
        
        # Load global ecoregions dataset for geographic boundaries
        ecoregions = ee.FeatureCollection('RESOLVE/ECOREGIONS/2017')
        
        # Load India country boundaries from US Department of State dataset
        india = ee.FeatureCollection('USDOS/LSIB_SIMPLE/2017') \
            .filter(ee.Filter.eq('country_co', 'IN'))
        
        # Define the study region geometry (currently set to India's boundaries)
        # Note: Variable name suggests focus on Malabar ecoregion, but code uses entire India
        ecoregion_geom = malabar_ecoregion = ee.FeatureCollection('USDOS/LSIB_SIMPLE/2017') \
                            .filter(ee.Filter.eq('country_co', 'IN')) \
                            .first()
        
        # Extract the geometry for spatial operations
        self.ecoregion_geom = ecoregion_geom.geometry()

    def reliability(self, presence_points_df, absence_point_dict):
        """
        Calculate the reliability of a pseudo-absence point based on its environmental
        dissimilarity to known presence points. Higher reliability means the point is
        more environmentally different from presence locations.
        
        Args:
            presence_points_df: DataFrame containing known presence points with features
            absence_point_dict: Dictionary containing environmental features for candidate absence point
        
        Returns:
            float: Reliability score (0-1), where higher values indicate better pseudo-absence points
        """
        # Extract only the environmental feature columns (skip longitude/latitude)
        presence_features = presence_points_df.iloc[:, 2:]

        # Convert absence point features to a list, filtering out None values
        # This handles cases where some environmental data might be missing
        absence_point_features = [float(absence_point_dict[i]) for i in absence_point_dict if absence_point_dict[i] is not None]
        
        # Calculate Euclidean distances between the absence point and all presence points
        # This measures how different the absence point is from known presence locations
        distances = cdist([absence_point_features], presence_features, metric='euclidean')

        # Convert distances to similarities using Gaussian kernel
        # Closer points (smaller distances) get higher similarity scores
        # The divisor (2 * number of features) normalizes for the feature space dimensionality
        similarities = np.exp(-distances**2 / (2 * presence_features.shape[1]))
        
        # Calculate mean similarity across all presence points
        mean_similarity = np.nanmean(similarities)
        
        # Reliability is inverse of similarity: less similar = more reliable as pseudo-absence
        reliability = 1 - mean_similarity
        return reliability

    def generate_batch_points(self, minx, miny, maxx, maxy, batch_size=1000):
        """
        Generate a batch of random coordinate points within specified bounding box.
        This creates candidate locations for pseudo-absence point evaluation.
        
        Args:
            minx, miny, maxx, maxy: Bounding box coordinates
            batch_size: Number of random points to generate
        
        Returns:
            list: List of (longitude, latitude) tuples
        """
        # Generate random longitude values within the x-bounds
        rand_lons = np.random.uniform(minx, maxx, batch_size)
        # Generate random latitude values within the y-bounds
        rand_lats = np.random.uniform(miny, maxy, batch_size)
        
        # Combine into coordinate pairs
        return list(zip(rand_lons, rand_lats))

    def generate_pseudo_absences(self, presence_df):
        """
        Main function to generate pseudo-absence points for species distribution modeling.
        The process involves:
        1. Generating random points within the study region
        2. Filtering points based on suitable habitat (LULC)
        3. Calculating environmental reliability scores
        4. Selecting points with high reliability (environmentally dissimilar to presences)
        
        Args:
            presence_df: DataFrame containing known species presence points with coordinates and features
        
        Returns:
            pandas.DataFrame: DataFrame containing generated pseudo-absence points with features
        """
        modelLULC = self.modeLULC
        # Generate same number of pseudo-absences as presence points (common practice)
        num_points = len(presence_df)
        # num_points=10  # Commented line for testing with smaller numbers
        ee = self.ee 
        ecoregion_geom = self.ecoregion_geom

        print(f"Target number of points to generate: {num_points}")

        # Initialize DataFrame to store all generated pseudo-absence points
        global_df = pd.DataFrame(columns=['longitude', 'latitude', 'reliability', 'normalized_features'])

        # Convert Earth Engine geometry to Shapely polygon for easier geometric operations
        eco_region_polygon = shape(ecoregion_geom.getInfo())
        # Get bounding box coordinates to constrain random point generation
        bounds = eco_region_polygon.bounds
        minx, miny, maxx, maxy = bounds

        # Initialize counters for monitoring generation process
        total_attempts = 0
        batch_size = 1000  # Process points in batches for efficiency

        # Progress bar to track point generation
        with tqdm(total=num_points, desc="Generating points") as pbar:
            # Continue generating until we have enough valid pseudo-absence points
            while len(global_df) < num_points:
                # Generate a batch of random candidate points within bounding box
                batch_points = self.generate_batch_points(minx, miny, maxx, maxy, batch_size)
                # TODO: Implement minimum distance constraint from presence points (e.g., 30m buffer)
                total_attempts += batch_size

                # Process batch points in parallel for efficiency
                # Using ThreadPoolExecutor to handle multiple points simultaneously
                with concurrent.futures.ThreadPoolExecutor(max_workers=16) as executor:
                    futures = []
                    # Submit each point for processing
                    for rand_lon, rand_lat in batch_points:
                        futures.append(executor.submit(self.process_single_point,
                                                    rand_lon,
                                                    rand_lat,
                                                    eco_region_polygon,
                                                    presence_df,
                                                    ))

                    # Collect results as they complete
                    for future in concurrent.futures.as_completed(futures):
                        result = future.result()
                        # Only add valid points (those that pass all filters)
                        if result is not None:
                            global_df = pd.concat([global_df, result], ignore_index=True)
                            pbar.update(1)

                            # Stop if we've generated enough points
                            if len(global_df) >= num_points:
                                break

        # Print final statistics about the generation process
        print(f"\nFinal Statistics:")
        print(f"Total points generated: {len(global_df)}")
        print(f"Total attempts made: {total_attempts}")
        print(f"Overall success rate: {(len(global_df)/total_attempts)*100:.2f}%")

        # Extract individual environmental features from the normalized_features dictionary
        # This creates separate columns for each environmental variable
        for i, col in enumerate(feature_cols):
            global_df[col] = global_df['normalized_features'].apply(lambda x: x[col])

        # Remove the original nested dictionary column now that features are in separate columns
        global_df = global_df.drop('normalized_features', axis=1)

        return global_df

    def process_single_point(self, rand_lon, rand_lat, eco_region_polygon, presence_df):
        """
        Process a single candidate point to determine if it's suitable as a pseudo-absence.
        The point must pass several criteria:
        1. Be within the study region boundaries
        2. Be in suitable habitat (LULC value = 1)
        3. Have sufficient environmental dissimilarity to presence points
        
        Args:
            rand_lon, rand_lat: Coordinates of candidate point
            eco_region_polygon: Shapely polygon defining study region boundaries
            presence_df: DataFrame containing known presence points for comparison
        
        Returns:
            pandas.DataFrame or None: Single-row DataFrame if point is valid, None otherwise
        """
        modelLULC = self.modeLULC
        ee = self.ee 
        
        try:
            # Check if the point falls within the study region boundaries
            if eco_region_polygon.contains(Point(rand_lon, rand_lat)):
                # Create Earth Engine point geometry for data extraction
                point = ee.Geometry.Point([rand_lon, rand_lat])
                
                # Extract land use/land cover value at this location
                # Using mode reducer to get the most common LULC class within the pixel
                lulc_value = modelLULC.reduceRegion(
                    ee.Reducer.mode(),
                    point,
                    scale=10,  # 10-meter resolution
                    maxPixels=1e9
                ).get('label').getInfo()

                # Check if the location is in suitable habitat (LULC value = 1)
                # Different LULC values represent different land cover types
                if lulc_value == 1:
                    # Extract environmental feature values at this location
                    bioclim_values_random = self.feature_extractor.get_feature_values_at_point(rand_lat, rand_lon)
                    
                    # Normalize the extracted values for consistent comparison
                    normalized_bioclim_values_random = self.feature_extractor.normalize_bioclim_values(bioclim_values_random)
                    
                    # Calculate how environmentally different this point is from known presences
                    reliability_value = self.reliability(presence_df, normalized_bioclim_values_random)

                    # Only accept points with sufficient environmental dissimilarity
                    # Threshold of 0.04 filters out points too similar to presence locations
                    if reliability_value > 0.04:
                        # Create a row with all relevant information for this pseudo-absence point
                        row = {
                            'longitude': rand_lon,
                            'latitude': rand_lat,
                            'reliability': reliability_value,
                            'normalized_features': normalized_bioclim_values_random
                        }
                        return pd.DataFrame([row])
                        
        except Exception as e:
            # Silently handle any errors (e.g., network issues, missing data)
            # This ensures the batch processing continues even if individual points fail
            pass
        
        # Return None if point doesn't meet criteria or encounters an error
        return None

# Generate pseudo-absence points
# (This comment suggests the class is ready to be used for generating pseudo-absence points)