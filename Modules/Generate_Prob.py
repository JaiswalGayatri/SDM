import rasterio
from shapely.geometry import Polygon
import numpy as np
from tqdm.notebook import tqdm
import time 
from . import features_extractor

class Generate_Prob:
    def __init__(self,ee):
        self.ee= ee 
        self.feature_extractor = features_extractor.Feature_Extractor(self.ee)
        self.assets = (self.feature_extractor).assets
        self.min_max_values = (self.feature_extractor).min_max_values
        return 
    


    def predict_eco_region(self,model):
        
        print("Preparing ecoregion bounds...")
        assets = self.assets
        min_max_dict = self.min_max_values
        
        # Get the ecoregion bounds geometry and extract coordinates
        ee_bounds_polygon = assets['malabar_ecoregion'].geometry().bounds()
        ee_bounds_coords = ee_bounds_polygon.coordinates().getInfo()[0]
        
        # Extract bounds from the coordinates (assumes rectangular bounds)
        min_lon = min(coord[0] for coord in ee_bounds_coords)
        max_lon = max(coord[0] for coord in ee_bounds_coords)
        min_lat = min(coord[1] for coord in ee_bounds_coords)
        max_lat = max(coord[1] for coord in ee_bounds_coords)
        bounds = (min_lon, min_lat, max_lon, max_lat)
        
        print("Eco-region bounds:")
        print(f"  Min Longitude: {bounds[0]}")
        print(f"  Max Longitude: {bounds[2]}")
        print(f"  Min Latitude: {bounds[1]}")
        print(f"  Max Latitude: {bounds[3]}")
        
        # Fixed resolution for WorldClim (30 arc-seconds)
        resolution = 0.25  # n
        
        # Calculate grid dimensions

        #todo: formalise this resolution parameter.
        width = int((bounds[2] - bounds[0]) / resolution)
        height = int((bounds[3] - bounds[1]) / resolution)
        
        print(f"Creating prediction grid with dimensions: {width}x{height}")
        
        # Create transform and mask
        transform = rasterio.transform.from_bounds(*bounds, width=width, height=height)
        eco_region_mask = rasterio.features.geometry_mask(
            [Polygon([(bounds[0], bounds[1]), (bounds[2], bounds[1]), (bounds[2], bounds[3]), (bounds[0], bounds[3]), (bounds[0], bounds[1])])],
            transform=transform,
            out_shape=(height, width),
            invert=True  # Mask points inside the polygon
        )
        
        # Debug mask
        valid_points_count = np.sum(eco_region_mask)
        print(f"Processing {valid_points_count} valid points...")
        
        if valid_points_count == 0:
            print("No valid points found in the mask. Check CRS and polygon geometry.")
            return None, None
        
        # Initialize probability map
        probability_map = np.zeros((height, width))
        
        # Collect all valid points
        all_points = [
            {'coords': (i, j), 'lat': lat, 'lon': lon}
            for i in range(height)
            for j in range(width)
            if eco_region_mask[i, j]
            for lon, lat in [rasterio.transform.xy(transform, i, j)]
        ]
        
        # Use tqdm to show progress for individual points
        for point in tqdm(all_points, desc="Processing points", unit="point"):
            feature_dict = self.feature_extractor.get_feature_values_at_point(point['lat'], point['lon'])
            if feature_dict is not None:
                # Normalize features
                normalized = self.feature_extractor.normalize_bioclim_values(feature_dict)
                feature_list = [normalized.get(key) for key in min_max_dict.keys()]
                
                if all(f is not None for f in feature_list):  # Ensure no missing features
                    X_pred = np.array([feature_list])
                    probability = model.predict_proba(X_pred)[:, 1]
                    
                    coord = point['coords']
                    probability_map[coord[0], coord[1]] = probability[0]
        
        # Save the GeoTIFF
        output_file = 'outputs/Probability_Distribution.tif'
        with rasterio.open(
            output_file,
            'w',
            driver='GTiff',
            height=height,
            width=width,
            count=1,
            dtype=probability_map.dtype,
            crs='EPSG:4326',
            transform=transform,
            nodata=0
        ) as dst:
            dst.write(probability_map, 1)
            dst.update_tags(
                TIFFTAG_COPYRIGHT='Generated using Earth Engine assets',
                TIFFTAG_DATETIME=time.strftime('%Y:%m:%d %H:%M:%S'),
                TIFFTAG_SOFTWARE='Species Distribution Model - Malabar Region'
            )
        
        print(f"Probability distribution saved to {output_file}")
        return probability_map, transform
