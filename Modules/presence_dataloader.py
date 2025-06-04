# Import necessary libraries for API requests, file handling, and data processing
import requests  # For making HTTP requests to GBIF API
import csv       # For reading and writing CSV files
from time import sleep  # For adding delays between API calls (rate limiting)
import ee        # Google Earth Engine (imported but not used in this class)
import pandas as pd  # For data manipulation and analysis

class Presence_dataloader():
    """
    A class for downloading and processing species occurrence data from GBIF (Global Biodiversity Information Facility).
    
    This class handles:
    1. Loading species occurrence points from GBIF API within a specified geographic region
    2. Filtering data by date range and taxonomic criteria
    3. Removing duplicate records
    4. Saving data to CSV files for further analysis
    
    GBIF is a major international biodiversity database that aggregates species occurrence
    records from museums, research institutions, and citizen science platforms worldwide.
    """
    
    def __init__(self):
        """
        Initialize the Presence_dataloader class.
        
        Note: Earth Engine initialization is commented out as it's not used in current implementation.
        The class is designed to work independently of Earth Engine for data downloading.
        """
        # self.ee = ee  # Earth Engine not needed for GBIF data loading
        return 
    
    def load_raw_presence_data(self, maxp=2000):
        """
        Download species occurrence data from GBIF API within a specified polygon boundary.
        
        The function performs the following steps:
        1. Reads the study area polygon and target genus from input files
        2. Makes paginated API calls to GBIF to retrieve occurrence records
        3. Filters out duplicate coordinates
        4. Saves unique occurrence points to a CSV file
        5. Continues until reaching the maximum number of points or exhausting available data
        
        Args:
            maxp (int): Maximum number of presence points to collect (default: 2000)
        
        Returns:
            set: Set of unique (longitude, latitude) tuples representing occurrence locations
        """
        # Read the study area polygon from Well-Known Text (WKT) format file
        # WKT is a standard format for representing geometric shapes
        with open("Inputs/polygon.wkt", "r") as input_polygon:
            polygon_wkt = input_polygon.read().strip()  # Remove whitespace/newlines

        # Read the target genus name from text file
        # This specifies which taxonomic group to search for
        with open("Inputs/genus_name.txt", "r") as genus:
            genus_name = genus.read().strip()  # Remove whitespace/newlines
        
        # Use a set to automatically handle duplicate coordinate pairs
        # Sets only store unique values, preventing duplicate locations
        occurrence_points = set()

        # Initialize the CSV file with headers for storing occurrence data
        # This creates a clean file each time the function runs
        try:
            with open("data/presence.csv", "w") as presence_data:
                writer = csv.writer(presence_data)
                writer.writerow(["longitude", "latitude"])  # Write column headers
        except FileNotFoundError:
            # Handle case where the data directory doesn't exist
            pass 

        # Set up pagination parameters for GBIF API
        # GBIF limits the number of records returned per request
        offset, limit = 0, 300  # Start at record 0, get 300 records per request
        
        print(f'Beginning to find at least {maxp} presence points for {genus_name} in input polygon')
        
        # Main loop to retrieve data from GBIF API with pagination
        while True:
            # GBIF API endpoint for occurrence data search
            gbif_url = "https://api.gbif.org/v1/occurrence/search"
            
            # Set up API request parameters
            params = {
                "scientificName": genus_name + "%",  # Search for genus (% is wildcard for species)
                "geometry": polygon_wkt,             # Limit search to specified polygon
                "limit": limit,                      # Number of records per request
                "offset": offset,                    # Starting position for pagination
                "eventDate": "2017-01-01,2023-12-31",  # Date range filter (recent data)
                "kingdomKey": 6                      # Kingdom filter (6 = Plantae for plants)
            }
            
            # Make the API request and handle potential errors
            response = requests.get(gbif_url, params=params)
            response.raise_for_status()  # Raise exception if HTTP error occurs

            # Process the response and extract coordinate information
            new_points = set()  # Store new points found in this API call
            
            # Iterate through each occurrence record in the API response
            for result in response.json()["results"]:
                # Extract longitude and latitude coordinates
                point = (result["decimalLongitude"], result["decimalLatitude"])
                
                # Only add points that haven't been seen before (avoid duplicates)
                if point not in occurrence_points:
                    new_points.add(point)
                    occurrence_points.add(point)
                    
                    # Print taxonomic information for monitoring progress
                    # Shows both genus and full scientific name for verification
                    print('genus is', result['genus'], 'species is', result['scientificName'])

            # Save new unique points to CSV file if any were found
            if new_points:
                with open("data/presence.csv", "a", newline="") as csvfile:  # Append mode
                    writer = csv.writer(csvfile)
                    writer.writerows(new_points)  # Write all new points at once
                print(f"Saved {len(new_points)} new unique occurrence points to presence.csv")

            # Check if we've reached the end of available data
            # If fewer results than requested limit, no more data available
            if len(response.json()["results"]) < limit:
                break

            # Move to the next page of results
            offset += limit

            # Stop if we've collected enough points
            if len(occurrence_points) >= maxp:
                break 

        return occurrence_points
    
    def load_unique_lon_lats(self):
        """
        Load and deduplicate occurrence data from the saved CSV file.
        
        This function provides a way to reload and clean previously downloaded data
        without making new API calls. It's useful for:
        1. Reprocessing existing data
        2. Removing any duplicates that might have been missed during download
        3. Getting a clean dataset for analysis
        
        Returns:
            pandas.DataFrame: DataFrame containing unique longitude/latitude pairs
        """
        # Read the previously saved occurrence data from CSV
        df = pd.read_csv("data/presence.csv")
    
        # Remove duplicate coordinate pairs
        # This ensures each geographic location appears only once in the dataset
        # Important for species distribution modeling to avoid spatial bias
        df_unique = df.drop_duplicates(subset=['longitude', 'latitude'])
        
        return df_unique