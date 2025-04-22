import ee 
import os 
from shapely.wkt import loads
import pandas as pd
import numpy as np
import signal
from contextlib import contextmanager
from Modules import presence_dataloader, features_extractor, LULC_filter, pseudo_absence_generator, models, Generate_Prob, utility
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from scipy.stats import spearmanr
import concurrent.futures
from functools import partial
import time
import traceback
import threading
import queue
ee.Authenticate()
ee.Initialize(project='sigma-bay-425614-a6')

@contextmanager
def timeout(time):
    """Raise TimeoutError if the block takes longer than 'time' seconds."""
    def raise_timeout(signum, frame):
        raise TimeoutError

    signal.signal(signal.SIGALRM, raise_timeout)
    signal.alarm(time)
    try:
        yield
    finally:
        signal.alarm(0)

# def test_model_on_all_ecoregions(clf, Features_extractor, modelss, output_file='data/avg_prob.txt'):
#     polygon_dir = 'data/eco_regions_polygon'

#     # Write the header for the output file if the file doesn't exist
#     if not os.path.exists(output_file):
#         with open(output_file, 'w') as out_file:
#             out_file.write('Ecoregion,Average_Probability\n')

#     cnt = 0

#     for filename in os.listdir(polygon_dir):
#         print(f'Starting for ecoregion {cnt + 1}')
#         if filename.endswith('.wkt'):  # Process only .wkt files
#             ecoregion_name = os.path.splitext(filename)[0]  # Get ecoregion name without extension
#             polygon_path = os.path.join(polygon_dir, filename)

#             try:
#                 with timeout(15):  # Set a 2-minute timeout
#                     # Read the polygon WKT
#                     with open(polygon_path, 'r') as file:
#                         polygon_wkt = file.read().strip()

#                     # Generate test data for the current ecoregion
#                     X_dissimilar = Features_extractor.add_features(
#                         utility.divide_polygon_to_grids(polygon_wkt, grid_size=1, points_per_cell=20)
#                     )
#                     test_presence_path = 'data/test_presence.csv'
#                     pd.DataFrame(X_dissimilar).to_csv(test_presence_path, index=False)

#                     X_test, y_test, _, _, _ = modelss.load_data(
#                         presence_path=test_presence_path,
#                         absence_path='data/test_absence.csv'
#                     )

#                     # Remove NaN and infinite values from test set
#                     X_test = np.array(X_test, dtype=float)
#                     mask = np.isfinite(X_test).all(axis=1)
#                     X_test = X_test[mask]

#                     if X_test.shape[0] == 0:  # If no valid samples remain
#                         print(f'No valid samples for {ecoregion_name}. Setting average probability to 0.')
#                         avg_probability = 0
#                     else:
#                         # Make predictions
#                         y_proba = clf.predict_proba(X_test)[:, 1]

#                         # Calculate the average probability
#                         avg_probability = y_proba.mean()
#             except TimeoutError:
#                 print(f'Timeout for {ecoregion_name}. Setting average probability to 0.')
#                 avg_probability = 0

#             # Write the result to the output file
#             with open(output_file, 'a') as out_file:
#                 out_file.write(f'{ecoregion_name},{avg_probability}\n')
#             cnt += 1
#             print(f'Done for ecoregion {cnt}')

#     print(f'Average probabilities saved to {output_file}')



def test_model_on_all_ecoregions(clf, Features_extractor, modelss, output_file='data/avg_prob.txt', num_workers=16):
    polygon_dir = 'data/eco_regions_polygon'
    
    # Write the header for the output file if the file doesn't exist
    if not os.path.exists(output_file):
        with open(output_file, 'w') as out_file:
            out_file.write('Ecoregion,Average_Probability\n')
    
    # Get all .wkt files
    wkt_files = [f for f in os.listdir(polygon_dir) if f.endswith('.wkt')]
    total_files = len(wkt_files)
    
    print(f'Starting processing of {total_files} ecoregions with {num_workers} workers')
    
    # Create a queue of files to process
    file_queue = queue.Queue()
    for filename in wkt_files:
        file_queue.put(filename)
    
    # Create a lock for file writing
    file_lock = threading.Lock()
    
    # Create a shared counter for completed files
    completed = [0]
    
    def worker():
        while not file_queue.empty():
            try:
                # Get a file from the queue with a timeout
                try:
                    filename = file_queue.get(timeout=1)
                except queue.Empty:
                    break
                
                # Process the file
                ecoregion_name = os.path.splitext(filename)[0]
                try:
                    avg_probability = process_single_ecoregion(
                        filename, polygon_dir, clf, Features_extractor, modelss
                    )
                except Exception as e:
                    print(f'Error processing {ecoregion_name}: {str(e)}')
                    avg_probability = 0.0
                
                # Write the result to the output file
                with file_lock:
                    with open(output_file, 'a') as out_file:
                        out_file.write(f'{ecoregion_name},{avg_probability}\n')
                    completed[0] += 1
                    print(f'Completed {completed[0]}/{total_files}: {ecoregion_name}')
                
                # Mark the task as done
                file_queue.task_done()
            except Exception as e:
                print(f'Worker error: {str(e)}')
    
    # Start worker threads
    threads = []
    for _ in range(num_workers):
        t = threading.Thread(target=worker)
        t.daemon = True
        t.start()
        threads.append(t)
    
    # Wait for all files to be processed or timeout
    timeout = 60 * 60  # 1 hour timeout
    start_time = time.time()
    while completed[0] < total_files and time.time() - start_time < timeout:
        time.sleep(1)
    
    # Check if all files were processed
    if completed[0] < total_files:
        print(f'WARNING: Only {completed[0]}/{total_files} files were processed before timeout')
        
        # Process any remaining files sequentially
        remaining_files = []
        while not file_queue.empty():
            try:
                remaining_files.append(file_queue.get(timeout=1))
            except queue.Empty:
                break
        
        print(f'Processing remaining {len(remaining_files)} files sequentially')
        for filename in remaining_files:
            ecoregion_name = os.path.splitext(filename)[0]
            try:
                avg_probability = process_single_ecoregion(
                    filename, polygon_dir, clf, Features_extractor, modelss
                )
            except Exception as e:
                print(f'Error processing {ecoregion_name}: {str(e)}')
                avg_probability = 0.0
            
            with open(output_file, 'a') as out_file:
                out_file.write(f'{ecoregion_name},{avg_probability}\n')
            completed[0] += 1
            print(f'Completed {completed[0]}/{total_files}: {ecoregion_name}')
    
    print(f'All ecoregions processed. Average probabilities saved to {output_file}')

def process_single_ecoregion(filename, polygon_dir, clf, Features_extractor, modelss):
    """Process a single ecoregion file and return the average probability."""
    ecoregion_name = os.path.splitext(filename)[0]
    polygon_path = os.path.join(polygon_dir, filename)
    
    # Read the polygon WKT
    with open(polygon_path, 'r') as file:
        polygon_wkt = file.read().strip()
    
    # Generate test data for the current ecoregion
    X_dissimilar = Features_extractor.add_features(
        utility.divide_polygon_to_grids(polygon_wkt, grid_size=1, points_per_cell=20)
    )
    
    # Create a unique temporary file for this thread
    thread_id = threading.get_ident()
    timestamp = int(time.time() * 1000)
    temp_file = f'data/temp_presence_{thread_id}_{timestamp}.csv'
    pd.DataFrame(X_dissimilar).to_csv(temp_file, index=False)
    
    try:
        X_test, y_test, _, _, _,bias_weights = modelss.load_data(
            presence_path=temp_file,
            absence_path='data/test_absence.csv'
        )
        
        # Remove NaN and infinite values from test set
        X_test = np.array(X_test, dtype=float)
        mask = np.isfinite(X_test).all(axis=1)
        X_test = X_test[mask]
        
        if X_test.shape[0] == 0:  # If no valid samples remain
            print(f'No valid samples for {ecoregion_name}. Setting average probability to 0.')
            avg_probability = 0
        else:
            # Make predictions
            y_proba = clf.predict_proba(X_test)[:, 1]
            
            # Calculate the average probability
            avg_probability = y_proba.mean()
    finally:
        # Clean up temporary file
        if os.path.exists(temp_file):
            try:
                os.remove(temp_file)
            except:
                pass
    
    return avg_probability


def main():
  
    # Presence_dataloader = presence_dataloader.Presence_dataloader()
    Features_extractor = features_extractor.Feature_Extractor(ee)
    # LULC_Filter = LULC_filter.LULC_Filter(ee)
    # Pseudo_absence = pseudo_absence_generator.PseudoAbsences(ee)
    modelss = models.Models()
    # # generate_prob = Generate_Prob.Generate_Prob(ee)
    
    
    # # raw_occurrences = Presence_dataloader.load_raw_presence_data()   #uncomment if want to use gbif api to generate presence points
    
    # # unique_presences = Presence_dataloader.load_unique_lon_lats()
    # # presences_filtered_LULC = LULC_Filter.filter_by_lulc(unique_presences)
    # # print(len(presences_filtered_LULC))
    # # presence_data_with_features  = Features_extractor.add_features(presences_filtered_LULC)
    # # presence_data_with_features.to_csv('data/presence.csv',index=False,mode='w')
    # # presence_data_with_features = pd.read_csv('data/presence.csv')
    # # pseudo_absence_points_with_features = Pseudo_absence.generate_pseudo_absences(presence_data_with_features)
    print('training model_random forest')
    X,y,_,_,sample_weights,bias_weights= modelss.load_data('data/testing_SDM/presence_points_dalbergia_all_india.csv','data/testing_SDM/absence_points_dalbergia_all_india.csv')
    # print(X.shape)
    # return
    print(bias_weights)
    clf, X_test, y_test, y_pred, y_proba = modelss.RandomForest(X,y,bias_weights)
    # avg=0

    metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'confusion_matrix': confusion_matrix(y_test, y_pred),
            'classification_report': classification_report(y_test, y_pred)
        }
        
    # # Print the results

    print(f"Accuracy: {metrics['accuracy']:.4f}")
    print("\nConfusion Matrix:")
    print(metrics['confusion_matrix'])
    print("\nClassification Report:")
    print(metrics['classification_report'])
    print('done predicting')
    test_model_on_all_ecoregions(clf,Features_extractor,modelss,output_file = 'outputs/testing_SDM_out/all_india_erythinia_avg_prob_RF.txt')

    print('##############################')
    # print('training model_ logistic regression')
    # X,y,_,_,_ = modelss.load_data('data/testing_SDM/presence_points_ficus_all_india.csv','data/testing_SDM/absence_points_ficus_all_india.csv')
    # # print(X.shape)
    # # # return
    # clf, X_test, y_test, y_pred, y_proba = modelss.logistic_regression_L2(X,y)
    # avg=0
    # metrics = {
    #         'accuracy': accuracy_score(y_test, y_pred),
    #         'confusion_matrix': confusion_matrix(y_test, y_pred),
    #         'classification_report': classification_report(y_test, y_pred)
    #     }
        
    # # # Print the results

    # print(f"Accuracy: {metrics['accuracy']:.4f}")
    # print("\nConfusion Matrix:")
    # print(metrics['confusion_matrix'])
    # print("\nClassification Report:")
    # print(metrics['classification_report'])
    # print('done predicting')
    # test_model_on_all_ecoregions(clf,Features_extractor,modelss,output_file = 'outputs/testing_SDM_out/all_indiar_trained_matrix_ficus_avg_prob_LR.txt')

    # print('##############################')
    # print('training model weighted logistic regression')
    # X,y,_,_,_ = modelss.load_data('data/testing_SDM/presence_points_ficus_all_india.csv','data/testing_SDM/absence_points_ficus_all_india.csv')
    # # print(X.shape)
    # # # return
    # clf, X_test, y_test, y_pred, y_proba = modelss.train_and_evaluate_model_logistic_weighted(X,y)
    # avg=0
    # metrics = {
    #         'accuracy': accuracy_score(y_test, y_pred),
    #         'confusion_matrix': confusion_matrix(y_test, y_pred),
    #         'classification_report': classification_report(y_test, y_pred)
    #     }
        
    # # # Print the results

    # print(f"Accuracy: {metrics['accuracy']:.4f}")
    # print("\nConfusion Matrix:")
    # print(metrics['confusion_matrix'])
    # print("\nClassification Report:")
    # print(metrics['classification_report'])
    # print('done predicting')

    # print('##############################')


    # test_model_on_all_ecoregions(clf,Features_extractor,modelss,output_file = 'outputs/testing_SDM_out/all_india_trained_matrix_ficus_avg_prob_WLR.txt')
   
    

    
  

   

    # X_test,y_test,_,_,_ = modelss.load_data(presence_path='data/test_presence.csv',absence_path='data/test_absence.csv')
    
    # print('testing data loaded')

    # y_pred = clf.predict(X_test)
    # y_proba = clf.predict_proba(X_test)[:, 1]
    # print('prediction stored')
    
    # avg=0
    # for i, prob in enumerate(y_proba):
    #     print(f"Sample {i}: {prob:.4f}")
    #     avg+=prob 
    # avg/=71
    # print('avg prob is',avg)

    # Print feature importances (Coefficients)
   
   
    # print(pseudo_absence_points_with_features.head(5))
    # pseudo_absence_points_with_features.to_csv('data/pseudo_absence.csv', index=False)

    # feature_vectors_df = utility.find_representive_vectors_from_files('data/eco_regions_polygon', ee)
    
    # # Step 2: Calculate similarity matrices
    # feature_vectors_df = pd.read_csv('data/representative_vectors_eco_region_wise.csv', index_col=0)
    # cosine_similarity_matrix = utility.calculate_cosine_similarity_matrix(feature_vectors_df)
    # euclidean_similarity_matrix = utility.calculate_euclidean_similarity_matrix(feature_vectors_df)
    
    # row_labels = feature_vectors_df.index.tolist()
    
    # # Print results
    # print("Cosine Similarity Matrix:")
    # cosine_df = pd.DataFrame(
    #     cosine_similarity_matrix, 
    #     index=row_labels, 
    #     columns=row_labels
    # )
    # print(cosine_df)
    
    # print("\nEuclidean Similarity Matrix:")
    # euclidean_df = pd.DataFrame(
    #     euclidean_similarity_matrix, 
    #     index=row_labels, 
    #     columns=row_labels
    # )
    # print(euclidean_df)
    
    # # Save matrices to text files
    # utility.save_matrix_to_text(
    #     cosine_similarity_matrix, 
    #     'data/cosine_similarity_matrix.txt', 
    #     row_labels
    # )
    # utility.save_matrix_to_text(
    #     euclidean_similarity_matrix, 
    #     'data/euclidean_similarity_matrix.txt', 
    #     row_labels


    # )

    # Example usage:
    # input_file = "data/eco_region_wise_genus.csv"  # Replace with your cleaned input file path
    # utility.jaccard_similarity(input_file)
    # with open('data/eco_regions_polygon/Terai_Duar_savanna_and_grasslands.wkt', 'r') as file:
    #     polygon_wkt1 = file.read().strip()
        # print(polygon_wkt)
    
    # # with open('data/eco_regions_polygon/South_Western_Ghats_moist_deciduous_forests.wkt', 'r') as file:
    # #     polygon_wkt2 = file.read().strip()

    # X_dissimilar = Features_extractor.add_features(utility.divide_polygon_to_grids(polygon_wkt1,grid_size=1,points_per_cell=20))
    # pd.DataFrame.to_csv(X_dissimilar,'data/test_presence.csv',index=False)
    # X_test,y_test,_,_,_ = modelss.load_data(presence_path='data/test_presence.csv',absence_path='data/test_absence.csv')

    # # print('predicting for a dissimilar reogionnn')
    # y_pred = clf.predict(X_test)
    # y_proba = clf.predict_proba(X_test)[:, 1]

    # print(f"Accuracy_RFC: {accuracy_score(y_test, y_pred):.4f}")
    # print("\nConfusion Matrix:")
    # print(confusion_matrix(y_test, y_pred))
    # print("\nClassification Report:")
    # print(classification_report(y_test, y_pred))


    # print("\nProbabilities on the test set:")
    # for i, prob in enumerate(y_proba):
    #     print(f"Sample {i}: {prob:.4f}")


    # X_dissimilar = Features_extractor.add_features(utility.divide_polygon_to_grids(polygon_wkt2,grid_size=12,points_per_cell=1))
    # print(X_dissimilar)
    # # print(X_similar)
    # pd.DataFrame.to_csv(X_dissimilar,'data/test_presence.csv',index=False)

# 




    


    # import os
    # import geopandas as gpd
    # import pandas as pd
    # from shapely import wkt, Point

    # print('Finding species count in each ecoregion...')

    # # Define paths
    # ecoregion_folder = "data/eco_regions_polygon"
    # presence_file = "data/testing_SDM/presence_points_Dalbergia_all_india.csv"
    # absence_file = "data/testing_SDM/absence_points_dalbergia_all_india.csv"

    # # Load presence data and convert to GeoDataFrame
    # presence_df = pd.read_csv(presence_file)
    # presence_df["geometry"] = presence_df.apply(lambda row: Point(row["longitude"], row["latitude"]), axis=1)
    # presence_gdf = gpd.GeoDataFrame(presence_df, geometry="geometry", crs="EPSG:4326")

    # # Load absence data and convert to GeoDataFrame
    # absence_df = pd.read_csv(absence_file)
    # absence_df["geometry"] = absence_df.apply(lambda row: Point(row["longitude"], row["latitude"]), axis=1)
    # absence_gdf = gpd.GeoDataFrame(absence_df, geometry="geometry", crs="EPSG:4326")

    # # Combine presence and absence GeoDataFrames
    # combined_gdf = pd.concat([presence_gdf, absence_gdf], ignore_index=True)
    # print(len(combined_gdf))

    # # Function to load ecoregion polygons from WKT files
    # def load_ecoregions(folder):
    #     ecoregions = []
    #     for file in os.listdir(folder):
    #         if file.endswith(".wkt"):  # Assuming WKT files
    #             with open(os.path.join(folder, file), "r") as f:
    #                 wkt_text = f.read().strip()
    #                 poly = wkt.loads(wkt_text)
    #                 ecoregions.append({"ecoregion": file.replace(".wkt", ""), "geometry": poly})
    #     return gpd.GeoDataFrame(ecoregions, geometry="geometry", crs="EPSG:4326")

    # # Load eco-region polygons
    # ecoregion_gdf = load_ecoregions(ecoregion_folder)

    # # Spatial join: assign each combined occurrence (presence and absence) to an eco-region
    # # Using predicate "within" to check if the point falls inside the polygon
    # combined_with_ecoregion = gpd.sjoin(combined_gdf, ecoregion_gdf, how="left", predicate="within")

    # # Count occurrences in each eco-region
    # ecoregion_counts = combined_with_ecoregion.groupby("ecoregion").size().reset_index(name="count")

    # # Save results to CSV
    # output_file = "outputs/testing_SDM_out/species_ecoregion_count_1.csv"
    # ecoregion_counts.to_csv(output_file, index=False)
    # print('Done. Species occurrence counts saved to:', output_file)


# print(ecoregion_counts)  # Print output

# def compute_concentration_index(csv_file):
#     """
#     Computes the concentration index (Ci) for a species based on its occurrence distribution across ecoregions.

#     Args:
#         csv_file (str): Path to the CSV file containing species occurrence counts per ecoregion.

#     Returns:
#         float: The concentration index (Ci).
#     """
#     # Load species occurrence counts
#     df = pd.read_csv(csv_file)

#     # Ensure the CSV has correct columns
#     if "count" not in df.columns:
#         raise ValueError("CSV file must contain a 'count' column with species occurrences.")

#     # Total occurrences of the species
#     total_occurrences = df["count"].sum()

#     # Compute probability pij for each ecoregion
#     df["pij"] = df["count"] / total_occurrences

#     # Compute concentration index (Ci)
#     df["pij_log_pij"] = df["pij"] * np.log(df["pij"])
#     Ci = -df["pij_log_pij"].sum()

#     return Ci

# # Example usage
# csv_file = "outputs/species_ecoregion_counts.csv"
# Ci = compute_concentration_index(csv_file)
# print(f"Concentration Index (Ci): {Ci:.4f}")


    
   



if __name__ == "__main__":
    main()

