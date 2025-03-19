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

def test_model_on_all_ecoregions(clf, Features_extractor, modelss,output_file='data/avg_prob.txt'):
    polygon_dir = 'data/eco_regions_polygon'
    # output_file = 'outputs/testing_SDM_out/all_india_trained_matrix_Dalbergia_avg_prob.txt'

    # Write the header for the output file
    with open(output_file, 'w') as out_file:
        out_file.write('Ecoregion,Average_Probability\n')

    write_header = not os.path.exists(output_file)
    cnt = 0

    for filename in os.listdir(polygon_dir):
        print(f'Starting for ecoregion {cnt + 1}')
        if filename.endswith('.wkt'):  # Process only .wkt files
            ecoregion_name = os.path.splitext(filename)[0]  # Get ecoregion name without extension
            polygon_path = os.path.join(polygon_dir, filename)

            try:
                with timeout(120):  # Set a 2-minute timeout
                    # Read the polygon WKT
                    with open(polygon_path, 'r') as file:
                        polygon_wkt = file.read().strip()

                    # Generate test data for the current ecoregion
                    X_dissimilar = Features_extractor.add_features(
                        utility.divide_polygon_to_grids(polygon_wkt, grid_size=1, points_per_cell=20)
                    )
                    test_presence_path = 'data/test_presence.csv'
                    pd.DataFrame(X_dissimilar).to_csv(test_presence_path, index=False)

                    X_test, y_test, _, _, _ = modelss.load_data(
                        presence_path=test_presence_path,
                        absence_path='data/test_absence.csv'
                    )

                    # Make predictions
                    y_proba = clf.predict_proba(X_test)[:, 1]

                    # Calculate the average probability
                    avg_probability = y_proba.mean()
            except TimeoutError:
                print(f'Timeout for {ecoregion_name}. Setting average probability to 0.')
                avg_probability = 0

            # Write the result to the output file
            with open(output_file, 'a') as out_file:
                out_file.write(f'{ecoregion_name},{avg_probability}\n')
            cnt += 1
            print(f'Done for ecoregion {cnt}')

    print(f'Average probabilities saved to {output_file}')


def main():
  
    Presence_dataloader = presence_dataloader.Presence_dataloader()
    Features_extractor = features_extractor.Feature_Extractor(ee)
    LULC_Filter = LULC_filter.LULC_Filter(ee)
    Pseudo_absence = pseudo_absence_generator.PseudoAbsences(ee)
    modelss = models.Models()
    # generate_prob = Generate_Prob.Generate_Prob(ee)
    
    
    # raw_occurrences = Presence_dataloader.load_raw_presence_data()   #uncomment if want to use gbif api to generate presence points
    
    # unique_presences = Presence_dataloader.load_unique_lon_lats()
    # presences_filtered_LULC = LULC_Filter.filter_by_lulc(unique_presences)
    # print(len(presences_filtered_LULC))
    # presence_data_with_features  = Features_extractor.add_features(presences_filtered_LULC)
    # presence_data_with_features.to_csv('data/presence.csv',index=False,mode='w')
    # presence_data_with_features = pd.read_csv('data/presence.csv')
    # pseudo_absence_points_with_features = Pseudo_absence.generate_pseudo_absences(presence_data_with_features)
    print('training model_random forest')
    X,y,_,_,_ = modelss.load_data('data/testing_SDM/presence_points_Dalbergia_all_india.csv','data/testing_SDM/absence_points_dalbergia_all_india.csv')
    # print(X.shape)
    # # return
    clf, X_test, y_test, y_pred, y_proba = modelss.RandomForest(X,y)
    avg=0
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

    print('##############################')
    # print('training model_ logistic regression')
    # X,y,_,_,_ = modelss.load_data('data/testing_SDM/presence_points_Dalbergia_all_india.csv','data/testing_SDM/absence_points_dalbergia_all_india.csv')
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

    # print('##############################')
    # print('training model weighted logistic regression')
    # X,y,_,_,_ = modelss.load_data('data/testing_SDM/presence_points_Dalbergia_all_india.csv','data/testing_SDM/absence_points_dalbergia_all_india.csv')
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
    test_model_on_all_ecoregions(clf,Features_extractor,modelss,output_file = 'outputs/testing_SDM_out/all_india_trained_matrix_Dalbergia_avg_prob_random_forest.txt')



    


    # import os
    # import geopandas as gpd
    # import pandas as pd
    # from shapely import wkt, Point
    # print('finding species count in each ecoregion')
    # # Path to folder containing WKT polygon files
    # ecoregion_folder = "data/eco_regions_polygon"

    # # Load species occurrence data (CSV with latitude & longitude)
    # species_df = pd.read_csv("data/testing_SDM/presence_points_erythinia_all_india.csv")

    # # Convert species points into a GeoDataFrame
    # species_df["geometry"] = species_df.apply(lambda row: Point(row["longitude"], row["latitude"]), axis=1)
    # species_gdf = gpd.GeoDataFrame(species_df, geometry="geometry", crs="EPSG:4326")

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

    # # Load all ecoregions into a GeoDataFrame
    # ecoregion_gdf = load_ecoregions(ecoregion_folder)

    # # Spatial join: Assign each species occurrence to an ecoregion
    # species_with_ecoregions = gpd.sjoin(species_gdf, ecoregion_gdf, how="left", predicate="within")

    # # Count occurrences in each ecoregion
    # ecoregion_counts = species_with_ecoregions.groupby("ecoregion").size().reset_index(name="count")

    # # Save results
    # ecoregion_counts.to_csv("outputs/testing_SDM_out/species_ecoregion_counts_erythinia.csv", index=False)
    # print('done')

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

