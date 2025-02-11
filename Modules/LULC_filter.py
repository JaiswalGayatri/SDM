import pandas as pd 

class LULC_Filter:
    def __init__(self,ee):
        self.ee = ee 
        self.modeLULC = self.load_modeLULC()


    def load_modeLULC(self):
        # Load the ecoregions dataset
        ee = self.ee 
        ecoregions = ee.FeatureCollection('RESOLVE/ECOREGIONS/2017')

        # Load the India boundaries dataset
        india = ee.FeatureCollection('USDOS/LSIB_SIMPLE/2017') \
            .filter(ee.Filter.eq('country_co', 'IN'))



        # Filter the ecoregions to get the Malabar Coast moist forests
        malabar_ecoregion = ecoregions \
            .filterBounds(india) \
            .filter(ee.Filter.eq('ECO_NAME', 'Malabar Coast moist forests')) \
            .first()

       

        # Get the LULC (Land Use/Land Cover) data for the eco region
        lulc = ee.ImageCollection("GOOGLE/DYNAMICWORLD/V1") \
            .filterBounds(malabar_ecoregion.geometry()) \
            .select('label')  # We only need the 'label' band for LULC classification

        # Get the most common LULC class (mode) for the Malabar region
        modeLULC = lulc.mode().clip(malabar_ecoregion.geometry())

       

        return modeLULC
    
    def filter_by_lulc(self,df):
        modeLULC = self.modeLULC
        print(len(df), 'points to be filtered')
        ee = self.ee
        features = []

        for _, row in df.iterrows():
            lon, lat = row['longitude'], row['latitude']
            point = ee.Geometry.Point([lon, lat])
            features.append(ee.Feature(point).set({'longitude': lon, 'latitude': lat}))
        # print(len(features),'this to be done')
        fc = ee.FeatureCollection(features)
        filtered_fc = fc.map(lambda feature: self.filter_point_by_lulc(feature)) \
                        .filter(ee.Filter.eq('lulc_label', 1))

        filtered_df = self.fc_to_dataframe(filtered_fc)
        return filtered_df

    def filter_point_by_lulc(self,feature):
        modeLULC = self.modeLULC
        ee = self.ee
        point = ee.Geometry.Point([ee.Number(feature.get('longitude')), ee.Number(feature.get('latitude'))])
        lulc_value = modeLULC.reduceRegion(
            reducer=ee.Reducer.mode(),
            geometry=point,
            scale=10,
            maxPixels=1e9
        ).get('label')
        return feature.set('lulc_label', lulc_value)

    def fc_to_dataframe(self,fc):
        ee = self.ee
        features = fc.getInfo()['features']
        data = []
        for feature in features:
            properties = feature['properties']
            data.append({
                'longitude': properties.get('longitude'),
                'latitude': properties.get('latitude'),
                'lulc_label': properties.get('lulc_label')
            })
        return pd.DataFrame(data)