from custom_transformers import *
from sklearn.pipeline import Pipeline
import pandas as pd

TO_DROP_LIST = ['Unnamed: 0', 
                'id', 
                'url', 
                'locality', 
                'type',
                'monthlyCost', 
                'hasBalcony', 
                'accessibleDisabledPeople', 
                'kitchenSurface', 
                'hasTerrace', 
                'hasGarden', 
                'gardenOrientation', 
                'roomCount', 
                'streetFacadeWidth', 
                'floorCount', 
                'floodZoneType', 
                'terraceOrientation', 
                'hasAttic', 
                'hasBasement', 
                'diningRoomSurface', 
                'hasDiningRoom', 
                'hasLift', 
                'heatingType', 
                'hasLivingRoom', 
                'livingRoomSurface', 
                'gardenSurface', 
                'parkingCountIndoor', 
                'hasAirConditioning', 
                'hasArmoredDoor', 
                'hasVisiophone', 
                'bathroomCount']

NA_BOOL_REPLACE_LIST = ['hasOffice', 
                        'hasPhotovoltaicPanels',
                        'hasHeatPump', 
                        'hasThermicPanels', 
                        'hasFireplace', 
                        'hasDressingRoom', 
                        'hasSwimmingPool']

NA_NUMERIC_REPLACE_LIST = ['terraceSurface', 'parkingCountOutdoor', 'landSurface']

MEDIAN_REPLACE_LIST = ['facedeCount', 'buildingConstructionYear', 'bedroomCount', 'habitableSurface', 'epc_kwh']

MODE_REPLACE_LIST = ['buildingCondition', 'subtype', 'kitchenType']


df = pd.read_csv('./data/Kangaroo.csv')

cleaning_pipe = Pipeline(steps=[
    ('drop_duplicates', DuplicateDropper(subset=['id'])),

    ('drop_columns', ColumnDropper(columns_to_drop=TO_DROP_LIST)), #Drop useless columns

    ('replace_na_bools', NAReplacer(column=NA_BOOL_REPLACE_LIST, new_value=False)), #Assume NaN is not having the feature
    ('replace_na_numerics', NAReplacer(column=NA_NUMERIC_REPLACE_LIST, new_value=0)), #Assume NaN is 0 for numerics (not present)

    #calculate epc_kwh
    ('epc_kwh_calculator', EpcKwhCalculator()), #Calculate epc_kwh from epcScore & province
    ('drop_epc_province_columns', ColumnDropper(columns_to_drop=['epcScore', 'province'])), #Drop columns used to calculate epc_kwh

    ('remove_outliers', OutlierRemover(columns=['price', 'habitableSurface'], multiplier=1.5, min_thresholds={'habitableSurface' : 25})), #Remove outliers from price column using IQR method

    ('replace_na_median', NAReplacer(column=MEDIAN_REPLACE_LIST, new_value='median')), #Fill NaN with mean value

    #change types to int
    ('convert_categories', CategoryTransformer(columns=['kitchenType', 'subtype', 'buildingCondition'])),
    ('convert_postal_code', PostalCodeTransformer()), #Convert postalCode to int

    ('replace_na_mode', NAReplacer(column=MODE_REPLACE_LIST, new_value='mode')), #Fill NaN with mode value

    ('drop_na', NADropper()), #Drop rows with NaN

    ('convert_bool_to_int', BooleanTransformer()), #Convert boolean strings to integers (True=1, False=0)
    ('convert_to_int', ToIntTransformer()) #Convert columns to int
])

cleaned_set = cleaning_pipe.fit_transform(df)

cleaned_set.to_csv('./data/kangaroo-cleaned.csv', index=False)
print(f"Dataset saved with {cleaned_set.shape[0]} rows and {cleaned_set.shape[1]} columns")