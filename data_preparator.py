import pandas as pd
from sklearn.model_selection import train_test_split

from pipeline.pipeline import prepare

df = pd.read_csv('./data/Kangaroo.csv')

# Split the dataset into training and testing sets (80% train, 20% test)
train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)

prepare(train_df, './data/train_data-cleaned.csv')
prepare(test_df, './data/test_data-cleaned.csv')
