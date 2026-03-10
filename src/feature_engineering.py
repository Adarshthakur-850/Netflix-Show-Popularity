import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder

def feature_engineering(df):
    print("Feature Engineering...")
    
    current_year = 2024
    df['content_age'] = current_year - df['release_year']
    
    df['cast_size'] = df['cast'].apply(lambda x: len(str(x).split(',')) if pd.notna(x) else 0)
    
    df['genre_count'] = df['listed_in'].apply(lambda x: len(str(x).split(',')) if pd.notna(x) else 0)
    
    df['is_movie'] = df['type'].apply(lambda x: 1 if x == 'Movie' else 0)
    df['is_us'] = df['country'].apply(lambda x: 1 if str(x) == 'United States' else 0)
    
    def parse_duration(val):
        try:
            return float(str(val).split()[0])
        except:
            return 0
    df['duration_num'] = df['duration'].apply(parse_duration)
    
    le = LabelEncoder()
    df['rating_encoded'] = le.fit_transform(df['rating'].fillna('Unknown'))
    
    return df
