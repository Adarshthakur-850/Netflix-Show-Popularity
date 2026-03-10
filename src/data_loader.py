import pandas as pd
import numpy as np
import random
import os
from datetime import datetime

def load_data(filepath="data/netflix_titles.csv"):
    if not os.path.exists(filepath):
        print("Generating synthetic Netflix data...")
        generate_synthetic_data(filepath)
    return pd.read_csv(filepath)

def generate_synthetic_data(filepath):
    np.random.seed(42)
    random.seed(42)
    
    n_samples = 2000
    
    types = ['Movie', 'TV Show']
    countries = ['United States', 'India', 'United Kingdom', 'Japan', 'South Korea']
    ratings = ['PG-13', 'TV-MA', 'TV-14', 'R', 'TV-PG']
    genres = ['Drama', 'Comedy', 'Action', 'Thriller', 'Documentary', 'Romance', 'Sci-Fi']
    
    data = []
    
    for _ in range(n_samples):
        title_type = random.choices(types, weights=[0.7, 0.3])[0]
        country = random.choices(countries, weights=[0.4, 0.2, 0.2, 0.1, 0.1])[0]
        release_year = random.randint(1990, 2024)
        
        # Synthesize duration
        if title_type == 'Movie':
            duration = f"{random.randint(80, 180)} min"
        else:
            duration = f"{random.randint(1, 10)} Season"
            
        genre_list = random.sample(genres, k=random.randint(1, 3))
        listed_in = ", ".join(genre_list)
        
        cast_size = random.randint(2, 10)
        cast = ", ".join([f"Actor {_}" for _ in range(cast_size)])
        
        # Calculate synthetic popularity (Target)
        # Newer content + specific genres + cast size -> higher popularity
        base_pop = 50
        age_factor = (release_year - 1990) * 1.5 
        genre_factor = 10 if 'Action' in genre_list or 'Drama' in genre_list else 5
        cast_factor = cast_size * 2
        noise = np.random.normal(0, 10)
        
        popularity = base_pop + age_factor + genre_factor + cast_factor + noise
        popularity = max(0, min(100, popularity))
        
        data.append({
            'show_id': f"s{_}",
            'type': title_type,
            'title': f"Show Title {_}",
            'director': f"Director {_%100}",
            'cast': cast,
            'country': country,
            'date_added': f"January {random.randint(1, 28)}, {release_year + random.randint(0, 2)}",
            'release_year': release_year,
            'rating': random.choice(ratings),
            'duration': duration,
            'listed_in': listed_in,
            'description': "A generic description of the show content.",
            'popularity_score': round(popularity, 1)
        })
        
    df = pd.DataFrame(data)
    
    # Save
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    df.to_csv(filepath, index=False)
    print(f"Data generated at {filepath}")
