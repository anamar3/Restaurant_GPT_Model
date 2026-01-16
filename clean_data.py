import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from sklearn.impute import SimpleImputer

import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer


def clean_data():
    df = pd.read_csv('../../../../Downloads/restaurant_clf.csv')
    print(f"Оригинален размер: {df.shape[0]:,} реда × {df.shape[1]} колони\n")

    # 1. Премахваме всичко без реална оценка
    initial = len(df)
    df = df[~df['rate'].astype(str).str.strip().str.upper().
            isin(['NEW', '-', ''])]
    df['rate'] = pd.to_numeric(df['rate'].astype(str).str.split('/').str[0],
                               errors='coerce')
    df = df.dropna(subset=['rate']).reset_index(drop=True)
    print(
        f"Остават {len(df):,} реда с реална оценка (премахнати {initial - len(df):,})\n"
    )

    # 2. Target
    median_rate = df['rate'].median()
    df['high_rating'] = (df['rate'] > median_rate).astype(int)

    # 3. Почистване на цена
    df['approx_cost'] = df['approx_cost(for two people)'].str.replace(
        ',', '').astype(float)

    # 4. Брой кухни
    df['cuisines_count'] = df['cuisines'].fillna('').apply(
        lambda x: len([i for i in str(x).split(',') if i.strip()]))

    # 5. Binary features
    df['has_dish_liked'] = df['dish_liked'].notna().astype(int)
    df['online_order'] = (df['online_order'] == 'Yes').astype(int)
    df['book_table'] = (df['book_table'] == 'Yes').astype(int)

    # 6. Log трансформации
    df['log_votes'] = np.log1p(df['votes'])
    df['log_cost'] = np.log1p(df['approx_cost'])

    numeric_cols = ['log_votes', 'log_cost', 'cuisines_count']
    df[numeric_cols] = SimpleImputer(strategy='median').fit_transform(
        df[numeric_cols])

    df.to_csv('rest_cleaned.csv', index=False)
    return df
