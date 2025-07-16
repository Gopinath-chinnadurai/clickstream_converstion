import pandas as pd

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    df['page'] = df['page'].fillna('unknown')
    df['price'] = pd.to_numeric(df['price'], errors='coerce').fillna(0)
    df['price_2'] = pd.to_numeric(df['price_2'], errors='coerce').fillna(0)

    df['session_length'] = df['price_2'] - df['price']

    df['num_clicks'] = df['page'].apply(lambda x: len(str(x).split(',')))

    for i in range(1, 5):
        df[f'category_{i}_clicks'] = df['page'].apply(lambda x: str(x).count(f'category{i}'))

    df['exit_page'] = df['page'].apply(lambda x: str(x).split(',')[-1] if pd.notna(x) else 'unknown')

    df['is_bounce'] = df['page'].apply(lambda x: 1 if len(str(x).split(',')) == 1 else 0)

    df['repeated_views'] = df['page'].apply(lambda x: 1 if len(set(str(x).split(','))) < len(str(x).split(',')) else 0)

    return df
