def basic_preprocess(df):
    df = df.copy()
    df.rename(columns={'sales': 'department'}, inplace=True)
    df['average_monthly_hours'] = df['average_monthly_hours'] / 100
    return df