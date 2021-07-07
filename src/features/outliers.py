def outliers_by_iqr(df, feature):
    """
    Returns the portion of a dataframe's feature that consists of outliers
    """
    q1 = df[feature].quantile(0.25)
    q3 = df[feature].quantile(0.75)
    iqr = q3-q1
    minx = q1 - 1.5*iqr
    maxx = q3 + 1.5*iqr
    return df[(df[feature] > maxx) | (df[feature] < minx)]