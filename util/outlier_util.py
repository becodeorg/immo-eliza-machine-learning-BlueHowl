#Thanks Copilot for this long piece of code 
def remove_outliers_iqr(df, columns, multiplier=1.5, min_thresholds={}):
    """
    Remove outliers from the dataframe using the IQR method.
    
    Parameters:
    df (pandas.DataFrame): The dataframe to process
    columns (list): List of column names to check for outliers
    multiplier (float): Multiplier for IQR to determine outlier threshold
    min_thresholds (dict): Dictionary with minimum threshold values for specific columns
    
    Returns:
    pandas.DataFrame: Dataframe with outliers removed
    """
    df_clean = df.copy()
    
    outlier_counts = {}
    total_outliers = 0
    
    for column in columns:
        # Calculate Q1, Q3 and IQR
        Q1 = df_clean[column].quantile(0.05)
        Q3 = df_clean[column].quantile(0.90)
        IQR = Q3 - Q1
        
        # Calculate the outlier thresholds
        lower_bound = Q1 - (multiplier * IQR)
        upper_bound = Q3 + (multiplier * IQR)
        
        # Apply minimum threshold if specified
        if column in min_thresholds:
            lower_bound = max(lower_bound, min_thresholds[column])
        
        # Identify outliers
        outliers = df_clean[(df_clean[column] < lower_bound) | (df_clean[column] > upper_bound)]
        outlier_count = len(outliers)
        outlier_counts[column] = outlier_count
        total_outliers += outlier_count
        
        # Filter out the outliers
        df_clean = df_clean[(df_clean[column] >= lower_bound) & (df_clean[column] <= upper_bound)]
    
    return df_clean