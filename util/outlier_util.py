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
    
    # if min_thresholds is None:
    #     min_thresholds = {}
    
    outlier_counts = {}
    total_outliers = 0
    
    for column in columns:
        # Calculate Q1, Q3 and IQR
        Q1 = df_clean[column].quantile(0.05) #0.15
        Q3 = df_clean[column].quantile(0.90) #0.85
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
    
    # print(f"Total rows before removing outliers: {len(df)}")
    # print(f"Total rows after removing outliers: {len(df_clean)}")
    # print(f"Total outliers removed: {len(df) - len(df_clean)}")
    # print(f"Percentage of data retained: {len(df_clean)/len(df)*100:.1f}%")
    # print("\nOutliers removed per column:")
    
    # for col, count in outlier_counts.items():
    #     print(f"  {col}: {count} ({count/len(df)*100:.1f}%)")
    
    return df_clean