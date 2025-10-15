def compare_dataframes(df_small, df_large, match_columns):
    """
    Compare two dataframes and find unique records in each
    
    Parameters:
    df_small: smaller dataframe
    df_large: larger dataframe  
    match_columns: list of column names used for matching
    
    Returns:
    dict containing:
    - 'large_only': records only in large dataframe
    - 'small_only': records only in small dataframe
    - 'common': records in both dataframes
    """
    # Check and report duplicates
    small_duplicates = df_small.duplicated(subset=match_columns).sum()
    large_duplicates = df_large.duplicated(subset=match_columns).sum()
    
    print(f"Small dataframe original records: {len(df_small)}")
    print(f"Small dataframe duplicates: {small_duplicates}")
    print(f"Large dataframe original records: {len(df_large)}")
    print(f"Large dataframe duplicates: {large_duplicates}")
    
    # Remove duplicates
    df_small_clean = df_small.drop_duplicates(subset=match_columns)
    df_large_clean = df_large.drop_duplicates(subset=match_columns)
    
    print(f"Small dataframe after deduplication: {len(df_small_clean)}")
    print(f"Large dataframe after deduplication: {len(df_large_clean)}")
    
    # Create matching keys
    small_keys = df_small_clean[match_columns]
    large_keys = df_large_clean[match_columns]
    
    # Find records only in large dataframe
    large_mask = ~large_keys.set_index(match_columns).index.isin(
        small_keys.set_index(match_columns).index
    )
    large_only = df_large_clean[large_mask]
    
    # Find records only in small dataframe
    small_mask = ~small_keys.set_index(match_columns).index.isin(
        large_keys.set_index(match_columns).index
    )
    small_only = df_small_clean[small_mask]
    
    # Find common records
    common_mask = small_keys.set_index(match_columns).index.isin(
        large_keys.set_index(match_columns).index
    )
    common = df_small_clean[common_mask]
    
    print(f"\nComparison results:")
    print(f"Records only in large dataframe: {len(large_only)}")
    print(f"Records only in small dataframe: {len(small_only)}")
    print(f"Common records: {len(common)}")
    
    return {
        'large_only': large_only,
        'small_only': small_only,
        'common': common
    }