def get_unique_names(data_dict):
    """Extract unique province names and city names from data dictionary"""
    all_pnames = set()
    all_citynames = set()
    
    for year, df in data_dict.items():
        if 'pname' in df.columns:
            all_pnames.update(df['pname'].dropna().unique())
        if 'cityname' in df.columns:
            all_citynames.update(df['cityname'].dropna().unique())
    
    return sorted(list(all_pnames)), sorted(list(all_citynames))