import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist
from sklearn.neighbors import NearestNeighbors
import matplotlib.pyplot as plt

def calculate_distance_with_xy(x1, y1, x2, y2):
    # Convert decimal degrees to radians
    lon1, lat1, lon2, lat2 = map(np.radians, [x1, y1, x2, y2])
    
    # Haversine formula
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
    c = 2 * np.arcsin(np.sqrt(a))
    
    # Radius of Earth in meters
    R = 6371.0 * 1000
    
    # Calculate the distance
    distance = R * c
    
    return distance


def haversine_vectorized(coords1, coords2):
    """
    向量化的haversine距离计算
    
    Parameters:
    -----------
    coords1 : array-like, shape (n, 2)
        第一组坐标 [longitude, latitude]
    coords2 : array-like, shape (m, 2) 
        第二组坐标 [longitude, latitude]
        
    Returns:
    --------
    ndarray, shape (n, m)
        距离矩阵
    """
    coords1 = np.array(coords1)
    coords2 = np.array(coords2)
    
    # 转换为弧度
    coords1_rad = np.radians(coords1)
    coords2_rad = np.radians(coords2)
    
    # 扩展维度进行广播计算
    lon1 = coords1_rad[:, 0:1]  # shape (n, 1)
    lat1 = coords1_rad[:, 1:2]  # 修复：应该是 1:2 而不是 1:1
    lon2 = coords2_rad[:, 0].reshape(1, -1)   # shape (1, m)
    lat2 = coords2_rad[:, 1].reshape(1, -1)   # shape (1, m)
    
    # Haversine公式向量化计算
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    
    a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
    c = 2 * np.arcsin(np.sqrt(a))
    
    # 地球半径（米）
    R = 6371.0 * 1000
    
    return R * c


def calculate_nearest_neighbor_distances(main_table, sub_table, x_col, y_col, match_columns=None):
    """
    计算副表中每一行到主表的最近邻距离（优化版本）
    
    Parameters:
    -----------
    main_table : DataFrame
        主表，包含参考点的坐标
    sub_table : DataFrame  
        副表，需要计算最近邻距离的点
    x_col : str
        x坐标列名（经度）
    y_col : str
        y坐标列名（纬度）
    match_columns : list, optional
        用于分组匹配的列名列表，如['province', 'city']
        
    Returns:
    --------
    list
        副表中每个点的最近邻距离列表
    """
    nearest_distances = []
    
    if match_columns is None:
        # 没有分组，直接计算所有点
        sub_coords = sub_table[[x_col, y_col]].values
        main_coords = main_table[[x_col, y_col]].values
        
        # 向量化计算距离矩阵
        distance_matrix = haversine_vectorized(sub_coords, main_coords)
        
        # 找每行的最小值
        nearest_distances = np.min(distance_matrix, axis=1).tolist()
        
    else:
        # 有分组匹配
        # 创建分组键
        if isinstance(match_columns, str):
            match_columns = [match_columns]
            
        # 为每个表创建分组键
        main_table = main_table.copy()
        sub_table = sub_table.copy()
        
        main_table['_group_key'] = main_table[match_columns].astype(str).agg('_'.join, axis=1)
        sub_table['_group_key'] = sub_table[match_columns].astype(str).agg('_'.join, axis=1)
        
        # 按原始索引排序以保持顺序
        sub_table_with_idx = sub_table.reset_index()
        sub_table_with_idx['_original_idx'] = sub_table_with_idx.index
        
        # 初始化结果数组
        result_distances = np.full(len(sub_table), np.inf)
        
        # 按组处理
        for group_key in sub_table['_group_key'].unique():
            # 获取当前组的数据
            sub_group = sub_table_with_idx[sub_table_with_idx['_group_key'] == group_key]
            main_group = main_table[main_table['_group_key'] == group_key]
            
            if len(main_group) == 0:
                # 如果主表中没有匹配的组，距离设为无穷大
                continue
                
            # 提取坐标
            sub_coords = sub_group[[x_col, y_col]].values
            main_coords = main_group[[x_col, y_col]].values
            
            # 向量化计算当前组的距离矩阵
            if len(sub_coords) > 0 and len(main_coords) > 0:
                distance_matrix = haversine_vectorized(sub_coords, main_coords)
                group_min_distances = np.min(distance_matrix, axis=1)
                
                # 将结果放回正确的位置
                original_indices = sub_group['_original_idx'].values
                result_distances[original_indices] = group_min_distances
        
        nearest_distances = result_distances.tolist()
    
    return nearest_distances


def analyze_coordinate_differences_and_correct_bias(main_table, sub_table, x_col, y_col, plot=True, ratio = 0.5):
    """
    Analyze coordinate differences between nearest neighbors and correct systematic bias
    
    Parameters:
    -----------
    main_table : DataFrame
        Reference table with coordinates
    sub_table : DataFrame  
        Table to find nearest neighbors for
    x_col : str
        Column name for x coordinates (longitude)
    y_col : str
        Column name for y coordinates (latitude)
    plot : bool, optional
        Whether to plot the distributions
        
    Returns:
    --------
    dict
        Dictionary containing:
        - 'original_distances': Original nearest neighbor distances
        - 'coordinate_differences': Coordinate differences (dx, dy)
        - 'systematic_bias': Estimated systematic bias (mean_dx, mean_dy)
        - 'corrected_distances': Distances after bias correction
        - 'bias_corrected_coords': Bias-corrected coordinates
    """
    
    # Get coordinates
    sub_coords = sub_table[[x_col, y_col]].values
    main_coords = main_table[[x_col, y_col]].values
    
    # Calculate distance matrix and find nearest neighbors
    distance_matrix = haversine_vectorized(sub_coords, main_coords)
    nearest_indices = np.argmin(distance_matrix, axis=1)
    original_distances = np.min(distance_matrix, axis=1)
    
    # Calculate coordinate differences
    nearest_main_coords = main_coords[nearest_indices]
    coord_differences_deg = nearest_main_coords - sub_coords  # (dx, dy) in degrees
    
    # Convert coordinate differences from degrees to meters
    # Approximate conversion: 1 degree lat ≈ 111.32 km, 1 degree lon ≈ 111.32 * cos(lat) km
    avg_lat = np.radians(np.mean([np.mean(sub_coords[:, 1]), np.mean(main_coords[:, 1])]))
    lat_to_meters = 111320  # meters per degree latitude
    lon_to_meters = 111320 * np.cos(avg_lat)  # meters per degree longitude at average latitude
    
    coord_differences = coord_differences_deg.copy()
    coord_differences[:, 0] *= lon_to_meters  # dx in meters
    coord_differences[:, 1] *= lat_to_meters  # dy in meters
    
    # Use local density method to find the densest region
    from sklearn.neighbors import NearestNeighbors
    
    # Calculate local density for each point using k nearest neighbors
    k_neighbors = min(int(0.1*len(coord_differences)), len(coord_differences) - 1)  # Use up to 10% of points, but not more than available points
    if k_neighbors < 1:
        k_neighbors = 1
    
    # Fit nearest neighbors model
    nbrs = NearestNeighbors(n_neighbors=k_neighbors + 1, algorithm='auto').fit(coord_differences)
    distances, indices = nbrs.kneighbors(coord_differences)
    
    # Calculate average distance to k nearest neighbors for each point (excluding itself)
    avg_distances = np.mean(distances[:, 1:], axis=1)  # Exclude the first column (distance to itself = 0)
    
    # Select the densest half of points (those with smallest average distances to neighbors)
    n_half = int(len(coord_differences) * ratio)
    densest_indices = np.argsort(avg_distances)[:n_half]
    dense_differences = coord_differences[densest_indices]
    
    # Calculate systematic bias as mean of densest points
    systematic_bias = np.mean(dense_differences, axis=0)
    
    # Convert systematic bias back to degrees for coordinate correction
    systematic_bias_deg = systematic_bias.copy()
    systematic_bias_deg[0] /= lon_to_meters  # convert dx back to degrees
    systematic_bias_deg[1] /= lat_to_meters  # convert dy back to degrees
    
    # Correct coordinates by adding systematic bias (in degrees)
    corrected_sub_coords = sub_coords + systematic_bias_deg
    
    # Recalculate distances with corrected coordinates
    corrected_distance_matrix = haversine_vectorized(corrected_sub_coords, main_coords)
    corrected_distances = np.min(corrected_distance_matrix, axis=1)
    
    if plot:
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # Plot coordinate differences (original scale)
        axes[0, 0].scatter(coord_differences[:, 0], coord_differences[:, 1], 
                          alpha=0.7, s=8, color='lightblue', label='All points')
        axes[0, 0].scatter(dense_differences[:, 0], dense_differences[:, 1], 
                          color='red', alpha=0.9, s=8, label=f'Dense region ({len(dense_differences)}/{len(coord_differences)} points)')
        axes[0, 0].scatter(systematic_bias[0], systematic_bias[1], 
                          color='black', s=15, marker='x', linewidth=3, 
                          label=f'Systematic bias ({systematic_bias[0]:.1f}m, {systematic_bias[1]:.1f}m)')
        axes[0, 0].set_xlabel('Longitude difference (m)')
        axes[0, 0].set_ylabel('Latitude difference (m)')
        axes[0, 0].set_title('Coordinate Differences Distribution')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Plot original distance distribution (log scale)
        log_original = np.log(original_distances + 1)  # Add small value to avoid log(0)
        axes[0, 1].hist(log_original, bins=50, alpha=0.7, color='blue', edgecolor='black')
        axes[0, 1].set_xlabel('Log Distance (m)')
        axes[0, 1].set_ylabel('Frequency')
        axes[0, 1].set_title('Original Distance Distribution (Log Scale)')
        axes[0, 1].grid(True, alpha=0.3)
        
        # Plot corrected distance distribution (log scale)
        log_corrected = np.log(corrected_distances + 1)  # Add small value to avoid log(0)
        axes[1, 0].hist(log_corrected, bins=50, alpha=0.7, color='green', edgecolor='black')
        axes[1, 0].set_xlabel('Log Distance (m)')
        axes[1, 0].set_ylabel('Frequency')
        axes[1, 0].set_title('Bias-Corrected Distance Distribution (Log Scale)')
        axes[1, 0].grid(True, alpha=0.3)
        
        # Plot comparison (log scale)
        axes[1, 1].hist(log_original, bins=50, alpha=0.5, color='blue', 
                       label='Original', edgecolor='black')
        axes[1, 1].hist(log_corrected, bins=50, alpha=0.5, color='green', 
                       label='Corrected', edgecolor='black')
        axes[1, 1].set_xlabel('Log Distance (m)')
        axes[1, 1].set_ylabel('Frequency')
        axes[1, 1].set_title('Distance Distribution Comparison (Log Scale)')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
    
    return {
        'original_distances': original_distances.tolist(),
        'coordinate_differences': coord_differences.tolist(),
        'systematic_bias': systematic_bias.tolist(),
        'corrected_distances': corrected_distances.tolist(),
        'bias_corrected_coords': corrected_sub_coords.tolist()
    }


