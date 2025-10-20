import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist

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

