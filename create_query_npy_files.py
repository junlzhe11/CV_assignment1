import numpy as np
import os

def create_empty_npy_files():
    """
    在指定路径创建空的0.npy到49.npy文件
    """
    # 目标路径
    feat_dir = r'D:\Dev\python_projects\Assignment1_export\CV_assignment1_guideline_CNN\assignment1_torch_code\data\query_feat'
    
    # 创建目录（如果不存在）
    os.makedirs(feat_dir, exist_ok=True)
    
    print(f"Creating empty npy files in: {feat_dir}")
    
    # 创建0.npy到49.npy
    for i in range(50):
        npy_path = os.path.join(feat_dir, f'{i}_feats.npy')
        
        # 创建空的numpy数组并保存
        empty_array = np.array([])
        np.save(npy_path, empty_array)
        
        print(f"Created: {npy_path} (empty)")

    print(f"✅ Successfully created 50 empty npy files (0_feats.npy to 49_feats.npy)")

def create_empty_npy_files_quick():
    """
    快速创建50个空的npy文件
    """
    feat_dir = r'D:\Dev\python_projects\Assignment1_export\CV_assignment1_guideline_CNN\assignment1_torch_code\data\query_feat'
    
    os.makedirs(feat_dir, exist_ok=True)
    
    print("Quick creating 50 empty npy files...")
    
    for i in range(50):
        npy_path = os.path.join(feat_dir, f'{i}_feats.npy')
        
        # 创建空的numpy数组
        empty_array = np.array([])
        np.save(npy_path, empty_array)
    
    print(f"✅ Successfully created 50 empty npy files in: {feat_dir}")

def create_zero_size_npy_files():
    """
    创建零字节的npy文件
    """
    feat_dir = r'D:\Dev\python_projects\Assignment1_export\CV_assignment1_guideline_CNN\assignment1_torch_code\data\query_feat'
    
    os.makedirs(feat_dir, exist_ok=True)
    
    print("Creating zero-size npy files...")
    
    for i in range(50):
        npy_path = os.path.join(feat_dir, f'{i}_feats.npy')
        
        # 创建空文件
        open(npy_path, 'w').close()
        
        print(f"Created: {npy_path} (0 bytes)")

if __name__ == '__main__':
    # 选择你想要使用的方法：
    
    # 方法1: 创建包含空数组的npy文件（推荐）
    create_empty_npy_files_quick()
    
    # 方法2: 基础版本
    # create_empty_npy_files()
    
    # 方法3: 创建零字节文件
    # create_zero_size_npy_files()