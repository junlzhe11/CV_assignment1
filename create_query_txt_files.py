import os

def create_empty_txt_files():
    """
    在指定路径创建空的0.txt到49.txt文件
    """
    # 目标路径
    txt_dir = r'D:\Dev\python_projects\Assignment1_export\CV_assignment1_guideline_CNN\assignment1_torch_code\data\query_txt'
    
    # 创建目录（如果不存在）
    os.makedirs(txt_dir, exist_ok=True)
    
    print(f"Creating empty txt files in: {txt_dir}")
    
    # 创建0.txt到49.txt
    for i in range(50):
        txt_path = os.path.join(txt_dir, f'{i}.txt')
        
        # 创建空的txt文件
        open(txt_path, 'w').close()
        
        print(f"Created: {txt_path} (empty)")

    print(f"✅ Successfully created 50 empty txt files (0.txt to 49.txt)")

def create_empty_txt_files_quick():
    """
    快速创建50个空的txt文件
    """
    txt_dir = r'D:\Dev\python_projects\Assignment1_export\CV_assignment1_guideline_CNN\assignment1_torch_code\data\query_txt'
    
    os.makedirs(txt_dir, exist_ok=True)
    
    print("Quick creating 50 empty txt files...")
    
    for i in range(50):
        txt_path = os.path.join(txt_dir, f'{i}.txt')
        
        # 创建空的txt文件
        open(txt_path, 'w').close()
    
    print(f"✅ Successfully created 50 empty txt files in: {txt_dir}")

if __name__ == '__main__':
    # 选择你想要使用的方法：
    
    # 方法1: 快速创建（推荐）
    create_empty_txt_files_quick()
    
    # 方法2: 基础版本
    # create_empty_txt_files()