import cv2
import os
import numpy as np

def analyze_query_images():
    """
    分析所有查询图像的实际尺寸
    """
    query_dir = './data/query/'
    sizes = []
    
    print("Analyzing query image sizes...")
    for i in range(50):
        query_path = os.path.join(query_dir, f'{i}.jpg')
        if os.path.exists(query_path):
            img = cv2.imread(query_path)
            if img is not None:
                height, width = img.shape[:2]
                sizes.append((width, height))
                print(f"{i}.jpg: {width}x{height}")
            else:
                print(f"{i}.jpg: Could not read")
        else:
            print(f"{i}.jpg: Not found")
    
    if sizes:
        avg_width = np.mean([w for w, h in sizes])
        avg_height = np.mean([h for w, h in sizes])
        print(f"\n📊 Statistics:")
        print(f"Average size: {avg_width:.0f}x{avg_height:.0f}")
        print(f"Min size: {min([w for w,h in sizes])}x{min([h for w,h in sizes])}")
        print(f"Max size: {max([w for w,h in sizes])}x{max([h for w,h in sizes])}")
    
    return sizes

def create_simple_bbox_files(sizes):
    """
    使用统一的65%覆盖比例创建边界框
    """
    query_dir = './data/query/'
    txt_dir = './data/query_txt/'
    
    os.makedirs(txt_dir, exist_ok=True)
    
    print("\nCreating bounding box files with 65% coverage...")
    
    for i in range(50):
        query_path = os.path.join(query_dir, f'{i}.jpg')
        txt_path = os.path.join(txt_dir, f'{i}.txt')
        
        if os.path.exists(query_path):
            img = cv2.imread(query_path)
            if img is not None:
                height, width = img.shape[:2]
                
                # 统一使用65%覆盖比例
                bbox_width = int(width * 0.65)
                bbox_height = int(height * 0.65)
                bbox_x = (width - bbox_width) // 2
                bbox_y = (height - bbox_height) // 2
                
                bbox = f"{bbox_x} {bbox_y} {bbox_width} {bbox_height}"
                
                with open(txt_path, 'w') as f:
                    f.write(bbox)
                
                print(f"{i}.txt: {bbox}")
            else:
                print(f"{i}.jpg: Could not read, skipping...")
        else:
            print(f"{i}.jpg: Not found, skipping...")
    
    print(f"\n✅ Successfully created bounding box files!")

# 运行分析并创建边界框文件
sizes = analyze_query_images()
create_simple_bbox_files(sizes)