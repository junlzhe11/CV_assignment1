#HOW TO INSTALL ANNACONDA: https://www.youtube.com/watch?v=YJC6ldI3hWk
#WHAT IS IMAGENET DATABASE: https://www.youtube.com/watch?v=gogV2wKKF_8
import torch
import cv2
import os
import numpy as np
import torchvision.models as models
import torchvision.transforms as transforms
from tqdm import tqdm
from ultralytics import YOLO

#crop the instance region. For the images containing two instances, you need to crop both of them.
def query_crops(query_path, txt_path, save_path):
    query_img = cv2.imread(query_path)
    query_img = query_img[:,:,::-1] #bgr2rgb
    txt = np.loadtxt(txt_path)     #load the coordinates of the bounding box
    
    crops = []  # Array to store all crops
    # Handle single bounding box (1D array)
    if txt.ndim == 1:
        if len(txt) == 4:
            # Single bounding box
            x, y, w, h = txt
            crop = query_img[int(y):int(y + h), int(x):int(x + w), :]
            cv2.imwrite(save_path, crop[:,:,::-1])
            crops.append(crop)
        else:
            # Multiple boxes in one row (shouldn't happen with 1-2 boxes)
            print(f"Unexpected number of coordinates: {len(txt)}")
    
    # Handle multiple bounding boxes (2D array)
    elif txt.ndim == 2:
        for i, bbox in enumerate(txt):
            x, y, w, h = bbox[:4]
            crop = query_img[int(y):int(y + h), int(x):int(x + w), :]
            
            # Save each crop with appropriate filename
            if len(txt) == 1:
                # Single box - use original save_path
                cv2.imwrite(save_path, crop[:,:,::-1])
            else:
                # Multiple boxes - add index to filename
                base_name = os.path.splitext(save_path)[0]
                ext = os.path.splitext(save_path)[1]
                individual_save_path = f"{base_name}_box{i}{ext}"
                cv2.imwrite(individual_save_path, crop[:,:,::-1])
            
            crops.append(crop)
    
    print(f"Cropped {len(crops)} instances from {query_path}")
    return crops

def vgg_11_extraction(img, featsave_path):
    # resnet_transform = transforms.Compose([
    #     transforms.ToTensor(),
    #     transforms.Normalize(mean=[0.485, 0.456, 0.406],
    #                          std=[0.229, 0.224, 0.225])])
    # img_transform = resnet_transform(img) #normalize the input image and transform it to tensor.
    # img_transform = torch.unsqueeze(img_transform, 0) #Set batchsize as 1. You can enlarge the batchsize to accelerate.

    # initialize the weights pretrained on the ImageNet dataset, you can also use other backbones (e.g. ResNet, XceptionNet, AlexNet, ...)
    # and extract features from more than one layer.
    yolo = YOLO("./data/model/yolo11n.pt").eval()
    net= yolo.model
    # vgg11 = models.vgg11(pretrained=True)
    # vgg11_feat_extractor = vgg11.features #define the feature extractor
    # vgg11_feat_extractor.eval()  #set the mode as evaluation
    features = {}
    def hook_fn(module, input, output):
        features['feat'] = output
    target_layer = net.model[22]
    target_layer.register_forward_hook(hook_fn)
    img_tensor = torch.from_numpy(img).permute(2,0,1).float().unsqueeze(0) / 255.0
    with torch.no_grad():
        _ = net(img_tensor)
    feat_map = features['feat']  # tensor shape: [1, C, H, W]
    # print("Feature map shape:", feat_map.shape)
    # feats = vgg11(img_transform) # extract feature
    feats = torch.mean(feat_map, dim=[2,3])
    feats_np = feats.cpu().detach().numpy() # convert tensor to numpy
    np.save(featsave_path, feats_np) # save the feature

# Note that I feed the whole image into the pretrained vgg11 model to extract the feature, which will lead to a poor retrieval performance.
# To extract more fine-grained features, you could preprocess the gallery images by cropping them using windows with different sizes and shapes.
# Hint: opencv provides some off-the-shelf tools for image segmentation.
def feat_extractor_gallery(gallery_dir, feat_savedir):
    for img_file in tqdm(os.listdir(gallery_dir)):
        img = cv2.imread(os.path.join(gallery_dir, img_file))
        img = img[:,:,::-1] #bgr2rgb
        img_resize = cv2.resize(img, (640, 640), interpolation=cv2.INTER_CUBIC) # resize the image
        featsave_path = os.path.join(feat_savedir, img_file.split('.')[0]+'.npy')
        vgg_11_extraction(img_resize, featsave_path)

def box_query(query_path, txt_path, box_path):
    # Read image
    img = cv2.imread(query_path)
    if img is None:
        print(f"Error: Could not load image {query_path}")
        return
    
    try:
        # Read bounding box coordinates
        bboxes = np.loadtxt(txt_path)
        
        # Handle all possible cases
        if bboxes.size == 0:
            # No bounding boxes
            print("No bounding boxes found")
            cv2.imwrite(box_path, img)
            return
            
        elif bboxes.ndim == 0:
            # Single scalar (unlikely for bbox data)
            print("Unexpected scalar data")
            return
            
        elif bboxes.ndim == 1:
            # 1D array - could be single bbox or multiple in one line
            if len(bboxes) == 4:
                # Single bounding box with 4 coordinates
                bboxes = [bboxes]
            else:
                # Multiple boxes concatenated in one row
                # Ensure total length is divisible by 4
                if len(bboxes) % 4 == 0:
                    bboxes = bboxes.reshape(-1, 4)
                else:
                    print(f"Warning: Invalid number of coordinates ({len(bboxes)})")
                    return
        
        print(f"Found {len(bboxes)} bounding boxes")
        
        # Define colors for different boxes
        colors = [
            (0, 0, 255),    # RED
            (255, 0, 0),    # BLUE  
            (0, 255, 0),    # GREEN
            (255, 255, 0),  # CYAN
            (255, 0, 255),  # MAGENTA
            (0, 255, 255),  # YELLOW
            (128, 0, 128),  # PURPLE
            (255, 165, 0)   # ORANGE
        ]
        
        thickness = 3
        
        # Draw each bounding box
        for i, bbox in enumerate(bboxes):
            # Format: x y width height
            x, y, width, height = map(int, bbox[:4])
            
            # Calculate end point (x2, y2)
            x2 = x + width
            y2 = y + height
            
            # Get color
            color = colors[i % len(colors)]
            
            # Draw rectangle
            cv2.rectangle(img, (x, y), (x2, y2), color, thickness)
            
            # Add box number label
            label = f"Box {i+1}"
            cv2.putText(img, label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 
                        0.7, color, 2)
            
            print(f"Box {i+1}: ({x}, {y}) width={width}, height={height} -> ({x2}, {y2})")
        
        # Save the image with all bounding boxes
        cv2.imwrite(box_path, img)
        print(f"Image with {len(bboxes)} bounding boxes saved to: {box_path}")
        
    except Exception as e:
        print(f"Error processing {query_path}: {e}")
        import traceback
        traceback.print_exc()
      
# Extract the query feature - 修改为处理50张查询图片
def feat_extractor_query():
    query_dir = './data/query/'  # 查询图片目录
    txt_dir = './data/query_txt/'  # 查询文本目录
    cropped_query_dir = './data/cropped_query/'  # 裁剪后的查询图片目录
    query_feat_dir = './data/query_feat/'  # 查询特征目录
    box_dir = './data/query_box' 
    
    # 创建目录（如果不存在）
    os.makedirs(cropped_query_dir, exist_ok=True)
    os.makedirs(query_feat_dir, exist_ok=True)
    os.makedirs(box_dir, exist_ok=True)
    
    # 处理50张查询图片 (0.jpg 到 49.jpg)
    for queryIndex in tqdm(range(50), desc="Processing query images"):
        # 构建文件路径
        query_path = os.path.join(query_dir, f'{queryIndex}.jpg')
        txt_path = os.path.join(txt_dir, f'{queryIndex}.txt')
        save_path = os.path.join(cropped_query_dir, f'{queryIndex}.jpg')
        box_path = os.path.join(box_dir, f'{queryIndex}_box.jpg')
        # box
        box_query(query_path, txt_path, box_path)
        
        try:
            # 裁剪和特征提取
            crops = query_crops(query_path, txt_path, save_path)  # 修正：query_crops -> query_crop
            
            # 檢查crops是否為空
            if crops is None or len(crops) == 0:
                print(f"No crops found for query image {queryIndex}")
                continue
                
            for cropIndex in range(len(crops)):
                crop = crops[cropIndex]
                crop_resize = cv2.resize(crop, (640, 640), interpolation=cv2.INTER_CUBIC)
                featsave_path = os.path.join(query_feat_dir, f'query{queryIndex}_feat{cropIndex}.npy')  # 修正：image{i} -> {i}
                vgg_11_extraction(crop_resize, featsave_path)
                print(f"Successfully processed image {queryIndex} crop {cropIndex} from query image {queryIndex}")  # 修正：更明確的日誌    
        except Exception as e:
            print(f"Error processing query image {queryIndex}: {e}")

def main():
    feat_extractor_query()
    gallery_dir = './data/gallery/'
    feat_savedir = './data/gallery_feature/'
    # feat_extractor_gallery(gallery_dir, feat_savedir)

if __name__=='__main__':
    main()