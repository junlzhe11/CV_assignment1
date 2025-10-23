# Retrieve the most similar images by measuring the similarity between features.
import numpy as np
import os
import cv2
from sklearn.metrics.pairwise import cosine_similarity
from matplotlib import pyplot as plt
from tqdm import tqdm

# Measure the similarity scores between query feature and gallery features.
# You could also use other metrics to measure the similarity scores between features.
def similarity(query_feat, gallery_feat):
    sim = cosine_similarity(query_feat, gallery_feat)
    sim = np.squeeze(sim)
    return sim

def get_ranklist(query_path, gallery_dir, query_index=1):
    rank_list_file='./data/rank_list.txt'
    query_feat = np.load(query_path)
    dict = {}
    for gallery_file in os.listdir(gallery_dir):
        gallery_feat = np.load(os.path.join(gallery_dir, gallery_file))
        gallery_index = gallery_file.split('.')[0] + '.jpg'
        sim = similarity(query_feat, gallery_feat)
        dict[gallery_index] = sim
    
    # Sort by similarity in DESCENDING order (highest similarity first)
    sorted_dict = sorted(dict.items(), key=lambda item: item[1], reverse=True)
    
    # Write complete rank list for ALL gallery images to file
    write_method = 'w' if query_index == 0 else 'a'  # Use first query to create file, others append
    
    with open(rank_list_file, write_method) as f:
        if query_index != 0:  # Add separator for all queries except the first one
            f.write("\n" + "=" * 60 + "\n\n")
        
        f.write(f"Query {query_index} - Complete Rank List\n")
        f.write("(Sorted by Similarity - Highest to Lowest)\n")
        f.write("-" * 50 + "\n")
        f.write("Rank\tImage Name\t\tSimilarity Score\n")
        f.write("-" * 50 + "\n")
        
        for rank, (image, score) in enumerate(sorted_dict, 1):
            f.write(f"{rank}\t{image}\t\t{score}\n")
    
    print(f"Rank list for query {query_index} saved to {rank_list_file}")
    print(f"Total images ranked: {len(sorted_dict)}")
    
    return sorted_dict

def retrival_index(query_path, gallery_dir, query_index=1):
    # Get the complete sorted dictionary from get_ranklist
    sorted_dict = get_ranklist(query_path, gallery_dir, query_index)
    
    # Get the best ten retrieved images (first 10 in descending order)
    best_ten = sorted_dict[:10]
    
    return best_ten

def visualization(retrieved, query):
    plt.figure(figsize=(15, 9))
    plt.subplot(3, 5, 1)
    plt.title('query')
    query_img = cv2.imread(query)
    img_rgb_rgb = query_img[:,:,::-1]
    plt.imshow(img_rgb_rgb)
    
    for i in range(10):
        img_path = './data/gallery/' + retrieved[i][0]
        img = cv2.imread(img_path)
        img_rgb = img[:,:,::-1]
        plt.subplot(3, 5, i + 6)
        plt.title(retrieved[i][1])
        plt.imshow(img_rgb)
    
    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    # Process 50 query images (0.jpg to 49.jpg)
    for i in range(50):
        # Build file paths
        query_feat_dir = './data/query_feat/'
        query_dir = './data/query/'
        gallery_dir = './data/gallery_feature/'

        query_feat_path = os.path.join(query_feat_dir, f'{i}_feats.npy')
        query_path = os.path.join(query_dir, f'{i}.jpg')
        
        # Check if files exist
        if not os.path.exists(query_feat_path):
            print(f"Query feature file not found: {query_feat_path}")
            continue
        
        # Option 1: Call get_ranklist directly to get all ranked results
        print(f"Processing query {i}...")
        all_ranked_results = get_ranklist(query_feat_path, gallery_dir, i)
        
        # Option 2: Get top 10 for visualization
        best_ten = all_ranked_results[:10]
        
        # Visualize the retrieval results
        # visualization(best_ten, query_path)