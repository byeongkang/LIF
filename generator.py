import cv2
import numpy as np
import torch
import snntorch as snn
import os
import matplotlib.pyplot as plt
import time


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using {device} for computation.")
def load_and_preprocess_image_opencv(image_path):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    image_tensor = torch.tensor(image, dtype=torch.float32) / 255.0
    return image_tensor


be = 0.5
th = 0.65
processed_images_count = 0
lif = snn.Leaky(beta=be, threshold=th).to(device)


kDir = 'input/images/'


output_dir = 'output/th0.65_be0.5_p8'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)


for image_name in os.listdir(kDir):
    if image_name.endswith('.jpg') or image_name.endswith('.png'):
        output_path = os.path.join(output_dir, image_name)
        if os.path.exists(output_path):
            processed_images_count += 1
            print(f"Skipping already processed file: {image_name}")
            continue  

        image_path = os.path.join(kDir, image_name)
        image_tensor = load_and_preprocess_image_opencv(image_path).to(device)

        
        mem = torch.zeros(1, device=device)
        spk_rec = torch.zeros_like(image_tensor)

        
        for i in range(image_tensor.shape[0]):
            for j in range(image_tensor.shape[1]):
                spk, mem = lif(image_tensor[i, j], mem)
                spk_rec[i, j] = spk

        
        spk_rec_img = spk_rec.cpu().numpy()  
        no_spike_color = np.random.choice([0, 255], spk_rec_img.shape, p=[0.8, 0.2])  
        spk_rec_img = np.where(spk_rec_img == 1, 200, no_spike_color).astype(np.uint8)  
        cv2.imwrite(os.path.join(output_dir, image_name), spk_rec_img)
        processed_images_count += 1
        print(f"Processed {processed_images_count}/{len(os.listdir(kDir))}: {image_name}")


print(f"Total {processed_images_count} images have been processed.")
