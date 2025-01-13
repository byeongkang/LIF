import os
import numpy as np
from PIL import Image


input_folder = './/'
seg_folder = './polarity_test/input_seg/'
output_folder = './LIF_rm_back/'


for img_filename in os.listdir(input_folder):
    if img_filename.endswith('.jpg'):  
        
        img_path = os.path.join(input_folder, img_filename)
        img = Image.open(img_path)
        img_array = np.array(img)

       
        seg_filename = img_filename.replace('.jpg', '.png')  # 파일명 변환
        seg_path = os.path.join(seg_folder, seg_filename)
        seg_img = Image.open(seg_path)
        seg_array = np.array(seg_img)

        
        img_array[seg_array == 0] = 200

        
        img_modified = Image.fromarray(img_array)
        save_path = os.path.join(output_folder, img_filename) 
        os.makedirs(os.path.dirname(save_path), exist_ok=True)  
        img_modified.save(save_path)
