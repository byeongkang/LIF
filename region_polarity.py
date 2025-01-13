import cv2
import numpy as np
import os


def paint_right_of_centroid_white_with_thickness(image, seg_image, values, thickness):
    mask = np.isin(seg_image, values)
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    output_image = image.copy()

    for contour in contours:
        M = cv2.moments(contour)
        if M['m00'] == 0: continue
        cx = int(M['m10'] / M['m00'])

      
        for i in range(len(contour)):
            x, y = contour[i][0][0], contour[i][0][1]
            if x > cx: 
                for thickness_offset in range(1, thickness + 1):  
                    output_image = cv2.circle(output_image, (x + thickness_offset, y), 1, (255, 255, 255), -1)

    return output_image


values = [ 4, 5, 10, 12, 13]
thickness = 3

input_folder_path = './LIF_rm_back_polarity/'
segmentation_folder_path = './polarity_test/input_seg/'
result_folder_path = './polarity_v2_tmp/'

if not os.path.exists(result_folder_path):
    os.makedirs(result_folder_path)

image_files = [f for f in os.listdir(input_folder_path) if f.endswith('.jpg')]

for image_file in image_files:
    base_filename = os.path.splitext(image_file)[0]
    image1_path = os.path.join(input_folder_path, image_file)
    image2_path = os.path.join(segmentation_folder_path, base_filename + '.png')

    if os.path.exists(image1_path) and os.path.exists(image2_path):
        image1 = cv2.imread(image1_path)
        image2_gray = cv2.cvtColor(cv2.imread(image2_path), cv2.COLOR_BGR2GRAY)

        image1_final = paint_right_of_centroid_white_with_thickness(image1, image2_gray, values, thickness)

        result_path = os.path.join(result_folder_path, base_filename + '.jpg')
        cv2.imwrite(result_path, image1_final)
        print(f"Processed and saved: {result_path}")
