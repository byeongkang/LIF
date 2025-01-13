def apply_custom_transformations(image1, image2_gray, mask_conditions, fill_values):
    image1_gray = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
    transformed = image1_gray.copy()

    for condition, color in mask_conditions.items():
        mask = np.isin(image2_gray, condition)
        mask &= ((image1_gray == 0) | (image1_gray == 255))  
        transformed = np.where(mask, color, transformed)

    for value in fill_values:
        mask = image2_gray == value
        contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        temp_mask = np.zeros_like(image2_gray)
        cv2.drawContours(temp_mask, contours, -1, 255, thickness=cv2.FILLED)
        inside_mask = (temp_mask == 255) & ~((image1_gray == 0) | (image1_gray == 255))
        transformed = np.where(inside_mask, 255, transformed)

    mask = image2_gray == 1
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        contour = max(contours, key=cv2.contourArea) 
        # 경계 기준 왼쪽과 오른쪽 변경
        for point in contour:
            x, y = point[0][0], point[0][1]
            if (image1_gray[x, y] == 0 or image1_gray[x, y] == 255):
                if x < image1_gray.shape[1] / 2:  
                    transformed[y, x] = 0 
                else:  # 오른쪽
                    transformed[y, x] = 255  

    return cv2.cvtColor(transformed.astype(np.uint8), cv2.COLOR_GRAY2BGR)


mask_conditions = {
    (2, 3): 0,
    12: 0,
    13: 255,
    10: 255
}
fill_values = [4, 5]

input_folder_path = './polarity_test/input/'
segmentation_folder_path = './polarity_test/input_seg/'

result_folder_path = './polarity_test/result_folder_boundary/'

if not os.path.exists(result_folder_path):
    os.makedirs(result_folder_path)

image_files = [f for f in os.listdir(input_folder_path) if f.endswith('.jpg')]

for image_file in image_files:

    base_filename = os.path.splitext(image_file)[0]

    image1_path = os.path.join(input_folder_path, image_file)
    image2_path = os.path.join(segmentation_folder_path, base_filename + '.png')

    if os.path.exists(image1_path) and os.path.exists(image2_path):
        image1 = cv2.imread(image1_path)
        image2 = cv2.imread(image2_path)
        image2_gray = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)

        image1_final = apply_custom_transformations(image1, image2_gray, mask_conditions, fill_values)

        result_path = os.path.join(result_folder_path, base_filename + '.jpg')  # 결과도 JPG 형식으로 저장
        cv2.imwrite(result_path, image1_final)
        print(f"Processed and saved: {result_path}")
