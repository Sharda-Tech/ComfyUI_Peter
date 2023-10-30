import os
import numpy as np
import cv2
from google.colab.patches import cv2_imshow

path = '/content/'
for file in os.listdir(path):
    if 'seg' in file and '.npy' in file:

        print(file)
        id = file.split('_')[1].split('.')[0]
        print(int(id))

        img_segm = np.load(os.path.join(path, file))
        img_segm_2 = np.load(os.path.join(path, file))
        img=np.load(os.path.join(path, file))
        
        orig_img = cv2.imread('/content/detectron2-barinov/projects/DensePose/image.png')

        # Invert the mask
        img_segm[(img_segm > 0) & (img_segm < 23)] = 0
        img_segm[img_segm >= 23] = 255
        inverse_mask = cv2.bitwise_not(img_segm)


        

        img[(img > 0) & (img< 3)] = 0
        img[img>=23]=0
        img[(img>=3) & (img<23)] = 255
        img_segm_2[img_segm_2>= 23] = 0
        img_segm_2[(img_segm_2 > 0) & (img_segm_2 < 23)] = 255
        for row in range(img.shape[0]):
                  row_pixels = np.where(img[row] > 0)[0]
                  if len(row_pixels)>0:
                    img[row,row_pixels[0]:row_pixels[-1]]=255

        img2=cv2.add(img,img_segm_2)

        h, w = img.shape[:2]
        i_h = int(1.5 * h / 100)
        i_w = int(5* w / 100)
        img_segm_2 = cv2.resize(img2, (w + i_w, h + i_h))
        dh, dw = int(i_h / 2), int(i_w / 2)

        img_segm_2 = img2[i_h:h - (i_h - dh)+i_h-dh, dw:w - (i_w - dw)]
        img_segm_2 = cv2.resize(img_segm_2, (w, h))
        
        
        
        inverse_mask_2 = cv2.bitwise_not(img)
        
        # Create a white image
        white_image = np.full_like(orig_img, (255, 255, 255))

        # Apply the inverted mask to the white image
        masked_area = cv2.bitwise_and(orig_img, orig_img, mask=inverse_mask_2)

        # Apply the mask to the original image
        masked_image = cv2.bitwise_and(white_image, white_image, mask=img_segm_2)

        # Combine the masked area and masked image
        result = cv2.add(masked_area, masked_image)

        cv2_imshow(result)

        cv2.imwrite('/content/detectron2-barinov/projects/DensePose/image_{}.png'.format(id), result)
