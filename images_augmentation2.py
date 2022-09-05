import albumentations as A
import cv2


image = cv2.imread("images/2.png")
#image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
transform = A.Compose([
    A.RandomCrop(width=256, height=256),
    A.HorizontalFlip(p=0.5),
    A.RandomBrightnessContrast(p=0.2),
])
transformed = transform(image=image)
transformed_image = transformed["image"]
cv2.imshow("image_augmented",transformed_image)
cv2.waitKey(0)
 #apply three augmentation on the same image 
transform = A.Compose([
    A.RandomBrightnessContrast(brightness_limit=1, contrast_limit=1, p=1.0),
])
transformed_image_1 = transform(image=image)['image']
transformed_image_2 = transform(image=image)['image']
transformed_image_3 = transform(image=image)['image']

cv2.imshow("image_augmented",transformed_image)
cv2.waitKey(0)