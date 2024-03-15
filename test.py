import cv2


img = cv2.imread('./plates/car.png')
cv2.imwrite('./data/output/output.jpg', img=img)
