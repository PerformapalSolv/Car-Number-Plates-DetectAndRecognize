# import required libraries
import cv2
import numpy as np


def detect_plate(img_path: str):
    # Read input image
    img = cv2.imread(img_path)

    # convert input image to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # read haarcascade for number plate detection
    cascade = cv2.CascadeClassifier('./model/haarcascade_russian_plate_number.xml')

    # Detect license number plates
    plates = cascade.detectMultiScale(gray, 1.2, 5)
    # print('Number of detected license plates:', len(plates))

    color_plates_list = []
    # loop over all plates
    for (x, y, w, h) in plates:
        # draw bounding rectangle around the license number plate
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
        gray_plates = gray[y:y + h, x:x + w]
        color_plates = img[y:y + h, x:x + w]
        color_plates_list.append(color_plates)
    return color_plates_list[0]


def main():
    img_path = './plates/car3.jpg'
    color_plates = detect_plate(img_path)
    # save number plate detected
    cv2.imwrite('data/output/output.jpg', color_plates)
    cv2.imshow('Number Plate', color_plates)
    cv2.imshow('origin Plate', cv2.imread(img_path))
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
