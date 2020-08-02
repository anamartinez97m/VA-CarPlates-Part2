import cv2
import os


def detect_and_display(frame):
    # Create the Cascade Classifier Object
    car_cascade = cv2.CascadeClassifier()

    # Load the cascades
    if not car_cascade.load("haar_opencv_4.1-4.2/coches.xml"):
        print("Error loading car cascade")
        exit(0)

    img_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    img_gray = cv2.equalizeHist(img_gray)

    cars = car_cascade.detectMultiScale(img_gray)
    for (x, y, z, w) in cars:
        frame = cv2.rectangle(frame, (x, y), (x+z, y+w), (0, 0, 255), 2)

    # cv2.imshow("Capture", frame)


def main():
    directory = './test'
    for filename in os.listdir(directory):
        if filename.endswith(".jpg"):
            img_route = os.path.join(directory, filename)
            img = cv2.imread(img_route)
            detect_and_display(img)
            cv2.waitKey()


if __name__ == "__main__":
    main()
