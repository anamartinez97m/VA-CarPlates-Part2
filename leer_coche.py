import cv2
import numpy as np
import os
import sys
import getopt
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.neighbors import KNeighborsClassifier
from deteccion_haar import detect_and_display as detect_and_display_car


def filter_by_width_and_height(width, height):
    return width < height


def filter_by_area(width, height):
    return 100 < width * height < 650


def filter_by_aspect_ratio(width, height):
    return float(width) / height < 0.9


def is_rectangle_inside_haar_rectangle(centers_x, centers_y, x_haar, y_haar, z_haar, w_haar):
    valid_positions_haar = []
    for i in range(len(centers_x)):
        if x_haar <= centers_x[i] <= x_haar + z_haar and y_haar <= centers_y[i] <= y_haar + w_haar:
            valid_positions_haar.append(i)
    return valid_positions_haar


def is_valid_rectangle(width, height):
    return filter_by_width_and_height(width, height) \
           and filter_by_area(width, height) \
           and filter_by_aspect_ratio(width, height)


def get_center_of_rectangle(x, y, width, height):
    return x + width // 2, y + height // 2


def detect_and_display(frame):
    # Create the Cascade Classifier Object
    plate_cascade = cv2.CascadeClassifier()

    # Load the cascades
    if not plate_cascade.load("haar_opencv_4.1-4.2/matriculas.xml"):
        print("Error loading plate cascade")
        exit(0)

    img_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    img_gray = cv2.equalizeHist(img_gray)

    plates = plate_cascade.detectMultiScale(img_gray)

    if len(plates) > 0:
        x_haar, y_haar, z_haar, w_haar = plates[0]
        cv2.rectangle(frame, (x_haar, y_haar), (x_haar + z_haar, y_haar + w_haar), (255, 255, 0), 3)

        return x_haar, y_haar, z_haar, w_haar
    else:
        return 0, 0, 0, 0


def train():
    matrix_characteristics_training_vectors = []

    directory_training = './training_ocr'
    for filenameTraining in os.listdir(directory_training):
        img_route_t = os.path.join(directory_training, filenameTraining)
        img_t = cv2.imread(img_route_t, 0)

        img_t = cv2.resize(img_t, (10, 10), interpolation=cv2.INTER_LINEAR)

        img_t = cv2.adaptiveThreshold(img_t, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 21, 15)

        np_img_t = np.array(img_t)
        training_vector = np_img_t.flatten()
        matrix_characteristics_training_vectors.append(training_vector)

    labels = []
    class_number = 0
    max_class_number = 37
    for i in range(max_class_number):
        for j in range(250):
            labels.append(class_number)
        class_number += 1

    matrix_characteristics_training_vectors = np.array(matrix_characteristics_training_vectors)

    lda = LinearDiscriminantAnalysis()
    lda.fit(matrix_characteristics_training_vectors, labels)
    matrix_reduced_characteristics = lda.transform(matrix_characteristics_training_vectors)

    matrix_reduced_characteristics = np.array(matrix_reduced_characteristics, dtype=np.float32)

    return lda, matrix_reduced_characteristics, labels


def predict(lda, rois, x, y):
    matrix_characteristics_test_vectors = []

    for i in range(len(rois)):
        roi_gray = cv2.cvtColor(rois[i], cv2.COLOR_BGR2GRAY)

        roi_gray = cv2.resize(roi_gray, (10, 10), interpolation=cv2.INTER_LINEAR)
        roi_gray = cv2.adaptiveThreshold(roi_gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 21, 15)

        np_img_t = np.array(roi_gray)
        test_vector = np_img_t.flatten()
        matrix_characteristics_test_vectors.append(test_vector)

    if np.array(matrix_characteristics_test_vectors).shape[0] == 0:
        return []
    test = lda.transform(matrix_characteristics_test_vectors)
    knn = KNeighborsClassifier(n_neighbors=6)
    knn.fit(x, y)
    predictions = knn.predict(test)

    return predictions


def get_label(num):
    characters = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'A', 'B', 'C', 'D', 'E', '', 'F', 'G', 'H', 'I',
                  'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']

    return characters[num]


def write_character_on_top(img, character, x, y):
    font = cv2.FONT_HERSHEY_SIMPLEX
    bottom_left = (x, y)
    font_scale = 0.5
    font_color = (125, 0, 0)
    line_type = 2

    cv2.putText(img, character, bottom_left, font, font_scale, font_color, line_type)


def create_result_file(filename):
    f = open(filename + '.txt', "w")
    f.close()


def write_plate_file(filename, line):
    f = open(filename + '.txt', "a+")
    f.write(line + "\n")
    f.close()


def sort_by_x_position(rois_plate, rois_x, rois_y):
    for i in range(len(rois_x)):
        swap = i + np.argmin(rois_x[i:])
        (rois_x[i], rois_x[swap]) = (rois_x[swap], rois_x[i])
        (rois_plate[i], rois_plate[swap]) = (rois_plate[swap], rois_plate[i])
        (rois_y[i], rois_y[swap]) = (rois_y[swap], rois_y[i])
    return rois_plate, rois_x, rois_y


def detect_number_letter_plates(absolute_path, visualization):
    lda_trained, x_lda, y_lda = train()

    directory = absolute_path
    filename_txt = os.path.basename(os.path.normpath(absolute_path))
    create_result_file(filename_txt)

    for filename in os.listdir(directory):
        if filename.endswith(".jpg"):
            img_route = os.path.join(directory, filename)
            img = cv2.imread(img_route, 0)
            img_color = cv2.imread(img_route)
            x_haar, y_haar, z_haar, w_haar = detect_and_display(img_color)

            x_center_plate, y_center_plate = get_center_of_rectangle(x_haar, y_haar, z_haar, w_haar)
            half_plate_length = abs(z_haar) / 2
            cv2.circle(img_color, (x_center_plate, y_center_plate), 10, (255, 0, 255), 2)

            line_file = filename + ' ' + str(x_center_plate) + ' ' + str(y_center_plate) + ' '

            thresh = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 21, 15)
            contours, hierarchy = cv2.findContours(thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)

            centers_x = []
            centers_y = []
            rectangles = []
            # computes the bounding box for the contour, and draws it on the frame,
            for c in contours:
                # get the bounding rect
                x, y, w, h = cv2.boundingRect(c)
                if is_valid_rectangle(w, h):
                    center_x, center_y = get_center_of_rectangle(x, y, w, h)
                    centers_x.append(center_x)
                    centers_y.append(center_y)
                    rectangles.append((x, y, w, h))

            inside_points_indexes = \
                is_rectangle_inside_haar_rectangle(centers_x, centers_y, x_haar, y_haar, z_haar, w_haar)

            rois_plate = []
            rois_x = []
            rois_y = []
            # Margin used to avoid having characters positioned directly on the contours
            margin = 3
            for i in range(len(inside_points_indexes)):
                x, y, w, h = rectangles[inside_points_indexes[i]]
                roi_character = img_color[y - margin:y + h + margin, x - margin:x + w + margin]
                rois_plate.append(roi_character)
                rois_x.append(x - margin)
                rois_y.append(y - margin)
                cv2.rectangle(img_color, (x, y), (x + w, y + h), (0, 255, 0), 1)

            rois_plate, rois_x, rois_y = sort_by_x_position(rois_plate, rois_x, rois_y)

            predicted = predict(lda_trained, rois_plate, x_lda, y_lda)

            for i in range(len(predicted)):
                line_file += get_label(predicted[i])
                write_character_on_top(img_color, get_label(predicted[i]), rois_x[i], rois_y[i])

            line_file += ' ' + str(half_plate_length)

            write_plate_file(filename_txt, line_file)
            if visualization == 'True':
                detect_and_display_car(img_color)
                cv2.imshow("Plates", img_color)
                cv2.waitKey()


def main():

    try:
        _, args = getopt.getopt(sys.argv[1:], '', [])
    except getopt.GetoptError as error:
        print(error)
        sys.exit(2)

    if len(args) == 1:
        absolute_path = args[0]
        visualization = False
    elif len(args) == 2:
        absolute_path = args[0]
        visualization = args[1]
    else:
        print("Error en el numero de argumentos")
        print(
            "Ejemplo de ejecución: python3 leer_coche.py /absolute/path True ")
        sys.exit(2)

    detect_number_letter_plates(absolute_path, visualization)


if __name__ == "__main__":
    main()
