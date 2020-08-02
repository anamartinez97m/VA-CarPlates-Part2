import csv
import math
import matplotlib.pyplot as plt
import numpy as np


def read_gt_csv_file(file, delim=" "):
    """

    """
    plates_info = dict()
    with open(file) as csv_file:
        csv_reader = csv.DictReader(csv_file, delimiter=delim)
        line_count = 0
        for row in csv_reader:
            image_name = row["image"]
            x_center = int(row["x_center"])
            y_center = int(row["y_center"])
            plate = row["plate"]
            half_length = float(row["half_plate_length"])
            if plates_info.get(image_name) is None:
                plates_info[image_name] = [(x_center, y_center, half_length, plate)]
            else:
                print('image=', image_name)
                l = plates_info[image_name]
                l.append([(x_center, y_center, half_length, plate)])
                plates_info[image_name] = l

            line_count += 1
    return plates_info


def read_csv_file(file, delim=" "):
    """

    """
    plates_info = dict()
    with open(file) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=delim)
        line_count = 0
        for row in csv_reader:
            image_name = row[0]
            x_center = int(row[1])
            y_center = int(row[2])
            plate = row[3]
            half_length = float(row[4])
            if plates_info.get(image_name) is None:
                plates_info[image_name] = [(x_center, y_center, half_length, plate)]
            else:
                print('image=', image_name)
                l = plates_info[image_name]
                l.append([(x_center, y_center, half_length, plate)])
                plates_info[image_name] = l

            line_count += 1
    return plates_info


def levenshtein_distance(str1, str2):
    """

    """
    d = dict()
    for i in range(len(str1) + 1):
        d[i] = dict()
        d[i][0] = i
    for i in range(len(str2) + 1):
        d[0][i] = i
    for i in range(1, len(str1) + 1):
        for j in range(1, len(str2) + 1):
            d[i][j] = min(d[i][j - 1] + 1, d[i - 1][j] + 1, d[i - 1][j - 1] + (not str1[i - 1] == str2[j - 1]))

    return d[len(str1)][len(str2)]


def compute_normalized_plate_distance(p_det, p_gt):
    """
        We compare detections with the distance of plate
        centers over the plate size.
    """

    x1, y1, r1 = p_det
    x2, y2, r2 = p_gt

    d = math.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)
    return d / (2 * r2)


def plot_plate_recognition_distances(p_gt, p):
    """

    """
    #iou_all = []
    norm_dist_all = []
    txt_distance_all = []
    for img_name in p_gt:
        p_info_gt = p_gt[img_name]

        if p.get(img_name) is None:  # Not found a plate in this image.
            norm_dist_all.append(-1) # -1 is not found plate
            txt_distance_all.append(-1) # -1 is not found plate
            continue

        p_info = p[img_name]

        # By now we assume only one detection for each image
        x_gt, y_gt, r_gt, plate_gt = p_info_gt[0]
        if len(p_info) >= 1:  # if we have at least one detection
            x, y, r, plate = p_info[0]
            norm_dist = compute_normalized_plate_distance((x, y, r), (x_gt, y_gt, r_gt))
            norm_dist_all.append(norm_dist)

            #area_inter = intersection_area_of_circles((x, y, r), (x_gt, y_gt, r_gt))
            #area_union = math.pi * r_gt ** 2 + math.pi * r ** 2 - area_inter
            #iou = area_inter / area_union
            #iou_all.append(iou)

            txt_distance = levenshtein_distance(plate_gt, plate)
            txt_distance_all.append(txt_distance)


    # Plot histogram
    plt.figure()
    hist, bin_edges = np.histogram(np.array(txt_distance_all),  density=False)
    plt.step(bin_edges[:-1], hist, where='mid')
    print("hist=", hist)
    print("bin_edges=", bin_edges)
    ax = plt.axes()
    ax.set_title("Distancia de Levenshtein: matrícula reconocida vs real")
    ax.set_ylabel("Núm. imágenes")
    ax.set_xlabel('Distancia de edición (en "número de operaciones")')


    # Now plot curve of distances to the ground truth plate center
    plt.figure()
    norm_dist_all.sort()
    plt.plot(norm_dist_all)
    ax = plt.axes()
    ax.set_title("Bondad de la detección de matrículas")
    ax.set_ylabel("Distancia normalizada: centro estimado a real")
    ax.set_xlabel("Núm. matrículas procesadas")
    plt.show()

if __name__ == "__main__":

    print('PROCESANDO testing_ocr ------------------------------')
    plates_gt = read_gt_csv_file('testing_ocr_ETIQUETADO.txt')
    print(plates_gt)
    plates = read_csv_file('testing_ocr.txt')
    print(plates)

    print('PROCESANDO testing_full_system ------------------------------')
    plates_gt = read_gt_csv_file('testing_full_system_ETIQUETADO.txt')
    print(plates_gt)
    plates = read_csv_file('testing_full_system.txt')
    print(plates)

    plot_plate_recognition_distances(plates_gt, plates)
