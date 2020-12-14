import cv2
import os
import numpy as np
import imutils

import tensorflow as tf
from tensorflow.keras.models import load_model
import psutil
import time

from solver import Sudoku_Solver


def resize(img):
    input = np.zeros((800, 800, 3), dtype=np.uint8)
    input[:, :] = (255, 255, 255)
    height = img.shape[0]
    width = img.shape[1]

    if height > width:

        new_height = 800
        new_width = int((800 * width) / height)

        img = cv2.resize(img, (new_width, 800))
    else:

        new_height = int((800 * height) / width)
        new_width = 800
        img = cv2.resize(img, (800, new_height))

    # print(img.shape)
    input[0:new_height, 0:new_width] = img
    return input, img

def filter(array, thresh, type, l=None):

    if type == "vertical":
        index = 0
    else:
        index = 1


    if l == None:
        l = len(array)
    i = 0
    while(i+1 != l):

        if array[i][-1] < 0:
            del array[i]
            i -= 1
            l -= 1

        elif (abs(array[i+1][-1] - array[i][-1]) < thresh):

            if array[i][index] > 400:
                del array[i+1]

            else:
                del array[i]

            i -= 1
            l -= 1
        i += 1

def preprocessing(input, ret = False):

    output = input.copy()

    gray = cv2.cvtColor(input, cv2.COLOR_BGR2GRAY)
    # blur = cv2.medianBlur(gray, 5)
    adapt_type = cv2.ADAPTIVE_THRESH_GAUSSIAN_C
    thresh_type = cv2.THRESH_BINARY_INV
    _, bin_img = cv2.threshold(gray, 100, 255, thresh_type)

    # bin_img = cv2.adaptiveThreshold(gray, 255, thresh_type, thresh_type, 5, 2)
    edges = cv2.Canny(gray, 50, 50, apertureSize=3)
    # edges = cv2.adaptiveThreshold(edges, 255, adapt_type, cv2.THRESH_BINARY, 11, 2)

    # cv2.imshow("bin_img", bin_img)
    # bin_img = cv2.bitwise_not(gray)
    cv2.imshow("inpufaft", gray)

    # cv2.imshow("img_1", img_1)



    rho, theta, thresh = 2, np.pi / 180, 375
    lines = cv2.HoughLines(edges, rho, theta, thresh)

    vertical_lines = list()
    horizontal_lines = list()

    for x in range(0, len(lines)):
        for rho, theta in lines[x]:
            a = np.cos(theta)
            b = np.sin(theta)
            x0 = a * rho
            y0 = b * rho
            x1 = int(x0 + 1000 * (-b))
            y1 = int(y0 + 1000 * (a))
            x2 = int(x0 - 1000 * (-b))
            y2 = int(y0 - 1000 * (a))

            # cv2.line(output, (x1, y1), (x2, y2), 255, 1)

            # print(rho / a, rho/b)
            # print(rho / a)

            if abs(a) > abs(b):
                vertical_lines.append([x1, y1, x2, y2, rho / a])
                # horizontal_distaces.append(rho/a)
                # print(rho / a)

            else:
                horizontal_lines.append([x1, y1, x2, y2, rho / b])

    vertical_lines.sort(key=lambda x: x[-1])
    horizontal_lines.sort(key=lambda x: x[-1])

    print(vertical_lines)
    print(horizontal_lines)

    print(len(vertical_lines))
    print(len(horizontal_lines))


    # for line in vertical_lines:
    #     output = cv2.line(output, (line[0], line[1]), (line[2], line[3]), 255, 1)
    #
    #
    # for line in horizontal_lines:
    #     output = cv2.line(output, (line[0], line[1]), (line[2], line[3]), 255, 1)

    filter(vertical_lines, 25, "vertical")
    filter(horizontal_lines, 25, "horizontal")

    print(vertical_lines)
    print(horizontal_lines)

    print(len(vertical_lines))
    print(len(horizontal_lines))

    print(output.shape)


    for line in vertical_lines:
        output = cv2.line(output, (line[0], line[1]), (line[2], line[3]), 255, 1)


    for line in horizontal_lines:
        output = cv2.line(output, (line[0], line[1]), (line[2], line[3]), 255, 1)



    sudoku_squares = list()
    if (len(horizontal_lines) == 10) and (len(vertical_lines) == 10):
        print("success")

    else:
        print("un-successiful filtering")
        cv2.imshow("edges", edges)
        cv2.imshow("input", gray)
        cv2.imshow("output", output)
        cv2.waitKey(0)
        exit()

    intersections = list()

    for hor_line in range(10):
        for ver_line in range(10):
            x1 = int(vertical_lines[ver_line][-1])
            y1 = int(horizontal_lines[hor_line][-1])
            output = cv2.circle(output, (x1, y1), radius=4, color=(0, 0, 255), thickness=-1)
            # intersections

            if (hor_line == 9) or (ver_line == 9):
                continue

            x2 = int(vertical_lines[ver_line + 1][-1])
            y2 = int(horizontal_lines[hor_line + 1][-1])

            output = cv2.circle(output, (x1, y2), radius=4, color=(255, 0, 0), thickness=-1)

            # print(x1, y1, x2, y2)
            cube = bin_img[y1: y2, x1: x2]
            # print(cube.shape)
            cube = cv2.resize(cube, (32, 32))

            cube = cube[2: 30, 2: 30]
            # print(cube.shape)

            # print(cube)
            sudoku_squares.append(cube)
            intersections.append((x1, y2))
    # cv2.imshow('out', output)

    if ret == True:
        return sudoku_squares, intersections, bin_img, edges, output
    else:
        return sudoku_squares, intersections


def nn_prediction(model, input):
    nn_input = input / 255.0
    nn_input = np.expand_dims(nn_input, axis=0)
    nn_input = np.expand_dims(nn_input, axis=-1)

    # print(nn_input.shape)
    result = np.argmax(model.predict(nn_input, batch_size=10), axis=-1) + 1

    return int(result)


def predictions(array, sudoku):
    blank_indices = list()
    model = tf.keras.models.load_model('digit_trained.h5')
    print("Loaded model from disk")
    index = 0

    for cube in array:

        whites = 0

        for row_index in range(4, 24):
            for col_index in range(4, 24):
                if cube[row_index][col_index] > 127:
                    whites += 1

        if whites <= 5:
            sudoku[index] = 0
            blank_indices.append(index)
        else:
            sudoku[index] = nn_prediction(model, cube)
        index += 1

    return blank_indices


def print_predictions(img, intersections, blank_indices, solved_array):
    itterator = 0
    # print(blank_indices)
    # print(intersections)
    for blank in blank_indices:
        img = cv2.putText(img, str(solved_array[itterator]), (intersections[blank][0] + 15, intersections[blank][1] - 15), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 2)
        itterator += 1
    return img

def main():


    #set this to s to save the cells as an image to create an dataset.

    user = "p"

    path = r"images"

    #output director to create the dataset
    output_path = r"Dataset"
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    files = os.listdir(path)

    print(files)
    file_number = 5
    img_path = os.path.join(path, files[file_number-1])
    img = cv2.imread(img_path)
    print(img_path)

    # input_img = resizing(img)

    # output = input_img.copy()
    input_img, img = resize(img)
    sudoku_squares, intersections, bin_img, edges, output = preprocessing(input_img, ret=True)

    cv2.imshow("output", output)
    cv2.imshow("bin_image", bin_img)
    cv2.imshow("edges", edges)
    #
    # cv2.waitKey(0)

    # print(sudoku_squares)
    if user == "s":
        i = 0
        for cubes in sudoku_squares:

            image_name = os.path.join(output_path, r"sudoku_{}_{}.jpg".format(file_number, i))
            # print(image_name)
            i += 1

            cv2.imwrite(image_name, cubes)
        exit()

    # sudoku_squares, intersections = preprocessing(input_img)
    # cv2.imshow("bin_img", bin_img)
    # cv2.imshow("input_img", edges)
    cv2.imshow("Sudoku", img)
    # cv2.imshow("input_img", input_img)

    # intersections = intersections.reshape(9, 9)
    sudoku = np.ones(81, dtype=np.int)

    blank_indices = predictions(sudoku_squares, sudoku)
    sudoku = sudoku.reshape(9, 9)
    print(sudoku)

    sudoku_obj = Sudoku_Solver(sudoku)
    print("Sudoku to Solve :")
    sudoku_obj.print()
    print("Number of Unknowns : {}".format(sudoku_obj.unknowns))
    input("Press enter to solve")
    sudoku_obj.solve()
    print("Solved Sudoku :")
    sudoku_obj.print()

    img = print_predictions(img, intersections, blank_indices, sudoku_obj.solved_array)

    cv2.imshow("solved", img)

    cv2.waitKey(0)



if __name__ == '__main__':
    process = psutil.Process(os.getpid())
    start = time.time()
    main()
    end = time.time()
    print(f"Runtime of the program is {end - start} sec")
    print(f"Memory used by the programme is {process.memory_info().rss} bytes")