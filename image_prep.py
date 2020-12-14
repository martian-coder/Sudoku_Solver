import cv2
import os
import numpy as np
import imutils

import tensorflow as tf
from tensorflow.keras.models import load_model
import psutil
import time

from solver import Sudoku_Solver


class Image_Predection(Sudoku_Solver):

    def __init__(self, path, output = "Dataset"):

        self.image_path = path
        self.output_path = output

        self.image = cv2.imread(self.image_path)
        self.input = np.zeros((800, 800, 3), dtype=np.uint8)
        self.input[:, :] = (255, 255, 255)
        self.output = self.input.copy()


        self.resize()

        self.gray = cv2.cvtColor(self.input, cv2.COLOR_BGR2GRAY)
        # blur = cv2.medianBlur(gray, 5)
        adapt_type = cv2.ADAPTIVE_THRESH_GAUSSIAN_C
        thresh_type = cv2.THRESH_BINARY_INV
        self.bin_img = cv2.adaptiveThreshold(self.gray, 255, adapt_type, thresh_type, 11, 2)
        self.edges = cv2.Canny(self.gray, 50, 50, apertureSize=3)

        self.sudoku_squares = list()

        self.sudoku = np.ones(81, dtype=np.int)
        self.model = tf.keras.models.load_model('digit_trained.h5')
        self.blank_indices = list()

        self.predictions()

    def resize(self):


        height = self.image.shape[0]
        width = self.image.shape[1]

        if height > width:

            new_height = 800
            new_width = int((800 * width) / height)

            self.image = cv2.resize(self.image, (new_width, 800))
        else:

            new_height = int((800 * height) / width)
            new_width = 800
            self.image = cv2.resize(self.image, (800, new_height))

        # print(img.shape)
        self.input[0:new_height, 0:new_width] = self.image

    def filter(self, array, thresh, l=None):
        if l == None:
            l = len(array)
        i = 0
        while (i + 1 != l):
            if (array[i + 1][-1] - array[i][-1]) < thresh:
                del array[i]
                i -= 1
                l -= 1
            i += 1

    def preprocessing(self):



        rho, theta, thresh = 2, np.pi / 180, 400
        lines = cv2.HoughLines(self.edges, rho, theta, thresh)

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

                if a > b:
                    vertical_lines.append([x1, y1, x2, y2, rho / a])
                    # horizontal_distaces.append(rho/a)
                    # print(rho / a)

                else:
                    horizontal_lines.append([x1, y1, x2, y2, rho / b])

        vertical_lines.sort(key=lambda x: x[-1])
        horizontal_lines.sort(key=lambda x: x[-1])

        # print(vertical_lines)
        # print(horizontal_lines)
        filter(vertical_lines, 20)
        filter(horizontal_lines, 20)

        for line in vertical_lines:
            cv2.line(self.output, (line[0], line[1]), (line[2], line[3]), 255, 1)

        # cv2.line(output_2, (line[0], line[1]), (line[2], line[3]), 255, 1)

        for line in horizontal_lines:
            cv2.line(self.output, (line[0], line[1]), (line[2], line[3]), 255, 1)

        sudoku_squares = list()
        if (len(horizontal_lines) == 10) and (len(vertical_lines) == 10):
            print("success")

        else:
            print("un-successiful filtering")
            return 1


        self.intersections = list()

        for hor_line in range(10):
            for ver_line in range(10):
                x1 = int(vertical_lines[ver_line][-1])
                y1 = int(horizontal_lines[hor_line][-1])
                self.output = cv2.circle(self.output, (x1, y1), radius=4, color=(0, 0, 255), thickness=-1)
                # intersections

                if (hor_line == 9) or (ver_line == 9):
                    continue

                x2 = int(vertical_lines[ver_line + 1][-1])
                y2 = int(horizontal_lines[hor_line + 1][-1])

                # output = cv2.circle(self.output, (x1, y2), radius=4, color=(255, 0, 0), thickness=-1)

                # print(x1, y1, x2, y2)
                cube = self.bin_img[y1: y2, x1: x2]
                # print(cube.shape)
                cube = cv2.resize(cube, (32, 32))

                cube = cube[2: 30, 2: 30]
                # print(cube.shape)

                # print(cube)
                self.sudoku_squares.append(cube)
                self.intersections.append((x1, y2))
        # cv2.imshow('out', output)

        return 0

    def nn_prediction(self, input):
        nn_input = input / 255.0
        nn_input = np.expand_dims(nn_input, axis=0)
        nn_input = np.expand_dims(nn_input, axis=-1)

        # print(nn_input.shape)
        result = np.argmax(self.model.predict(nn_input, batch_size=10), axis=-1) + 1

        return int(result)

    def predictions(self):


        print("Loaded model from disk")
        index = 0

        for cube in self.sudoku_squares:

            whites = 0

            for row_index in range(4, 14):
                for col_index in range(4, 24):
                    if cube[row_index][col_index] > 127:
                        whites += 1

            if whites <= 10:
                self.sudoku[index] = 0
                self.blank_indices.append(index)
            else:
                self.sudoku[index] = self.nn_prediction(cube)
            index += 1

        self.sudoku = self.sudoku.reshape(9, 9)

        # self.Solve_Sudoku()

    def Solve_Sudoku(self):
        super().__init__(self.sudoku)

        super().print()
        # print("Number of Unknowns : {}".format(sudoku_obj.unknowns))
        super().solve()
        print("Solved Sudoku :")
        super().print()

        # print(self.sudoku_matrix_2d)


def main():

    path = r"images"
    # output_path = r"Dataset"
    files = os.listdir(path)

    print(files)
    file_number = 1
    img_path = os.path.join(path, files[file_number-1])

    pred_obj = Image_Predection(img_path)
    status = pred_obj.predictions()

    if status == 1:
        print("image could not be segmented successfully")

    pred_obj.Solve_Sudoku()
    print(pred_obj.sudoku)


if __name__ == '__main__':
    process = psutil.Process(os.getpid())
    start = time.time()
    main()
    end = time.time()
    print(f"Runtime of the program is {end - start} sec")
    print(f"Memory used by the programme is {process.memory_info().rss} bytes")