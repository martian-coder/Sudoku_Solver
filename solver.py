import numpy as np
import os
import psutil
import time

sudoku_matrix_old = np.array([[[[0, 8, 6], [0, 0, 0], [0, 0, 0]],
                           [[1, 4, 9], [7, 0, 0], [0, 3, 0]],
                           [[0, 5, 3], [9, 8, 4], [0, 7, 0]]],
                          [[[0, 0, 0], [0, 5, 0], [4, 0, 0]],
                           [[0, 6, 3], [0, 0, 0], [9, 1, 0]],
                           [[0, 0, 1], [0, 3, 0], [0, 0, 0]]],
                          [[[0, 4, 0], [3, 1, 2], [6, 7, 0]],
                           [[0, 7, 0], [0, 0, 5], [8, 2, 4]],
                           [[0, 0, 0], [0, 0, 0], [3, 1, 0]]]])

sudoku_matrix = np.array([[0, 8, 6, 1, 4, 9, 0, 5, 3],
                           [0, 0, 0, 7, 0, 0, 9, 8, 4],
                           [0, 0, 0, 0, 3, 0, 0, 7, 0],
                           [0, 0, 0, 0, 6, 3, 0, 0, 1],
                           [0, 5, 0, 0, 0, 0, 0, 3, 0],
                           [4, 0, 0, 9, 1, 0, 0, 0, 0],
                           [0, 4, 0, 0, 7, 0, 0, 0, 0],
                           [3, 1, 2, 0, 0, 5, 0, 0, 0],
                           [6, 7, 0, 8, 2, 4, 3, 1, 0]])
class Sudoku_Solver:

    full_set = {1, 2, 3, 4, 5, 6, 7, 8, 9}

    def __init__(self, sudoku_matrix_2d):
        self.sudoku_matrix_2d = sudoku_matrix_2d
        self.sudoku_transpose = np.transpose(self.sudoku_matrix_2d)
        self.sudoku_submatrix = self.submatrix_generator()
        self.leadger = self.leadger_generator()
        self.unknowns = self.leadger[-1][-1]
        self.solved_array = np.zeros(self.unknowns + 1, dtype=np.int)

    def submatrix_generator(self):

        submatrix = list()

        for outer_row in range(3):
            for col in range(3):
                temp = set()
                for row in range(3):
                    temp = temp.union(self.sudoku_matrix_2d[outer_row * 3 + row][col * 3: col * 3 + 3])
                # print(temp)
                submatrix.append(temp)
        return submatrix

    def leadger_generator(self):
        leadger = list()
        index = 0
        for row in range(9):
            for column in range(9):
                if self.sudoku_matrix_2d[row][column] == 0:
                    submatrix = ((column//3) + 3*(row//3))
                    leadger.append([row, column, submatrix, index])
                    index += 1
        return leadger


    def solve(self):

        # intrest = list()

        while(self.unknowns!=-1):

            for item in self.leadger:
                if item == 0:
                    continue
                temp = set()
                row = item[0]
                column = item[1]
                submatrix = item[2]
                temp = temp.union(self.sudoku_matrix_2d[row])
                temp = temp.union(self.sudoku_transpose[column])
                temp = temp.union(self.sudoku_submatrix[submatrix])
                temp = Sudoku_Solver.full_set.difference(temp)
                # intrest.append(temp)
                if len(temp) == 1:
                    number = temp.pop()
                    # print(item[3])
                    self.update(item[3], number)
                    self.solved_array[int(item[3])] = number
                    break
        # print("intrest")
        # print(intrest)

    def update(self, index, number):

        # print("updating index: {}, number {}".format(index, number))
        # print("leadger")
        # print(self.leadger)
        row = self.leadger[index][0]
        column = self.leadger[index][1]
        submatrix = self.leadger[index][2]



        self.sudoku_matrix_2d[row][column] = number
        self.sudoku_transpose[column][row] = number

        # num_s = set(number)
        # print(num_s)
        # print(self.sudoku_submatrix[submatrix])
        # print("duh")
        self.sudoku_submatrix[submatrix].add(number)
        # print(self.sudoku_submatrix[submatrix])
        self.leadger[index] = 0
        self.unknowns -= 1


    def print_dash(self, count):
        for i in range(count):
            print("=", end="")

    def print(self, type = "n", matrix = None):
        self.print_dash(29)

        if type == "n":
            matrix = self.sudoku_matrix_2d
        elif type == "t":
            matrix = self.sudoku_transpose
        # elif type == "s":
        #     matrix = self.sudoku_submatrix
        elif matrix != None:
            matrix = matrix

        print()

        for row in range(9):
            print("|| ", end="")
            for column in range(9):
                print(matrix[row][column], end=" ")
                if (column == 2) or (column == 5):
                    print("|| ", end="")
            print("||")

            if (row % 3) == 2:
                self.print_dash(29)
                print()



    def print_old(self):

        self.print_dash(29)
        print()
        for outer_row in range(3):
            for outer_coulumn in range(3):
                print("|| ", end="")
                for inner_row in range(3):
                    for inner_column in range(3):
                        print(self.sudoku_matrix_2d[outer_row][inner_row][outer_coulumn][inner_column], end=" ")

                    if inner_row < 2:
                        print("|| ", end="")

                print("||")
            self.print_dash(29)
            print()


def main():
    sudoku_obj = Sudoku_Solver(sudoku_matrix)
    print("Sudoku to Solve :")
    sudoku_obj.print()
    print("Number of Unknowns : {}".format(sudoku_obj.unknowns))
    # sudoku_obj.update(1, 3)
    # sudoku_obj.print()
    # sudoku_obj.print(type="t")
    # sudoku_obj.print(type="s")
    # print(sudoku_obj.leadger)
    # print(sudoku_obj.sudoku_submatrix)
    sudoku_obj.solve()
    print("Solved Sudoku :")
    sudoku_obj.print()
    print(sudoku_obj.solved_array)
    # print(sudoku_obj.unknowns)
    #
    # l1 = [0, 1]
    # set1 = set(l1)
    # print(set1)
    # set2 = {9, 8}
    # set3 = set2.union([9, 8, 1, 2])
    # # set3.remove(0)
    # print(set3)


if __name__ == '__main__':
    process = psutil.Process(os.getpid())
    start = time.time()
    main()
    end = time.time()
    print(f"Runtime of the program is {end - start} sec")
    print(f"Memory used by the programme is {process.memory_info().rss} bytes")
