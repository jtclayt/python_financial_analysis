'''
    Numpy is a linear algebra library for multi dimensional arrays, good for:
        - computational efficiency
        - easy python syntax
'''
import numpy as np


'''Built in functions of numpy'''
# my_list = [1,2,3,4]
# arr = np.array(my_list)
# print(arr)

# nested_list = [
#     [1,2],
#     [3,4]
# ]

# matrix = np.array(nested_list)

# print(matrix)
# print(np.random.rand(5,5))
# print(np.random.randint(1, 5, 10))
# print(np.arange(1, 50, 2)) # array range 1 to 50 in stepsize 2
# print(np.eye(5)) # 5x5 identity matrix
# print(np.ones((3, 3))) # create a 3x3 matrix of ones
# print(np.zeros((2, 2))) # create a 2x2 matrix of 0

# challenge
# upper_bound = int(input('Enter a max for random number: ')) + 1
# print(np.random.randint(0, upper_bound, 20))


'''Shape Length and Type'''
# my_list = [1, 2, 3, 4, 5, 6, 7, 8, 9]
# np_arr = np.array(my_list)
# print(len(np_arr)) # get length of num py array
# print(np_arr.shape) # get dimensions of array
# print(np_arr.dtype) # get data type of array
# np_matrix = np_arr.reshape(3, 3) # change array from 1x9 to 3x3 matrix
# print(np_matrix)
# print(len(np_matrix)) # number of 'rows' in matrix
# print(np_matrix.shape) # dimesnions of matrix
# print(np_matrix.max()) # get max in array
# print(np_matrix.min()) # get min in array
# print(np_matrix.argmax()) # get position of max element
# print(np_matrix.argmin()) # get position of min element

# challenge 1
# x = np.arange(300, 500, 10).reshape(4, 5)
# print(x)

# challenge 2
# y = np.random.uniform(-1000, 1000, (20, 20))
# print(f'Min: {y.min()}')
# print(f'Max: {y.max()}')
# print(f'Mean: {y.mean()}')


'''Math operations in numpy'''
# x = np.arange(1, 4)
# y = np.arange(1, 4)
# print(x**2) # elementwise square an array
# print(np.sqrt(x)) # elementwise square root an array
# print(x + y) # add two linear array
# print(x * y) # elementwise multiply to arrays
# print(np.dot(x, y)) # dot product of two arrays
# print(np.cross(y, x)) # cross product of two arrays (must be length 2 or 3)

# # challenge one
# x = np.array([3, 20, 30])
# y = np.array([4, 6, 7])
# vect = x - y
# print(np.sqrt(vect.dot(vect)))
# distances = np.sqrt(x**2 + y**2) # distance from the origin to each point
# print(distances)

# A = np.ones((3, 3))
# B = np.ones((3, 3))

# print(A)
# print(A + B) # elementwise addition of matrices
# print(A * B) # element wise multiplication of matrices


'''Slicing and indexing'''
# x = np.array([1, 2, 3, 4, 5])
# print(x[1]) # Same way to access as a regular list
# print(x[0:3]) # slice from array
# x[1] = 5
# x[2:] = 1 # broadcasting: change multiple values at once
# print(x)

# matrix = np.random.randint(1, 10, (5, 5))
# # make a matrix of random ints from 1 to 9
# print(matrix)
# print(matrix[0]) # get whole first row
# print(matrix[:, 0]) # get first column as an array
# print(matrix[0][0]) # get element in first row first col
# print(matrix[:2, :2]) # get a sub matrix
# matrix

# # challenge
# X = np.array([
#     [2, 30, 20, -2, -4],
#     [3, 4, 40, -3, -2],
#     [-3, 4, -6, 90, 10],
#     [25, 45, 34, 22, 12],
#     [13, 24, 22, 32, 37]
# ])
# X[-1] = -1 # change bottom row to -1
# X[:2, -2:] *= 2 # multiply upper right 2x2 by 2
# print(X)

'''Element selection (conditional)'''
matrix = np.random.randint(1, 10, (5, 5))
print(matrix)
print(matrix[matrix > 3])  # selects all elements greater than 3 from matrix
print(matrix[matrix % 2 == 0])  # select all even elements from matrix

# challenge
X = np.array([
    [2, 30, 20, -2, -4],
    [3, 4, 40, -3, -2],
    [-3, 4, -6, 90, 10],
    [25, 45, 34, 22, 12],
    [13, 24, 22, 32, 37]
])

X[X < 0] = 0  # replace neg elements with 0
X[X % 2 == 1] = 25  # replace odd elements with 25

print(X)
