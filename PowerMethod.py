import numpy as np


def matrix_initializer(row):
    '''this function initializes the matrix elements with user's input'''
    print("----> Enter matrix element in a single line and separate each element by space!!! <----)")
    element = [int(i) for i in input().split()]
    matrix = np.array(element).reshape(row, row)
    return matrix


def power_method_inputs_initializer(vector_size, epsilon):
    ''' this function initialize starter vector(V0) and epsilon(has default value) '''
    print("---- > Enter the V0 vector element in a single line(separated each element by space) <----")
    element = [int(i) for i in input().split()]
    V0 = np.array(element).reshape(vector_size)
    return V0, epsilon


def approximate_largest_eigen_value(matrix, v0, epsilon):
    """ this function implements power iteration(power method) algorithm for target matrix
        and returns an approximation of largest eigen value and eigen vector at the end
    """
    # if you want see approximation of eigen vector for largest eigen value return this ==> (vk_list[k-2])

    vk_list = []
    vk_old_list = []
    uk = []

    vk_old_list.append(np.matmul(matrix, v0))

    k = 1
    while True:
        uk.append(max(vk_old_list[k-1], key=abs))
        vk_list.append(vk_old_list[k-1]/uk[k-1])
        k = k+1
        vk_old_list.append(np.matmul(matrix, vk_list[k-2]))
        if k >= 3:
            if abs(uk[k-2]-uk[k-3]) <= epsilon:
                break
    return uk[k-2]


def smallest_eigen_value(matrix, lambda_one, v_zero, epsilon):
    """
    This function calcaulates lambda_n which represent the smallest eigen value
    It only functions for positive symmetric matrix(vital rule!)
    """
    helper_matrix = matrix - (lambda_one * np.identity(3))
    helper_matrix_dominant_eigen_value = approximate_largest_eigen_value(
        helper_matrix, v_zero, epsilon=1/100000)
    return (helper_matrix_dominant_eigen_value + lambda_one)


if __name__ == "__main__":
    # If you wanna test program you can simply start with sample matrix (n*n) in bottom
    # first one matrix is symmetric positive definite matrix
    # -------- matrix = np.array([[2,-1,0], [-1, 2, -1], [0, -1, 2]])--------
    # -------- matrix = np.array([[-100, 3, 5], [7, 6, 5], [7, 8, 9]])--------
    # row => 3 / matrix must be square
    n = int(input("Enter the number of rows=columns => "))

    matrix = matrix_initializer(n)

    # you can test V0 with this sample vector v0 = np.array([1, 1, 1])
    # power_method_inputs_initializer get input from user for V0 and you can initialize some value for epsilon
    v_zero, epsilon = power_method_inputs_initializer(n, epsilon=1/100000)

    lambda_one = approximate_largest_eigen_value(
        matrix, v_zero, epsilon)
    print('Largest eigen value ===> ', lambda_one)

    lambda_n = smallest_eigen_value(matrix, lambda_one, v_zero, epsilon)
    # this gives only the smallest eigen value for symmetric postive definit matrices
    print('Smallest eigen value ===> ', lambda_n)
