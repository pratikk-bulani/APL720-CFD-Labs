import numpy as np, matplotlib.pyplot as plt

L = 1 # Length along X and Y direction
N = 4 # Number of samples along X and Y direction

def create_grid():
    grid = np.linspace(0, L, N)
    return grid
def grid_points():
    x = create_grid()
    y = create_grid()
    return np.meshgrid(x, y)
def find_variable_index(r, c):
    return r * (N-2) + c
def initial_condition1():
    xx, yy = grid_points()
    return (np.sin(xx * np.pi) + np.sin(yy * np.pi))[1:-1, 1:-1]
def initial_condition2():
    xx, yy = grid_points()
    return (np.sin(8 * xx * np.pi) + np.sin(8 * yy * np.pi))[1:-1, 1:-1]
def create_A():
    A = []
    b = []
    aP = 4; aE = 1; aW = 1; aN = 1; aS  = 1
    for row in range(0, N-2):
        for col in range(0, N-2):
            tempA = [0] * ((N-2)*(N-2))
            tempA[find_variable_index(row, col)] = aP
            if row == 0:
                tempA[find_variable_index(row+1, col)] = -aS
            elif row == N-3:
                tempA[find_variable_index(row-1, col)] = -aN
            else:
                tempA[find_variable_index(row-1, col)] = -aN
                tempA[find_variable_index(row+1, col)] = -aS
            if col == 0:
                tempA[find_variable_index(row, col+1)] = -aE
            elif col == N-3:
                tempA[find_variable_index(row, col-1)] = -aW
            else:
                tempA[find_variable_index(row, col+1)] = -aE
                tempA[find_variable_index(row, col-1)] = -aW
            A.append(tempA)
            b.append(0)
    return np.array(A), np.array(b)
def gauss_seidel_step(initial_condition, A, b):
    for i in range(len(A)):
        temp = (A[i, :].reshape(1,-1) @ initial_condition)[0, 0] - A[i,i]*initial_condition[i,0]
        initial_condition[i,0] = (b[i] - temp) / A[i,i]
def epsilon_k(phi_k_new, phi_k_old):
    return np.sum((phi_k_new - phi_k_old) ** 2) ** 0.5
def gauss_seidel(initial_condition,A,b):
    ic = initial_condition().flatten().reshape(-1,1)
    iterative_values = [np.copy(ic)]
    relative_errors = []
    while(True):
        gauss_seidel_step(ic, A, b)
        iterative_values.append(np.copy(ic))
        relative_errors.append(epsilon_k(iterative_values[-1], iterative_values[-2]) / epsilon_k(iterative_values[1], iterative_values[0]))
        if(relative_errors[-1] <= 1e-2):
            break
    print(f"Took {len(relative_errors)} iterations to converge")
    return relative_errors

def experiment1():
    A,b = create_A()
    relative_errors1 = gauss_seidel(initial_condition1,A,b)
    relative_errors2 = gauss_seidel(initial_condition2,A,b)
    plt.plot(relative_errors1, label = "IC1")
    plt.plot(relative_errors2, label = "IC2")
    plt.legend()
    plt.xlabel("#iters")
    plt.ylabel("Relative Error")
    plt.grid()
    plt.show()

def experiment2():
    global N
    iterations1 = []; iterations2 = []
    grid_size = [16, 32, 64]
    for i in grid_size:
        N = i
        A,b = create_A()
        relative_errors = gauss_seidel(initial_condition1,A,b)
        iterations1.append(len(relative_errors))
        relative_errors = gauss_seidel(initial_condition2,A,b)
        iterations2.append(len(relative_errors))
    plt.plot(grid_size, iterations1, label = "IC1")
    plt.plot(grid_size, iterations2, label = "IC2")
    plt.legend()
    plt.xlabel("Grid Size")
    plt.ylabel("Iterations")
    plt.grid()
    plt.show()

if __name__ == "__main__":
    experiment1()
    # experiment2()