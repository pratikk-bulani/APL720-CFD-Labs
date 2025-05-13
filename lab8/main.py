import numpy as np, matplotlib.pyplot as plt

L = 1.0 # Length of rod
N = 20 # initial grid points

def initial_values(grid_points):
    return np.linspace(0, 1, grid_points)
def create_A_b(grid_points):
    grid_space_x = L / grid_points
    A = []; b = [grid_space_x * grid_space_x] * grid_points; b[-1] += 2
    # Middle points
    for i in range(1, grid_points-1):
        A.append([0]*grid_points)
        A[-1][i] = 2
        A[-1][i-1] = -1
        A[-1][i+1] = -1
    # Leftmost point
    for i in [0]:
        A.insert(0, [0]*grid_points)
        A[0][i] = 3
        A[0][i+1] = -1
    # Rightmost point
    for i in [grid_points-1]:
        A.append([0]*grid_points)
        A[-1][i] = 3
        A[-1][i-1] = -1
    return np.array(A), np.array(b)
def gauss_seidel_step(initial_condition, A, b):
    for i in range(len(A)):
        temp = (A[i, :].reshape(1,-1) @ initial_condition.reshape(-1, 1))[0, 0] - A[i,i]*initial_condition[i]
        initial_condition[i] = (b[i] - temp) / A[i,i]
def fine_grid_iterations():
    A, b = create_A_b(N)
    y = initial_values(N)
    residuals = []
    for i in range(10):
        gauss_seidel_step(y, A, b)
        residuals.append(np.linalg.norm(b - A @ y))
    r = b - A @ y
    return r, y, A, b, residuals
def prolongation(e_2h):
    e_h_dash = 0.75 * e_2h
    e_h_dash = np.stack((e_h_dash, e_h_dash)).T.flatten()
    for i in range(1, e_h_dash.shape[0]-1):
        if(i%2 == 0):
            e_h_dash[i] += e_2h[i // 2 - 1] * 0.25
        else:
            e_h_dash[i] += e_2h[i // 2 + 1] * 0.25
    return e_h_dash
def restriction(r_h, divide_factor):
    r_2h = r_h.reshape(-1, 2).sum(axis = 1)
    A_2h, b_2h = create_A_b(N // (2 ** divide_factor))
    e_2h = np.zeros_like(b_2h)
    for i in range(40):
        gauss_seidel_step(e_2h, A_2h, r_2h)
    if divide_factor == 2:
        return prolongation(e_2h)
    r_2h -= A_2h @ e_2h
    e_2h_dash = restriction(r_2h, divide_factor+1)
    e_2h_corrected = e_2h + e_2h_dash
    for i in range(5):
        gauss_seidel_step(e_2h_corrected, A_2h, b_2h)
    return prolongation(e_2h_corrected)

def multi_grid():
    r, y, A, b, residuals = fine_grid_iterations()
    e_h = restriction(r, 1)
    y_corrected = y + e_h
    new_residuals = []
    for i in range(100):
        gauss_seidel_step(y_corrected, A, b)
        new_residuals.append(np.linalg.norm(b - A @ y_corrected))
    return y_corrected, residuals + new_residuals[-10:]
def gauss_seidel():
    A, b = create_A_b(N)
    y = initial_values(N)
    residuals = []
    for i in range(110):
        gauss_seidel_step(y, A, b)
        residuals.append(np.linalg.norm(b - A @ y))
    return y, residuals
def analytical():
    grid_space_x = L / N
    grid_point_coordinates = np.linspace(0, 1, N+1) + grid_space_x / 2
    grid_point_coordinates = np.delete(grid_point_coordinates, -1, axis=0)
    return -0.5 * np.power(grid_point_coordinates, 2) + 1.5 * grid_point_coordinates, grid_point_coordinates

def experiment():
    y_multi_grid, residuals_multi_grid = multi_grid()
    y_gauss_seidel, residuals_gauss_seidel = gauss_seidel()
    y_analytical, grid_point_coordinates = analytical()
    plt.plot(residuals_gauss_seidel, label = "gauss seidel")
    plt.plot(residuals_multi_grid, label = "multi grid")
    plt.xlabel("iterations")
    plt.ylabel("residuals")
    plt.legend()
    plt.show()
    plt.plot(grid_point_coordinates, y_multi_grid, label = 'multi grid')
    plt.plot(grid_point_coordinates, y_gauss_seidel, label = 'gauss seidel')
    plt.plot(grid_point_coordinates, y_analytical, label = 'Analytical')
    plt.legend()
    plt.xlabel('rod length')
    plt.ylabel('value')
    plt.show()
    
if __name__ == "__main__":
    experiment()