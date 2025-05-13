# Rod Length
L_x = 1
L_y = 1
# Number of control volumes
N_x = 51
N_y = 51
# Velocities
u = 2
v = 2
# Dirichlet condition
C_0 = 1
sigma = L_y / 5
# Diffusion
D = 0.01

# Extra variables (No need to change)
rho = 1
Tau = D
time_scaling_factor = 0.1

import numpy as np, matplotlib.pyplot as plt

def create_grid():
    x = np.linspace(0, L_x, num=N_x+1)
    delta_x = x[1] - x[0]
    x -= delta_x / 2
    x[0] = 0.0
    x = np.concatenate((x, np.array([L_x])))

    y = np.linspace(0, L_y, num=N_y+1)
    delta_y = y[1] - y[0]
    y -= delta_y / 2
    y[0] = 0.0
    y = np.concatenate((y, np.array([L_y])))

    return np.meshgrid(x, y), delta_x, delta_y

def left_boundary_pollutants():
    [xx, yy], delta_x, delta_y = create_grid()
    return C_0 * np.exp(-((yy[:, 0] - L_y / 2) ** 2) / (2 * (sigma ** 2))), delta_x, delta_y, xx, yy

def get_variable_index(row, col):
    return row * N_x + col

def upwind(u, v):
    left_boundary_condition, delta_x, delta_y, xx, yy = left_boundary_pollutants()

    # In between CVs
    a_E = Tau / delta_x
    a_W = Tau / delta_x + rho * u
    a_N = Tau / delta_y
    a_S = Tau / delta_y + rho * v
    a_P = a_E + a_W + a_N + a_S
    A = []
    b = []
    for j in range(1, N_y-1): # row
        for i in range(1, N_x-1): # col
            temp = [0]*(N_x*N_y)
            temp[get_variable_index(j, i)] = a_P
            temp[get_variable_index(j+1, i)] = -a_S
            temp[get_variable_index(j-1, i)] = -a_N
            temp[get_variable_index(j, i+1)] = -a_E
            temp[get_variable_index(j, i-1)] = -a_W
            A.append(temp)
            b.append(0)
    
    # Leftmost CVs
    a_W = 2 * Tau / delta_x + rho * u
    a_P = a_E + a_W + a_N + a_S
    for j in range(1, N_y-1):
        temp = [0]*(N_x*N_y)
        temp[get_variable_index(j, 0)] = a_P
        temp[get_variable_index(j, 1)] = -a_E
        temp[get_variable_index(j-1, 0)] = -a_N
        temp[get_variable_index(j+1, 0)] = -a_S
        A.append(temp)
        b.append(a_W * left_boundary_condition[1+j])
    a_W = Tau / delta_x + rho * u

    # Rightmost CVs
    a_E = 0
    a_P = a_E + a_W + a_N + a_S
    for j in range(1, N_y-1):
        temp = [0]*(N_x*N_y)
        temp[get_variable_index(j, N_x-1)] = a_P
        temp[get_variable_index(j+1, N_x-1)] = -a_S
        temp[get_variable_index(j-1, N_x-1)] = -a_N
        temp[get_variable_index(j, N_x-2)] = -a_W
        A.append(temp)
        b.append(0)
    a_E = Tau / delta_x

    # Topmost CVs
    a_N = 0
    a_P = a_E + a_W + a_N + a_S
    for i in range(1, N_x-1):
        temp = [0]*(N_x*N_y)
        temp[get_variable_index(0, i)] = a_P
        temp[get_variable_index(0, i+1)] = -a_E
        temp[get_variable_index(0, i-1)] = -a_W
        temp[get_variable_index(1, i)] = -a_S
        A.append(temp)
        b.append(0)
    a_N = Tau / delta_y

    # Bottomost CVs 
    a_S = rho * v
    a_P = a_E + a_W + a_N + a_S
    for i in range(1, N_x-1):
        temp = [0]*(N_x*N_y)
        temp[get_variable_index(N_y-1, i)] = a_P-a_S
        temp[get_variable_index(N_y-1, i+1)] = -a_E
        temp[get_variable_index(N_y-2, i)] = -a_N
        temp[get_variable_index(N_y-1, i-1)] = -a_W
        A.append(temp)
        b.append(0)
    a_S = Tau / delta_y + rho * v

    # Top left corner
    a_N = 0
    a_W = 2 * Tau / delta_x + rho * u
    a_P = a_E + a_W + a_N + a_S
    temp = [0]*(N_x*N_y)
    temp[get_variable_index(0, 0)] = a_P
    temp[get_variable_index(0, 1)] = -a_E
    temp[get_variable_index(1, 0)] = -a_S
    A.append(temp)
    b.append(a_W * left_boundary_condition[1])
    a_N = Tau / delta_y
    a_W = Tau / delta_x + rho * u

    # Top right corner
    a_E = 0.0
    a_N = 0.0
    a_P = a_E + a_W + a_N + a_S
    temp = [0]*(N_x*N_y)
    temp[get_variable_index(0, N_x-1)] = a_P
    temp[get_variable_index(0, N_x-2)] = -a_W
    temp[get_variable_index(1, N_x-1)] = -a_S
    A.append(temp)
    b.append(0)
    a_E = Tau / delta_x
    a_N = Tau / delta_y

    # Bottom left corner
    a_S = rho * v
    a_W = 2 * Tau / delta_x + rho * u
    temp = [0]*(N_x*N_y)
    temp[get_variable_index(N_y-1, 0)] = a_P - a_S
    temp[get_variable_index(N_y-1, 1)] = -a_E
    temp[get_variable_index(N_y-2, 0)] = -a_N
    A.append(temp)
    b.append(a_W * left_boundary_condition[-2])
    a_S = Tau / delta_y + rho * v
    a_W = Tau / delta_x + rho * u

    # Bottom right corner
    a_S = rho * v
    a_E = 0
    a_P = a_E + a_W + a_N + a_S
    temp = [0]*(N_x*N_y)
    temp[get_variable_index(N_y-1, N_x-1)] = a_P - a_S
    temp[get_variable_index(N_y-1, N_x-2)] = -a_W
    temp[get_variable_index(N_y-2, N_x-1)] = -a_N
    A.append(temp)
    b.append(0)
    a_S = Tau / delta_y + rho * v
    a_E = Tau / delta_x

    A = np.array(A)
    b = np.array(b)
    return np.linalg.solve(A, b), xx, yy, left_boundary_condition

def upwind_iterations():
    print("Upwind")
    for i in [100]:
        local_u = i*time_scaling_factor*u
        local_v = i*time_scaling_factor*v
        pollutants, xx, yy, left_boundary_condition = upwind(local_u, local_v)
        pollutants = pollutants.reshape(N_y, N_x)
        pollutants -= np.min(pollutants[:, -1])
        plt.clf()
        plt.contourf(xx[1:-1, 1:-1], yy[1:-1, 1:-1][::-1, :], pollutants)
        plt.colorbar()
        plt.xlabel("X")
        plt.ylabel("Y")
        plt.title(f"Velocity = ({local_u}, {local_v})")
        plt.savefig(f"./upwind/concentration_profile_{i}")

        plt.clf()
        plt.plot(yy[1:-1, 0], left_boundary_condition[1:-1], label = "left boundary")
        plt.plot(xx[0, 1:-1], pollutants[0, :], label = "right boundary")
        plt.xlabel("Y")
        plt.ylabel("Pollutants")
        plt.title(f"Velocity = ({local_u}, {local_v})")
        # plt.colorbar()
        plt.savefig(f"./upwind/boundary_profile_{i}")

def central_difference(u, v):
    left_boundary_condition, delta_x, delta_y, xx, yy = left_boundary_pollutants()

    A = []
    b = []

    # In between CVs
    a_E = Tau / delta_x - rho * u / 2
    a_W = Tau / delta_x + rho * u / 2
    a_N = Tau / delta_y - rho * v / 2
    a_S = Tau / delta_y + rho * v / 2
    a_P = a_E + a_W + a_N + a_S
    for j in range(1, N_y-1): # row
        for i in range(1, N_x-1): # col
            temp = [0]*(N_x*N_y)
            temp[get_variable_index(j, i)] = a_P
            temp[get_variable_index(j+1, i)] = -a_S
            temp[get_variable_index(j-1, i)] = -a_N
            temp[get_variable_index(j, i+1)] = -a_E
            temp[get_variable_index(j, i-1)] = -a_W
            A.append(temp)
            b.append(0)
    
    # Leftmost CVs
    a_W = 2 * Tau / delta_x + rho * u / 2
    a_P = a_E + a_W + a_N + a_S
    for j in range(1, N_y-1):
        temp = [0]*(N_x*N_y)
        temp[get_variable_index(j, 0)] = a_P
        temp[get_variable_index(j+1, 0)] = -a_S
        temp[get_variable_index(j-1, 0)] = -a_N
        temp[get_variable_index(j, 1)] = -a_E
        A.append(temp)
        b.append(a_W * left_boundary_condition[j+1])
    a_W = Tau / delta_x + rho * u / 2

    # Rightmost CVs
    a_E = - rho * u / 2
    a_P = a_E + a_W + a_N + a_S
    for j in range(1, N_y-1):
        temp = [0]*(N_x*N_y)
        temp[get_variable_index(j, N_x-1)] = a_P - a_E
        temp[get_variable_index(j+1, N_x-1)] = -a_S
        temp[get_variable_index(j-1, N_x-1)] = -a_N
        temp[get_variable_index(j, N_x-2)] = -a_W
        A.append(temp)
        b.append(0)
    a_E = Tau / delta_x - rho * u / 2

    # Topmost CVs
    a_N = - rho * v / 2
    a_P = a_E + a_W + a_N + a_S
    for i in range(1, N_x-1):
        temp = [0]*(N_x*N_y)
        temp[get_variable_index(0, i)] = a_P - a_N
        temp[get_variable_index(0, i+1)] = -a_E
        temp[get_variable_index(0, i-1)] = -a_W
        temp[get_variable_index(1, i)] = -a_S
        A.append(temp)
        b.append(0)
    a_N = Tau / delta_y - rho * v / 2

    # Bottommost CVs
    a_S = rho * v / 2
    a_P = a_E + a_W + a_N + a_S
    for i in range(1, N_x-1):
        temp = [0]*(N_x*N_y)
        temp[get_variable_index(N_y-1, i)] = a_P - a_S
        temp[get_variable_index(N_y-1, i+1)] = -a_E
        temp[get_variable_index(N_y-1, i-1)] = -a_W
        temp[get_variable_index(N_y-2, i)] = -a_N
        A.append(temp)
        b.append(0)
    a_S = Tau / delta_y + rho * v / 2

    # Topleft corner
    a_N = - rho * v / 2
    a_W = 2 * Tau / delta_x + rho * u / 2
    a_P = a_E + a_W + a_N + a_S
    temp = [0]*(N_x*N_y)
    temp[get_variable_index(0, 0)] = a_P - a_N
    temp[get_variable_index(0, 1)] = -a_E
    temp[get_variable_index(1, 0)] = -a_S
    A.append(temp)
    b.append(a_W * left_boundary_condition[1])
    a_N = Tau / delta_y - rho * v / 2
    a_W = Tau / delta_x + rho * u / 2

    # Topright corner
    a_E = - rho * u / 2
    a_N = - rho * v / 2
    a_P = a_E + a_W + a_N + a_S
    temp = [0]*(N_x*N_y)
    temp[get_variable_index(0, N_x-1)] = a_P - a_E - a_N
    temp[get_variable_index(1, N_x-1)] = -a_S
    temp[get_variable_index(0, N_x-2)] = -a_W
    A.append(temp)
    b.append(0)
    a_E = Tau / delta_x - rho * u / 2
    a_N = Tau / delta_y - rho * v / 2

    # Bottomleft corner
    a_S = rho * v / 2
    a_W = 2 * Tau / delta_x + rho * u / 2
    temp = [0]*(N_x*N_y)
    temp[get_variable_index(N_y-1, 0)] = a_P - a_S
    temp[get_variable_index(N_y-2, 0)] = -a_N
    temp[get_variable_index(N_y-1, 1)] = -a_E
    A.append(temp)
    b.append(a_W * left_boundary_condition[-2])
    a_S = Tau / delta_y + rho * v / 2
    a_W = Tau / delta_x + rho * u / 2

    # Bottomright corner
    a_E = - rho * u / 2
    a_S = rho * v / 2
    temp = [0]*(N_x*N_y)
    temp[get_variable_index(N_y-1, N_x-1)] = a_P - a_E - a_S
    temp[get_variable_index(N_y-2, N_x-1)] = -a_N
    temp[get_variable_index(N_y-1, N_x-2)] = -a_W
    A.append(temp)
    b.append(0)
    a_E = Tau / delta_x - rho * u / 2
    a_S = Tau / delta_y + rho * v / 2

    A = np.array(A)
    b = np.array(b)
    return np.linalg.solve(A, b), xx, yy, left_boundary_condition, delta_x, delta_y

def central_difference_iterations():
    print("Central Difference")
    for i in [100]:
        local_u = i*time_scaling_factor*u
        local_v = i*time_scaling_factor*v
        pollutants, xx, yy, left_boundary_condition, delta_x, delta_y = central_difference(local_u, local_v)
        pollutants = pollutants.reshape(N_y, N_x)
        pollutants -= np.min(pollutants[:, -1])
        print("Peclet Number x =", rho*local_u*delta_x/Tau, "Peclet Number y =", rho*local_v*delta_y/Tau)

        plt.clf()
        plt.contourf(xx[1:-1, 1:-1], yy[1:-1, 1:-1][::-1, :], pollutants)
        plt.xlabel("X")
        plt.ylabel("Y")
        plt.title(f"Velocity = ({local_u}, {local_v})")
        # plt.colorbar()
        plt.savefig(f"./central_difference/concentration_profile_{i}")

        plt.clf()
        plt.plot(yy[1:-1, 0], left_boundary_condition[1:-1], label = "left boundary")
        plt.plot(yy[1:-1, 0], pollutants[:, -1], label = "right boundary")
        plt.xlabel("Y")
        plt.ylabel("Pollutants")
        plt.title(f"Velocity = ({local_u}, {local_v})")
        # plt.colorbar()
        plt.savefig(f"./central_difference/boundary_profile_{i}")

if __name__ == "__main__":
    upwind_iterations()
    central_difference_iterations()
