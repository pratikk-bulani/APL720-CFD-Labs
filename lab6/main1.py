import numpy as np, matplotlib.pyplot as plt

C0 = 1.0 # peak concentration
Lx = 1 # Length of domain along x
Ly = 1 # Length of domain along y
sigma = Ly / 5 
D = 1 # diffusion coefficient
u = 0.1 # velocity in X
v = 0.1 # velocity in Y
Nx = 40 # Number of CVs in X direction
Ny = 50 # Number of CVs in Y direction

time_scaling_factor = 0.5

def create_grid(L, N):
    grid_space = L / N
    grid = np.linspace(0, L, N+1) + grid_space / 2
    return np.delete(grid, -1, 0)
def grid_points():
    x = create_grid(Lx, Nx)
    y = create_grid(Ly, Ny)
    return np.meshgrid(x, y)
def left_boundary_condition():
    y = create_grid(Ly, Ny)
    return C0 * np.exp((-1 * (y - y[len(y) // 2]) ** 2) / (2 * sigma * sigma))
def find_variable_index(r, c):
    return r * Nx + c

def quick_scheme(u, v):
    grid_space_x = Lx / Nx; grid_space_y = Ly / Ny
    Ax = grid_space_y; Ay = grid_space_x
    A = []; b = []

    # All the nodes 
    aE = -((3 * u * Ax / 8) - (D * Ax / grid_space_x))
    aN = -((3 * v * Ay / 8) - (D * Ay / grid_space_y))
    aW = -((-7 * u * Ax / 8) - (D * Ax / grid_space_x))
    aS = -((-7 * v * Ay / 8) - (D * Ay / grid_space_y))
    aWW = -(1 * u * Ax / 8)
    aSS = -(1 * v * Ay / 8)
    aP = aE + aW + aN + aS + aSS + aWW
    for row in range(1, Ny-2):
        for col in range(2, Nx-1):
            tempA = [0] * (Nx * Ny)
            tempA[find_variable_index(row, col)] = aP
            tempA[find_variable_index(row, col+1)] = -aE
            tempA[find_variable_index(row-1, col)] = -aN
            tempA[find_variable_index(row, col-1)] = -aW
            tempA[find_variable_index(row+1, col)] = -aS
            tempA[find_variable_index(row, col-2)] = -aWW
            tempA[find_variable_index(row+2, col)] = -aSS
            A.append(tempA)
            b.append(0)
    
    # 2nd last row
    row = Ny - 2
    aS = (D * Ay / grid_space_y) + (6 * v * Ay / 8)
    aP = aE + aN + aW + aS + aWW
    for col in range(2, Nx-1):
        tempA = [0] * (Nx * Ny)
        tempA[find_variable_index(row, col)] = aP
        tempA[find_variable_index(row, col+1)] = -aE
        tempA[find_variable_index(row-1, col)] = -aN
        tempA[find_variable_index(row, col-1)] = -aW
        tempA[find_variable_index(row+1, col)] = -aS
        tempA[find_variable_index(row, col-2)] = -aWW
        A.append(tempA)
        b.append(0)
    aS = -((-7 * v * Ay / 8) - (D * Ay / grid_space_y))

    # 2nd column
    col = 1
    aW = (D * Ax / grid_space_x) + (6 * u * Ax / 8)
    aP = aE + aN + aW + aS + aSS
    for row in range(1, Ny-2):
        tempA = [0] * (Nx * Ny)
        tempA[find_variable_index(row, col)] = aP
        tempA[find_variable_index(row, col+1)] = -aE
        tempA[find_variable_index(row-1, col)] = -aN
        tempA[find_variable_index(row, col-1)] = -aW
        tempA[find_variable_index(row+1, col)] = -aS
        tempA[find_variable_index(row+2, col)] = -aSS
        A.append(tempA)
        b.append(0)
    aW = -((-7 * u * Ax / 8) - (D * Ax / grid_space_x))

    # 2nd last row, 2nd column corner point
    row = Ny-2; col = 1
    aW = (D * Ax / grid_space_x) + (6 * u * Ax / 8)
    aS = (D * Ay / grid_space_y) + (6 * v * Ay / 8)
    aP = aE + aN + aW + aS
    tempA = [0] * (Nx * Ny)
    tempA[find_variable_index(row, col)] = aP
    tempA[find_variable_index(row, col+1)] = -aE
    tempA[find_variable_index(row-1, col)] = -aN
    tempA[find_variable_index(row, col-1)] = -aW
    tempA[find_variable_index(row+1, col)] = -aS
    A.append(tempA)
    b.append(0)
    aW = -((-7 * u * Ax / 8) - (D * Ax / grid_space_x))
    aS = -((-7 * v * Ay / 8) - (D * Ay / grid_space_y))

    # 1st row
    row = 0
    aP = aE + aW + aS + aSS + aWW
    for col in range(2, Nx-1):
        tempA = [0] * (Nx * Ny)
        tempA[find_variable_index(row, col)] = aP
        tempA[find_variable_index(row, col+1)] = -aE
        tempA[find_variable_index(row, col-1)] = -aW
        tempA[find_variable_index(row+1, col)] = -aS
        tempA[find_variable_index(row, col-2)] = -aWW
        tempA[find_variable_index(row+2, col)] = -aSS
        A.append(tempA)
        b.append(0)
    
    # last column
    col = Nx-1
    aP = aW + aN + aS + aSS + aWW
    for row in range(1, Ny-2):
        tempA = [0] * (Nx * Ny)
        tempA[find_variable_index(row, col)] = aP
        tempA[find_variable_index(row-1, col)] = -aN
        tempA[find_variable_index(row, col-1)] = -aW
        tempA[find_variable_index(row+1, col)] = -aS
        tempA[find_variable_index(row, col-2)] = -aWW
        tempA[find_variable_index(row+2, col)] = -aSS
        A.append(tempA)
        b.append(0)
    
    # 1st row, 2nd colum corner
    row = 0; col = 1
    aW = (D * Ax / grid_space_x) + (6 * u * Ax / 8)
    aP = aE + aW + aS + aSS
    tempA = [0] * (Nx * Ny)
    tempA[find_variable_index(row, col)] = aP
    tempA[find_variable_index(row, col+1)] = -aE
    tempA[find_variable_index(row, col-1)] = -aW
    tempA[find_variable_index(row+1, col)] = -aS
    tempA[find_variable_index(row+2, col)] = -aSS
    A.append(tempA)
    b.append(0)
    aW = -((-7 * u * Ax / 8) - (D * Ax / grid_space_x))

    # 2nd last row, last column corner
    row = Ny-2; col = Nx-1
    aS = (D * Ay / grid_space_y) + (6 * v * Ay / 8)
    aP = aN + aS + aW + aWW
    tempA = [0] * (Nx * Ny)
    tempA[find_variable_index(row, col)] = aP
    tempA[find_variable_index(row, col-1)] = -aW
    tempA[find_variable_index(row+1, col)] = -aS
    tempA[find_variable_index(row-1, col)] = -aN
    tempA[find_variable_index(row, col-2)] = -aWW
    A.append(tempA)
    b.append(0)
    aS = -((-7 * v * Ay / 8) - (D * Ay / grid_space_y))

    # last row
    row = Ny-1
    aP = aW + aWW + aN + aE
    for col in range(2, Nx-1):
        tempA = [0] * (Nx * Ny)
        tempA[find_variable_index(row, col)] = aP
        tempA[find_variable_index(row, col-1)] = -aW
        tempA[find_variable_index(row-1, col)] = -aN
        tempA[find_variable_index(row, col-2)] = -aWW
        tempA[find_variable_index(row, col+1)] = -aE
        A.append(tempA)
        b.append(0)
    
    # last row, 2nd column corner
    row = Ny-1; col = 1
    aW = (D * Ax / grid_space_x) + (6 * u * Ax / 8)
    aP = aE + aN + aW
    tempA = [0] * (Nx * Ny)
    tempA[find_variable_index(row, col)] = aP
    tempA[find_variable_index(row, col+1)] = -aE
    tempA[find_variable_index(row-1, col)] = -aN
    tempA[find_variable_index(row, col-1)] = -aW
    A.append(tempA)
    b.append(0)
    aW = -((-7 * u * Ax / 8) - (D * Ax / grid_space_x))

    # 1st row, last column corner
    row=0; col=Nx-1
    aP = aW + aS + aSS + aWW
    tempA = [0] * (Nx * Ny)
    tempA[find_variable_index(row, col)] = aP
    tempA[find_variable_index(row, col-1)] = -aW
    tempA[find_variable_index(row+1, col)] = -aS
    tempA[find_variable_index(row, col-2)] = -aWW
    tempA[find_variable_index(row+2, col)] = -aSS
    A.append(tempA)
    b.append(0)

    # last row, last column corner
    row=Ny-1; col=Nx-1
    aP = aW + aN + aWW
    tempA = [0] * (Nx * Ny)
    tempA[find_variable_index(row, col)] = aP
    tempA[find_variable_index(row, col-1)] = -aW
    tempA[find_variable_index(row-1, col)] = -aN
    tempA[find_variable_index(row, col-2)] = -aWW
    A.append(tempA)
    b.append(0)

    # 1st column
    left_boundary_values = left_boundary_condition()
    aE = (D * Ax / grid_space_x) - (3 * u * Ax / 8)
    aN = (D * Ay / grid_space_y) - (3 * v * Ay / 8)
    aS = (D * Ay / grid_space_y) + (7 * v * Ay / 8)
    aSS = (-1 * v * Ay / 8)
    aP = aE + aN + aS + aSS + (10 * u * Ax / 8) + (2 * D * Ax / grid_space_x)
    col = 0
    for row in range(1, Ny-2):
        tempA = [0] * (Nx * Ny)
        tempA[find_variable_index(row, col)] = aP
        tempA[find_variable_index(row, col+1)] = -aE
        tempA[find_variable_index(row-1, col)] = -aN
        tempA[find_variable_index(row+1, col)] = -aS
        tempA[find_variable_index(row+2, col)] = -aSS
        A.append(tempA)
        b.append(10 * u * Ax * left_boundary_values[row] / 8 + 2 * D * Ax * left_boundary_values[row] / grid_space_x)
    
    # 2nd last row, 1st column corner
    row = Ny-2; col=0
    aS = (D * Ay / grid_space_y) + (6 * v * Ay / 8)
    aP = aE + aN + aS + (10 * u * Ax / 8) + (2 * D * Ax / grid_space_x)
    tempA = [0] * (Nx * Ny)
    tempA[find_variable_index(row, col)] = aP
    tempA[find_variable_index(row, col+1)] = -aE
    tempA[find_variable_index(row-1, col)] = -aN
    tempA[find_variable_index(row+1, col)] = -aS
    A.append(tempA)
    b.append(10 * u * Ax * left_boundary_values[row] / 8 + 2 * D * Ax * left_boundary_values[row] / grid_space_x)
    aS = -((-7 * v * Ay / 8) - (D * Ay / grid_space_y))

    # 1st row, 1st column corner
    row=0; col=0
    aP = aE + aS + aSS + (10 * u * Ax / 8) + (2 * D * Ax / grid_space_x)
    tempA = [0] * (Nx * Ny)
    tempA[find_variable_index(row, col)] = aP
    tempA[find_variable_index(row, col+1)] = -aE
    tempA[find_variable_index(row+1, col)] = -aS
    tempA[find_variable_index(row+2, col)] = -aSS
    A.append(tempA)
    b.append(10 * u * Ax * left_boundary_values[row] / 8 + 2 * D * Ax * left_boundary_values[row] / grid_space_x)

    # last row, 1st column corner
    row=Ny-1; col=0
    aP = aE + aN + (10 * u * Ax / 8) + (2 * D * Ax / grid_space_x)
    tempA = [0] * (Nx * Ny)
    tempA[find_variable_index(row, col)] = aP
    tempA[find_variable_index(row, col+1)] = -aE
    tempA[find_variable_index(row-1, col)] = -aN
    A.append(tempA)
    b.append(10 * u * Ax * left_boundary_values[row] / 8 + 2 * D * Ax * left_boundary_values[row] / grid_space_x)

    return np.linalg.solve(np.array(A), np.array(b))

def find_right_boundary():
    xx, yy = grid_points()
    yy = yy[::-1, :]
    left_boundary_values = left_boundary_condition()
    for i in range(10, 1001, 25):
        local_u = i*time_scaling_factor*u
        local_v = i*time_scaling_factor*v
        final_output = quick_scheme(local_u, local_v).reshape(Ny, Nx)
        final_output[-1,-2] = C0
        
        plt.clf()
        plt.contourf(xx, yy, final_output)
        plt.colorbar()
        plt.savefig(f"./quick/concentration_profile_{i}.png")

        plt.clf()
        plt.plot(yy[:, -1], final_output[:, -1], label = "right boundary")
        plt.plot(yy[:, -1], left_boundary_values, label = "left boundary")
        plt.legend()
        plt.grid()
        plt.ylabel("Concentration")
        plt.xlabel("Y")
        plt.title("Concentration Profile on right boundary")
        plt.savefig(f"./quick/right_boundary_{i}.png")

        plt.clf()
        plt.plot(xx[0, :], final_output[0, :], label = "top boundary")
        plt.plot(yy[:, -1], left_boundary_values, label = "left boundary")
        plt.legend()
        plt.grid()
        plt.ylabel("Concentration")
        plt.xlabel("X")
        plt.title("Concentration Profile on top boundary")
        plt.savefig(f"./quick/top_boundary_{i}.png")

if __name__ == "__main__":
    find_right_boundary()