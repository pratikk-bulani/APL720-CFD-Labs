import numpy as np, matplotlib.pyplot as plt

Ly = 1 # Length in y direction
ratio_lengths = 3 / 1
Lx = ratio_lengths * Ly # Length in x direction

Re = 50 # Reynold's number
U_in = 1.0 # Left boundary velocity

Ny = 40 # Number of samples in y direction
Nx = 120 # Number of samples in x direction
# There are (Ny, Nx) P-CVs, (Ny+1, Nx) v-CVs and (Nx+1, Ny) u-CVs

"""
    Please do not change the below global variables
"""
grid_space_x = Lx / Nx # delta_x
grid_space_y = Ly / Ny # delta_y
Ax = grid_space_y # Area x
Ay = grid_space_x # Area y
rho = 1 # Density (from the equations given)
Mu = 1 / Re # Diffusion coefficient (from the equations given)

def initial_guess(Nx, Ny):
    result = []
    for i in range(Ny):
        result.append([0 for j in range(Nx)])
    return result
p_initial_guess = lambda : initial_guess(Nx, Ny)
def u_initial_guess():
    result = initial_guess(Nx+1, Ny)
    for r in result:
        r[0] = U_in
    return result
v_initial_guess = lambda : initial_guess(Nx, Ny+1)

create_grid = lambda N, L: np.linspace(0, L, N)
def u_create_grid():
    x = create_grid(Nx+1, Lx)
    y = create_grid(Ny+1, Ly) + grid_space_y / 2
    y = np.delete(y, -1, 0)
    return x, y
def v_create_grid():
    x = create_grid(Nx+1, Lx) + grid_space_x / 2
    x = np.delete(x, -1, 0)
    y = create_grid(Ny+1, Ly)
    return x, y
def p_create_grid():
    x = create_grid(Nx+1, Lx) + grid_space_x / 2
    x = np.delete(x, -1, 0)
    y = create_grid(Ny+1, Ly) + grid_space_y / 2
    y = np.delete(y, -1, 0)
    return x, y

def fetch_index(r, c, cols):
    assert(r >= 0 and c >= 0 and cols >= 0)
    result = r*cols + c
    assert(result >= 0)
    return result
def calculate_u_star(guess_u, guess_v, guess_p):
    Fe = []
    for J in range(Ny):
        Fe.append([])
        for i in range(1, Nx-1):
            Fe[-1].append((guess_u[J][i+1] + guess_u[J][i]) * rho / 2.0)
        Fe[-1].append(guess_u[J][-2] * rho)
    # print(np.array(Fe).shape)

    Fw = []
    for J in range(Ny):
        Fw.append([])
        for i in range(1, Nx):
            Fw[-1].append((guess_u[J][i] + guess_u[J][i-1]) * rho / 2.0)
    # print(np.array(Fw).shape)
    # print(Fw)

    Fn = []
    for j in range(0, Ny):
        Fn.append([])
        for I in range(1, Nx):
            Fn[-1].append((guess_v[j+1][I] + guess_v[j+1][I-1]) * rho / 2.0)
    # print(np.array(Fn).shape)

    Fs = []
    for j in range(0, Ny):
        Fs.append([])
        for I in range(1, Nx):
            Fs[-1].append((guess_v[j][I] + guess_v[j][I-1]) * rho / 2.0)
    # print(np.array(Fs).shape)

    u_A = []
    for J in range(Ny):
        u_A.append([0] * (Nx-1))

    # Remaining elements
    A = []; b = []
    for J in range(1, Ny-1):
        for i in range(1, Nx-2):
            a_iJ = Fe[J][i] * Ax + Fn[J][i] * Ay + 2 * Mu * Ax / grid_space_x + 2 * Mu * Ay / grid_space_y
            u_A[J][i] = a_iJ
            a_ip1J = Mu * Ax / grid_space_x
            a_im1J = Mu * Ax / grid_space_x + Fw[J][i] * Ax
            a_iJp1 = Mu * Ay / grid_space_y
            a_iJm1 = Mu * Ay / grid_space_y + Fs[J][i] * Ay
            tempA = [0] * (Ny * (Nx-1))
            tempA[fetch_index(J, i, Nx-1)] = a_iJ
            tempA[fetch_index(J, i+1, Nx-1)] = -a_ip1J
            tempA[fetch_index(J, i-1, Nx-1)] = -a_im1J
            tempA[fetch_index(J+1, i, Nx-1)] = -a_iJp1
            tempA[fetch_index(J-1, i, Nx-1)] = -a_iJm1
            A.append(tempA)
            b.append(-(guess_p[J][i+1] - guess_p[J][i]) * Ax)
    # Top row
    for J in [Ny-1]:
        for i in range(1, Nx-2):
            a_iJ = Fe[J][i] * Ax + 2 * Mu * Ax / grid_space_x + 2 * Mu * Ay / grid_space_y
            u_A[J][i] = a_iJ
            a_ip1J = Mu * Ax / grid_space_x
            a_im1J = Mu * Ax / grid_space_x + Fw[J][i] * Ax
            a_iJp1 = 0
            a_iJm1 = Mu * Ay / grid_space_y + Fs[J][i] * Ay
            tempA = [0] * (Ny * (Nx-1))
            tempA[fetch_index(J, i, Nx-1)] = a_iJ
            tempA[fetch_index(J, i+1, Nx-1)] = -a_ip1J
            tempA[fetch_index(J, i-1, Nx-1)] = -a_im1J
            # tempA[fetch_index(J+1, i, Nx-1)] = -a_iJp1
            tempA[fetch_index(J-1, i, Nx-1)] = -a_iJm1
            A.append(tempA)
            b.append(-(guess_p[J][i+1] - guess_p[J][i]) * Ax)
    # Bottom row
    for J in [0]:
        for i in range(1, Nx-2):
            a_iJ = Fe[J][i] * Ax + Fn[J][i] * Ay + 2 * Mu * Ax / grid_space_x + 2 * Mu * Ay / grid_space_y
            u_A[J][i] = a_iJ
            a_ip1J = Mu * Ax / grid_space_x
            a_im1J = Mu * Ax / grid_space_x + Fw[J][i] * Ax
            a_iJp1 = Mu * Ay / grid_space_y
            a_iJm1 = 0
            tempA = [0] * (Ny * (Nx-1))
            tempA[fetch_index(J, i, Nx-1)] = a_iJ
            tempA[fetch_index(J, i+1, Nx-1)] = -a_ip1J
            tempA[fetch_index(J, i-1, Nx-1)] = -a_im1J
            tempA[fetch_index(J+1, i, Nx-1)] = -a_iJp1
            # tempA[fetch_index(J-1, i, Nx-1)] = -a_iJm1
            A.append(tempA)
            b.append(-(guess_p[J][i+1] - guess_p[J][i]) * Ax)
    # left col
    for J in range(1, Ny-1):
        for i in [0]:
            a_iJ = Fe[J][i] * Ax + Fn[J][i] * Ay + 2 * Mu * Ax / grid_space_x + 2 * Mu * Ay / grid_space_y
            u_A[J][i] = a_iJ
            a_ip1J = Mu * Ax / grid_space_x
            a_im1J = Mu * Ax / grid_space_x + Fw[J][i] * Ax
            a_iJp1 = Mu * Ay / grid_space_y
            a_iJm1 = Mu * Ay / grid_space_y + Fs[J][i] * Ay
            tempA = [0] * (Ny * (Nx-1))
            tempA[fetch_index(J, i, Nx-1)] = a_iJ
            tempA[fetch_index(J, i+1, Nx-1)] = -a_ip1J
            tempA[fetch_index(J+1, i, Nx-1)] = -a_iJp1
            tempA[fetch_index(J-1, i, Nx-1)] = -a_iJm1
            A.append(tempA)
            b.append(a_im1J * U_in - (guess_p[J][i+1] - guess_p[J][i]) * Ax)
    # right col
    for J in range(1, Ny-1):
        for i in [Nx-2]:
            a_iJ = Fe[J][i] * Ax + Fn[J][i] * Ay + 2 * Mu * Ax / grid_space_x + 2 * Mu * Ay / grid_space_y
            u_A[J][i] = a_iJ
            a_ip1J = Mu * Ax / grid_space_x
            a_im1J = Mu * Ax / grid_space_x + Fw[J][i] * Ax
            a_iJp1 = Mu * Ay / grid_space_y
            a_iJm1 = Mu * Ay / grid_space_y + Fs[J][i] * Ay
            tempA = [0] * (Ny * (Nx-1))
            tempA[fetch_index(J, i, Nx-1)] = a_iJ - a_ip1J
            tempA[fetch_index(J, i-1, Nx-1)] = -a_im1J
            tempA[fetch_index(J+1, i, Nx-1)] = -a_iJp1
            tempA[fetch_index(J-1, i, Nx-1)] = -a_iJm1
            A.append(tempA)
            b.append(-(guess_p[J][i+1] - guess_p[J][i]) * Ax)
    # top left corner
    for J in [Ny-1]:
        for i in [0]:
            a_iJ = Fe[J][i] * Ax + 2 * Mu * Ax / grid_space_x + 2 * Mu * Ay / grid_space_y
            u_A[J][i] = a_iJ
            a_ip1J = Mu * Ax / grid_space_x
            a_im1J = Mu * Ax / grid_space_x + Fw[J][i] * Ax
            a_iJp1 = 0
            a_iJm1 = Mu * Ay / grid_space_y + Fs[J][i] * Ay
            tempA = [0] * (Ny * (Nx-1))
            tempA[fetch_index(J, i, Nx-1)] = a_iJ
            tempA[fetch_index(J, i+1, Nx-1)] = -a_ip1J
            # tempA[fetch_index(J+1, i, Nx-1)] = -a_iJp1
            tempA[fetch_index(J-1, i, Nx-1)] = -a_iJm1
            A.append(tempA)
            b.append(a_im1J * U_in - (guess_p[J][i+1] - guess_p[J][i]) * Ax)
    # top right corner
    for J in [Ny-1]:
        for i in [Nx-2]:
            a_iJ = Fe[J][i] * Ax + 2 * Mu * Ax / grid_space_x + 2 * Mu * Ay / grid_space_y
            u_A[J][i] = a_iJ
            a_ip1J = Mu * Ax / grid_space_x
            a_im1J = Mu * Ax / grid_space_x + Fw[J][i] * Ax
            a_iJp1 = 0
            a_iJm1 = Mu * Ay / grid_space_y + Fs[J][i] * Ay
            tempA = [0] * (Ny * (Nx-1))
            tempA[fetch_index(J, i, Nx-1)] = a_iJ - a_ip1J
            tempA[fetch_index(J, i-1, Nx-1)] = -a_im1J
            # tempA[fetch_index(J+1, i, Nx-1)] = -a_iJp1
            tempA[fetch_index(J-1, i, Nx-1)] = -a_iJm1
            A.append(tempA)
            b.append(-(guess_p[J][i+1] - guess_p[J][i]) * Ax)
    # bottom left corner
    for J in [0]:
        for i in [0]:
            a_iJ = Fe[J][i] * Ax + Fn[J][i] * Ay + 2 * Mu * Ax / grid_space_x + 2 * Mu * Ay / grid_space_y
            u_A[J][i] = a_iJ
            a_ip1J = Mu * Ax / grid_space_x
            a_im1J = Mu * Ax / grid_space_x + Fw[J][i] * Ax
            a_iJp1 = Mu * Ay / grid_space_y
            a_iJm1 = 0
            tempA = [0] * (Ny * (Nx-1))
            tempA[fetch_index(J, i, Nx-1)] = a_iJ
            tempA[fetch_index(J, i+1, Nx-1)] = -a_ip1J
            tempA[fetch_index(J+1, i, Nx-1)] = -a_iJp1
            # tempA[fetch_index(J-1, i, Nx-1)] = -a_iJm1
            A.append(tempA)
            b.append(a_im1J * U_in - (guess_p[J][i+1] - guess_p[J][i]) * Ax)
    # bottom right corner
    for J in [0]:
        for i in [Nx-2]:
            a_iJ = Fe[J][i] * Ax + Fn[J][i] * Ay + 2 * Mu * Ax / grid_space_x + 2 * Mu * Ay / grid_space_y
            u_A[J][i] = a_iJ
            a_ip1J = Mu * Ax / grid_space_x
            a_im1J = Mu * Ax / grid_space_x + Fw[J][i] * Ax
            a_iJp1 = Mu * Ay / grid_space_y
            a_iJm1 = 0
            tempA = [0] * (Ny * (Nx-1))
            tempA[fetch_index(J, i, Nx-1)] = a_iJ - a_ip1J
            tempA[fetch_index(J, i-1, Nx-1)] = -a_im1J
            tempA[fetch_index(J+1, i, Nx-1)] = -a_iJp1
            # tempA[fetch_index(J-1, i, Nx-1)] = -a_iJm1
            A.append(tempA)
            b.append(-(guess_p[J][i+1] - guess_p[J][i]) * Ax)
    # print(np.array(A).shape, np.array(b).shape)

    u_star = np.linalg.solve(np.array(A), np.array(b)).reshape(Ny, -1)

    # Rectifying the u_A
    for J in range(Ny):
        Fe = (guess_u[J][0] + guess_u[J][1]) * rho / 2.0
        Fn = (guess_v[J+1][0]) * rho / 2.0
        u_A[J].insert(0, Fe*Ax + Fn*Ay + 2*Mu*Ax/grid_space_x + 2*Mu*Ay/grid_space_y)
        Fe = (guess_u[J][-1]) * rho / 2.0
        Fn = (guess_v[J+1][-1] * rho)
        u_A[J].append(Fe*Ax + Fn*Ay + 2*Mu*Ax/grid_space_x + 2*Mu*Ay/grid_space_y)
    
    # Rectifying the u_star
    u_star = np.concatenate((np.array([U_in for i in range(Ny)]).reshape(-1, 1), u_star, u_star[:, [-1]]), axis = 1)

    return u_star, u_A
def calculate_v_star(guess_u, guess_v, guess_p):
    Fn = []
    for j in range(1, Ny-1):
        Fn.append([])
        for I in range(Nx):
            Fn[-1].append((guess_v[j+1][I] + guess_v[j][I]) * rho / 2.0)
    Fn.append([guess_v[Ny-1][I] * rho / 2.0 for I in range(Nx)])
    # print(np.array(Fn).shape)

    Fs = []
    for j in range(2, Ny):
        Fs.append([])
        for I in range(Nx):
            Fs[-1].append((guess_v[j][I] + guess_v[j-1][I]) * rho / 2.0)
    Fs.insert(0, [guess_v[1][I] * rho / 2.0 for I in range(Nx)])
    # print(np.array(Fs).shape)

    Fe = []
    for J in range(1, Ny):
        Fe.append([])
        for i in range(0, Nx):
            Fe[-1].append((guess_u[J][i+1] + guess_u[J-1][i+1]) * rho / 2.0)
    # print(np.array(Fe).shape)

    Fw = []
    for J in range(1, Ny):
        Fw.append([])
        for i in range(0, Nx):
            Fw[-1].append((guess_u[J][i] + guess_u[J-1][i]) * rho / 2.0)
    # print(np.array(Fw).shape)

    v_A = []
    for j in range(Ny-1):
        v_A.append([0] * Nx)

    # Remaining elements
    A = []; b = []
    for j in range(1, Ny-2):
        for I in range(1, Nx-1):
            a_Ij = Fe[j][I] * Ax + Fn[j][I] * Ay + 2 * Mu * Ax / grid_space_x + 2 * Mu * Ay / grid_space_y
            v_A[j][I] = a_Ij
            a_Ip1j = Mu * Ax / grid_space_x
            a_Im1j = Mu * Ax / grid_space_x + Fw[j][I] * Ax
            a_Ijp1 = Mu * Ay / grid_space_y
            a_Ijm1 = Mu * Ay / grid_space_y + Fs[j][I] * Ay
            tempA = [0] * ((Ny-1) * Nx)
            tempA[fetch_index(j, I, Nx)] = a_Ij
            tempA[fetch_index(j, I+1, Nx)] = -a_Ip1j
            tempA[fetch_index(j, I-1, Nx)] = -a_Im1j
            tempA[fetch_index(j+1, I, Nx)] = -a_Ijp1
            tempA[fetch_index(j-1, I, Nx)] = -a_Ijm1
            A.append(tempA)
            b.append(-(guess_p[j+1][I] - guess_p[j][I]) * Ay)
    # top row
    for j in [Ny-2]:
        for I in range(1, Nx-1):
            a_Ij = Fe[j][I] * Ax + Fn[j][I] * Ay + 2 * Mu * Ax / grid_space_x + 2 * Mu * Ay / grid_space_y
            v_A[j][I] = a_Ij
            a_Ip1j = Mu * Ax / grid_space_x
            a_Im1j = Mu * Ax / grid_space_x + Fw[j][I] * Ax
            a_Ijp1 = 0
            a_Ijm1 = Mu * Ay / grid_space_y + Fs[j][I] * Ay
            tempA = [0] * ((Ny-1) * Nx)
            tempA[fetch_index(j, I, Nx)] = a_Ij
            tempA[fetch_index(j, I+1, Nx)] = -a_Ip1j
            tempA[fetch_index(j, I-1, Nx)] = -a_Im1j
            # tempA[fetch_index(j+1, I, Nx)] = -a_Ijp1
            tempA[fetch_index(j-1, I, Nx)] = -a_Ijm1
            A.append(tempA)
            b.append(-(guess_p[j+1][I] - guess_p[j][I]) * Ay)
    # bottom row
    for j in [0]:
        for I in range(1, Nx-1):
            a_Ij = Fe[j][I] * Ax + Fn[j][I] * Ay + 2 * Mu * Ax / grid_space_x + 2 * Mu * Ay / grid_space_y
            v_A[j][I] = a_Ij
            a_Ip1j = Mu * Ax / grid_space_x
            a_Im1j = Mu * Ax / grid_space_x + Fw[j][I] * Ax
            a_Ijp1 = Mu * Ay / grid_space_y
            a_Ijm1 = 0
            tempA = [0] * ((Ny-1) * Nx)
            tempA[fetch_index(j, I, Nx)] = a_Ij
            tempA[fetch_index(j, I+1, Nx)] = -a_Ip1j
            tempA[fetch_index(j, I-1, Nx)] = -a_Im1j
            tempA[fetch_index(j+1, I, Nx)] = -a_Ijp1
            # tempA[fetch_index(j-1, I, Nx)] = -a_Ijm1
            A.append(tempA)
            b.append(-(guess_p[j+1][I] - guess_p[j][I]) * Ay)
    # left col
    for j in range(1, Ny-2):
        for I in [0]:
            a_Ij = Fe[j][I] * Ax + Fn[j][I] * Ay + 2 * Mu * Ax / grid_space_x + 2 * Mu * Ay / grid_space_y
            v_A[j][I] = a_Ij
            a_Ip1j = Mu * Ax / grid_space_x
            a_Im1j = 0
            a_Ijp1 = Mu * Ay / grid_space_y
            a_Ijm1 = Mu * Ay / grid_space_y + Fs[j][I] * Ay
            tempA = [0] * ((Ny-1) * Nx)
            tempA[fetch_index(j, I, Nx)] = a_Ij
            tempA[fetch_index(j, I+1, Nx)] = -a_Ip1j
            # tempA[fetch_index(j, I-1, Nx)] = -a_Im1j
            tempA[fetch_index(j+1, I, Nx)] = -a_Ijp1
            tempA[fetch_index(j-1, I, Nx)] = -a_Ijm1
            A.append(tempA)
            b.append(-(guess_p[j+1][I] - guess_p[j][I]) * Ay)
    # right col
    for j in range(1, Ny-2):
        for I in [Nx-1]:
            a_Ij = Fe[j][I] * Ax + Fn[j][I] * Ay + 2 * Mu * Ax / grid_space_x + 2 * Mu * Ay / grid_space_y
            v_A[j][I] = a_Ij
            a_Ip1j = Mu * Ax / grid_space_x
            a_Im1j = Mu * Ax / grid_space_x + Fw[j][I] * Ax
            a_Ijp1 = Mu * Ay / grid_space_y
            a_Ijm1 = Mu * Ay / grid_space_y + Fs[j][I] * Ay
            tempA = [0] * ((Ny-1) * Nx)
            tempA[fetch_index(j, I, Nx)] = a_Ij - a_Ip1j
            tempA[fetch_index(j, I-1, Nx)] = -a_Im1j
            tempA[fetch_index(j+1, I, Nx)] = -a_Ijp1
            tempA[fetch_index(j-1, I, Nx)] = -a_Ijm1
            A.append(tempA)
            b.append(-(guess_p[j+1][I] - guess_p[j][I]) * Ay)
    # top left corner
    for j in [Ny-2]:
        for I in [0]:
            a_Ij = Fe[j][I] * Ax + Fn[j][I] * Ay + 2 * Mu * Ax / grid_space_x + 2 * Mu * Ay / grid_space_y
            v_A[j][I] = a_Ij
            a_Ip1j = Mu * Ax / grid_space_x
            a_Im1j = 0
            a_Ijp1 = 0
            a_Ijm1 = Mu * Ay / grid_space_y + Fs[j][I] * Ay
            tempA = [0] * ((Ny-1) * Nx)
            tempA[fetch_index(j, I, Nx)] = a_Ij
            tempA[fetch_index(j, I+1, Nx)] = -a_Ip1j
            # tempA[fetch_index(j, I-1, Nx)] = -a_Im1j
            # tempA[fetch_index(j+1, I, Nx)] = -a_Ijp1
            tempA[fetch_index(j-1, I, Nx)] = -a_Ijm1
            A.append(tempA)
            b.append(-(guess_p[j+1][I] - guess_p[j][I]) * Ay)
    # top right corner
    for j in [Ny-2]:
        for I in [Nx-1]:
            a_Ij = Fe[j][I] * Ax + Fn[j][I] * Ay + 2 * Mu * Ax / grid_space_x + 2 * Mu * Ay / grid_space_y
            v_A[j][I] = a_Ij
            a_Ip1j = Mu * Ax / grid_space_x
            a_Im1j = Mu * Ax / grid_space_x + Fw[j][I] * Ax
            a_Ijp1 = 0
            a_Ijm1 = Mu * Ay / grid_space_y + Fs[j][I] * Ay
            tempA = [0] * ((Ny-1) * Nx)
            tempA[fetch_index(j, I, Nx)] = a_Ij - a_Ip1j
            tempA[fetch_index(j, I-1, Nx)] = -a_Im1j
            # tempA[fetch_index(j+1, I, Nx)] = -a_Ijp1
            tempA[fetch_index(j-1, I, Nx)] = -a_Ijm1
            A.append(tempA)
            b.append(-(guess_p[j+1][I] - guess_p[j][I]) * Ay)
    # bottom left corner
    for j in [0]:
        for I in [0]:
            a_Ij = Fe[j][I] * Ax + Fn[j][I] * Ay + 2 * Mu * Ax / grid_space_x + 2 * Mu * Ay / grid_space_y
            v_A[j][I] = a_Ij
            a_Ip1j = Mu * Ax / grid_space_x
            a_Im1j = 0
            a_Ijp1 = Mu * Ay / grid_space_y
            a_Ijm1 = 0
            tempA = [0] * ((Ny-1) * Nx)
            tempA[fetch_index(j, I, Nx)] = a_Ij
            tempA[fetch_index(j, I+1, Nx)] = -a_Ip1j
            # tempA[fetch_index(j, I-1, Nx)] = -a_Im1j
            tempA[fetch_index(j+1, I, Nx)] = -a_Ijp1
            # tempA[fetch_index(j-1, I, Nx)] = -a_Ijm1
            A.append(tempA)
            b.append(-(guess_p[j+1][I] - guess_p[j][I]) * Ay)
    # bottom right corner
    for j in [0]:
        for I in [Nx-1]:
            a_Ij = Fe[j][I] * Ax + Fn[j][I] * Ay + 2 * Mu * Ax / grid_space_x + 2 * Mu * Ay / grid_space_y
            v_A[j][I] = a_Ij
            a_Ip1j = Mu * Ax / grid_space_x
            a_Im1j = Mu * Ax / grid_space_x + Fw[j][I] * Ax
            a_Ijp1 = Mu * Ay / grid_space_y
            a_Ijm1 = 0
            tempA = [0] * ((Ny-1) * Nx)
            tempA[fetch_index(j, I, Nx)] = a_Ij - a_Ip1j
            tempA[fetch_index(j, I-1, Nx)] = -a_Im1j
            tempA[fetch_index(j+1, I, Nx)] = -a_Ijp1
            # tempA[fetch_index(j-1, I, Nx)] = -a_Ijm1
            A.append(tempA)
            b.append(-(guess_p[j+1][I] - guess_p[j][I]) * Ay)
    # print(np.array(A).shape, np.array(b).shape)

    v_star = np.linalg.solve(np.array(A), np.array(b)).reshape(-1, Nx)

    # Rectifying the v_A
    v_A.append([1]*Nx)
    v_A.insert(0, [1]*Nx)
    for I in range(Nx):
        Fe = (guess_u[0][I+1]) * rho / 2.0
        Fn = (guess_v[0][I] + guess_v[1][I]) * rho / 2.0
        v_A[0][I] = Fe * Ax + Fn * Ay + 2 * Mu * Ax / grid_space_x + 2 * Mu * Ay / grid_space_y
        Fe = (guess_u[-1][I+1]) * rho / 2.0
        Fn = (guess_v[-1][I]) * rho / 2.0
        v_A[-1][I] = Fe * Ax + Fn * Ay + 2 * Mu * Ax / grid_space_x + 2 * Mu * Ay / grid_space_y
    # print(np.array(v_A).shape)

    # Rectifying the v_star
    v_star = np.concatenate((np.array([0]*Nx).reshape(1, -1), v_star, np.array([0]*Nx).reshape(1, -1)), axis = 0)

    return v_star, v_A
def calculate_p_dash(u_star, v_star, u_A, v_A):
    A = []; b = []
    # Remaining elements
    for J in range(1, Ny-1):
        for I in range(1, Nx-1):
            a_IJ = Ax * Ax / u_A[J][I+1] + Ax * Ax / u_A[J][I] + Ay * Ay / v_A[J+1][I] + Ay * Ay / v_A[J][I]
            a_Ip1J = Ax * Ax / u_A[J][I+1]
            a_Im1J = Ax * Ax / u_A[J][I]
            a_IJp1 = Ay * Ay / v_A[J+1][I]
            a_IJm1 = Ay * Ay / v_A[J][I]
            tempA = [0] * (Nx * Ny)
            tempA[fetch_index(J, I, Nx)] = a_IJ
            tempA[fetch_index(J, I+1, Nx)] = -a_Ip1J
            tempA[fetch_index(J, I-1, Nx)] = -a_Im1J
            tempA[fetch_index(J+1, I, Nx)] = -a_IJp1
            tempA[fetch_index(J-1, I, Nx)] = -a_IJm1
            A.append(tempA)
            b.append((u_star[J][I] - u_star[J][I+1]) * Ax + (v_star[J][I] - v_star[J+1][I]) * Ay)
    # Top row
    for J in [Ny-1]:
        for I in range(1, Nx-1):
            a_IJ = Ax * Ax / u_A[J][I+1] + Ax * Ax / u_A[J][I] + Ay * Ay / v_A[J+1][I] + Ay * Ay / v_A[J][I]
            a_Ip1J = Ax * Ax / u_A[J][I+1]
            a_Im1J = Ax * Ax / u_A[J][I]
            a_IJp1 = 0
            a_IJm1 = Ay * Ay / v_A[J][I]
            tempA = [0] * (Nx * Ny)
            tempA[fetch_index(J, I, Nx)] = a_IJ
            tempA[fetch_index(J, I+1, Nx)] = -a_Ip1J
            tempA[fetch_index(J, I-1, Nx)] = -a_Im1J
            # tempA[fetch_index(J+1, I, Nx)] = -a_IJp1
            tempA[fetch_index(J-1, I, Nx)] = -a_IJm1
            A.append(tempA)
            b.append((u_star[J][I] - u_star[J][I+1]) * Ax + (v_star[J][I] - v_star[J+1][I]) * Ay)
    # Bottom row
    for J in [0]:
        for I in range(1, Nx-1):
            a_IJ = Ax * Ax / u_A[J][I+1] + Ax * Ax / u_A[J][I] + Ay * Ay / v_A[J+1][I] + Ay * Ay / v_A[J][I]
            a_Ip1J = Ax * Ax / u_A[J][I+1]
            a_Im1J = Ax * Ax / u_A[J][I]
            a_IJp1 = Ay * Ay / v_A[J+1][I]
            a_IJm1 = 0
            tempA = [0] * (Nx * Ny)
            tempA[fetch_index(J, I, Nx)] = a_IJ
            tempA[fetch_index(J, I+1, Nx)] = -a_Ip1J
            tempA[fetch_index(J, I-1, Nx)] = -a_Im1J
            tempA[fetch_index(J+1, I, Nx)] = -a_IJp1
            # tempA[fetch_index(J-1, I, Nx)] = -a_IJm1
            A.append(tempA)
            b.append((u_star[J][I] - u_star[J][I+1]) * Ax + (v_star[J][I] - v_star[J+1][I]) * Ay)
    # Left col
    for J in range(1, Ny-1):
        for I in [0]:
            a_IJ = Ax * Ax / u_A[J][I+1] + Ax * Ax / u_A[J][I] + Ay * Ay / v_A[J+1][I] + Ay * Ay / v_A[J][I]
            a_Ip1J = Ax * Ax / u_A[J][I+1]
            a_Im1J = 0
            a_IJp1 = Ay * Ay / v_A[J+1][I]
            a_IJm1 = Ay * Ay / v_A[J][I]
            tempA = [0] * (Nx * Ny)
            tempA[fetch_index(J, I, Nx)] = a_IJ
            tempA[fetch_index(J, I+1, Nx)] = -a_Ip1J
            # tempA[fetch_index(J, I-1, Nx)] = -a_Im1J
            tempA[fetch_index(J+1, I, Nx)] = -a_IJp1
            tempA[fetch_index(J-1, I, Nx)] = -a_IJm1
            A.append(tempA)
            b.append((u_star[J][I] - u_star[J][I+1]) * Ax + (v_star[J][I] - v_star[J+1][I]) * Ay)
    # Right col
    for J in range(1, Ny-1):
        for I in [Nx-1]:
            a_IJ = Ax * Ax / u_A[J][I+1] + Ax * Ax / u_A[J][I] + Ay * Ay / v_A[J+1][I] + Ay * Ay / v_A[J][I]
            a_Ip1J = 0
            a_Im1J = Ax * Ax / u_A[J][I]
            a_IJp1 = Ay * Ay / v_A[J+1][I]
            a_IJm1 = Ay * Ay / v_A[J][I]
            tempA = [0] * (Nx * Ny)
            tempA[fetch_index(J, I, Nx)] = a_IJ
            # tempA[fetch_index(J, I+1, Nx)] = -a_Ip1J
            tempA[fetch_index(J, I-1, Nx)] = -a_Im1J
            tempA[fetch_index(J+1, I, Nx)] = -a_IJp1
            tempA[fetch_index(J-1, I, Nx)] = -a_IJm1
            A.append(tempA)
            b.append((u_star[J][I] - u_star[J][I+1]) * Ax + (v_star[J][I] - v_star[J+1][I]) * Ay)
    # for J in range(1, Ny-1):
    #     for I in [Nx-1]:
    #         a_IJ = 1
    #         tempA = [0] * (Nx * Ny)
    #         tempA[fetch_index(J, I, Nx)] = a_IJ
    #         A.append(tempA)
    #         b.append(0)
    # Top left corner
    for J in [Ny-1]:
        for I in [0]:
            a_IJ = Ax * Ax / u_A[J][I+1] + Ax * Ax / u_A[J][I] + Ay * Ay / v_A[J+1][I] + Ay * Ay / v_A[J][I]
            a_Ip1J = Ax * Ax / u_A[J][I+1]
            a_Im1J = 0
            a_IJp1 = 0
            a_IJm1 = Ay * Ay / v_A[J][I]
            tempA = [0] * (Nx * Ny)
            tempA[fetch_index(J, I, Nx)] = a_IJ
            tempA[fetch_index(J, I+1, Nx)] = -a_Ip1J
            # tempA[fetch_index(J, I-1, Nx)] = -a_Im1J
            # tempA[fetch_index(J+1, I, Nx)] = -a_IJp1
            tempA[fetch_index(J-1, I, Nx)] = -a_IJm1
            A.append(tempA)
            b.append((u_star[J][I] - u_star[J][I+1]) * Ax + (v_star[J][I] - v_star[J+1][I]) * Ay)
    # Top right corner
    for J in [Ny-1]:
        for I in [Nx-1]:
            a_IJ = Ax * Ax / u_A[J][I+1] + Ax * Ax / u_A[J][I] + Ay * Ay / v_A[J+1][I] + Ay * Ay / v_A[J][I]
            a_Ip1J = 0
            a_Im1J = Ax * Ax / u_A[J][I]
            a_IJp1 = 0
            a_IJm1 = Ay * Ay / v_A[J][I]
            tempA = [0] * (Nx * Ny)
            tempA[fetch_index(J, I, Nx)] = a_IJ
            # tempA[fetch_index(J, I+1, Nx)] = -a_Ip1J
            tempA[fetch_index(J, I-1, Nx)] = -a_Im1J
            # tempA[fetch_index(J+1, I, Nx)] = -a_IJp1
            tempA[fetch_index(J-1, I, Nx)] = -a_IJm1
            A.append(tempA)
            b.append((u_star[J][I] - u_star[J][I+1]) * Ax + (v_star[J][I] - v_star[J+1][I]) * Ay)
    # for J in [Ny-1]:
    #     for I in [Nx-1]:
    #         a_IJ = 1
    #         tempA = [0] * (Nx * Ny)
    #         tempA[fetch_index(J, I, Nx)] = a_IJ
    #         A.append(tempA)
    #         b.append(0)
    # Bottom left corner
    for J in [0]:
        for I in [0]:
            a_IJ = Ax * Ax / u_A[J][I+1] + Ax * Ax / u_A[J][I] + Ay * Ay / v_A[J+1][I] + Ay * Ay / v_A[J][I]
            a_Ip1J = Ax * Ax / u_A[J][I+1]
            a_Im1J = 0
            a_IJp1 = Ay * Ay / v_A[J+1][I]
            a_IJm1 = 0
            tempA = [0] * (Nx * Ny)
            tempA[fetch_index(J, I, Nx)] = a_IJ
            tempA[fetch_index(J, I+1, Nx)] = -a_Ip1J
            # tempA[fetch_index(J, I-1, Nx)] = -a_Im1J
            tempA[fetch_index(J+1, I, Nx)] = -a_IJp1
            # tempA[fetch_index(J-1, I, Nx)] = -a_IJm1
            A.append(tempA)
            b.append((u_star[J][I] - u_star[J][I+1]) * Ax + (v_star[J][I] - v_star[J+1][I]) * Ay)
    # Bottom right corner
    for J in [0]:
        for I in [Nx-1]:
            a_IJ = Ax * Ax / u_A[J][I+1] + Ax * Ax / u_A[J][I] + Ay * Ay / v_A[J+1][I] + Ay * Ay / v_A[J][I]
            a_Ip1J = 0
            a_Im1J = Ax * Ax / u_A[J][I]
            a_IJp1 = Ay * Ay / v_A[J+1][I]
            a_IJm1 = 0
            tempA = [0] * (Nx * Ny)
            tempA[fetch_index(J, I, Nx)] = a_IJ
            # tempA[fetch_index(J, I+1, Nx)] = -a_Ip1J
            tempA[fetch_index(J, I-1, Nx)] = -a_Im1J
            tempA[fetch_index(J+1, I, Nx)] = -a_IJp1
            # tempA[fetch_index(J-1, I, Nx)] = -a_IJm1
            A.append(tempA)
            b.append((u_star[J][I] - u_star[J][I+1]) * Ax + (v_star[J][I] - v_star[J+1][I]) * Ay)
    # for J in [0]:
    #     for I in [Nx-1]:
    #         a_IJ = 1
    #         tempA = [0] * (Nx * Ny)
    #         tempA[fetch_index(J, I, Nx)] = a_IJ
    #         A.append(tempA)
    #         b.append(0)
    # print(np.array(A).shape, np.array(b).shape)

    return np.linalg.solve(np.array(A), np.array(b)).reshape(-1, Nx)
def calculate_u(u_star, p_dash, u_A):
    # print(u_star.shape, np.array(u_A).shape)
    u_new = []
    for J in range(Ny):
        u_new.append([])
        for I in range(1, Nx):
            u_new[-1].append(u_star[J][I] - (p_dash[J][I] - p_dash[J][I-1]) * Ax / u_A[J][I])
    for k in u_new:
        k.append(k[-1])
        k.insert(0, U_in)
    return u_new
def calculate_v(v_star, p_dash, v_A):
    # print(v_star.shape, np.array(v_A).shape)
    v_new = []
    for J in range(1, Ny):
        v_new.append([])
        for I in range(Nx):
            v_new[-1].append(v_star[J][I] - (p_dash[J][I] - p_dash[J-1][I]) * Ay / v_A[J][I])
    v_new.append([0]*Nx)
    v_new.insert(0, [0]*Nx)
    return v_new

def simple_iteration():
    guess_u = u_initial_guess()
    # print(np.array(guess_u).shape)
    guess_v = v_initial_guess()
    # print(np.array(guess_v).shape)
    guess_p = p_initial_guess()
    # print(np.array(guess_p).shape)
    i = 0
    residue = []
    while(True):
        u_star, u_A = calculate_u_star(guess_u, guess_v, guess_p)
        # print(u_star.shape, np.array(u_A).shape)
        v_star, v_A = calculate_v_star(guess_u, guess_v, guess_p)
        # print(v_star.shape, np.array(v_A).shape)
        p_dash = calculate_p_dash(u_star, v_star, u_A, v_A)
        # print(p_dash.shape)
        p_new = (np.array(guess_p) + p_dash).tolist()
        # print(np.array(p_new).shape, type(p_new), type(p_new[0]))
        u_new = calculate_u(u_star, p_dash, u_A)
        # print(np.array(u_new).shape, type(u_new))
        v_new = calculate_v(v_star, p_dash, v_A)
        # print(np.array(v_new).shape, type(v_new))
        guess_u = u_new; guess_v = v_new; guess_p = p_new
        i += 1
        if(True):
            u_x, u_y = u_create_grid()
            v_x, v_y = v_create_grid()
            p_x, p_y = p_create_grid()

            plt.clf()
            plt.contourf(*np.meshgrid(u_x, u_y), np.array(u_new))
            plt.colorbar()
            plt.savefig(f"./outputs/u_{i}.png")

            # plt.clf()
            # plt.contourf(*np.meshgrid(v_x, v_y), v_new)
            # plt.colorbar()
            # plt.savefig(f"./outputs/v_{i}.png")

            plt.clf()
            plt.contourf(*np.meshgrid(p_x, p_y), p_new)
            plt.colorbar()
            plt.savefig(f"./outputs/p_{i}.png")

            plt.clf()
            for N in [Nx//4, Nx//2, Nx]:
                plt.plot(u_y, np.array(u_new)[:, N], label=f"{N}")
            plt.legend()
            plt.grid()
            plt.xlabel("Y-length")
            plt.ylabel("U")
            plt.savefig(f"./outputs/different_cols_{i}.png")
            
        residue.append(np.linalg.norm(p_dash, ord="fro"))
        if(i == 3): break
    plt.clf()
    plt.plot([i for i in range(1, len(residue)+1)], residue)
    plt.xlabel("#iteration")
    plt.ylabel("Residue")
    plt.grid()
    plt.savefig(f"./outputs/residue.png")

if __name__ == "__main__":
    simple_iteration()