import numpy as np, sys, matplotlib.pyplot as plt

# Global variables
L = 1 # rod length in meters

def finite_difference_method(N, q, k, T_0, T_L, h, partial_result = False):
    A = np.diag([-2.0] * N, 0) + np.diag([1.0] * (N-1), 1) + np.diag([1.0] * (N-1), -1)
    
    b = (-q*h*h/k) * np.ones(shape = (N, 1))
    if(partial_result):
        return A, b
    b[0, 0] -= T_0
    b[N-1, 0] -= T_L

    temps = np.linalg.inv(A) @ b
    return temps

def closed_form_solution(N, q, k, T_0, T_L, h):
    C_2 = T_0
    C_0 = -q/(2*k)
    C_1 = (T_L - C_0*L*L - C_2) / L
    
    temps = np.arange(1, N+0.1, 1)
    temps = (C_0 * ((temps*h) ** 2) + (C_1*h) * temps + C_2).reshape(N, 1)
    return temps

def experiment_1(N, q, k, T_0, T_L):
    h = L / (N+1) # grid-space size
    fdm_temps = finite_difference_method(N, q, k, T_0, T_L, h)
    cfs_temps = closed_form_solution(N, q, k, T_0, T_L, h)

    # Graph plot
    plt.plot([i*h for i in range(N+2)], [T_0] + fdm_temps.flatten().tolist() + [T_L], label = "FDM")
    plt.plot([i*h for i in range(N+2)], [T_0] + cfs_temps.flatten().tolist() + [T_L], linestyle="dashed", label = "CFS")
    plt.xlabel("Rod distance (x)")
    plt.ylabel("Temperature (T)")
    plt.title("Finite Difference Method (FDM) vs Closed Form Solution (CFS)")
    plt.legend()
    plt.grid()
    plt.show()

def experiment_2(q, k, T_0, T_L):
    Ns = [11, 21, 41, 81]
    rmse_errors = []
    log_Hs = []
    for N in Ns:
        h = L / (N+1) # grid-space size
        fdm_temps = finite_difference_method(N, q, k, T_0, T_L, h)
        cfs_temps = closed_form_solution(N, q, k, T_0, T_L, h)
        rmse_errors.append(np.log10(((fdm_temps - cfs_temps) ** 2).mean() ** 0.5))
        log_Hs.append(np.log10(h))
    
    # Graph plot
    plt.plot(log_Hs, rmse_errors, marker='o')
    plt.xlabel("log(delta_x)")
    plt.ylabel("log(RMSE)")
    plt.grid()
    plt.show()
    """
        The slope of this line is 2. So, the order of accuracy is 2.
    """

def experiment_3(N, q, k, T_0):
    h = L / (N+1) # grid-space size
    A, b = finite_difference_method(N, q, k, T_0, None, h, partial_result = True)
    b[N-1, 0] -= T_L
    b[0, 0] += T_0
    A[0, 0] = 0.0
    """
        dT/dx = 0 at x=L implies T_N = T_L.
        Proof: Applying 1st order accurate backward difference formula. This gives, T_i = T_(i-1).
        Code changes: The last linear equation got changed. Keeping T_L = T_N in the final linear equation and not subtracting -T_L from the vector b.
    """
    fdm_temps = np.linalg.inv(A) @ b

    # Graph plot
    plt.plot([i*h for i in range(N+2)], [T_0] + fdm_temps.flatten().tolist() + [fdm_temps[N-1][0]], label = "FDM")
    plt.xlabel("Rod distance (x)")
    plt.ylabel("Temperature (T)")
    plt.title("Finite Difference Method (FDM) (dT/dx=0)|x=L")
    plt.legend()
    plt.grid()
    plt.show()

if __name__ == "__main__":
    if(len(sys.argv) != 6):
        raise Exception("Invalid arguments passed. Run like this: python3 main.py <N> <q> <k> <T_0> <T_L>")
    N = int(sys.argv[1]) # number of samples
    q = float(sys.argv[2]) # uniform heat generation rate
    k = float(sys.argv[3]) # thermal conductivity of the material
    T_0 = float(sys.argv[4]) # Boundary x=0 value
    T_L = float(sys.argv[5]) # Boundary x=L value
    experiment_1(N, q, k, T_0, T_L)
    experiment_2(q, k, T_0, T_L)
    experiment_3(N, q, k, T_0)
# Run command: python3 main.py 100 2 1 0.64 0.84
