import numpy as np, matplotlib.pyplot as plt, sys, math

# Global Variables
L = 1 # rod length in meters
alpha = 1.5 # thermal diffusivity of the material

def initial_condition(N, T_0, T_L, h):
    temps_0 = np.arange(0, N+2, 1, dtype=np.float64)
    temps_0 = T_0 * ((1 - temps_0 * (h / L)) ** 2)
    temps_0[-1] = T_L
    return temps_0

def explicit_euler(N, t_f, T_0, T_L, h, t, record_halfway_temps = False):
    temps_0 = initial_condition(N, T_0, T_L, h)
    if record_halfway_temps:
        halfway_index = math.ceil(N/2)
        halfway_temps = [temps_0[halfway_index]]
        halfway_times = [0.]
    constant_used = alpha * t / (h * h) # should be F_o for experiment 1

    def update_step(index):
        if index == 0 or index == N+1:
            return temps_0[index]
        else:
            return constant_used * temps_0[index+1] + (1 - 2*constant_used) * temps_0[index] + constant_used * temps_0[index-1]
    
    time_step = t
    while(time_step <= t_f):
        temps_1 = np.arange(0, N+2, 1, dtype=np.int32)
        temps_1 = np.vectorize(update_step)(temps_1)
        del temps_0
        temps_0 = temps_1
        if record_halfway_temps:
            halfway_temps.append(temps_0[halfway_index])
            halfway_times.append(time_step)
        time_step += t
    
    if record_halfway_temps:
        return halfway_temps, halfway_times
    else:
        return temps_0

def implicit_euler(N, t_f, T_0, T_L, h, t, record_halfway_temps = False):
    temps_0 = initial_condition(N, T_0, T_L, h)
    if record_halfway_temps:
        halfway_index = math.ceil(N/2)
        halfway_temps = [temps_0[halfway_index]]
        halfway_times = [0.]
    constant_used = alpha * t / (h * h) # should be F_o for experiment 1

    A = np.diag([1+2*constant_used] * N, 0) + np.diag([-constant_used] * (N-1), 1) + np.diag([-constant_used] * (N-1), -1)
    
    time_step = t
    while(time_step <= t_f):
        b = temps_0[1:-1].copy()
        b[0] += constant_used * temps_0[0]
        b[-1] += constant_used * temps_0[-1]
        temps_0[1: -1] = (np.linalg.inv(A) @ b.reshape(N, 1)).flatten()
        if record_halfway_temps:
            halfway_temps.append(temps_0[halfway_index])
            halfway_times.append(time_step)
        time_step += t
    
    if record_halfway_temps:
        return halfway_temps, halfway_times
    else:
        return temps_0

def crank_nicholson(N, t_f, T_0, T_L, h, t, record_halfway_temps = False):
    temps_0 = initial_condition(N, T_0, T_L, h)
    if record_halfway_temps:
        halfway_index = math.ceil(N/2)
        halfway_temps = [temps_0[halfway_index]]
        halfway_times = [0.]
    constant_used = alpha * t / (h * h) # should be F_o for experiment 1

    A = np.diag([1+constant_used] * N, 0) + np.diag([-constant_used/2] * (N-1), 1) + np.diag([-constant_used/2] * (N-1), -1)

    def update_b(index):
        return (constant_used/2)*temps_0[index-1] + (1-constant_used)*temps_0[index] + (constant_used/2)*temps_0[index+1]

    time_step = t
    while(time_step <= t_f):
        b = np.arange(1, N+1, 1, dtype = np.int32)
        b = np.vectorize(update_b)(b)
        b[0] += (constant_used/2) * temps_0[0]
        b[-1] += (constant_used/2) * temps_0[-1]
        temps_0[1: -1] = (np.linalg.inv(A) @ b.reshape(N, 1)).flatten()
        if record_halfway_temps:
            halfway_temps.append(temps_0[halfway_index])
            halfway_times.append(time_step)
        time_step += t
    
    if record_halfway_temps:
        return halfway_temps, halfway_times
    else:
        return temps_0

def experiment_1(N, t_f, T_0, T_L):
    h = L / (N+1)
    F_o = 0.4 # Fourier number (should be < 0.5 for explicit Euler for stability)
    t = F_o * h * h / alpha # delta_t
    explicit_euler_temps = explicit_euler(N, t_f, T_0, T_L, h, t)

    F_o = 50 # Fourier number (can be anything for implicit Euler. For crank nicholson method should not be very high)
    t = F_o * h * h / alpha # delta_t value is different for implicit Euler
    crank_nicholson_temps = crank_nicholson(N, t_f, T_0, T_L, h, t)
    implicit_euler_temps = implicit_euler(N, t_f, T_0, T_L, h, t)

    plt.plot(h*np.arange(0, N+2, 1), explicit_euler_temps, label = "Explicit Euler")
    plt.plot(h*np.arange(0, N+2, 1), implicit_euler_temps, linestyle="dashed", label = "Implicit Euler")
    plt.plot(h*np.arange(0, N+2, 1), crank_nicholson_temps, linestyle="dashed", label = "Crank Nicholson")
    plt.grid()
    plt.legend()
    plt.xlabel("Rod Length (x)")
    plt.ylabel("Temperature (T)")
    plt.title("Temperature Distribution on the Rod")
    plt.show()

def experiment_2(N, t_f, T_0, T_L):
    h = L / (N+1)

    # Explicit Euler
    for F_o in [0.49, 0.5, 0.51]:
        t = F_o * h * h / alpha # delta_t
        explicit_euler_temps = explicit_euler(N, t_f, T_0, T_L, h, t)

        plt.plot(h*np.arange(0, N+2, 1), explicit_euler_temps, label = "Explicit Euler")
        plt.grid()
        plt.legend()
        plt.xlabel("Rod Length (x)")
        plt.ylabel("Temperature (T)")
        plt.title(f"Explicit Euler (delta_t = {t} seconds)")
        plt.show()
    
    # Crank Nicholson
    for F_o in [50, 100, 150, 200, 250, 2000, 2000000]:
        t = F_o * h * h / alpha # delta_t
        crank_nicholson_temps = crank_nicholson(N, t_f, T_0, T_L, h, t)

        plt.plot(h*np.arange(0, N+2, 1), crank_nicholson_temps, label = "Crank Nicholson")
        plt.grid()
        plt.legend()
        plt.xlabel("Rod Length (x)")
        plt.ylabel("Temperature (T)")
        plt.title(f"Crank Nicholson (delta_t = {t} seconds)")
        plt.show()

def experiment_3(N, t_f, T_0, T_L):
    h = L / (N+1)
    F_o = 0.49 # Fourier number (should be < 0.5 for explicit Euler for stability)
    t = F_o * h * h / alpha # delta_t
    halfway_temps_explicit, halfway_times_explicit = explicit_euler(N, t_f, T_0, T_L, h, t, record_halfway_temps=True)
    halfway_temps_implicit, halfway_times_implicit = implicit_euler(N, t_f, T_0, T_L, h, t, record_halfway_temps=True)
    halfway_temps_crank, halfway_times_crank = crank_nicholson(N, t_f, T_0, T_L, h, t, record_halfway_temps=True)

    plt.plot(halfway_times_explicit, halfway_temps_explicit, label = "Explicit Euler")
    plt.plot(halfway_times_implicit, halfway_temps_implicit, label = "Implicit Euler")
    plt.plot(halfway_times_crank, halfway_temps_crank, linestyle="dashed", label = "Crank Nicholson")
    plt.legend()
    plt.grid()
    plt.xlabel("Time (secs)")
    plt.ylabel("Temperature (T)")
    plt.title("Temperature Variation at L/2 of the Rod")
    plt.show()

if __name__ == "__main__":
    if len(sys.argv) != 5:
        raise Exception("Invalid arguments passed. Run like this: python3 main.py <N> <t_f> <T_0> <T_L>")
    N = int(sys.argv[1])
    t_f = float(sys.argv[2])
    T_0 = float(sys.argv[3])
    T_L = float(sys.argv[4])
    experiment_1(N, t_f, T_0, T_L)
    # experiment_2(N, t_f, T_0, T_L)
    # experiment_3(N, t_f, T_0, T_L)
# Run the command: python3 main.py 51 0.9 50 10
