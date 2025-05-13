import numpy as np, matplotlib.pyplot as plt, math

c = -1 # wave speed
L = 1 # length of rod

def initial_condition(N):
    grid = np.linspace(0, L, N+2, dtype=np.float128)
    wave = np.sin(2*np.pi*grid/L, dtype=np.float128)
    wave[-1] = 0.0
    return wave

def analytical_solution(N, time_step):
    grid = np.linspace(0, L, N+2, dtype=np.float128)
    wave = np.sin(2*np.pi*(grid - c*time_step) / L, dtype=np.float128)
    return wave

def ftfs(N, lambd, wave_0):
    A = np.diag([1+lambd]*N, k=0) + np.diag([-lambd]*(N-1), k=1)
    wave_1 = wave_0.copy()
    wave_1[1:-1] = A @ wave_0[1:-1]
    wave_1[-2] -= lambd * wave_0[-1]
    return wave_1

def ftbs(N, lambd, wave_0):
    A = np.diag([1-lambd]*N, k=0) + np.diag([lambd]*(N-1), k=-1)
    wave_1 = wave_0.copy()
    wave_1[1:-1] = A @ wave_0[1:-1]
    wave_1[1] += lambd * wave_0[0]
    return wave_1

def save_graph(X, Y, title, time_step, i, lambd = ""):
    if(i%10 == 0):
        plt.clf()
        plt.plot(X, Y, label = title)
        plt.legend()
        plt.grid()
        plt.xlabel("rod length")
        plt.ylabel("wave u")
        plt.title(f"{title} {time_step}secs")
        plt.savefig(f"./graphs/{title}/{lambd}_{i}.png")

def experiment2(N, t_f):
    h = L / (N+1)

    # Analytical Solution
    analytical_solution_waves = [analytical_solution(N, 0)]
    save_graph(np.linspace(0, L, N+2, dtype=np.float128), analytical_solution_waves[-1], "Analytical", 0.0, 0)
    lambd = 1.0
    t = lambd * h / c
    time_step = t
    index = 1
    while(time_step <= t_f):
        analytical_solution_waves.append(analytical_solution(N, time_step))
        save_graph(np.linspace(0, L, N+2, dtype=np.float128), analytical_solution_waves[-1], "Analytical", time_step, index)
        index += 1
        time_step += t

def experiment1(N, t_f):
    h = L / (N+1)

    for lambd in [-0.5]:
        ftfs_waves = [initial_condition(N)]
        ftbs_waves = [ftfs_waves[0]]
        
        save_graph(np.linspace(0, L, N+2, dtype=np.float128), ftfs_waves[-1], "FTFS", 0.0, 0, lambd)
        save_graph(np.linspace(0, L, N+2, dtype=np.float128), ftbs_waves[-1], "FTBS", 0.0, 0, lambd)
        t = lambd * h / c
        time_step = t
        index = 1
        while(time_step <= t_f):
            ftfs_waves.append(ftfs(N, lambd, ftfs_waves[-1]))
            ftbs_waves.append(ftbs(N, lambd, ftbs_waves[-1]))
            save_graph(np.linspace(0, L, N+2, dtype=np.float128), ftfs_waves[-1], "FTFS", time_step, index, lambd)
            save_graph(np.linspace(0, L, N+2, dtype=np.float128), ftbs_waves[-1], "FTBS", time_step, index, lambd)
            index += 1
            time_step += t
        assert len(ftfs_waves) == len(ftbs_waves)

if __name__ == "__main__":
    N = 101
    t_f = 0.5
    experiment1(N, t_f)
    experiment2(N, t_f)