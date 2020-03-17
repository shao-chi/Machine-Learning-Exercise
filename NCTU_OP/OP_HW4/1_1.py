import numpy as np
from numpy.random import rand
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm

def real_num_GA(constraint_set, N, p_m, g_iter, func):
    """
    constraint_set: array type
    N: population size
    p_m: mutation rate
    iter: generations
    func: function for finding maximum
    """
    def roulette_wheel(fit, N, P):
        M = list()
        cum_fit = np.cumsum(fit)
        for _ in range(N):
            index = np.where(cum_fit - rand() > 0)[0]
            M.append(P[index[0]])
        return np.array(M)

    def tournament(fit, N, P):
        M = list()
        for _ in range(N):
            fight1 = int(rand() * N)
            fight2 = int(rand() * N)
            if fit[fight1] > fit[fight2]:
                M.append(P[fight1])
            else:
                M.append(P[fight2])
        return np.array(M)

    num_argu = len(constraint_set)
    p_c = 0.75
    lower_bound = constraint_set[:, 0]
    upper_bound = constraint_set[:, 1]
    best_fit = list()
    avg_fit = list()
    worst_fit = list()

    # initial
    P = list()
    for i in range(N):
        tmp = lower_bound + rand(1, num_argu) * (upper_bound - lower_bound)
        P.append(tmp[0])
    P = np.array(P)
    fit = func(P)
    maximum = max(fit)
    maximum_domain = P[np.argmax(fit)]
    best_fit.append(maximum)
    avg_fit.append(np.mean(fit))
    worst_fit.append(min(fit))

    for i in range(g_iter - 1):
        # selection
        fit = fit - min(fit)
        if sum(fit) == 0:
            print('stop at {} generations'.format(i+1))
            for _ in range(i, g_iter - 1):
                best_fit.append(best_fit[-1])
                avg_fit.append(avg_fit[-1])
                worst_fit.append(worst_fit[-1])
            break
        else:
            fit = fit / sum(fit)

        M = roulette_wheel(fit, N, P)
        # M = tournament(fit, N, P)

        # crossover
        m = M
        for _ in range(N//2):
            index1 = int(rand() * N)
            index2 = int(rand() * N)
            parent1 = M[index1, :]
            parent2 = M[index2, :]
            if rand() < p_c:
                alpha = rand()
                offspring1 = (alpha * parent1 + (1 - alpha) * parent2 + (rand(1, num_argu) - 0.5) * (upper_bound - lower_bound) / 10)[0]
                offspring2 = (alpha * parent2 + (1 - alpha) * parent1 + (rand(1, num_argu) - 0.5) * (upper_bound - lower_bound) / 10)[0]
                ## projection
                for j in range(num_argu):
                    if offspring1[j] < lower_bound[j]:
                        offspring1[j] = lower_bound[j]
                    elif offspring1[j] > upper_bound[j]:
                        offspring1[j] = upper_bound[j]
                    if offspring2[j] < lower_bound[j]:
                        offspring2[j] = lower_bound[j]
                    elif offspring2[j] > upper_bound[j]:
                        offspring2[j] = upper_bound[j]
                m[index1] = offspring1
                m[index2] = offspring2

        # mutation
        for j in range(N):
            if rand() < p_m:
                alpha = rand()
                m[j] = alpha * m[j] + (1 - alpha) * (lower_bound + rand(1, num_argu) * (upper_bound - lower_bound))
        
        P = m
        # evaluation
        fit = func(P)

        if max(fit) > maximum:
            maximum = max(fit)
            maximum_domain = P[np.argmax(fit)]

        best_fit.append(max(fit))
        avg_fit.append(np.mean(fit))
        worst_fit.append(min(fit))

    plt.suptitle('population size = {}, mutation rate = {}'.format(N, p_m))
    plt.title('maximum = {}, x = {}'.format(maximum, maximum_domain))
    plt.xlabel('Generations')
    plt.ylabel('Objective Function Value')
    x_axis = np.linspace(1, g_iter, g_iter)
    plt.plot(x_axis, best_fit, c='b', marker='D', label="best", linewidth=0.7)
    plt.plot(x_axis, avg_fit, c='g', marker='o', label="average", linewidth=0.7)
    plt.plot(x_axis, worst_fit, c='r', marker='X', label="worst", linewidth=0.7)
    plt.show()

    return maximum, maximum_domain

def f_x(domain):
    """
    domain: [[x1, x2], .....] 
    (size: n * 2)
    (type: array)
    """
    x1 = domain[:, 0]
    x2 = domain[:, 1]
    return x1 * np.sin(x1) + x2 * np.sin(5 * x2)

constraint_set = np.array([[0, 10], [4, 6]])

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
X1 = np.linspace(0, 10, 500)
X2 = np.linspace(4, 6, 500)
X1, X2 = np.meshgrid(X1, X2)
Y = X1 * np.sin(X1) + X2 * np.sin(5 * X2)
ax.plot_surface(X1, X2, Y, cmap=cm.coolwarm, linewidth=0, antialiased=False)
ax.set_xlabel('x1')
ax.set_ylabel('x2')
ax.set_zlabel('f(x1,x2)')
plt.show()

N = [2, 5, 10, 30, 50, 100, 300, 500]
p_m = [0, 0.00075, 0.0075, 0.075, 0.75, 1]

for n in N:
    maximum, maximum_domain = real_num_GA(constraint_set, n, 0.00075, 50, f_x)
    print(maximum_domain)
    print(maximum)
for p in p_m:
    maximum, maximum_domain = real_num_GA(constraint_set, 50, p, 50, f_x)
    print(maximum_domain)
    print(maximum)

# maximum, maximum_domain = real_num_GA(constraint_set, 50, 0.00075, 50, f_x)