import numpy as np
from numpy.random import rand
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm

def real_num_GA(func, constraint_set, N, p_m, p_c, g_iter, selection):
    """
    constraint_set: array type
    N: population size
    p_m: mutation rate
    p_c: 
    iter: generations
    selection: 0 -> roulette-wheel,1 -> tournament
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
            if fit[fight1] < fit[fight2]:
                M.append(P[fight1])
            else:
                M.append(P[fight2])
        return np.array(M)

    num_argu = len(constraint_set)
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
    minimum = min(fit)
    minimum_domain = P[np.argmin(fit)]
    # best_fit.append(minimum)
    # avg_fit.append(np.mean(fit))
    # worst_fit.append(max(fit))

    for i in range(g_iter):
        # selection
        fit = -(fit - max(fit)) # positive
        if sum(fit) == 0:
            print('stop at {} generations'.format(i+1))
            for _ in range(i, g_iter):
                best_fit.append(best_fit[-1])
                avg_fit.append(avg_fit[-1])
                worst_fit.append(worst_fit[-1])
            break
        else:
            fit = fit/sum(fit)

        if selection == 0:
            M = roulette_wheel(fit, N, P)
        elif selection == 1:
            M = tournament(fit, N, P)
        else:
            assert False, 'The argument of selection is not correct. Only 0 or 1'

        # crossover
        m = M
        for _ in range(N//2):
            index1 = int(rand() * N)
            index2 = int(rand() * N)
            parent1 = M[index1, :]
            parent2 = M[index2, :]
            if rand() < p_c:
                alpha = rand()
                offspring1 = (alpha * parent1 + (1 - alpha) * parent2 - (rand(1, num_argu) - 0.5) * (upper_bound - lower_bound) / 10)[0]
                offspring2 = (alpha * parent2 + (1 - alpha) * parent1 - (rand(1, num_argu) - 0.5) * (upper_bound - lower_bound) / 10)[0]
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
                m[j] = alpha * m[j] - (1 - alpha) * (lower_bound + rand(1, num_argu) * (upper_bound - lower_bound))
        
        P = m
        # evaluation
        fit = func(P)

        if min(fit) < minimum:
            minimum = min(fit)
            minimum_domain = P[np.argmin(fit)]

        best_fit.append(min(fit))
        avg_fit.append(np.mean(fit))
        worst_fit.append(max(fit))

    if selection == 0:
        plt.suptitle('population size = {}, selection = roulette wheel\nmutation rate = {}, crossover rate = {}'.format(N, p_m, p_c))
    elif selection == 1:
        plt.suptitle('population size = {}, selection = tournament\nmutation rate = {}, crossover rate = {}'.format(N, p_m, p_c))
    # plt.title('minimum = {}, x = {}'.format(minimum, minimum_domain))
    plt.xlabel('Generations\nminimum = {}, x = {}'.format(minimum, minimum_domain))
    plt.ylabel('Objective Function Value')
    x_axis = np.linspace(1, g_iter, g_iter)
    plt.plot(x_axis, best_fit, c='b', marker='D', label="best", linewidth=0.7)
    plt.plot(x_axis, avg_fit, c='g', marker='o', label="average", linewidth=0.7)
    plt.plot(x_axis, worst_fit, c='r', marker='X', label="worst", linewidth=0.7)
    plt.show()

    return minimum, minimum_domain

def a(domain):
    """
    domain: [[x, y], .....] 
    (size: n * 2)
    (type: array)
    """
    x = domain[:, 0]
    y = domain[:, 1]
    return -np.exp(0.2 * np.sqrt(np.power(x - 1, 2) + np.power(y - 1, 2)) + np.cos(2 * x) + np.sin(2 * x))

def b(domain):
    """
    domain: [[x, y], .....] 
    (size: n * 2)
    (type: array)
    """
    x = domain[:, 0]
    y = domain[:, 1]
    tmp = np.power(x, 2) + np.power(y, 2)
    return 0.5 + ((np.sin(tmp) - 0.5) / (1 + 0.1 * tmp))

constraint_set = np.array([[-5, 5], [-5, 5]])

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
X1 = np.linspace(-5, 5, 100)
X2 = np.linspace(-5, 5, 100)
X1, X2 = np.meshgrid(X1, X2)
Ya = -np.exp(0.2 * np.sqrt(np.power(X1 - 1, 2) + np.power(X2 - 1, 2)) + np.cos(2 * X1) + np.sin(2 * X1))
Yb = 0.5 + ((np.sin(np.power(X1, 2) + np.power(X2, 2)) - 0.5) / (1 + 0.1 * (np.power(X1, 2) + np.power(X2, 2))))
ax.plot_surface(X1, X2, Ya, cmap=cm.coolwarm, linewidth=0, antialiased=False)
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('f(x,y)')
plt.show()

N = [2, 5, 10, 30, 50, 100, 300, 500, 1000]
p_m = [0, 0.00075, 0.0075, 0.075, 0.75, 1]

print('Problem a')
for n in N:
    print('N = {}'.format(n))
    minimum, minimum_domain = real_num_GA(a, constraint_set, n, 0.00075, 0.75, 50, 0)
    print(minimum_domain)
    print(minimum)
for p in p_m:
    print('mutation rate = {}'.format(p))
    minimum, minimum_domain = real_num_GA(a, constraint_set, 50, p, 0.75, 50, 0)
    print(minimum_domain)
    print(minimum)

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(X1, X2, Yb, cmap=cm.coolwarm, linewidth=0, antialiased=False)
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('f(x,y)')
plt.show()

print('Problem b')
for n in N:
    print('N = {}'.format(n))
    minimum, minimum_domain = real_num_GA(b, constraint_set, n, 0.00075, 0.75, 50, 0)
    print(minimum_domain)
    print(minimum)
for p in p_m:
    print('mutation rate = {}'.format(p))
    minimum, minimum_domain = real_num_GA(b, constraint_set, 50, p, 0.75, 50, 0)
    print(minimum_domain)
    print(minimum)