import numpy as np
import kmeans
import common
import naive_em
import em


X = np.loadtxt("toy_data.txt")
K = [1, 2, 3, 4]
seed = [0, 1, 2, 3, 4]

# min_cost = np.ndarray(2)

# # TODO: Your code here
# for K in range(4):
#     for seed in range(5):
#         gm, p = common.init(X,K+1,seed)     
#         mixture, post, cost = kmeans.run(X, gm, p)
#         naive_em.estep(X,gm)
#         naive_em.mstep(X,gm)
#         #common.plot(X, mixture, post, K+1)
#         #min_cost = np.append([K+1,cost])
#         print('Cost of K=',K+1,': ', cost)
#         #print(min_cost)

def run_kmeans(X, plot=False):
    """ My solution:
    for i in range(len(K)):
        for j in range(len(seed)):
            mixture, post = common.init(X, K[i], seed[j])
            mixture, post, cost = kmeans.run(X, mixture, post)
            print("K = {}, seed = {}, cost = {}".format(K[i], seed[j], cost))
            if plot:
                common.plot(X, mixture, post, "K={}, seed={}".format(K[i], seed[j]))
    """
    # Instructor's solution:
    for K in range(1, 5):
        min_cost = None
        best_seed = None
        for seed in range(0, 5):
            mixture, post = common.init(X, K, seed)
            mixture, post, cost = kmeans.run(X, mixture, post)
            if min_cost is None or cost < min_cost:
                min_cost = cost
                best_seed = seed

        mixture, post = common.init(X, K, best_seed)
        mixture, post, cost = kmeans.run(X, mixture, post)
        title = "K-means for K=, seed=, cost=".format(K, best_seed,
                                                      min_cost)
        print(title)
        common.plot(X, mixture, post, title)

def run_with_bic():
    max_bic = None
    for K in range(1, 5):
        max_ll = None
        best_seed = None
        for seed in range(0, 5):
            mixture, post = common.init(X, K, seed)
            mixture, post, ll = naive_em.run(X, mixture, post)
            if max_ll is None or ll > max_ll:
                max_ll = ll
                best_seed = seed

        mixture, post = common.init(X, K, best_seed)
        mixture, post, ll = naive_em.run(X, mixture, post)
        bic = common.bic(X, mixture, ll)
        if max_bic is None or bic > max_bic:
            max_bic = bic
        title = "EM for K={}, seed={}, ll={}, bic={}".format(K, best_seed, ll, bic)
        print(title)
        common.plot(X, mixture, post, title)

def main():
    print('Start main()...')

    # run_kmeans(X, plot=False)
    # run_with_bic()


    print('Finish main()')



if __name__ == "__main__":
    main()