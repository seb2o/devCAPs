import numpy as np
import spams

rng = np.random.default_rng()
n = 11000
d = 2000
k = 6
a = np.empty((d, n), dtype=np.double, order='F')
rng.random(a.shape, out=a)

for i in range(5):
    col_ids = np.random.default_rng(seed=i).choice(
            a.shape[1],
            size=k,
            replace=False
        )
    dict_inits = a[:, col_ids]

    D = spams.trainDL(
            a,
            K=k,
            D=dict_inits,
            mode=3,
            lambda1=950,
            numThreads=1,
            batchsize=512,
            verbose=True,
            iter=2,
            posD=True,
            posAlpha=False,
            return_model=False,
        ) # D shape (n_voxels, n_components) (each column is a component)

    omp_L, omp_eps, omp_lambda1 = 950, None, None

    Alpha = spams.omp(
            a,
            D,
            L=omp_L,
            eps=omp_eps,
            lambda1=omp_lambda1,
            numThreads=1
        )

