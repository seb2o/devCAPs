import numpy as np
import spams

rng = np.random.default_rng()
n = 11000
k = 2000
a = np.empty((k, n), dtype=np.double, order='F')
rng.random(a.shape, out=a)

for i in range(5):
    col_ids = np.random.default_rng(seed=init).choice(
            reshaped_stacked_frames.shape[1],
            size=n_comps,
            replace=False
        )
    dict_inits = reshaped_stacked_frames[:, col_ids]

    D = spams.trainDL(
            reshaped_stacked_frames,
            K=n_comps,
            D=dict_inits,
            mode=3,
            lambda1=alpha,
            numThreads=subject_loading_n_workers,
            batchsize=512,
            verbose=True,
            iter=n_iters,
            posD=positive_atoms,
            posAlpha=positive_code,
            return_model=False,
        ) # D shape (n_voxels, n_components) (each column is a component)


    Alpha = spams.omp(
            reshaped_stacked_frames,
            D,
            L=omp_L,
            eps=omp_eps,
            lambda1=omp_lambda1,
            numThreads=subject_loading_n_workers
        )

