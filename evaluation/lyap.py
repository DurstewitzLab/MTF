import torch as tc
import numpy as np

"""
Compute the Lyapunov spectrum of the PLRNN model given initial condition `z₁`.

The system is first evolved for `Tₜᵣ` steps to reach the attractor,

and then the spectrum is computed across `T` steps. 

"""

def lyapunov_spectrum(model,initial_state, T, T_trans=100, ons=5):
    tc.set_num_threads(1)

    # evolve for transient time Tᵣ

    tmp, latent = model.generate_free_trajectory(initial_state, T_trans)

    # initialize
    z=latent[-1]

    y = tc.zeros(z.shape[0])

    # initialize as Identity matrix

    Q = tc.eye((z.shape[-1]))
    

    for t in range(T):
        
        z=model.latent_model.generate_step(z)
        inputs = (z)
        jacobians=tc.autograd.functional.jacobian(model.latent_model.generate_step,inputs)

        # compute jacobian

        Q = tc.matmul(jacobians,Q)

        if (t%ons == 0):

            # reorthogonalize

            Q, R = tc.linalg.qr(Q)

            # accumulate lyapunov exponents

            y += tc.log(tc.abs(tc.diag(R)))

    return np.asarray(y) / T