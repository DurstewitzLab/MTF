import torch as tc
from torch.linalg import pinv


def get_input(inputs, step, n_steps):
    if inputs is not None:
        input_for_step = inputs[step:(-n_steps + step), :]
    else:
        input_for_step = None
    return input_for_step


def get_ahead_pred_obs(model, data, n_steps, do):
    # dims
    T, dx = data.size()

    warmup=10
    # true data
    time_steps = T - n_steps
    x_data = data[:time_steps, :].to(model.device)

    # latent model
    lat = model.latent_model
    z_enc, _ = model.E(x_data.view(-1, dx))
    df=z_enc.shape[-1]

    # initial state. Uses x0 of the dataset
    # batch dim of z holds all T - n_steps time steps
    
    B_PI = None
    dz = lat.d_z
    z = tc.randn((time_steps, dz), device=model.device)
    z = lat.teacher_force(z, z_enc, B_PI)
    

    X_pred = tc.empty((n_steps, time_steps, do), device=model.device)
    params = model.get_latent_parameters()
    for step in range(n_steps):
        # latent step performs ahead prediction on every
        # time step here
        z = lat.latent_step(z, *params)
        x = model.D_ordinal(z[:, :df])
        X_pred[step] = x

    return X_pred
    
def construct_ground_truth(data, n_steps, dg, do):
    T, dx = data.size()
    time_steps = T - n_steps
    X_true = tc.empty((n_steps, time_steps, do))
    for step in range(1, n_steps + 1):
        X_true[step - 1] = data[step : time_steps + step, dg:dg+do]
    return X_true


def linear_error(x_pred, x_true):
    return tc.abs(x_pred - x_true)
   # return tc.pow(x_pred - x_true, 2)

@tc.no_grad()
def n_steps_ahead_pred_ope(model, data, n_steps, dg, do):
    x_pred = get_ahead_pred_obs(model, data, n_steps, do)
    x_true = construct_ground_truth(data, n_steps, dg, do).to(model.device)
    mse = linear_error(x_pred, x_true).mean([1, 2]).cpu().numpy()
    return mse
