import torch as tc
from torch.linalg import pinv


def get_input(inputs, step, n_steps):
    if inputs is not None:
        input_for_step = inputs[step:(-n_steps + step), :]
    else:
        input_for_step = None
    return input_for_step
    
    
def get_ahead_pred_obs_warm_up(model, data, n_steps, dg, warm_up_steps, warm_up_sequence):
    # dims
    T, dx = data.size()
    # true data
    #warm_up
    warm_up_length=warm_up_steps*warm_up_sequence
    
    time_steps = T - n_steps
    total_length=T - n_steps-warm_up_length
    
    x_data = data[:time_steps, :].to(model.device)

    # latent model
    lat = model.latent_model
    
    z_enc, entropy = model.E(x_data.view(-1, dx))
    
    df=z_enc.shape[-1]

    # initial state. Uses x0 of the dataset
    # batch dim of z holds all T - n_steps time steps
    
    B_PI = None
    dz = lat.d_z
    z = tc.randn((time_steps, dz), device=model.device)
    z = lat.teacher_force(z, z_enc, B_PI)
    

    X_pred = tc.empty((n_steps, total_length, dg), device=model.device)
    params = model.get_latent_parameters()
    
    for step in range(warm_up_steps):
      for l in range(warm_up_sequence):
        z = lat.latent_step(z, *params)
    #force after warm-up iteration
      z = lat.teacher_force(z[warm_up_sequence:], z_enc[(step+1)*warm_up_sequence:], B_PI)
      
    for step in range(n_steps):
        # latent step performs ahead prediction on every
        # time step here
        z = lat.latent_step(z, *params)
        x = model.D_gaussian(z[:, :df])

        X_pred[step] = x

    return X_pred
    


def get_ahead_pred_obs(model, data, n_steps, dg):
    # dims
    T, dx = data.size()
    # true data
    time_steps = T - n_steps
    x_data = data[:time_steps, :].to(model.device)

    # latent model
    lat = model.latent_model
    
    z_enc, entropy = model.E(x_data.view(-1, dx))
    
    df=z_enc.shape[-1]

    # initial state. Uses x0 of the dataset
    # batch dim of z holds all T - n_steps time steps
    
    B_PI = None
    dz = lat.d_z
    z = tc.randn((time_steps, dz), device=model.device)
    z = lat.teacher_force(z, z_enc, B_PI)
    

    X_pred = tc.empty((n_steps, time_steps, dg), device=model.device)
    params = model.get_latent_parameters()
    for step in range(n_steps):
        # latent step performs ahead prediction on every
        # time step here
        z = lat.latent_step(z, *params)
        x = model.D_gaussian(z[:, :df])
        X_pred[step] = x

    return X_pred
    
def construct_ground_truth(data, n_steps, dg):
    T, dx = data.size()
    time_steps = T - n_steps
    X_true = tc.empty((n_steps, time_steps, dg))
    for step in range(1, n_steps + 1):
        X_true[step - 1] = data[step : time_steps + step, :dg]
    return X_true
    
def construct_ground_truth_warm_up(data, n_steps, dg, warm_up_steps, warm_up_sequence):
    T, dx = data.size()
    warm_up_length=warm_up_steps*warm_up_sequence
    time_steps = T - n_steps-warm_up_length
    X_true = tc.empty((n_steps, time_steps, dg))
    for step in range(1, n_steps + 1):
        X_true[step - 1] = data[warm_up_length+step : warm_up_length+time_steps + step, :dg]
    return X_true


def squared_error(x_pred, x_true):
    return tc.pow(x_pred - x_true, 2)

@tc.no_grad()
def n_steps_ahead_pred_mse(model, data, n_steps, dg):
    x_pred = get_ahead_pred_obs(model, data, n_steps, dg)
    x_true = construct_ground_truth(data, n_steps, dg).to(model.device)
    mse = squared_error(x_pred, x_true).mean([1, 2]).cpu().numpy()
    return mse
    
def n_steps_ahead_pred_mse_warm_up(model, data, n_steps, dg, warm_up_steps, warm_up_sequence):
    x_pred = get_ahead_pred_obs_warm_up(model, data, n_steps, dg, warm_up_steps, warm_up_sequence)
    x_true = construct_ground_truth_warm_up(data, n_steps, dg, warm_up_steps, warm_up_sequence).to(model.device)
    mse = squared_error(x_pred, x_true).mean([1, 2]).detach().cpu().numpy()
    return mse
