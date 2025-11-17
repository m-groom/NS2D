"""
Compare the prediction of the FNO model with the ground truth
"""
import numpy as np
import h5py
from tqdm import tqdm
import torch
import matplotlib.pyplot as plt

def downsample_x(u, downsample_spatial=(1, 1)):
    """
    Downsample a real-valued input using FFT, matching DedalusDataset2D.
    Args:
        u: Input tensor of shape (T, H, W)
        N: Target size for downsampling (assumes square target N x N)
    Returns:
        Downsampled tensor of shape (T, N, N)
    """
    # Get original size
    u = torch.from_numpy(u)
    
    T, H, W = u.shape
    N = H//downsample_spatial[0]
    # Compute FFT
    u_hat = torch.fft.rfft2(u, norm='forward')
    
    # Create frequency selection mask
    freqs_h = torch.fft.fftfreq(H, d=1/H)
    freqs_w = torch.fft.rfftfreq(W, d=1/W)
    
    # Select frequencies within [-N/2, N/2-1] range
    sel_h = torch.logical_and(freqs_h >= -N/2, freqs_h <= N/2-1)
    sel_w = torch.logical_and(freqs_w >= -N/2, freqs_w <= N/2-1)
    
    # Apply frequency selection
    u_hat_down = u_hat[:, sel_h][:, :, sel_w]
    
    # Compute inverse FFT
    u_down = torch.fft.irfft2(u_hat_down, s=(N, N), norm='forward')
    u_down = u_down.numpy()
    return u_down



def get_groundtruth_from_h5(h5_path, snapshot_idx=None, downsample_spatial=(1, 1)):
    true_u = []
    true_v = []
    true_pressure = []
    with h5py.File(h5_path, 'r') as f:
        for idx in tqdm(snapshot_idx):
            u = np.array(f['tasks/u'][idx])
            v = np.array(f['tasks/v'][idx])
            pressure = np.array(f['tasks/p'][idx])
            true_u.append(u)
            true_v.append(v)
            true_pressure.append(pressure)
    true_u = np.array(true_u)
    true_v = np.array(true_v)
    true_pressure = np.array(true_pressure)
    data = np.concatenate([true_u, true_v, true_pressure], axis=0)
    if downsample_spatial[0] != (1,1):
        data = downsample_x(data, downsample_spatial)
        n_snapshots = len(true_u)
        true_u = data[:n_snapshots]
        true_v = data[n_snapshots:2*n_snapshots]
        true_pressure = data[2*n_snapshots:]
    return true_u, true_v, true_pressure
    


def load_data(pred_path, h5_path=None, load_both=True):
    data = np.load(pred_path, allow_pickle=True)
    print("data keys: ", data.keys())
    pred_u = data['pred_velocity_x']  # (T, Nx, Ny)
    pred_v = data['pred_velocity_y']  # (T, Nx, Ny)
    pred_pressure = data['pred_pressure']  # (T, Nx, Ny)
    if load_both:
        true_u = data['output_velocity_x']
        true_v = data['output_velocity_y']
        true_pressure = data['output_pressure']
    else:
        # start_idx = 7500
        # end_idx = 8500
        # stride = 4  # the model was trained to predict the solution after 4 steps
        # downsample_spatial = (2, 2)  # downsample the spatial resolution from 256 to 128
        # skip_steps = 7  # the first 7 steps were used as input to FNO
        # snapshot_idx = np.arange(start_idx, end_idx, stride)[skip_steps:]  # 243 snapshots
        # true_u, true_v, true_pressure = get_groundtruth_from_h5(h5_path, snapshot_idx, downsample_spatial)
        raise ValueError("No ground truth data provided")
    print("pred u shape: ", pred_u.shape, "pred v shape: ", pred_v.shape, "pred pressure shape: ", pred_pressure.shape)
    print("true u shape: ", true_u.shape, "true v shape: ", true_v.shape, "true pressure shape: ", true_pressure.shape)

    # Compute MSE between ground truth and prediction
    mse_u = np.mean((pred_u - true_u)**2)
    mse_v = np.mean((pred_v - true_v)**2)
    mse_pressure = np.mean((pred_pressure - true_pressure)**2)
    print("MSE between u: ", mse_u, "MSE between v: ", mse_v, "MSE between pressure: ", mse_pressure)

    # Compute MSE between ground truth and prediction for each time step
    mse_u_each_step = np.mean((pred_u - true_u)**2, axis=(1, 2))
    mse_v_each_step = np.mean((pred_v - true_v)**2, axis=(1, 2))
    mse_pressure_each_step = np.mean((pred_pressure - true_pressure)**2, axis=(1, 2))

    # Divide by variance of ground truth for each time step
    mse_u_each_step = mse_u_each_step / np.var(true_u, axis=(1, 2))
    mse_v_each_step = mse_v_each_step / np.var(true_v, axis=(1, 2))
    mse_pressure_each_step = mse_pressure_each_step / np.var(true_pressure, axis=(1, 2))

    # Plot the MSE between ground truth and prediction for each time step
    plt.figure(figsize=(15, 5))
    plt.subplot(1, 3, 1)
    plt.plot(mse_u_each_step, label='Normalised MSE for u')
    plt.xlabel('Time step')
    plt.ylabel('MSE / Var')
    plt.legend()
    plt.subplot(1, 3, 2)
    plt.plot(mse_v_each_step, label='Normalised MSE for v')
    plt.xlabel('Time step')
    plt.ylabel('MSE / Var')
    plt.legend()
    plt.subplot(1, 3, 3)
    plt.plot(mse_pressure_each_step, label='Normalised MSE for pressure')
    plt.xlabel('Time step')
    plt.ylabel('MSE / Var')
    plt.legend()
    plt.tight_layout()
    plt.savefig('mse_between_ground_truth_and_prediction.png')
    plt.close()

    return pred_u, pred_v, pred_pressure, true_u, true_v, true_pressure



if __name__ == "__main__":
    pred_path  ='/datasets/work/oa-tcch/work/forMichael/test_data_prediction_long_spectral_reg.npz'
    print("Loading both pred and ground truth from npz ", pred_path)
    load_data(pred_path, h5_path=None, load_both=True)
