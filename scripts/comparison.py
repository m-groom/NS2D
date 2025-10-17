"""
Compare the prediciton of the FNO model with the ground truth
# (250 snapshots in total, the first 7 steps were used as input to FNO, the rest (243) were obtained by One step ahead prediction)
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
    true_vorticity = []
    true_streamfunction = []
    with h5py.File(h5_path, 'r') as f:
        for idx in tqdm(snapshot_idx):
            vorticity = np.array(f['tasks/vorticity'][idx])
            streamfunction = np.array(f['tasks/streamfunction'][idx])
            true_vorticity.append(vorticity)
            true_streamfunction.append(streamfunction)
    true_vorticity = np.array(true_vorticity)
    true_streamfunction = np.array(true_streamfunction)
    data = np.concatenate([true_vorticity, true_streamfunction], axis=0)
    if downsample_spatial[0] != (1,1):
        data = downsample_x(data, downsample_spatial)
        true_vorticity, true_streamfunction = data[:len(true_vorticity)], data[len(true_vorticity):]
    return true_vorticity, true_streamfunction
    


def load_data(pred_path, h5_path=None, load_both=True):
    data = np.load(pred_path, allow_pickle=True)
    print("data keys: ",data.keys())
    pred_vorticity = data['pred_vorticity'] # (T, Nx, Ny) (243, 128, 128)
    pred_streamfunction = data['pred_streamfunction'] # (T, Nx, Ny) (243, 128, 128)
    if load_both:    
        true_vorticity = data['output_vorticity']
        true_streamfunction = data['output_streamfunction']
    else:
        start_idx = 7500
        end_idx = 8500
        stride = 4 # the model was trained to predict the solution after 4 steps
        downsample_spatial = (2, 2) # downsample the spatial resolution from 256 to 128 
        skip_steps = 7 # the first 7 steps were used as input to FNO
        snapshot_idx = np.arange(start_idx, end_idx, stride)[skip_steps:] # 243 snapshots
        true_vorticity, true_streamfunction = get_groundtruth_from_h5(h5_path, snapshot_idx, downsample_spatial)
    print("pred vorticity shape: ", pred_vorticity.shape, "pred streamfunction shape: ", pred_streamfunction.shape)
    print("true vorticity shape: ", true_vorticity.shape, "true streamfunction shape: ", true_streamfunction.shape)
    # compute MSE between ground truth and prediction
    mse_vorticity = np.mean((pred_vorticity - true_vorticity)**2)
    mse_streamfunction = np.mean((pred_streamfunction - true_streamfunction)**2)
    print("MSE between vorticity: ", mse_vorticity, "MSE between streamfunction: ", mse_streamfunction)
    # compute MSE between ground truth and prediction for each time step
    mse_vorticity_each_step = np.mean((pred_vorticity - true_vorticity)**2, axis=(1,2))
    mse_streamfunction_each_step = np.mean((pred_streamfunction - true_streamfunction)**2, axis=(1,2))
    # Divide by std of ground truth for each time step
    mse_vorticity_each_step = mse_vorticity_each_step / np.var(true_vorticity, axis=(1,2))
    mse_streamfunction_each_step = mse_streamfunction_each_step / np.var(true_streamfunction, axis=(1,2))
    # print("MSE between vorticity for each time step: ", mse_vorticity_each_step)
    # print("MSE between streamfunction for each time step: ", mse_streamfunction_each_step)
    # Plot the MSE between ground truth and prediction for each time step on separate plots
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.plot(mse_vorticity_each_step, label='MSE between vorticity')
    plt.legend()
    plt.subplot(1, 2, 2)
    plt.plot(mse_streamfunction_each_step, label='MSE between streamfunction')
    plt.legend()
    plt.savefig('mse_between_ground_truth_and_prediction.png')
    plt.close()
    return pred_vorticity, pred_streamfunction, true_vorticity, true_streamfunction



if __name__ == "__main__":
    # pred_path = '/scratch3/wan410/operator_learning_model/FNO_ns2d_dedalus_ntrain4968/test_data_prediction.npz'
    pred_path  ='/datasets/work/oa-tcch/work/forMichael/test_data_prediction.npz'
    print("Loading both pred and ground truth from npz ", pred_path)
    load_data(pred_path, h5_path=None, load_both=True)
    
    # h5_path = '/datasets/work/oa-tcch/work/forXuesong/new/realisation_0000/snapshots/snapshots_s1.h5'
    # print("Loading ground truth from the original h5 file, should give you the same results as the npz file") # only if you have torch installed ,because the torch fft didn't get the same results as np fft
    # load_data(pred_path, h5_path=h5_path, load_both=False)
