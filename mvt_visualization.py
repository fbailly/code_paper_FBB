import scipy.io as sio
import seaborn
from utils import *
import bioviz
import matplotlib.pyplot as plt

biorbd_model = biorbd.Model("arm_wt_rot_scap.bioMod")

# Define folder and status file
fold_w_emg = f"solutions/w_track_emg_rt_exc/"
# Get data for optimal (track EMG) movement
mat_content = sio.loadmat(
    f"{fold_w_emg}track_mhe_w_EMG_excitation_driven_co_lvl0_noise_lvl_0_0.mat"
)
X_est = mat_content["X_est"]
q_est = np.mean(X_est[:, :biorbd_model.nbQ(), :], axis=0)
b = bioviz.Viz("arm_wt_rot_scap.bioMod")
b.load_movement(q_est)
b.vtk_window.change_background_color((1, 1, 1))
b.exec()


