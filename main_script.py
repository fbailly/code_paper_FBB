from generate_movement import *
from funct_ocp_MHE import *
from utils import *
import scipy.io as sio
import os

# Configuration of the problem
model_path = "arm_wt_rot_scap.bioMod"
biorbd_model = biorbd.Model(model_path)

conf = {
    "load_data": True,
    "use_torque": False,
    "use_activation": False,
    "save_status": False,
    "save_results": False,
    "WRITE_STATS": False,
    "TRACK_EMG": False,
    "plot": True,
    "use_noise": False,
    "use_co": False,
    "use_try": False,
    "with_low_weight": False,
    "full_windows": False  # full windows was tested only for null co_lvl and noise_lvl and low_weigth = false.
}
use_N_elec = True if conf["use_activation"] else False

# Choose the folder to save data
if conf["TRACK_EMG"]:
    if conf["use_activation"]:
        fold = "solutions/w_track_emg_rt_act/"
    else:
        if conf["with_low_weight"]:
            fold = "solutions/w_track_emg_rt_exc_low_weight/"
        else:
            fold = "solutions/w_track_emg_rt_exc/"
else:
    if conf["use_activation"]:
        fold = "solutions/wt_track_emg_rt_act/"
    else:
        if conf["with_low_weight"]:
            fold = "solutions/wt_track_emg_rt_exc_low_weight/"
        else:
            fold = "solutions/wt_track_emg_rt_exc/"

# Variables of the problem
if conf["use_activation"] is not True:
    set_ratio = [3, 3, 3, 3, 4, 5, 5, 6, 6, 6, 6, 6, 6, 6]  # For Nmhe = [3, 16]
else:
    set_ratio = [3, 3, 3, 3, 3, 3, 4, 4, 4, 4, 4, 4, 4, 5, 5, 5, 5, 5, 5, 5]  # For Nmhe = [3, 22]

var = {
    "T_ref": 8,
    "start_delay": 25,  # Start movement after the 25 first nodes
    "N_elec": 2,  # Set how much node represent well the electromechanical delay (~0.02s)
    "final_offset": 30,  # Number of last nodes to ignore when calculate RMSE
    "init_offset": 5,  # Number of first nodes to ignore when calculate RMSE
    "Ns_mhe": 7,
    "nb_try": 30,
    "marker_noise_lvl": [0, 0.002, 0.005, 0.01],
    "EMG_noise_lvl": [0, 1, 1.5, 2],
    "set_ratio": set_ratio,
    "nb_co_lvl": 4,
    "Ns_ref": 800,
}  # Set the size of MHE windows

Ns_mhe = "Full" if conf["full_windows"] else var["Ns_mhe"]
nbQ, nbMT = biorbd_model.nbQ(), biorbd_model.nbMuscles()
Ns = var["Ns_ref"] - var["start_delay"]
T = var["T_ref"] * Ns / var["Ns_ref"]
nbGT = biorbd_model.nbGeneralizedTorque() if conf["use_torque"] else 0
rt_ratio = set_ratio[Ns_mhe - 3] if conf["full_windows"] is not True else 1  # Get ratio from the setup
T_mhe = 0 if conf["full_windows"] else T / (Ns / rt_ratio) * Ns_mhe   # Compute the new time of OCP
var["T"], var["Ns"], var["rt_ratio"], var["Ns_mhe"] = T, Ns, rt_ratio, Ns_mhe

# Get reference data
nb_co_lvl = 4
X_ref = np.ndarray((nb_co_lvl, biorbd_model.nbQ() * 2 + biorbd_model.nbMuscles(), var["Ns_ref"] + 1))
U_ref = np.ndarray((nb_co_lvl, biorbd_model.nbMuscles(), var["Ns_ref"] + 1))
if conf["load_data"]:
    for i in range(nb_co_lvl):
        file_path = f"solutions/sim_ac_8000ms_800sn_REACH2_co_level_{i}.bob"
        X_ref[i, :, :], U_ref[i, :, :] = get_reference_movement(file_path, conf["use_torque"], nbGT, var["Ns_ref"])
else:
    T_mvt = 1
    Ns_mvt = 100
    nb_phase = 8
    x_phase = np.array(
        [
            [-0.1, -0.3, -0.1, 0.1, 0, 0, 0, 0],
            [-0.2, -1.3, -0.5, 0.5, 0, 0, 0, 0],
            [-0.1, -0.3, -0.1, 0.1, 0, 0, 0, 0],
            [-0.2, -1.3, -0.5, 0.5, 0, 0, 0, 0],
            [-0.1, -0.3, -0.1, 0.1, 0, 0, 0, 0],
            [-0.2, -1.3, -0.5, 0.5, 0, 0, 0, 0],
            [-0.1, -0.3, -0.1, 0.1, 0, 0, 0, 0],
            [-0.2, -1.3, -0.5, 0.5, 0, 0, 0, 0],
            [-0.1, -0.3, -0.1, 0.1, 0, 0, 0, 0],
            [-0.2, -1.3, -0.5, 0.5, 0, 0, 0, 0],
        ]
    )
    X_1 = generate_state(model_path, T_mvt, Ns_mvt, nb_phase, x_phase)
    rate = int(Ns_mvt / T_mvt)
    low_exc = [0, 0.1, 0.2, 0.3]
    X_2, U_2 = generate_exc_low_bounds(rate, model_path, X_1, low_exc)
    X_ref, U_ref = generate_final_data(rate, X_2, U_2, save_data=False, plot=True)
X_ref, U_ref = X_ref[:, :, var["start_delay"] :], U_ref[:, :, var["start_delay"] :]

# Noise informations
marker_noise_lvl = var["marker_noise_lvl"] if conf["use_noise"] else [0]
EMG_noise_lvl = var["EMG_noise_lvl"] if conf["use_noise"] else [0]

len_sol_x = Ns + 1 if conf["full_windows"] else ceil((Ns + 1) / rt_ratio) - Ns_mhe
len_sol_u = Ns if conf["full_windows"] else ceil(Ns / rt_ratio) - Ns_mhe
# Set size of optimal states and controls
if conf["use_activation"]:
    var["X_est"] = np.zeros((biorbd_model.nbQ() * 2, len_sol_x))
else:
    var["X_est"] = np.zeros((biorbd_model.nbQ() * 2 + biorbd_model.nbMuscles(), len_sol_x))
var["U_est"] = np.zeros((nbGT + biorbd_model.nbMuscleTotal(), len_sol_u))

# Set number of co-contraction level
nb_co_lvl = var["nb_co_lvl"] if conf["use_co"] else 1
T_ocp = T if conf["full_windows"] else T_mhe
N_ocp = Ns if conf["full_windows"] else Ns_mhe
# Build the graph
ocp = prepare_ocp(
    biorbd_model=biorbd_model,
    final_time=T_ocp,
    x0=var["X_est"][:, 0],
    nbGT=nbGT,
    number_shooting_points=N_ocp,
    use_torque=conf["use_torque"],
    use_activation=conf["use_activation"],
    use_SX=True,
)

# Initialize files where status of optimisation are stored
if conf["TRACK_EMG"] and conf["save_status"]:
    f = open(f"{fold}status_track_rt_EMG{conf['TRACK_EMG']}.txt", "a")
    f.write("Ns_mhe;  Co_lvl;  Marker_noise;  EMG_noise;  nb_try;  iter\n")
    f.close()
elif conf["TRACK_EMG"] is not True and conf["save_status"]:
    f = open(f"{fold}status_track_rt_EMG{conf['TRACK_EMG']}.txt", "a")
    f.write("Ns_mhe;  Co_lvl;  Marker_noise;  EMG_noise;  nb_try;  iter\n")
    f.close()

# Graph to compute muscle force
var["get_force"] = force_func(biorbd_model, use_activation=conf["use_activation"])
# Solve in a loop for each co-contraction level
for co in range(0, nb_co_lvl):
    var["co"] = co
    # Get data of reference movement, depend of co-contraction level
    q_ref = X_ref[co, : biorbd_model.nbQ(), :]
    u_ref = U_ref[co, :, :]

    # Loop for marker and EMG noise
    for marker_lvl in range(len(marker_noise_lvl)):
        var["marker_lvl"] = marker_lvl
        emg_range = range(len(EMG_noise_lvl)) if conf["TRACK_EMG"] else range(0, 1)
        for EMG_lvl in emg_range:
            var["EMG_lvl"] = EMG_lvl
            # Get targets
            get_markers = markers_fun(biorbd_model)
            markers_target = np.zeros((3, biorbd_model.nbMarkers(), Ns + 1))
            for i in range(Ns + 1):
                markers_target[:, :, i] = get_markers(q_ref[:, i])
            var["markers_target"] = markers_target
            muscles_target = u_ref

            # Add electromechanical delay to the control target
            if use_N_elec:
                muscles_target = np.ndarray((u_ref.shape[0], u_ref.shape[1]))
                for i in range(var["N_elec"], u_ref.shape[1]):
                    muscles_target[:, i] = u_ref[:, i - var["N_elec"]]
                for i in range(var["N_elec"]):
                    muscles_target[:, i] = muscles_target[:, var["N_elec"]]
            var["muscles_target"] = muscles_target
            var["x_ref"], var["u_ref"] = X_ref[co, :, :], U_ref[co, :, :]
            # ------------------- RUN MHE ------------------#
            (
                err_tries,
                force_est,
                force_ref,
                X_est_tries,
                U_est_tries,
                muscles_target_tries,
                markers_target_tries,
                toc,
            ) = run_mhe(model_path, ocp, var, conf, fold)
            # ----------------------------------------------#

            # Store stats (informations on RMSE, force and time to solve) in .mat file if flag is on True
            err_dic = {"err_tries": err_tries, "force_est": force_est, "force_ref": force_ref}
            if conf["WRITE_STATS"]:
                if os.path.isfile(f"solutions/stats_rt_test_activation_driven{conf['use_activation']}.mat"):
                    matcontent = sio.loadmat(f"solutions/stats_rt_test_activation_driven{conf['use_activation']}.mat")
                    err_mat = np.concatenate((matcontent["err_tries"], err_tries))
                    err_dic = {"err_tries": err_mat, "force_est": force_est, "force_ref": force_ref}
                    sio.savemat(f"solutions/stats_rt_test_activation_driven{conf['use_activation']}.mat", err_dic)
                else:
                    sio.savemat(f"solutions/stats_rt_test_activation_driven{conf['use_activation']}.mat", err_dic)

            # Store results in .mat file for all tries if flag is on True
            if conf["save_results"]:
                dic = {
                    "X_est": X_est_tries,
                    "U_est": U_est_tries,
                    "x_sol": var["x_ref"],
                    "u_sol": var["u_ref"],
                    "markers_target": markers_target_tries,
                    "u_target": muscles_target_tries,
                    "time_per_mhe": toc / ceil(Ns / rt_ratio - Ns_mhe),
                    "time_tot": toc,
                    "co_lvl": co,
                    "marker_noise_lvl": marker_noise_lvl[marker_lvl],
                    "EMG_noise_lvl": EMG_noise_lvl[EMG_lvl],
                    "N_mhe": Ns_mhe,
                    "N_tot": Ns,
                    "rt_ratio": rt_ratio,
                    "f_est": force_est,
                    "f_ref": force_ref,
                }
                mhe = "mhe" if conf["full_windows"] is not True else "full"
                track = "track" if conf["TRACK_EMG"] is not True else "min"
                sio.savemat(
                    f"{fold}{mhe}_{track}_EMG_excitation_driven_co_lvl{co}_noise_lvl_{marker_noise_lvl[marker_lvl]}_{EMG_noise_lvl[EMG_lvl]}.mat",
                    dic,
                )
