import seaborn
import matplotlib.pyplot as plt
import pickle
from utils import *
import matplotlib.ticker as ticker

# Variables of the problem
biorbd_model = biorbd.Model("arm_wt_rot_scap.bioMod")
T = 8
start_delay = 25  # Start movement after 25 first nodes
Ns = 800 - start_delay
T = T * (Ns) / 800
final_offset = 5  # Ignore last 5 node when plot EMG
tau_init = 0
muscle_init = 0.5
nbMT = biorbd_model.nbMuscleTotal()
nbQ = biorbd_model.nbQ()
nbGT = 0
nb_try = 1
co = "2"
motion = "REACH2"
muscles_names = [
    "Pec sternal",
    "Pec ribs",
    "Lat thoracic",
    "Lat lumbar",
    "Lat iliac",
    "Delt posterior",
    "Tri long",
    "Tri lat",
    "Tri med",
    "Brachial",
    "Brachioradial",
    "Pec clavicular",
    "Delt anterior",
    "Delt middle",
    "Supraspin",
    "Infraspin",
    "Subscap",
    "Bic long",
    "Bic short",
]

# Noises informations
marker_noise_lvl = [0, 0.002, 0.005, 0.01]
EMG_noise_lvl = [0, 1, 1.5, 2]
EMG_noise = np.ndarray((nb_try, len(EMG_noise_lvl), nbMT, Ns + 1))
marker_noise = np.ndarray((nb_try, len(marker_noise_lvl), 3, biorbd_model.nbMarkers(), Ns + 1))

# Get reference movement
with open(f"solutions/sim_ac_8000ms_800sn_{motion}_co_level_{co}.bob", "rb") as file:
    data = pickle.load(file)
states = data["data"][0]
controls = data["data"][1]
q_ref = states["q"][:, start_delay:]
dq_ref = states["q_dot"][:, start_delay:]
a_ref = states["muscles"][:, start_delay:]
u_ref = controls["muscles"][:, start_delay:]
w_tau = "tau" in controls.keys()  # Look if there are residual torque in the reference movement
if w_tau:
    tau = controls["tau"]
else:
    tau = np.zeros((nbGT, Ns + 1))

# Generate EMG for all noise levels. Loop for all trials, markers and EMG noise
for tries in range(nb_try):
    for marker_lvl in range(1):
        for EMG_lvl in range(len(EMG_noise_lvl)):
            get_markers = markers_fun(biorbd_model)
            markers_target = np.zeros((3, biorbd_model.nbMarkers(), Ns + 1))
            for i in range(Ns + 1):
                markers_target[:, :, i] = get_markers(q_ref[:, i])
            EMG_noise[tries, EMG_lvl, :, :] = u_ref

            x_ref = np.concatenate((q_ref, dq_ref, a_ref))
            if marker_lvl != 0:
                marker_noise[tries, marker_lvl, :, :, :] = generate_noise(
                    biorbd_model, q_ref, u_ref, marker_noise_lvl[marker_lvl], EMG_noise_lvl[EMG_lvl]
                )[0]

            if EMG_lvl != 0:
                EMG_noise[tries, EMG_lvl, :, :] = generate_noise(
                    biorbd_model, q_ref, u_ref, marker_noise_lvl[marker_lvl], EMG_noise_lvl[EMG_lvl]
                )[1]

# Plot EMG for all levels of noises
t = np.linspace(0, T, Ns + 1 - final_offset)
seaborn.set_style("whitegrid")
seaborn.color_palette()

mean_emg = np.real(EMG_noise).mean(axis=0)
std_emg = np.real(EMG_noise).std(axis=0)
minors = np.linspace(0, 1, 2)
fig = plt.figure("Muscles controls")
plt.gcf().subplots_adjust(left=0.06, right=0.99, wspace=0.25, hspace=0.2)
plt.gca().yaxis.set_major_formatter(ticker.StrMethodFormatter("{x:,.3f}"))
plt.gca().yaxis.set_major_locator(ticker.FixedLocator(minors))
emg_lvl = ["low", "mid", "high"]
for i in range(biorbd_model.nbMuscles()):
    fig = plt.subplot(4, 5, i + 1)
    if i in [14, 15, 16, 17, 18]:
        plt.xlabel("Time(s)")
        plt.gca().yaxis.set_major_formatter(ticker.StrMethodFormatter("{x:,.3f}"))
    else:
        fig.set_xticklabels([])
    if i in [0, 5, 10, 15]:
        plt.ylabel("EMG to track")
        plt.gca().yaxis.set_major_formatter(ticker.StrMethodFormatter("{x:,.3f}"))

    if i == 18:
        for k in range(1, len(EMG_noise_lvl)):
            plt.plot(t, mean_emg[k, i, :-final_offset], label=f"EMG noise level: {emg_lvl[k-1]}")
        plt.gca().yaxis.set_major_formatter(ticker.StrMethodFormatter("{x:,.3f}"))
        # plt.gca().yaxis.set_major_locator(ticker.MultipleLocator(0.5))
    else:
        for k in range(1, len(EMG_noise_lvl)):
            plt.plot(t, mean_emg[k, i, :-final_offset], label=f"EMG noise level: {emg_lvl[k-1]}")
        plt.gca().yaxis.set_major_formatter(ticker.StrMethodFormatter("{x:,.3f}"))
    plt.title(muscles_names[i])

for i in range(biorbd_model.nbMuscles()):
    fig = plt.subplot(4, 5, i + 1)
    if i == 18:
        for k in range(0, 1):
            plt.plot(t, mean_emg[k, i, :-final_offset], "red", label=f"Reference")
    else:
        for k in range(0, 1):
            plt.plot(t, mean_emg[k, i, :-final_offset], "red", label=f"Reference")
plt.legend(bbox_to_anchor=(1.05, 0.80), loc="upper left", frameon=False)
plt.show()
