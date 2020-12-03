from time import time
import pickle
from utils import *
import os
import scipy.io as sio
from bioptim import (
    OptimalControlProgram,
    ObjectiveList,
    Objective,
    Data,
    DynamicsTypeList,
    DynamicsType,
    BoundsList,
    QAndQDotBounds,
    InitialGuessOption,
    Solver,
    InterpolationType,
    Bounds,
    ShowResult,
)


def prepare_ocp(
    biorbd_model,
    final_time,
    x0,
    nbGT,
    number_shooting_points,
    use_SX=False,
    nb_threads=1,
    use_activation=True,
    use_torque=True,
):
    # --- Options --- #
    # Model path
    biorbd_model = biorbd_model
    nbQ = biorbd_model.nbQ()
    nbGT = nbGT
    nbMT = biorbd_model.nbMuscleTotal()
    tau_min, tau_max, tau_init = -100, 100, 0
    muscle_min, muscle_max, muscle_init = 0, 1, 0.5
    activation_min, activation_max, activation_init = 0, 1, 0.2

    # Add objective functions
    objective_functions = ObjectiveList()

    # Dynamics
    dynamics = DynamicsTypeList()
    if use_activation and use_torque:
        dynamics.add(DynamicsType.MUSCLE_ACTIVATIONS_AND_TORQUE_DRIVEN)
    elif use_activation is not True and use_torque:
        dynamics.add(DynamicsType.MUSCLE_EXCITATIONS_AND_TORQUE_DRIVEN)
    elif use_activation and use_torque is not True:
        dynamics.add(DynamicsType.MUSCLE_ACTIVATIONS_DRIVEN)
    elif use_activation is not True and use_torque is not True:
        dynamics.add(DynamicsType.MUSCLE_EXCITATIONS_DRIVEN)

    # State path constraint
    x_bounds = BoundsList()
    x_bounds.add(QAndQDotBounds(biorbd_model))
    if use_activation is not True:
        x_bounds[0].concatenate(
            Bounds([activation_min] * biorbd_model.nbMuscles(), [activation_max] * biorbd_model.nbMuscles())
        )

    # Control path constraint
    u_bounds = BoundsList()
    u_bounds.add(
        [
            [tau_min] * nbGT + [muscle_min] * biorbd_model.nbMuscleTotal(),
            [tau_max] * nbGT + [muscle_max] * biorbd_model.nbMuscleTotal(),
        ]
    )

    # Initial guesses
    x_init = InitialGuessOption(
        np.tile(x0, (number_shooting_points + 1, 1)).T, interpolation=InterpolationType.EACH_FRAME
    )

    u0 = np.array([tau_init] * nbGT + [muscle_init] * nbMT) + 0.1
    u_init = InitialGuessOption(np.tile(u0, (number_shooting_points, 1)).T, interpolation=InterpolationType.EACH_FRAME)
    # ------------- #

    return OptimalControlProgram(
        biorbd_model,
        dynamics,
        number_shooting_points,
        final_time,
        x_init,
        u_init,
        x_bounds,
        u_bounds,
        objective_functions,
        use_SX=use_SX,
        nb_threads=nb_threads,
    )


if __name__ == "__main__":
    # Configuration of the problem
    use_activation = False
    use_torque = False
    use_ACADOS = True
    use_bash = True
    save_stats = True
    use_noise = False
    plot_and_animate = False
    save_results = False
    TRACK_EMG = True
    if use_activation:
        use_N_elec = True  # Use an electromechanical delay when using activation driven
    else:
        use_N_elec = False

    # Variable of the problem
    biorbd_model = biorbd.Model("arm_wt_rot_scap.bioMod")
    T = 8
    start_delay = 25  # Start movement after 25 first nodes
    Ns = 800 - start_delay
    T = T * (Ns) / 800
    N_elec = 2  # Set how much node represent well the electromechanical delay (~0.02s)
    final_offset = 30  # Number of last nodes to ignore when calculate RMSE
    init_offset = 5  # Number of first nodes to ignore when calculate RMSE
    Ns_mhe = 16  # Set the size of MHE windows
    tau_init = 0
    muscle_init = 0.5
    motion = "REACH2"
    i = "0"

    # Get data of reference movement, depend of co-contraction level
    with open(f"solutions/sim_ac_8000ms_800sn_{motion}_co_level_{i}.bob", "rb") as file:
        data = pickle.load(file)
    states = data["data"][0]
    controls = data["data"][1]
    q_ref = states["q"][:, start_delay:]
    dq_ref = states["q_dot"][:, start_delay:]
    a_ref = states["muscles"][:, start_delay:]
    u_ref = controls["muscles"][:, start_delay:]
    if use_torque:
        nbGT = biorbd_model.nbGeneralizedTorque()
    else:
        nbGT = 0
    nbMT = biorbd_model.nbMuscleTotal()
    nbQ = biorbd_model.nbQ()
    w_tau = "tau" in controls.keys()  # Check if there are residuals torques
    if w_tau:
        tau = controls["tau"]
    else:
        tau = np.zeros((nbGT, Ns + 1))

    if use_activation:
        x0 = np.hstack([q_ref[:, 0], dq_ref[:, 0]])
    else:
        x0 = np.hstack([q_ref[:, 0], dq_ref[:, 0], a_ref[:, 0]])

    # Get targets
    get_markers = markers_fun(biorbd_model)
    markers_target = np.zeros((3, biorbd_model.nbMarkers(), Ns + 1))
    for i in range(Ns + 1):
        markers_target[:, :, i] = get_markers(q_ref[:, i])
    muscles_target = u_ref

    # Add electromechanical delay to the control target
    muscles_target_real = np.ndarray((u_ref.shape[0], u_ref.shape[1]))
    for i in range(N_elec, u_ref.shape[1]):
        muscles_target_real[:, i] = u_ref[:, i - N_elec]
    for i in range(N_elec):
        muscles_target_real[:, i] = muscles_target_real[:, N_elec]

    # Build the graph
    ocp = prepare_ocp(
        biorbd_model=biorbd_model,
        final_time=T,
        x0=x0,
        nbGT=nbGT,
        number_shooting_points=Ns,
        use_torque=use_torque,
        use_activation=use_activation,
        use_SX=use_ACADOS,
    )

    # Set initial state
    ocp.nlp[0].x_bounds.min[:, 0] = x0
    ocp.nlp[0].x_bounds.max[:, 0] = x0

    # Set initial guess
    x_init = InitialGuessOption(x0, interpolation=InterpolationType.CONSTANT)
    u0 = np.array([tau_init] * nbGT + [muscle_init] * nbMT)
    u_init = InitialGuessOption(u0, interpolation=InterpolationType.CONSTANT)
    ocp.update_initial_guess(x_init, u_init)

    # Set objectives functions
    objectives = ObjectiveList()
    if use_activation:
        if TRACK_EMG:
            objectives.add(Objective.Lagrange.TRACK_MUSCLES_CONTROL, weight=100000, target=muscles_target_real[:, :-1])
        else:
            objectives.add(Objective.Lagrange.MINIMIZE_MUSCLES_CONTROL, weight=100000)
    else:
        if TRACK_EMG:
            objectives.add(Objective.Lagrange.TRACK_MUSCLES_CONTROL, weight=100000, target=muscles_target_real[:, :-1])
        else:
            objectives.add(Objective.Lagrange.MINIMIZE_MUSCLES_CONTROL, weight=100000)
    objectives.add(Objective.Lagrange.TRACK_MARKERS, weight=100000000, target=markers_target[:, :, :])
    objectives.add(Objective.Lagrange.MINIMIZE_STATE, weight=1, index=np.array(range(nbQ)))
    objectives.add(Objective.Lagrange.MINIMIZE_STATE, weight=10, index=np.array(range(nbQ, nbQ * 2)))
    if use_activation is not True:
        objectives.add(Objective.Lagrange.MINIMIZE_STATE, weight=1, index=np.array(range(nbQ * 2, nbQ * 2 + nbMT)))
    if use_torque:
        objectives.add(Objective.Lagrange.MINIMIZE_TORQUE, weight=100000000)
    ocp.update_objectives(objectives)

    # Solve the OCP (using Acados or Ipopt, depend of use_ACADOS flag)
    if use_ACADOS:
        tic = time()
        sol = ocp.solve(
            solver=Solver.ACADOS,
            show_online_optim=False,
            solver_options={
                "nlp_solver_tol_comp": 1e-6,
                "nlp_solver_tol_eq": 1e-6,
                "nlp_solver_tol_stat": 1e-6,
                "integrator_type": "IRK",
                "nlp_solver_type": "SQP",
                "sim_method_num_steps": 1,
            },
        )
        print(f"Time to solve with ACADOS : {time()-tic} s")

    else:
        tic = time()
        sol = ocp.solve(
            solver=Solver.IPOPT,
            show_online_optim=False,
            solver_options={
                "tol": 1e-4,
                "dual_inf_tol": 1e-6,
                "constr_viol_tol": 1e-6,
                "compl_inf_tol": 1e-6,
                "linear_solver": "ma57",
                "max_iter": 500,
                "hessian_approximation": "exact",
            },
        )
        print(f"Time to solve with IPOPT : {time() - tic} s")
    toc = time() - tic
    print(f"Total time to solve with ACADOS : {toc} s")

    # Store optimals data
    data_est = Data.get_data(ocp, sol)
    if use_activation:
        X_est = np.vstack([data_est[0]["q"], data_est[0]["q_dot"]])
    else:
        X_est = np.vstack([data_est[0]["q"], data_est[0]["q_dot"], data_est[0]["muscles"]])
    if use_torque:
        U_est = np.vstack([data_est[1]["tau"], data_est[1]["muscles"]])
    else:
        U_est = data_est[1]["muscles"]

    # Compute and print RMSE
    err = compute_err(
        init_offset,
        final_offset,
        X_est,
        U_est,
        Ns,
        biorbd_model,
        q_ref,
        dq_ref,
        tau,
        a_ref,
        u_ref,
        nbGT,
        use_activation=use_activation,
    )
    print(err)

    # Get force (estimate and reference)
    q_est = X_est[: biorbd_model.nbQ(), :]
    dq_est = X_est[biorbd_model.nbQ() : biorbd_model.nbQ() * 2, :]
    if use_activation:
        a_est = np.zeros((nbMT, Ns))
    else:
        a_est = X_est[-nbMT:, :]
    force_ref = np.ndarray((biorbd_model.nbMuscles(), Ns))
    force_est = np.ndarray((biorbd_model.nbMuscles(), Ns))
    get_force = force_func(biorbd_model, use_activation=use_activation)
    for i in range(biorbd_model.nbMuscles()):
        for j in range(Ns):
            force_est[i, j] = get_force(q_est[:, j], dq_est[:, j], a_est[:, j], U_est[nbGT:, j])[i, :]

    get_force = force_func(biorbd_model, use_activation=False)
    for i in range(biorbd_model.nbMuscles()):
        for k in range(Ns):
            force_ref[i, k] = get_force(q_ref[:, k], dq_ref[:, k], a_ref[:, k], u_ref[:, k])[i, :]
    err_tmp = np.array(
        [[Ns, 1, toc, toc, err["q"], err["q_dot"], err["tau"], err["muscles"], err["markers"], err["force"]]]
    )

    # Store stats (informations on RMSE, force and time to solve) in .mat file if flag is on True
    if save_stats:
        if os.path.isfile(f"solutions/stats_rt_activation_driven{use_activation}.mat"):
            matcontent = sio.loadmat(f"solutions/stats_rt_activation_driven{use_activation}.mat")
            err_mat = np.concatenate((matcontent["err_tries"], err_tmp))
            err_dic = {"err_tries": err_mat, "force_est": force_est, "force_ref": force_ref}
            sio.savemat(f"solutions/stats_rt_activation_driven{use_activation}.mat", err_dic)

    # Store results in .mat file for all tries if flag is on True
    if save_results:
        dic = {
            "X_est": X_est,
            "U_est": U_est,
            "x_sol": np.concatenate((q_ref, dq_ref)),
            "u_sol": u_ref,
            "markers_target": markers_target,
            "u_target": muscles_target,
            "time_per_mhe": toc,
            "time_tot": toc,
            "co_lvl": int(i),
            "marker_noise_lvl": 0,
            "EMG_noise_lvl": 0,
            "N_mhe": Ns,
            "N_tot": Ns,
            "rt_ratio": 1,
            "f_est": force_est,
            "f_ref": force_ref,
        }
        if TRACK_EMG:
            sio.savemat(
                f"solutions/track_full_w_EMG_excitation_driven_co_lvl{int(i)}_noise_lvl_0_0.mat",
                dic,
            )
        else:
            sio.savemat(
                f"solutions/track_full_wt_EMG_excitation_driven_co_lvl{int(i)}_noise_lvl_0_0.mat",
                dic,
            )

    # Plot and animate results if flag is on True
    if plot_and_animate:
        # --- Show results --- #
        result = ShowResult(ocp, sol)
        result.graphs()
        result.animate()
