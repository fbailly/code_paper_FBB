from time import time
import matplotlib.pyplot as plt
import pickle
import scipy.io as sio
import sys
import os
from utils import *
from bioptim import (
    OptimalControlProgram,
    ObjectiveList,
    Objective,
    DynamicsTypeList,
    DynamicsType,
    BoundsList,
    QAndQDotBounds,
    InitialGuessOption,
    Solver,
    InterpolationType,
    Bounds,
)


def prepare_ocp(
    biorbd_model, final_time, x0, nbGT, number_shooting_points, use_SX=False, nb_threads=8, use_torque=False
):
    # --- Options --- #
    # Model path
    biorbd_model = biorbd_model
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
    biorbd_model = biorbd.Model("arm_wt_rot_scap.bioMod")
    use_torque = False
    use_activation = False
    use_ACADOS = True
    WRITE_STATS = False
    save_status = False
    save_results = False
    TRACK_EMG = True
    plot = True
    with_low_weight = False
    use_noise = False
    use_co = False
    use_bash = False
    use_try = False
    use_N_elec = False  # Use an electromechanical delay when using activation driven
    if use_activation:
        use_N_elec = True

    # Choose the folder to save data
    if TRACK_EMG:
        if use_activation:
            fold = "solutions/w_track_emg_rt_act/"
        else:
            if with_low_weight:
                fold = "solutions/w_track_emg_rt_exc_low_weight/"
            else:
                fold = "solutions/w_track_emg_rt_exc/"
    else:
        if use_activation:
            fold = "solutions/wt_track_emg_rt_act/"
        else:
            if with_low_weight:
                fold = "solutions/wt_track_emg_rt_exc_low_weight/"
            else:
                fold = "solutions/wt_track_emg_rt_exc/"

    # Variable of the problem
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
    nbMT = biorbd_model.nbMuscleTotal()
    nbQ = biorbd_model.nbQ()
    if use_activation:
        x_ref = np.zeros((biorbd_model.nbQ() * 2))
    else:
        x_ref = np.zeros((biorbd_model.nbQ() * 2 + biorbd_model.nbMuscles()))

    if use_torque:
        nbGT = biorbd_model.nbGeneralizedTorque()
    else:
        nbGT = 0

    if use_bash:
        Ns_mhe = int(sys.argv[1])

    # Set ratio to be in real time (depend of the PC configuration and the problem complexity)
    if use_activation is not True:
        rt_ratio_tot = [3, 3, 3, 3, 3, 5, 5, 6, 6, 6, 6, 6, 6, 6]  # For Nmhe = [3, 16]
    else:
        rt_ratio_tot = [2, 2, 2, 3, 3, 3, 4, 4, 4, 4, 4, 4, 4, 5, 5, 5, 5, 5, 5, 5]  # For Nmhe = [3, 22]
    rt_ratio = rt_ratio_tot[Ns_mhe - 3]  # Get ratio from the list above
    T_mhe = T / (Ns / rt_ratio) * Ns_mhe  # Compute the new time of OCP

    # Set number of tries
    if use_try:
        nb_try = 30
    else:
        nb_try = 1

    # Noise informations
    if use_noise:
        marker_noise_lvl = [0, 0.002, 0.005, 0.01]
        EMG_noise_lvl = [0, 1, 1.5, 2]
    else:
        marker_noise_lvl = [0]
        EMG_noise_lvl = [0]

    # Set size of optimal states and controls
    if use_activation:
        X_est = np.zeros((biorbd_model.nbQ() * 2, ceil((Ns + 1) / rt_ratio) - Ns_mhe))
    else:
        X_est = np.zeros((biorbd_model.nbQ() * 2 + biorbd_model.nbMuscles(), ceil((Ns + 1) / rt_ratio) - Ns_mhe))
    U_est = np.zeros((nbGT + biorbd_model.nbMuscleTotal(), ceil(Ns / rt_ratio) - Ns_mhe))

    # Set number of co-contraction level
    nb_co_lvl = 1
    if use_co:
        nb_co_lvl = 4

    # Build the graph
    ocp = prepare_ocp(
        biorbd_model=biorbd_model,
        final_time=T_mhe,
        x0=x_ref,
        nbGT=nbGT,
        number_shooting_points=Ns_mhe,
        use_torque=use_torque,
        use_SX=use_ACADOS,
    )

    # Initialize files where to save the status of optimisations
    if TRACK_EMG and save_status:
        f = open(f"{fold}status_track_rt_EMG{TRACK_EMG}.txt", "a")
        f.write("Ns_mhe;  Co_lvl;  Marker_noise;  EMG_noise;  nb_try;  iter\n")
        f.close()
    elif TRACK_EMG is not True and save_status:
        f = open(f"{fold}status_track_rt_EMG{TRACK_EMG}.txt", "a")
        f.write("Ns_mhe;  Co_lvl;  Marker_noise;  EMG_noise;  nb_try;  iter\n")
        f.close()

    # Solve in a loop for each co-contraction level
    for co in range(0, nb_co_lvl):

        # Get data of reference movement, depend of co-contraction level
        motion = "REACH2"
        with open(f"solutions/sim_ac_8000ms_800sn_{motion}_co_level_{co}.bob", "rb") as file:
            data = pickle.load(file)
        states = data["data"][0]
        controls = data["data"][1]
        q_ref = states["q"][:, start_delay:]
        dq_ref = states["q_dot"][:, start_delay:]
        a_ref = states["muscles"][:, start_delay:]
        w_tau = "tau" in controls.keys()  # Check if there are residuals torques
        if w_tau:
            tau = controls["tau"]
        else:
            tau = np.zeros((nbGT, Ns + 1))

        if use_torque:
            u_ref = np.concatenate((tau, controls["muscles"][:, start_delay:]))
        else:
            u_ref = controls["muscles"][:, start_delay:]

        # Loop for marker and EMG noise
        for marker_lvl in range(len(marker_noise_lvl)):
            if TRACK_EMG is not True:
                emg_range = range(0, 1)  # No need to solve minimize excitations with noise on EMG
            else:
                emg_range = range(len(EMG_noise_lvl))
            for EMG_lvl in emg_range:

                # Get targets
                get_markers = markers_fun(biorbd_model)
                markers_target = np.zeros((3, biorbd_model.nbMarkers(), Ns + 1))
                for i in range(Ns + 1):
                    markers_target[:, :, i] = get_markers(q_ref[:, i])
                muscles_target = u_ref

                # Add electromechanical delay to the control target
                if use_N_elec:
                    muscles_target = np.ndarray((u_ref.shape[0], u_ref.shape[1]))
                    for i in range(N_elec, u_ref.shape[1]):
                        muscles_target[:, i] = u_ref[:, i - N_elec]
                    for i in range(N_elec):
                        muscles_target[:, i] = muscles_target[:, N_elec]

                # Set variables for all tries
                X_est_tries = np.ndarray((nb_try, X_est.shape[0], X_est.shape[1]))
                U_est_tries = np.ndarray((nb_try, U_est.shape[0], U_est.shape[1]))
                markers_target_tries = np.ndarray((nb_try, markers_target.shape[0], markers_target.shape[1], Ns + 1))
                muscles_target_tries = np.ndarray((nb_try, muscles_target.shape[0], Ns + 1))
                force_ref = np.ndarray((biorbd_model.nbMuscles(), Ns))
                force_est = np.ndarray((nb_try, biorbd_model.nbMuscles(), int(ceil(Ns / rt_ratio) - Ns_mhe)))
                if use_activation:
                    x_ref = np.concatenate((q_ref, dq_ref))
                else:
                    x_ref = np.concatenate((q_ref, dq_ref, a_ref))
                err_tries = np.ndarray((nb_try, 10))

                # Loop for simulate some tries, generate new random noise to each try
                for tries in range(nb_try):
                    # Print which optimisation is running
                    print(
                        f"--- Ns_mhe = {Ns_mhe}; Co_lvl: {co}; Marker_noise: {marker_lvl}; EMG_noise : {EMG_lvl}; nb_try : {tries} ---"
                    )

                    # Generate data with noise
                    if use_noise:
                        if marker_lvl != 0:
                            markers_target = generate_noise(
                                biorbd_model, q_ref, u_ref, marker_noise_lvl[marker_lvl], EMG_noise_lvl[EMG_lvl]
                            )[0]

                        if EMG_lvl != 0:
                            muscles_target = generate_noise(
                                biorbd_model, q_ref, u_ref, marker_noise_lvl[marker_lvl], EMG_noise_lvl[EMG_lvl]
                            )[1]

                    # Reload the model with the original markers
                    biorbd_model = biorbd.Model("arm_wt_rot_scap.bioMod")

                    # Update bounds
                    x_bounds = BoundsList()
                    x_bounds.add(QAndQDotBounds(biorbd_model))
                    if use_activation is not True:
                        x_bounds[0].concatenate(Bounds([0] * biorbd_model.nbMuscles(), [1] * biorbd_model.nbMuscles()))

                    x_bounds[0].min[: biorbd_model.nbQ() * 2, 0] = x_ref[: biorbd_model.nbQ() * 2, 0] - 0.1
                    x_bounds[0].max[: biorbd_model.nbQ() * 2, 0] = x_ref[: biorbd_model.nbQ() * 2, 0] + 0.1
                    ocp.update_bounds(x_bounds)

                    # Update initial guess
                    x_init = InitialGuessOption(x_ref[:, 0], interpolation=InterpolationType.CONSTANT)
                    u0 = muscles_target
                    u_init = InitialGuessOption(u0[:, 0], interpolation=InterpolationType.CONSTANT)
                    ocp.update_initial_guess(x_init, u_init)

                    # Update objectives functions
                    objectives = ObjectiveList()
                    if TRACK_EMG:
                        if with_low_weight:
                            w_marker = 10000000
                            w_control = 1000000
                        else:
                            w_marker = 1000000000
                            w_control = 1000000
                        w_torque = 100000000
                        objectives.add(
                            Objective.Lagrange.TRACK_MUSCLES_CONTROL,
                            weight=w_control,
                            target=muscles_target[nbGT:, 0 : Ns_mhe * rt_ratio : rt_ratio],
                        )
                        if use_torque:
                            objectives.add(Objective.Lagrange.MINIMIZE_TORQUE, weight=w_torque)
                        objectives.add(Objective.Lagrange.MINIMIZE_MUSCLES_CONTROL, weight=10)

                    else:
                        if with_low_weight:
                            w_marker = 1000000
                        else:
                            w_marker = 10000000
                        w_control = 10000
                        w_torque = 10000000
                        objectives.add(Objective.Lagrange.MINIMIZE_MUSCLES_CONTROL, weight=w_control)
                        if use_torque:
                            objectives.add(Objective.Lagrange.MINIMIZE_TORQUE, weight=w_torque)

                    objectives.add(
                        Objective.Lagrange.TRACK_MARKERS,
                        weight=w_marker,
                        target=markers_target[:, :, 0 : (Ns_mhe + 1) * rt_ratio : rt_ratio],
                    )
                    objectives.add(
                        Objective.Lagrange.MINIMIZE_STATE, weight=10, index=np.array(range(biorbd_model.nbQ()))
                    )
                    objectives.add(
                        Objective.Lagrange.MINIMIZE_STATE,
                        weight=10,
                        index=np.array(range(biorbd_model.nbQ(), biorbd_model.nbQ() * 2)),
                    )
                    if use_activation is not True:
                        objectives.add(
                            Objective.Lagrange.MINIMIZE_STATE,
                            weight=10,
                            index=np.array(
                                range(biorbd_model.nbQ() * 2, biorbd_model.nbQ() * 2 + biorbd_model.nbMuscles())
                            ),
                        )
                    ocp.update_objectives(objectives)

                    # Initialize the solver options
                    if co == 0 and marker_lvl == 0 and EMG_lvl == 0 and tries == 0:
                        sol = ocp.solve(
                            solver=Solver.ACADOS,
                            show_online_optim=False,
                            solver_options={
                                "nlp_solver_tol_comp": 1e-4,
                                "nlp_solver_tol_eq": 1e-4,
                                "nlp_solver_tol_stat": 1e-4,
                                "integrator_type": "IRK",
                                "nlp_solver_type": "SQP",
                                "sim_method_num_steps": 1,
                                "print_level": 0,
                                "nlp_solver_max_iter": 15,
                            },
                        )

                    # Update all allowed solver options
                    else:
                        sol = ocp.solve(
                            solver=Solver.ACADOS,
                            show_online_optim=False,
                            solver_options={
                                "nlp_solver_tol_comp": 1e-4,
                                "nlp_solver_tol_eq": 1e-4,
                                "nlp_solver_tol_stat": 1e-4,
                            },
                        )

                    # Save status of optimisation
                    if sol["status"] != 0 and save_status:
                        if TRACK_EMG:
                            f = open(f"{fold}status_track_rt_EMG{TRACK_EMG}.txt", "a")
                        else:
                            f = open(f"{fold}status_track_rt_EMG{TRACK_EMG}.txt", "a")
                        f.write(f"{Ns_mhe}; {co}; {marker_lvl}; {EMG_lvl}; {tries}; " f"'init'\n")
                        f.close()

                    # Set solutions and set initial guess for next optimisation
                    x0, u0, x_out, u_out = warm_start_mhe(ocp, sol, use_activation=use_activation)
                    X_est[:, 0] = x_out
                    U_est[:, 0] = u_out

                    tic = time()  # Save initial time
                    for iter in range(1, ceil((Ns + 1) / rt_ratio - Ns_mhe)):

                        # set initial state
                        ocp.nlp[0].x_bounds.min[:, 0] = x0[:, 0]
                        ocp.nlp[0].x_bounds.max[:, 0] = x0[:, 0]

                        # Update initial guess
                        x_init = InitialGuessOption(x0, interpolation=InterpolationType.EACH_FRAME)
                        u_init = InitialGuessOption(u0, interpolation=InterpolationType.EACH_FRAME)
                        ocp.update_initial_guess(x_init, u_init)

                        # Update objectives functions
                        objectives = ObjectiveList()
                        if TRACK_EMG:
                            objectives.add(
                                Objective.Lagrange.TRACK_MUSCLES_CONTROL,
                                weight=w_control,
                                target=muscles_target[nbGT:, iter * rt_ratio : (Ns_mhe + iter) * rt_ratio : rt_ratio],
                            )
                            objectives.add(Objective.Lagrange.MINIMIZE_MUSCLES_CONTROL, weight=10)
                            if use_torque:
                                objectives.add(Objective.Lagrange.MINIMIZE_TORQUE, weight=w_torque)
                        else:
                            w_torque = 1000000
                            objectives.add(
                                Objective.Lagrange.MINIMIZE_MUSCLES_CONTROL,
                                weight=w_control,
                            )
                            if use_torque:
                                objectives.add(Objective.Lagrange.MINIMIZE_TORQUE, weight=w_torque)

                        objectives.add(
                            Objective.Lagrange.TRACK_MARKERS,
                            weight=w_marker,
                            target=markers_target[:, :, iter * rt_ratio : (Ns_mhe + iter + 1) * rt_ratio : rt_ratio],
                        )
                        objectives.add(
                            Objective.Lagrange.MINIMIZE_STATE, weight=1, index=np.array(range(biorbd_model.nbQ()))
                        )
                        objectives.add(
                            Objective.Lagrange.MINIMIZE_STATE,
                            weight=10,
                            index=np.array(range(biorbd_model.nbQ(), biorbd_model.nbQ() * 2)),
                        )
                        if use_activation is not True:
                            objectives.add(
                                Objective.Lagrange.MINIMIZE_STATE,
                                weight=1,
                                index=np.array(
                                    range(biorbd_model.nbQ() * 2, biorbd_model.nbQ() * 2 + biorbd_model.nbMuscles())
                                ),
                            )
                        ocp.update_objectives(objectives)

                        # Solve problem
                        sol = ocp.solve(
                            solver=Solver.ACADOS,
                            show_online_optim=False,
                            solver_options={
                                "nlp_solver_tol_comp": 1e-4,
                                "nlp_solver_tol_eq": 1e-4,
                                "nlp_solver_tol_stat": 1e-4,
                            },
                        )

                        # Save status of optimisation
                        if sol["status"] != 0 and save_status:
                            if TRACK_EMG:
                                f = open(f"{fold}status_track_rt_EMG{TRACK_EMG}.txt", "a")
                            else:
                                f = open(f"{fold}status_track_rt_EMG{TRACK_EMG}.txt", "a")
                            f.write(f"{Ns_mhe}; {co}; {marker_lvl}; {EMG_lvl}; {tries}; " f"{iter}\n")
                            f.close()

                        # Set solutions and set initial guess for next optimisation
                        x0, u0, x_out, u_out = warm_start_mhe(ocp, sol, use_activation=use_activation)
                        X_est[:, iter] = x_out
                        if iter < ceil(Ns / rt_ratio) - Ns_mhe:
                            U_est[:, iter] = u_out

                    toc = time() - tic  # Save total time to solve

                    # Store data
                    X_est_tries[tries, :, :] = X_est
                    U_est_tries[tries, :, :] = U_est
                    markers_target_tries[tries, :, :, :] = markers_target
                    muscles_target_tries[tries, :, :] = muscles_target

                    # Compute muscular force
                    q_est = X_est[: biorbd_model.nbQ(), :]
                    dq_est = X_est[biorbd_model.nbQ() : biorbd_model.nbQ() * 2, :]
                    if use_activation:
                        a_est = np.zeros((nbMT, Ns))
                    else:
                        a_est = X_est[-nbMT:, :]
                    get_force = force_func(biorbd_model, use_activation=use_activation)
                    for i in range(biorbd_model.nbMuscles()):
                        for j in range(int(ceil(Ns / rt_ratio) - Ns_mhe)):
                            force_est[tries, i, j] = get_force(q_est[:, j], dq_est[:, j], a_est[:, j], U_est[nbGT:, j])[
                                i, :
                            ]

                    get_force = force_func(biorbd_model, use_activation=False)
                    for i in range(biorbd_model.nbMuscles()):
                        for k in range(Ns):
                            force_ref[i, k] = get_force(q_ref[:, k], dq_ref[:, k], a_ref[:, k], u_ref[nbGT:, k])[i, :]

                    # Print some informations about optimisations
                    print(f"nb loops: {iter}")
                    print(f"Total time to solve with ACADOS : {toc} s")
                    print(f"Time per MHE iter. : {toc/iter} s")

                    # Compute and print RMSE
                    err_offset = Ns_mhe
                    err = compute_err_mhe(
                        init_offset,
                        final_offset,
                        err_offset,
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
                        ratio=rt_ratio,
                        use_activation=use_activation,
                    )
                    err_tmp = [
                        Ns_mhe,
                        rt_ratio,
                        toc,
                        toc / iter,
                        err["q"],
                        err["q_dot"],
                        err["tau"],
                        err["muscles"],
                        err["markers"],
                        err["force"],
                    ]
                    err_tries[int(tries), :] = err_tmp
                    print(err)

                    # Plot results if flag is on True
                    if plot:
                        plt.figure("q")
                        for i in range(biorbd_model.nbQ()):
                            plt.subplot(3, 2, i + 1)
                            plt.plot(X_est[i, :], "x")
                            plt.plot(q_ref[i, 0 : Ns + 1 : rt_ratio])
                        plt.legend(
                            labels=["q_est", "q_ref"], bbox_to_anchor=(1.05, 1), loc="upper left", borderaxespad=0.0
                        )

                        plt.figure("qdot")
                        for i in range(biorbd_model.nbQ(), biorbd_model.nbQ() * 2):
                            plt.subplot(3, 2, i - nbQ + 1)
                            plt.plot(X_est[i, :], "x")
                            plt.plot(dq_ref[i - nbQ, 0 : Ns + 1 : rt_ratio])
                        plt.legend(
                            labels=["q_est", "q_ref"], bbox_to_anchor=(1.05, 1), loc="upper left", borderaxespad=0.0
                        )

                        plt.figure("Tau")
                        for i in range(biorbd_model.nbQ()):
                            plt.subplot(3, 2, i + 1)
                            plt.plot(U_est[i, :], "x")
                            plt.plot(u_ref[i, 0 : Ns + 1 : rt_ratio])
                            plt.plot(muscles_target[i, :], "k--")
                        plt.legend(
                            labels=["Tau_est", "Tau_ref"], bbox_to_anchor=(1.05, 1), loc="upper left", borderaxespad=0.0
                        )

                        plt.figure("Muscles excitations")
                        for i in range(biorbd_model.nbMuscles()):
                            plt.subplot(4, 5, i + 1)
                            plt.plot(U_est[nbGT + i, :])
                            plt.plot(u_ref[nbGT + i, 0:Ns:rt_ratio], c="red")
                            plt.plot(muscles_target[nbGT + i, 0:Ns:rt_ratio], "k--")
                            plt.title(biorbd_model.muscleNames()[i].to_string())
                        plt.legend(
                            labels=["u_est", "u_ref", "u_with_noise"],
                            bbox_to_anchor=(1.05, 1),
                            loc="upper left",
                            borderaxespad=0.0,
                        )

                        plt.figure("Muscles_force")
                        for i in range(biorbd_model.nbMuscles()):
                            plt.subplot(4, 5, i + 1)
                            plt.plot(force_est[tries, i, :])
                            plt.plot(force_ref[i, 0:Ns:rt_ratio], c="red")
                            plt.title(biorbd_model.muscleNames()[i].to_string())
                        plt.legend(
                            labels=["f_est", "f_ref"], bbox_to_anchor=(1.05, 1), loc="upper left", borderaxespad=0.0
                        )

                        n_mark = biorbd_model.nbMarkers()
                        get_markers = markers_fun(biorbd_model)
                        markers = np.zeros((3, biorbd_model.nbMarkers(), q_ref.shape[1]))
                        for i in range(q_ref.shape[1]):
                            markers[:, :, i] = get_markers(q_ref[:, i])
                        markers_est = np.zeros((3, biorbd_model.nbMarkers(), X_est.shape[1]))
                        for i in range(X_est.shape[1]):
                            markers_est[:, :, i] = get_markers(X_est[: biorbd_model.nbQ(), i])

                        plt.figure("Markers")
                        for i in range(markers_target.shape[1]):
                            plt.plot(markers_target[:, i, 0:Ns:rt_ratio].T, "k")
                            plt.plot(markers[:, i, 0:Ns:rt_ratio].T, "r--")
                            plt.plot(markers_est[:, i, :].T, "b")
                        plt.xlabel("Time")
                        plt.ylabel("Markers Position")
                        plt.show()

                # Store stats (informations on RMSE, force and time to solve) in .mat file if flag is on True
                err_dic = {"err_tries": err_tries, "force_est": force_est, "force_ref": force_ref}
                if WRITE_STATS:
                    if os.path.isfile(f"solutions/stats_rt_activation_driven{use_activation}.mat"):
                        matcontent = sio.loadmat(f"solutions/stats_rt_activation_driven{use_activation}.mat")
                        err_mat = np.concatenate((matcontent["err_tries"], err_tries))
                        err_dic = {"err_tries": err_mat, "force_est": force_est, "force_ref": force_ref}
                        sio.savemat(f"solutions/stats_rt_activation_driven{use_activation}.mat", err_dic)
                    else:
                        sio.savemat(f"solutions/stats_rt_activation_driven{use_activation}.mat", err_dic)

                # Store results in .mat file for all tries if flag is on True
                if save_results:
                    dic = {
                        "X_est": X_est_tries,
                        "U_est": U_est_tries,
                        "x_sol": x_ref,
                        "u_sol": u_ref,
                        "markers_target": markers_target_tries,
                        "u_target": muscles_target_tries,
                        "time_per_mhe": toc / (iter),
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
                    if TRACK_EMG:
                        sio.savemat(
                            f"{fold}track_mhe_w_EMG_excitation_driven_co_lvl{co}_noise_lvl_{marker_noise_lvl[marker_lvl]}_{EMG_noise_lvl[EMG_lvl]}.mat",
                            dic,
                        )
                    else:
                        sio.savemat(
                            f"{fold}track_mhe_wt_EMG_excitation_driven_co_lvl{co}_noise_lvl_{marker_noise_lvl[marker_lvl]}_{EMG_noise_lvl[EMG_lvl]}.mat",
                            dic,
                        )
