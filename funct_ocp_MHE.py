# ----------------------------------------------------------------------------------------------------------------------
# Run OCP in markers tracking and excitations tracking/minimizing for several sizes of windows
# ----------------------------------------------------------------------------------------------------------------------
from time import time
from utils import *
from bioptim import (
    BoundsList,
    QAndQDotBounds,
    InitialGuess,
    Solver,
    InterpolationType,
    Bounds,
)


def run_mhe(model_path, ocp, var, conf, fold):
    # Set variables
    Ns, T, Ns_mhe, rt_ratio = var["Ns"], var["T"], var["Ns_mhe"], var["rt_ratio"]
    co, marker_lvl, EMG_lvl = var["co"], var["marker_lvl"], var["EMG_lvl"]
    X_est, U_est = var["X_est"], var["U_est"]
    markers_target, muscles_target = var["markers_target"], var["muscles_target"]
    marker_noise_lvl, EMG_noise_lvl = var["marker_noise_lvl"], var["EMG_noise_lvl"]
    x_ref, u_ref = var["x_ref"], var["u_ref"]
    biorbd_model = biorbd.Model(model_path)
    nbQ, nbMT = biorbd_model.nbQ(), biorbd_model.nbMuscles()
    nbGT = biorbd_model.nbGeneralizedTorque() if conf["use_torque"] else 0
    q_ref, dq_ref, a_ref = x_ref[:nbQ, :], x_ref[nbQ : nbQ * 2, :], x_ref[nbQ * 2 :, :]
    if "TRACK_LESS_EMG" in conf.keys():
        TRACK_LESS_EMG = conf["TRACK_LESS_EMG"]
        idx = var["idx_muscle_track"] if conf["TRACK_LESS_EMG"] else False
    else:
        idx = None
        TRACK_LESS_EMG = False
    # Set number of tries
    nb_try = var["nb_try"] if conf["use_try"] else 1

    # Set variables' shape for all tries
    X_est_tries = np.ndarray((nb_try, X_est.shape[0], X_est.shape[1]))
    U_est_tries = np.ndarray((nb_try, U_est.shape[0], U_est.shape[1]))
    markers_target_tries = np.ndarray((nb_try, markers_target.shape[0], markers_target.shape[1], Ns + 1))
    muscles_target_tries = np.ndarray((nb_try, muscles_target.shape[0], Ns + 1))
    force_ref = np.ndarray((biorbd_model.nbMuscles(), Ns))
    force_est = np.ndarray((nb_try, biorbd_model.nbMuscles(), int(ceil(Ns / rt_ratio) - Ns_mhe)))

    err_tries = np.ndarray((nb_try, 10))
    # Loop for simulate some tries, generate new random noise to each try
    for tries in range(nb_try):
        # Print current optimisation configuration
        print(
            f"- Ns_mhe = {Ns_mhe}; Co_lvl: {co}; Marker_noise: {marker_lvl}; EMG_noise : {EMG_lvl}; nb_try : {tries} -"
        )
        # Generate data with noise
        if conf["use_noise"]:
            if marker_lvl != 0:
                markers_target = generate_noise(
                    biorbd_model, q_ref, u_ref, marker_noise_lvl[marker_lvl], EMG_noise_lvl[EMG_lvl]
                )[0]

            if EMG_lvl != 0:
                muscles_target = generate_noise(
                    biorbd_model, q_ref, u_ref, marker_noise_lvl[marker_lvl], EMG_noise_lvl[EMG_lvl]
                )[1]

        # Reload the model with the original markers
        biorbd_model = biorbd.Model(model_path)

        # Update bounds
        x_bounds = BoundsList()
        x_bounds.add(bounds=QAndQDotBounds(biorbd_model))
        if conf["use_activation"] is not True:
            x_bounds[0].concatenate(Bounds([0] * biorbd_model.nbMuscles(), [1] * biorbd_model.nbMuscles()))
        x_bounds[0].min[: biorbd_model.nbQ() * 2, 0] = x_ref[: biorbd_model.nbQ() * 2, 0] - 0.1
        x_bounds[0].max[: biorbd_model.nbQ() * 2, 0] = x_ref[: biorbd_model.nbQ() * 2, 0] + 0.1
        ocp.update_bounds(x_bounds)

        # Update initial guess
        x0 = x_ref[: biorbd_model.nbQ() * 2, 0] if conf["use_activation"] else x_ref[:, 0]
        x_init = InitialGuess(x0, interpolation=InterpolationType.CONSTANT)
        u0 = muscles_target
        u_init = InitialGuess(u0[:, 0], interpolation=InterpolationType.CONSTANT)
        ocp.update_initial_guess(x_init, u_init)

        # Update objectives functions
        ocp.update_objectives(
            define_objective(
                conf["use_activation"],
                conf["use_torque"],
                conf["TRACK_EMG"],
                0,
                rt_ratio,
                nbGT,
                Ns_mhe,
                muscles_target,
                markers_target,
                conf["with_low_weight"],
                biorbd_model,
                idx=idx,
                TRACK_LESS_EMG=TRACK_LESS_EMG,
            )
        )

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
        if sol["status"] != 0 and conf["save_status"]:
            if conf["TRACK_EMG"]:
                f = open(f"{fold}status_track_rt_EMG{conf['TRACK_EMG']}.txt", "a")
            else:
                f = open(f"{fold}status_track_rt_EMG{conf['TRACK_EMG']}.txt", "a")
            f.write(f"{Ns_mhe}; {co}; {marker_lvl}; {EMG_lvl}; {tries}; " f"'init'\n")
            f.close()

        # Set solutions and set initial guess for next optimisation
        x0, u0, X_est[:, 0], U_est[:, 0] = warm_start_mhe(ocp, sol, use_activation=conf["use_activation"])

        tic = time()  # Save initial time
        for iter in range(1, ceil(Ns / rt_ratio - Ns_mhe)):
            # set initial state
            ocp.nlp[0].x_bounds.min[:, 0] = x0[:, 0]
            ocp.nlp[0].x_bounds.max[:, 0] = x0[:, 0]

            # Update initial guess
            x_init = InitialGuess(x0, interpolation=InterpolationType.EACH_FRAME)
            u_init = InitialGuess(u0, interpolation=InterpolationType.EACH_FRAME)
            ocp.update_initial_guess(x_init, u_init)

            # Update objectives functions
            ocp.update_objectives(
                define_objective(
                    conf["use_activation"],
                    conf["use_torque"],
                    conf["TRACK_EMG"],
                    iter,
                    rt_ratio,
                    nbGT,
                    Ns_mhe,
                    muscles_target,
                    markers_target,
                    conf["with_low_weight"],
                    biorbd_model,
                    idx,
                    TRACK_LESS_EMG=TRACK_LESS_EMG,
                )
            )

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
            # Set solutions and set initial guess for next optimisation
            x0, u0, X_est[:, iter], u_out = warm_start_mhe(ocp, sol, use_activation=conf["use_activation"])
            if iter < int((Ns / rt_ratio) - Ns_mhe):
                U_est[:, iter] = u_out

            # Compute muscular force at each iteration
            q_est = X_est[: biorbd_model.nbQ(), :]
            dq_est = X_est[biorbd_model.nbQ() : biorbd_model.nbQ() * 2, :]
            a_est = np.zeros((nbMT, Ns)) if conf["use_activation"] else X_est[-nbMT:, :]
            for i in range(biorbd_model.nbMuscles()):
                for j in [iter]:
                    force_est[tries, i, j] = var["get_force"](q_est[:, j], dq_est[:, j], a_est[:, j], U_est[nbGT:, j])[
                        i, :
                    ]
            # Save status of optimisation
            if sol["status"] != 0 and conf["save_status"]:
                if conf["TRACK_EMG"]:
                    f = open(f"{fold}status_track_rt_EMG{conf['TRACK_EMG']}.txt", "a")
                else:
                    f = open(f"{fold}status_track_rt_EMG{conf['TRACK_EMG']}.txt", "a")
                f.write(f"{Ns_mhe}; {co}; {marker_lvl}; {EMG_lvl}; {tries}; " f"{iter}\n")
                f.close()

        toc = time() - tic  # Save total time to solve
        # Store data
        X_est_tries[tries, :, :], U_est_tries[tries, :, :] = X_est, U_est
        markers_target_tries[tries, :, :, :] = markers_target
        muscles_target_tries[tries, :, :] = muscles_target

        # Compute reference muscular force
        get_force = force_func(biorbd_model, use_activation=False)
        for i in range(biorbd_model.nbMuscles()):
            for k in range(Ns):
                force_ref[i, k] = get_force(q_ref[:, k], dq_ref[:, k], a_ref[:, k], u_ref[nbGT:, k])[i, :]

        # Print some informations about optimisations
        print(f"nb loops: {iter}")
        print(f"Total time to solve with ACADOS : {toc} s")
        print(f"Time per MHE iter. : {toc/iter} s")
        tau = np.zeros((nbGT, Ns + 1))
        # Compute and print RMSE
        err_offset = Ns_mhe
        err = compute_err_mhe(
            var["init_offset"],
            var["final_offset"],
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
            use_activation=conf["use_activation"],
        )
        err_tries[int(tries), :] = [
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
        print(err)
        if conf["plot"]:
            plot_MHE_results(
                biorbd_model,
                X_est,
                q_ref,
                Ns,
                rt_ratio,
                nbQ,
                dq_ref,
                U_est,
                u_ref,
                nbGT,
                muscles_target,
                force_est,
                force_ref,
                tries,
                markers_target,
                conf["use_torque"],
            )

    return err_tries, force_est, force_ref, X_est_tries, U_est_tries, muscles_target_tries, markers_target_tries, toc
