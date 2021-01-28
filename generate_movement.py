# ----------------------------------------------------------------------------------------------------------------------
# The aim of this script is to generate a reference joint kinematics. Next step (generate_excitation_with_low_bounds.py)
# will be to track this kinematics to generate optimal excitations for several levels of co-contraction.
# ----------------------------------------------------------------------------------------------------------------------
from utils import *
from time import time
from bioptim import (
    OptimalControlProgram,
    ObjectiveList,
    ObjectiveFcn,
    DynamicsList,
    DynamicsFcn,
    BoundsList,
    QAndQDotBounds,
    InitialGuess,
    Solver,
    InterpolationType,
    Bounds,
)


def prepare_ocp(biorbd_model, final_time, number_shooting_points, x0, use_SX=False, nb_threads=8):
    # --- Options --- #
    # Model path
    activation_min, activation_max, activation_init = 0, 1, 0.1
    excitation_min, excitation_max, excitation_init = 0, 1, 0.2

    # Add objective functions
    objective_functions = ObjectiveList()

    # Dynamics
    dynamics = DynamicsList()
    dynamics.add(DynamicsFcn.MUSCLE_EXCITATIONS_DRIVEN)

    # State path constraint
    x_bounds = BoundsList()
    x_bounds.add(bounds=QAndQDotBounds(biorbd_model))
    # add muscle activation bounds
    x_bounds[0].concatenate(
        Bounds([activation_min] * biorbd_model.nbMuscles(), [activation_max] * biorbd_model.nbMuscles())
    )

    # Control path constraint
    u_bounds = BoundsList()
    u_bounds.add([excitation_min] * biorbd_model.nbMuscleTotal(), [excitation_max] * biorbd_model.nbMuscleTotal())

    # Initial guesses
    x_init = InitialGuess(
        np.tile(np.concatenate((x0, [activation_init] * biorbd_model.nbMuscles())), (number_shooting_points + 1, 1)).T,
        interpolation=InterpolationType.EACH_FRAME,
    )
    u0 = np.array([excitation_init] * biorbd_model.nbMuscles())
    u_init = InitialGuess(np.tile(u0, (number_shooting_points, 1)).T, interpolation=InterpolationType.EACH_FRAME)
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


def generate_state(model_path, T, Ns, nb_phase, x_phase):
    # Parameters of the problem
    biorbd_model = biorbd.Model(model_path)
    X_est = np.zeros((biorbd_model.nbQ() * 2 + biorbd_model.nbMuscleTotal(), nb_phase * Ns + 1))
    U_est = np.zeros((biorbd_model.nbMuscleTotal(), nb_phase * Ns))

    # Set the joint angles target for each phase
    x0 = x_phase[0, :]
    # Build graph
    ocp = prepare_ocp(biorbd_model=biorbd_model, final_time=T, number_shooting_points=Ns, x0=x0, use_SX=True)
    x0 = np.concatenate((x0, np.array([0.2] * biorbd_model.nbMuscles())))
    # Solve for each phase
    for phase in range(1, nb_phase + 1):
        # impose it as first state of next solve
        ocp.nlp[0].x_bounds.min[:, 0] = x0
        ocp.nlp[0].x_bounds.max[:, 0] = x0

        # update initial guess on states
        x_init = InitialGuess(np.tile(x0, (ocp.nlp[0].ns + 1, 1)).T, interpolation=InterpolationType.EACH_FRAME)
        ocp.update_initial_guess(x_init=x_init)

        # Update objectives functions
        objectives = ObjectiveList()
        objectives.add(ObjectiveFcn.Lagrange.MINIMIZE_STATE, weight=10, index=np.array(range(biorbd_model.nbQ())))
        objectives.add(
            ObjectiveFcn.Lagrange.MINIMIZE_STATE,
            weight=10,
            index=np.array(range(biorbd_model.nbQ(), biorbd_model.nbQ() * 2)),
        )
        objectives.add(
            ObjectiveFcn.Lagrange.MINIMIZE_STATE,
            weight=10,
            index=np.array(range(biorbd_model.nbQ() * 2, biorbd_model.nbQ() * 2 + biorbd_model.nbMuscles())),
        )
        objectives.add(ObjectiveFcn.Lagrange.MINIMIZE_MUSCLES_CONTROL, weight=10)

        objectives.add(
            ObjectiveFcn.Mayer.TRACK_STATE,
            weight=10000,
            target=np.array([x_phase[phase, : biorbd_model.nbQ() * 2]]).T,
            index=np.array(range(biorbd_model.nbQ() * 2)),
        )
        ocp.update_objectives(objectives)
        sol = ocp.solve(
            solver=Solver.ACADOS,
            show_online_optim=False,
            solver_options={
                # "nlp_solver_max_iter": 50,
                "nlp_solver_tol_comp": 1e-7,
                "nlp_solver_tol_eq": 1e-7,
                "nlp_solver_tol_stat": 1e-7,
            },
        )

        # get last state of previous solve
        x_out, u_out, x0 = switch_phase(ocp, sol)

        # Store optimal solution
        X_est[:, (phase - 1) * Ns : phase * Ns] = x_out
        U_est[:, (phase - 1) * Ns : phase * Ns] = u_out

    # Collect last state
    X_est[:, -1] = x0

    return X_est


def generate_exc_low_bounds(rate, model_path, X_est, low_exc):
    # Variable of the problem
    biorbd_model = biorbd.Model(model_path)
    q_ref = X_est[: biorbd_model.nbQ(), :]
    dq_ref = X_est[biorbd_model.nbQ() : biorbd_model.nbQ() * 2, :]
    Ns = q_ref.shape[1] - 1
    T = Ns / rate
    # Set lower bounds for excitations on muscles in co-contraction (here triceps). Depends on co-contraction level
    excitations_max = [1] * biorbd_model.nbMuscleTotal()
    excitations_init = []
    excitations_min = []
    for i in range(len(low_exc)):
        excitations_init.append([[0.05] * 6 + [low_exc[i]] * 3 + [0.05] * 10])
        excitations_min.append([[0] * 6 + [low_exc[i]] * 3 + [0] * 10])
    x0 = np.hstack([q_ref[:, 0], dq_ref[:, 0]])
    # Build the graph
    ocp = prepare_ocp(biorbd_model=biorbd_model, final_time=T, x0=x0, number_shooting_points=Ns, use_SX=True)
    X_est_co = np.ndarray((len(excitations_init), biorbd_model.nbQ() * 2 + biorbd_model.nbMuscles(), X_est.shape[1]))
    U_est_co = np.ndarray((len(excitations_init), biorbd_model.nbMuscles(), Ns + 1))
    # Solve for all co-contraction levels
    for co in range(len(excitations_init)):
        # Update excitations and to correspond to the co-contraction level
        u_i = excitations_init[co]
        u_mi = excitations_min[co]
        u_ma = excitations_max

        # Update initial guess
        u_init = InitialGuess(np.tile(u_i, (ocp.nlp[0].ns, 1)).T, interpolation=InterpolationType.EACH_FRAME)
        x_init = InitialGuess(
            np.tile(X_est[:, 0], (ocp.nlp[0].ns + 1, 1)).T, interpolation=InterpolationType.EACH_FRAME
        )
        ocp.update_initial_guess(x_init, u_init=u_init)

        # Update bounds
        u_bounds = BoundsList()
        u_bounds.add(u_mi[0], u_ma, interpolation=InterpolationType.CONSTANT)
        ocp.update_bounds(u_bounds=u_bounds)

        # Update objectives functions
        objectives = ObjectiveList()
        objectives.add(ObjectiveFcn.Lagrange.MINIMIZE_MUSCLES_CONTROL, weight=1000)
        objectives.add(
            ObjectiveFcn.Lagrange.TRACK_STATE,
            weight=100000,
            target=X_est[: biorbd_model.nbQ(), :],
            index=range(biorbd_model.nbQ()),
        )
        objectives.add(
            ObjectiveFcn.Lagrange.MINIMIZE_STATE,
            weight=100,
            index=range(biorbd_model.nbQ() * 2, biorbd_model.nbQ() * 2 + biorbd_model.nbMuscles()),
        )
        objectives.add(
            ObjectiveFcn.Lagrange.MINIMIZE_STATE, weight=1000, index=range(biorbd_model.nbQ(), biorbd_model.nbQ() * 2)
        )
        ocp.update_objectives(objectives)

        # Solve the OCP
        tic = time()
        sol = ocp.solve(
            solver=Solver.ACADOS,
            show_online_optim=False,
            solver_options={
                # "nlp_solver_max_iter": 50,
                "nlp_solver_tol_comp": 1e-5,
                "nlp_solver_tol_eq": 1e-5,
                "nlp_solver_tol_stat": 1e-5,
            },
        )
        print(f"Time to solve with ACADOS : {time() - tic} s")
        toc = time() - tic
        print(f"Total time to solve with ACADOS : {toc} s")

        # Get optimal solution
        data_est = Data.get_data(ocp, sol)

        X_est_co[co, :, :] = np.vstack([data_est[0]["q"], data_est[0]["qdot"], data_est[0]["muscles"]])
        U_est_co[co, :, :] = data_est[1]["muscles"]
    return X_est_co, U_est_co


def generate_final_data(rate, X_ref, U_ref, save_data, plot):
    # Variable of the problem
    biorbd_model = biorbd.Model("arm_wt_rot_scap.bioMod")
    Ns = X_ref.shape[2] - 1
    T = int(Ns / rate)
    x0 = X_ref[0, : -biorbd_model.nbMuscles(), 0]

    # Build the graph
    ocp = prepare_ocp(biorbd_model=biorbd_model, final_time=T, number_shooting_points=Ns, x0=x0, use_SX=True)
    X_ref_fin = X_ref
    U_ref_fin = U_ref
    q_init = X_ref[0, : biorbd_model.nbQ(), :]
    # Run problem for all co-contraction level
    for co in range(X_ref.shape[0]):
        # Get reference data for co-contraction level ranging from 1 to 3
        q_ref = X_ref[co, : biorbd_model.nbQ(), :]
        dq_ref = X_ref[co, biorbd_model.nbQ() : biorbd_model.nbQ() * 2, :]
        u_ref = U_ref[co, :, :]
        x_ref_0 = np.hstack([q_ref[:, 0], dq_ref[:, 0], [0.3] * biorbd_model.nbMuscles()])

        # Update initial guess
        x_init = InitialGuess(np.tile(x_ref_0, (Ns + 1, 1)).T, interpolation=InterpolationType.EACH_FRAME)
        u_init = InitialGuess(u_ref[:, :-1], interpolation=InterpolationType.EACH_FRAME)
        ocp.update_initial_guess(x_init, u_init)

        # Update bounds on state
        x_bounds = BoundsList()
        x_bounds.add(bounds=QAndQDotBounds(biorbd_model))
        x_bounds[0].concatenate(Bounds([0] * biorbd_model.nbMuscles(), [1] * biorbd_model.nbMuscles()))
        x_bounds[0].min[: biorbd_model.nbQ() * 2, 0] = x_ref_0[: biorbd_model.nbQ() * 2]
        x_bounds[0].max[: biorbd_model.nbQ() * 2, 0] = x_ref_0[: biorbd_model.nbQ() * 2]
        x_bounds[0].min[biorbd_model.nbQ() * 2 : biorbd_model.nbQ() * 2 + biorbd_model.nbMuscles(), 0] = [
            0.1
        ] * biorbd_model.nbMuscles()
        x_bounds[0].max[biorbd_model.nbQ() * 2 : biorbd_model.nbQ() * 2 + biorbd_model.nbMuscles(), 0] = [
            1
        ] * biorbd_model.nbMuscles()
        ocp.update_bounds(x_bounds=x_bounds)

        # Update Objectives
        objective_functions = ObjectiveList()
        objective_functions.add(
            ObjectiveFcn.Lagrange.MINIMIZE_STATE, weight=10, index=np.array(range(biorbd_model.nbQ()))
        )
        objective_functions.add(
            ObjectiveFcn.Lagrange.MINIMIZE_STATE,
            weight=100,
            index=np.array(range(biorbd_model.nbQ(), biorbd_model.nbQ() * 2)),
        )
        objective_functions.add(
            ObjectiveFcn.Lagrange.MINIMIZE_STATE,
            weight=100,
            index=np.array(range(biorbd_model.nbQ() * 2, biorbd_model.nbQ() * 2 + biorbd_model.nbMuscles())),
        )
        objective_functions.add(ObjectiveFcn.Lagrange.MINIMIZE_MUSCLES_CONTROL, weight=100)
        objective_functions.add(
            ObjectiveFcn.Lagrange.TRACK_MUSCLES_CONTROL,
            weight=10000,
            target=u_ref[[9, 10, 17, 18], :-1],
            index=np.array([9, 10, 17, 18]),
        )
        objective_functions.add(
            ObjectiveFcn.Lagrange.TRACK_STATE, weight=100000, target=q_init, index=np.array(range(biorbd_model.nbQ()))
        )

        # Solve OCP
        ocp.update_objectives(objective_functions)
        sol = ocp.solve(
            solver=Solver.ACADOS,
            show_online_optim=False,
            solver_options={
                # "nlp_solver_max_iter": 40,
                "nlp_solver_tol_comp": 1e-4,
                "nlp_solver_tol_eq": 1e-4,
                "nlp_solver_tol_stat": 1e-4,
            },
        )

        # Store optimal solutions
        states, controls = Data.get_data(ocp, sol)
        X_ref_fin[co, :, :] = np.concatenate((states["q"], states["qdot"], states["muscles"]))
        U_ref_fin[co, :, :] = controls["muscles"]
        # Save results in .bob file if flag is True
        if save_data:
            ocp.save_get_data(sol, f"solutions/sim_ac_{int(T * 1000)}ms_{Ns}sn_REACH2_co_level_{co}.bob")
    if plot:
        t = np.linspace(0, T, Ns + 1)
        plt.figure("Muscles controls")
        for i in range(biorbd_model.nbMuscles()):
            plt.subplot(4, 5, i + 1)
            for k in range(X_ref_fin.shape[0]):
                plt.step(t, U_ref[k, i, :])

        plt.figure("Q")
        for i in range(biorbd_model.nbQ()):
            plt.subplot(2, 3, i + 1)
            for k in range(X_ref_fin.shape[0]):
                plt.plot(X_ref[k, i, :])
            plt.figure("Q")

        plt.figure("dQ")
        for i in range(biorbd_model.nbQ()):
            plt.subplot(2, 3, i + 1)
            for k in range(X_ref_fin.shape[0]):
                plt.plot(X_ref[k, i + biorbd_model.nbQ(), :])
        plt.show()

    return X_ref_fin, U_ref_fin
