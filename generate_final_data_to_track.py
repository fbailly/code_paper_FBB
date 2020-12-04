# ----------------------------------------------------------------------------------------------------------------------
# The aim of the script is to generate reference data to track for OCP. We track the states and antagonist muscles
# excitations from generate_excitations_with_lower_bounds.py, and we minimize all others muscles excitations to have
# the optimal movement.
# ----------------------------------------------------------------------------------------------------------------------
import biorbd
import numpy as np
import pickle
import matplotlib.pyplot as plt
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
    Data,
)


def prepare_ocp(
    biorbd_model,
    final_time,
    number_shooting_points,
    x0,
    use_SX=False,
    nb_threads=8,
):
    # --- Options --- #
    # Model path
    biorbd_model = biorbd_model
    activation_min, activation_max, activation_init = 0, 1, 0.1
    excitation_min, excitation_max, excitation_init = 0, 1, 0.2

    # Add objective functions
    objective_functions = ObjectiveList()

    # Dynamics
    dynamics = DynamicsTypeList()
    dynamics.add(DynamicsType.MUSCLE_EXCITATIONS_DRIVEN)

    # State path constraint
    x_bounds = BoundsList()
    x_bounds.add(QAndQDotBounds(biorbd_model))

    # add muscle activation bounds
    x_bounds[0].concatenate(
        Bounds([activation_min] * biorbd_model.nbMuscles(), [activation_max] * biorbd_model.nbMuscles())
    )

    # Control path constraint
    u_bounds = BoundsList()
    u_bounds.add(
        [
            [excitation_min] * biorbd_model.nbMuscleTotal(),
            [excitation_max] * biorbd_model.nbMuscleTotal(),
        ]
    )

    # Initial guesses
    x_init = InitialGuessOption(
        np.tile(np.concatenate((x0, [activation_init] * biorbd_model.nbMuscles())), (number_shooting_points + 1, 1)).T,
        interpolation=InterpolationType.EACH_FRAME,
    )
    u0 = np.array([excitation_init] * biorbd_model.nbMuscles())
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


# Configuration of the problem
save_data = True
plot = True

# Variable of the problem
biorbd_model = biorbd.Model("arm_wt_rot_scap.bioMod")
T = 8
Ns = 800
x0 = np.array([0.0, -0.2, 0, 0, 0, 0, 0, 0])
xT = np.array([-0.2, -1.3, -0.5, 0.5, 0, 0, 0, 0])

# Get reference data with no co-contraction level to track corresponding states
with open(f"solutions/sim_ac_{int(T * 1000)}ms_{Ns}sn_REACH2_co_level_0.bob", "rb") as file:
    data = pickle.load(file)
states = data["data"][0]
controls = data["data"][1]
q_init = states["q"]

# Build the graph
ocp = prepare_ocp(biorbd_model=biorbd_model, final_time=T, number_shooting_points=Ns, x0=x0, use_SX=True)

# Run problem for all co-contraction level
for co in range(1, 4):
    # Get reference data for co-contraction level ranging from 1 to 3
    with open(f"solutions/sim_ac_{int(T * 1000)}ms_{Ns}sn_REACH2_co_level_{co}_tmp.bob", "rb") as file:
        data = pickle.load(file)
    states = data["data"][0]
    controls = data["data"][1]
    q_ref = states["q"]
    dq_ref = states["q_dot"]
    a_ref = states["muscles"]
    u_ref = controls["muscles"]
    x_ref = np.hstack([q_ref[:, 0], dq_ref[:, 0], [0.3] * biorbd_model.nbMuscles()])

    # Update initial guess
    x_init = InitialGuessOption(np.tile(x_ref, (ocp.nlp[0].ns + 1, 1)).T, interpolation=InterpolationType.EACH_FRAME)
    u0 = u_ref[:, :-1]
    u_init = InitialGuessOption(u0, interpolation=InterpolationType.EACH_FRAME)
    ocp.update_initial_guess(x_init, u_init)

    # Update bounds on state
    x_bounds = BoundsList()
    x_bounds.add(QAndQDotBounds(biorbd_model))
    x_bounds[0].concatenate(Bounds([0] * biorbd_model.nbMuscles(), [1] * biorbd_model.nbMuscles()))
    x_bounds[0].min[: biorbd_model.nbQ() * 2, 0] = x_ref[: biorbd_model.nbQ() * 2]
    x_bounds[0].max[: biorbd_model.nbQ() * 2, 0] = x_ref[: biorbd_model.nbQ() * 2]
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
        Objective.Lagrange.MINIMIZE_STATE, weight=10, states_idx=np.array(range(biorbd_model.nbQ()))
    )
    objective_functions.add(
        Objective.Lagrange.MINIMIZE_STATE,
        weight=100,
        states_idx=np.array(range(biorbd_model.nbQ(), biorbd_model.nbQ() * 2)),
    )
    objective_functions.add(
        Objective.Lagrange.MINIMIZE_STATE,
        weight=100,
        states_idx=np.array(range(biorbd_model.nbQ() * 2, biorbd_model.nbQ() * 2 + biorbd_model.nbMuscles())),
    )
    objective_functions.add(
        Objective.Lagrange.MINIMIZE_MUSCLES_CONTROL,
        weight=100,
    )
    objective_functions.add(
        Objective.Lagrange.TRACK_MUSCLES_CONTROL,
        weight=10000,
        target=u_ref[[9, 10, 17, 18], :-1],
        muscles_idx=np.array([9, 10, 17, 18]),
    )
    objective_functions.add(
        Objective.Lagrange.TRACK_STATE, weight=100000, target=q_init, states_idx=np.array(range(biorbd_model.nbQ()))
    )

    # Solve OCP
    ocp.update_objectives(objective_functions)
    sol = ocp.solve(
        solver=Solver.ACADOS,
        show_online_optim=False,
        solver_options={
            "nlp_solver_max_iter": 40,
            "nlp_solver_tol_comp": 1e-4,
            "nlp_solver_tol_eq": 1e-4,
            "nlp_solver_tol_stat": 1e-4,
            "integrator_type": "IRK",
            "nlp_solver_type": "SQP",
            "sim_method_num_steps": 1,
        },
    )

    # Store optimal solutions
    states, controls = Data.get_data(ocp, sol)
    u_est = controls["muscles"]
    q_est = states["q"]
    dq_est = states["q_dot"]

    # Plot solutions if flag is True
    if plot:
        t = np.linspace(0, T, Ns + 1)
        plt.figure("Muscles controls")
        for i in range(biorbd_model.nbMuscles()):
            plt.subplot(4, 5, i + 1)
            plt.step(t, u_est[i, :])
            plt.step(t, u_ref[i, :], c="red")

        plt.figure("Q")
        for i in range(biorbd_model.nbQ()):
            plt.subplot(2, 3, i + 1)
            plt.plot(q_est[i, :])
            plt.plot(q_init[i, :], "r")
            plt.figure("Q")
        plt.figure("dQ")
        for i in range(biorbd_model.nbQ()):
            plt.subplot(2, 3, i + 1)
            plt.plot(dq_est[i, :])
            plt.plot(dq_ref[i, :], "r")
        plt.show()

    # Save results in .bob file if flag is True
    if save_data:
        ocp.save_get_data(sol, f"solutions/sim_ac_{int(T * 1000)}ms_{Ns}sn_REACH2_co_level_{co}.bob")
