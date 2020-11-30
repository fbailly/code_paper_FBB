import biorbd
from time import time
import numpy as np
from casadi import MX, Function
import bioviz
import pickle
import os.path
import matplotlib.pyplot as plt
# from generate_data_noise_funct import generate_noise
from bioptim import (
    OptimalControlProgram,
    ObjectiveList,
    Objective,
    DynamicsTypeList,
    DynamicsType,
    BoundsList,
    QAndQDotBounds,
    InitialGuessList,
    InitialGuessOption,
    ShowResult,
    Solver,
    InterpolationType,
    Bounds,
    BoundsOption,
    Instant,
    Data
)
def prepare_ocp(
        biorbd_model,
        final_time,
        number_shooting_points,
        x0,
        xT,
        use_SX=False,
        nb_threads=8,
        ):
    # --- Options --- #
    # Model path
    biorbd_model = biorbd_model
    nbQ = biorbd_model.nbQ()

    # tau_min, tau_max, tau_init = -10, 10, 0
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
    # x_bounds[0].min[:nbQ, 0] = [-0.1, -0.3, 0.1, -0.3]
    # x_bounds[0].max[:nbQ, 0] = [-0.1, 0, 0.3, 0]
    # Control path constraint
    u_bounds = BoundsList()
    u_bounds.add(
        [
            [excitation_min] * biorbd_model.nbMuscleTotal(),
            [excitation_max] * biorbd_model.nbMuscleTotal(),
        ]
    )

    # Initial guesses
    x_init = InitialGuessOption(np.tile(np.concatenate(
        (x0, [activation_init] * biorbd_model.nbMuscles()))
        , (number_shooting_points+1, 1)).T, interpolation=InterpolationType.EACH_FRAME)
    u0 = np.array([excitation_init]*biorbd_model.nbMuscles())
    u_init = InitialGuessOption(np.tile(u0, (number_shooting_points, 1)).T,
                                interpolation=InterpolationType.EACH_FRAME)
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

T = 8
Ns = 800
save_data = True
biorbd_model = biorbd.Model("arm_wt_rot_scap.bioMod")
x0 = np.array([0., -0.2, 0, 0, 0, 0, 0, 0])
xT = np.array([-0.2, -1.3, -0.5, 0.5, 0, 0, 0, 0])
# muscle_idx = [9, 10, 17, 18]
with open(
        f"solutions/sim_ac_{int(T * 1000)}ms_{Ns}sn_REACH2_co_level_0.bob", 'rb'
) as file:
    data = pickle.load(file)
states = data['data'][0]
controls = data['data'][1]
q_ref = states['q']
# muscle_idx_tot = [i for i in range(biorbd_model.nbMuscles())]
ocp = prepare_ocp(biorbd_model=biorbd_model, final_time=T, number_shooting_points=Ns, x0=x0, xT=xT, use_SX=True)
for co in range(2, 4):
    with open(
            f"solutions/sim_ac_{int(T * 1000)}ms_{Ns}sn_REACH2_co_level_{co}_tmp.bob", 'rb'
    ) as file:
        data = pickle.load(file)
    states = data['data'][0]
    controls = data['data'][1]
    q_sol = states['q']
    dq_sol = states['q_dot']
    a_sol = states['muscles']
    u_sol = controls['muscles']
    u_co = u_sol
    t = np.linspace(0, T, u_co.shape[1])
    x_ref = np.hstack([q_sol[:, 0], dq_sol[:, 0], [0.3] * biorbd_model.nbMuscles()])
    # x_ref = np.concatenate((q_sol, dq_sol, a_sol))
    x_init = InitialGuessOption(np.tile(x_ref, (ocp.nlp[0].ns+1, 1)).T, interpolation=InterpolationType.EACH_FRAME)
    u0 = u_sol[:, :-1]
    u_init = InitialGuessOption(
        u0, interpolation=InterpolationType.EACH_FRAME)
    ocp.update_initial_guess(x_init, u_init)

    x_bounds = BoundsList()
    x_bounds.add(QAndQDotBounds(biorbd_model))
    x_bounds[0].concatenate(
        Bounds([0] * biorbd_model.nbMuscles(), [1] * biorbd_model.nbMuscles()))
    x_bounds[0].min[:biorbd_model.nbQ() * 2, 0] = x_ref[:biorbd_model.nbQ() * 2]
    x_bounds[0].max[:biorbd_model.nbQ() * 2, 0] = x_ref[:biorbd_model.nbQ() * 2]
    x_bounds[0].min[biorbd_model.nbQ() * 2:biorbd_model.nbQ() * 2 + biorbd_model.nbMuscles(), 0] = [0.1] * biorbd_model.nbMuscles()
    x_bounds[0].max[biorbd_model.nbQ() * 2:biorbd_model.nbQ() * 2 + biorbd_model.nbMuscles(), 0] = [1] * biorbd_model.nbMuscles()
    ocp.update_bounds(x_bounds=x_bounds)

    # Update Objectives
    objective_functions = ObjectiveList()
    objective_functions.add(
        Objective.Lagrange.MINIMIZE_STATE, weight=10, states_idx=np.array(range(biorbd_model.nbQ()))
    )
    objective_functions.add(Objective.Lagrange.MINIMIZE_STATE, weight=100,
                            states_idx=np.array(range(biorbd_model.nbQ(), biorbd_model.nbQ() * 2)))
    objective_functions.add(
        Objective.Lagrange.MINIMIZE_STATE,
        weight=100,
        states_idx=np.array(
            range(biorbd_model.nbQ() * 2, biorbd_model.nbQ() * 2 + biorbd_model.nbMuscles()))
    )
    objective_functions.add(
        Objective.Lagrange.MINIMIZE_MUSCLES_CONTROL,
        weight=100,
        # muscles_idx=np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 11, 12, 13, 14, 15, 16])
    )
    objective_functions.add(
        Objective.Lagrange.TRACK_MUSCLES_CONTROL,
        weight=10000,
        target=u_co[[9, 10, 17, 18], :-1],
        muscles_idx=np.array([9, 10, 17, 18]),
    )
    objective_functions.add(
        Objective.Lagrange.TRACK_STATE,
        weight=100000,
        target=q_ref,
        # target=np.tile(xT[:biorbd_model.nbQ()], (u_co.shape[1] + 1, 1)).T,
        states_idx=np.array(range(biorbd_model.nbQ()))
    )
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
        })
    states, controls = Data.get_data(ocp, sol)
    u_final = controls['muscles']
    q_co = states['q']
    dq_co = states['q_dot']
    t = np.linspace(0, T, Ns + 1)
    plt.figure("Muscles controls")
    for i in range(biorbd_model.nbMuscles()):
        plt.subplot(4, 5, i + 1)
        plt.step(t, u_final[i, :])
        plt.step(t, u_co[i, :], c='red')

    plt.figure("Q")
    for i in range(biorbd_model.nbQ()):
        plt.subplot(2, 3, i + 1)
        plt.plot(q_co[i, :])
        plt.plot(q_ref[i, :], 'r')
        plt.figure("Q")
    plt.figure("dQ")
    for i in range(biorbd_model.nbQ()):
        plt.subplot(2, 3, i + 1)
        plt.plot(dq_co[i, :])
        plt.plot(dq_sol[i, :], 'r')
            # plt.title(q_name[i])
    # plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
    plt.show()
    if save_data:
        ocp.save_get_data(
            sol, f"solutions/sim_ac_{int(T * 1000)}ms_{Ns}sn_REACH2_co_level_{co}.bob"
        )