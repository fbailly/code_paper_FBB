import biorbd
from time import time
import numpy as np
from casadi import MX, Function
import bioviz
import scipy.io as sio
import pickle
import os.path
import matplotlib.pyplot as plt
# from generate_data_noise_funct import generate_noise
from utils import *
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
    Simulate,
    Data
)

def switch_phase(ocp, sol):
    data = Data.get_data(ocp, sol)
    q = data[0]["q"]
    dq = data[0]["q_dot"]
    act = data[0]["muscles"]
    exc = data[1]["muscles"]
    x = np.vstack([q, dq, act])
    return x[:, :-1], exc[:, :-1], x[:, -1]

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
    objective_functions.add(
        Objective.Lagrange.MINIMIZE_STATE, weight=10, states_idx=np.array(range(biorbd_model.nbQ())))
    objective_functions.add(Objective.Lagrange.MINIMIZE_STATE, weight=10,
                            states_idx=np.array(range(biorbd_model.nbQ(), biorbd_model.nbQ() * 2)))
    objective_functions.add(
        Objective.Lagrange.MINIMIZE_STATE,
        weight=10,
        states_idx=np.array(range(biorbd_model.nbQ() * 2, biorbd_model.nbQ() * 2 + biorbd_model.nbMuscles()))
    )
    objective_functions.add(Objective.Lagrange.MINIMIZE_MUSCLES_CONTROL, weight=10)

    # objective_functions.add(Objective.Lagrange.MINIMIZE_MUSCLES_CONTROL_DERIVATIVE, weight=100)

    objective_functions.add(
        Objective.Mayer.TRACK_STATE,
        weight=100000,
        target=np.array([xT[:biorbd_model.nbQ()*2]]).T,
        states_idx=np.array(range(biorbd_model.nbQ()*2))
    )
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


if __name__ == "__main__":
    biorbd_model = biorbd.Model("arm_wt_rot_scap.bioMod")
    T = 1
    Ns = 100
    co_value = []
    motion = 'REACH2'
    x_phase = np.array([[-0.1, -0.3, -0.1, 0.1, 0, 0, 0, 0],
                        [-0.2, -1.3, -0.5, 0.5, 0, 0, 0, 0],
                        [-0.1, -0.3, -0.1, 0.1, 0, 0, 0, 0],
                        [-0.2, -1.3, -0.5, 0.5, 0, 0, 0, 0],
                        [-0.1, -0.3, -0.1, 0.1, 0, 0, 0, 0],
                        [-0.2, -1.3, -0.5, 0.5, 0, 0, 0, 0],
                        [-0.1, -0.3, -0.1, 0.1, 0, 0, 0, 0],
                        [-0.2, -1.3, -0.5, 0.5, 0, 0, 0, 0],
                        [-0.1, -0.3, -0.1, 0.1, 0, 0, 0, 0],
                        [-0.2, -1.3, -0.5, 0.5, 0, 0, 0, 0],
                        ])

    # x_phase = np.array([[-0.1, -0.3, -0.1, 0.1, 0, 0, 0, 0],
    #                     [-0.2, -1.3, -0.5, 0.5, 0, 0, 0, 0],
    #                     [0.89, -0.5, -0.5, 0.8, 0, 0, 0, 0],
    #                     [-0.2, -1.3, -0.5, 0.5, 0, 0, 0, 0],
    #                     [-0.1, -0.3, -0.1, 0.1, 0, 0, 0, 0],
    #                     [-0.2, -1.3, -0.5, 0.5, 0, 0, 0, 0],
    #                     [0.89, -0.5, -0.5, 0.8, 0, 0, 0, 0],
    #                     [-0.2, -1.3, -0.5, 0.5, 0, 0, 0, 0],
    #                     [-0.1, -0.3, -0.1, 0.1, 0, 0, 0, 0],
    #                     [-0.2, -1.3, -0.5, 0.5, 0, 0, 0, 0],
    #                     [0.89, -0.5, -0.5, 0.8, 0, 0, 0, 0],
    #                     [-0.1, -0.3, -0.1, 0.1, 0, 0, 0, 0],
    #                     [-0.2, -1.3, -0.5, 0.5, 0, 0, 0, 0],
    #                     [0.89, -0.5, -0.5, 0.8, 0, 0, 0, 0],
    #                     [-0.1, -0.3, -0.1, 0.1, 0, 0, 0, 0],
    #                     [-0.2, -1.3, -0.5, 0.5, 0, 0, 0, 0],
    #                     [-0.1, -0.3, -0.1, 0.1, 0, 0, 0, 0],
    #                     ])
    nb_phase = 8
    X_est = np.zeros((biorbd_model.nbQ()*2+biorbd_model.nbMuscleTotal(), nb_phase*Ns+1))
    U_est = np.zeros((biorbd_model.nbMuscleTotal(), nb_phase*Ns))
    use_ACADOS = True
    use_CO = False
    save_data = True

    excitations_max = [1] * biorbd_model.nbMuscleTotal()
    if use_CO is not True:
        excitations_init = [[0.05] * biorbd_model.nbMuscleTotal()]
        excitations_min = [[0] * biorbd_model.nbMuscleTotal()]

    else:
        excitations_init = [
            [0.05] * biorbd_model.nbMuscleTotal(),
            [0.05] * 6 + [0.1] * 3 + [0.05] * 10,
            [0.05] * 6 + [0.2] * 3 + [0.05] * 10,
            [0.05] * 6 + [0.3] * 3 + [0.05] * 10,
            # [0.05] * 6 + [0.4] * 3 + [0.05] * 10
        ]
        excitations_min = [
            [0] * biorbd_model.nbMuscleTotal(),
            [0] * 6 + [0.1] * 3 + [0] * 10,
            [0] * 6 + [0.2] * 3 + [0] * 10,
            [0] * 6 + [0.3] * 3 + [0] * 10,
            # [0] * 6 + [0.4] * 3 + [0] * 10
        ]

    ocp = prepare_ocp(biorbd_model=biorbd_model, final_time=T, number_shooting_points=Ns,
                      x0=x_phase[0, :], xT=x_phase[1, :], use_SX=True)

    for co in range(0, len(excitations_init)):
        u_i = excitations_init[co]
        u_mi = excitations_min[co]
        u_ma = excitations_max

        # Update u_init and u_bounds
        u_init = InitialGuessOption(np.tile(u_i, (ocp.nlp[0].ns, 1)).T, interpolation=InterpolationType.EACH_FRAME)
        x_init = InitialGuessOption(np.tile(np.concatenate(
            (x_phase[0, :], [0.5] * biorbd_model.nbMuscles())), (ocp.nlp[0].ns + 1, 1)).T,
                                    interpolation=InterpolationType.EACH_FRAME)
        ocp.update_initial_guess(x_init, u_init=u_init)

        u_bounds = BoundsOption([u_mi, u_ma], interpolation=InterpolationType.CONSTANT)
        x_bounds = BoundsList()
        x_bounds.add(QAndQDotBounds(biorbd_model))
        # add muscle activation bounds
        x_bounds[0].concatenate(
            Bounds([0] * biorbd_model.nbMuscles(), [1] * biorbd_model.nbMuscles())
        )
        x_bounds[0].min[:biorbd_model.nbQ(), 0] = x_phase[0, :biorbd_model.nbQ()]-0.1
        x_bounds[0].max[:biorbd_model.nbQ(), 0] = x_phase[0, :biorbd_model.nbQ()]+0.1
        ocp.update_bounds(x_bounds=x_bounds, u_bounds=u_bounds)

        for phase in range(1, nb_phase+1):
            # sol = ocp.solve(solver=Solver.IPOPT)
            sol = ocp.solve(
                solver=Solver.ACADOS,
                show_online_optim=False,
                solver_options={
                    "nlp_solver_max_iter": 100,
                    "nlp_solver_tol_comp": 1e-7,
                    "nlp_solver_tol_eq": 1e-7,
                    "nlp_solver_tol_stat": 1e-7,
                    "integrator_type": "IRK",
                    "nlp_solver_type": "SQP",
                    "sim_method_num_steps": 1,
                })

            # get last state of previous solve
            x_out, u_out, x0 = switch_phase(ocp, sol)

            # impose it as first state of next solve
            ocp.nlp[0].x_bounds.min[:, 0] = x0
            ocp.nlp[0].x_bounds.max[:, 0] = x0

            # update initial guess, bounds stay the same
            x_init = InitialGuessOption(np.tile(x0, (ocp.nlp[0].ns + 1, 1)).T,
                                        interpolation=InterpolationType.EACH_FRAME)
            ocp.update_initial_guess(x_init, u_init=u_init)
            ocp.update_bounds(u_bounds=u_bounds)

            objectives = ObjectiveList()
            objectives.add(
                Objective.Lagrange.MINIMIZE_STATE, weight=10, states_idx=np.array(range(biorbd_model.nbQ())))
            objectives.add(Objective.Lagrange.MINIMIZE_STATE, weight=10,
                           states_idx=np.array(range(biorbd_model.nbQ(), biorbd_model.nbQ() * 2)))
            objectives.add(
                Objective.Lagrange.MINIMIZE_STATE,
                weight=10,
                states_idx=np.array(range(biorbd_model.nbQ() * 2, biorbd_model.nbQ() * 2 + biorbd_model.nbMuscles()))
            )
            objectives.add(Objective.Lagrange.MINIMIZE_MUSCLES_CONTROL, weight=10)

            objectives.add(
                Objective.Mayer.TRACK_STATE,
                weight=10000,
                target=np.array([x_phase[phase+1, :biorbd_model.nbQ()*2]]).T,
                states_idx=np.array(range(biorbd_model.nbQ()*2))
            )

            # objectives.add(Objective.Lagrange.MINIMIZE_MUSCLES_CONTROL_DERIVATIVE, weight=100)
            print(x_phase[phase+1, :biorbd_model.nbQ()])
            ocp.update_objectives(objectives)

            X_est[:, (phase-1)*Ns:phase*Ns] = x_out
            U_est[:, (phase - 1) * Ns:phase * Ns] = u_out
# Collect last state
X_est[:, -1] = x0

dic = {'state': X_est, 'controls': U_est}
sio.savemat(f"solutions/state_to_track_{phase}phase.mat", dic)

plt.subplot(211)
for est, name in zip(X_est[:biorbd_model.nbQ(), :], biorbd_model.nameDof()):
    plt.plot(est, 'x', label=name.to_string() + '_q_est')
plt.legend()

plt.subplot(212)
for est, name in zip(X_est[biorbd_model.nbQ():biorbd_model.nbQ() * 2, :], biorbd_model.nameDof()):
    plt.plot(est, 'x', label=name.to_string() + '_qdot_est')
plt.legend()
plt.tight_layout()

plt.figure('Muscles excitations')
for i in range(biorbd_model.nbMuscles()):
    plt.subplot(4, 5, i + 1)
    plt.plot(U_est[i, :])
    plt.title(biorbd_model.muscleNames()[i].to_string())
plt.legend(labels=['u_est'], bbox_to_anchor=(1.05, 1), loc='upper left',
           borderaxespad=0.)
print()
plt.show()

# b = bioviz.Viz(model_path="arm_wt_rot_scap.bioMod")
# b.load_movement(X_est[:biorbd_model.nbQ(), :])
# b.exec()



