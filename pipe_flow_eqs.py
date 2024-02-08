import jax
from jax import grad,jit, vmap, pmap
import jax.numpy as jnp
jax.config.update("jax_enable_x64", True)

import numpy as np
from tqdm import tqdm

# # Geometry Definition
x_e = 2

k = 13 / 100
a = k * (2 + 1 / 10)
b = 0.9 + 8 / 100

c = 2 - b

# # Thermodynamic properties of the fluid
gamma = 1.4
R = 287.05
c_v = R / (gamma - 1)
c_p = gamma * c_v

# Ideal gas law
@jit
def density_ideal_gas(T, P):
    return P / (R * T)


@jit
def get_local_Mach(U):
    rho, u, p = U
    return u / jnp.sqrt(gamma * p / rho)

# # Boundary Conditions
# mine
p_input = 1e5
p_exit = p_input - 0.1 * 8e5
T_input = 273 + 8

rho_input = p_input / (R * T_input)

@jit
def S_x(x):
    x2 = x - c
    return k + a * (x2) ** 2 * (x2**2 - b**2)

def setup_vars(N):
    x = jnp.linspace(0, x_e, N)
    S = S_x(x)
    dx = x[1:] - x[:-1]
    dx = jnp.insert(dx, 0, dx[0], axis=0)
    dS_dx = np.gradient(S, x)
    return x, S, dx, dS_dx


@jit
def energy_from_U(U):
    rho, u, p = U
    # Find the temperature from the ideal gas law
    T = p / (rho * R)
    # return energy
    return c_v * T + 0.5 * u**2


@jit
def total_enthalpy_from_U(U):
    rho, u, p = U
    T = p / (rho * R)
    # return energy
    return c_v * T + 0.5 * u**2 + p / rho

def get_initial_conditions(N):
    # # Initial Conditions
    T0 = jnp.ones(N) * (T_input - 10)
    u0 = jnp.zeros(N)
    p0 = jnp.ones(N) * p_exit

    # Initial Discontinuity
    p0 = p0.at[0].set(p_input)
    T0 = T0.at[0].set(T_input)

    rho0 = density_ideal_gas(T0, p0)
    U0 = jnp.array([rho0, u0, p0])
    return U0

# # Conversions
# From conservative to primitive variables
@jit
def Uc_to_U(UC, S):
    rho = UC[0] / S
    u = UC[1] / UC[0]
    E = UC[2] / UC[0]

    p = (gamma - 1) * (rho * E - 0.5 * rho * u**2)
    U = jnp.array([rho, u, p])
    return U


# From primitive to conservative variables
@jit
def U_to_Uc(U, S):
    rho = U[0]
    u = U[1]
    p = U[2]

    E = p / (gamma - 1) / rho + 0.5 * u**2
    UC1 = rho * S
    UC2 = UC1 * u
    UC3 = UC1 * E
    UC = jnp.array([UC1, UC2, UC3])
    return UC


# # Calculate Fluxes
@jit
def get_input_ghost_cells(U):
    u = U[1, 0]
    return jnp.expand_dims(jnp.array([rho_input, u, p_input]), axis=1)


@jit
def get_output_ghost_cells(U):
    rho, u, _ = U[:, -1]
    return jnp.expand_dims(jnp.array([rho, u, p_exit]), axis=1)


@jit
def get_sources(U, ds_dx):
    _, _, p = U
    return jnp.array([jnp.zeros_like(p), p * ds_dx, jnp.zeros_like(p)])


@jit
def get_fluxes(U, S):
    H = total_enthalpy_from_U(U)
    UC = U_to_Uc(U, S)
    # _ , u, p = U

    return jnp.array([UC[1], UC[1] * U[1] + U[2] * UC[0] / U[0], UC[1] * H])


@jit
def A_abs(A_R, L, A_L, U_right, U_left):
    return jnp.matmul(A_R, jnp.matmul(L, jnp.matmul(A_L, (U_right - U_left))))


@jit
def roe_scheme(U_L, U_R,S):
    H_L = total_enthalpy_from_U(U_L)
    H_R = total_enthalpy_from_U(U_R)

    U1_L, U2_L, U3_L = U_L
    U1_R, U2_R, U3_R = U_R

    # Roe Variables
    rho_mean = jnp.sqrt(U1_L * U1_R)

    u_mean = (jnp.sqrt(U1_L) * U2_L + jnp.sqrt(U1_R) * U2_R) / (
        jnp.sqrt(U1_L) + jnp.sqrt(U1_R)
    )

    H_mean = (jnp.sqrt(U1_L) * H_L + jnp.sqrt(U1_R) * H_R) / (
        jnp.sqrt(U1_L) + jnp.sqrt(U1_R)
    )

    c_mean = jnp.sqrt((gamma - 1) * (H_mean - 0.5 * u_mean**2))

    # The A matrix is:
    # [ u   rho     0       ]
    # [ 0  u        1/rho   ]
    # [ 0  rho*c^2  u       ]

    # Create A matrix according to Roe's method
    A_R = jnp.array(
        [
            [jnp.ones_like(u_mean), jnp.ones_like(u_mean), jnp.ones_like(u_mean)],
            [u_mean - c_mean, u_mean, u_mean + c_mean],
            [H_mean - u_mean * c_mean, 0.5 * u_mean**2, H_mean + u_mean * c_mean],
        ]
    ).transpose(2, 0, 1)

    a1 = (gamma - 1) * u_mean**2 / (2 * c_mean**2)
    a2 = (gamma - 1) / c_mean**2

    A_L = jnp.array(
        [
            [
                1 / 2 * (a1 + u_mean / c_mean),
                -1 / 2 * (a2 * u_mean + 1 / c_mean),
                a2 / 2,
            ],
            [1 - a1, a2 * u_mean, -a2],
            [
                1 / 2 * (a1 - u_mean / c_mean),
                -1 / 2 * (a2 * u_mean - 1 / c_mean),
                a2 / 2,
            ],
        ]
    ).transpose(2, 0, 1)

    # Entropy Corrections
    delta = 0.05 * jnp.abs(c_mean)

    cond1 = [jnp.abs(u_mean - c_mean) > delta, jnp.abs(u_mean - c_mean) < delta]
    num = (jnp.abs(u_mean - c_mean) ** 2 + delta**2) / (2 * delta)
    l1 = jnp.select(cond1, [jnp.abs(u_mean - c_mean), num], num)

    cond2 = [jnp.abs(u_mean) > delta, jnp.abs(u_mean) < delta]
    num = (jnp.abs(u_mean) ** 2 + delta**2) / (2 * delta)
    l2 = jnp.select(cond2, [jnp.abs(u_mean), num], num)

    cond3 = [jnp.abs(u_mean + c_mean) > delta, jnp.abs(u_mean + c_mean) < delta]
    num = (jnp.abs(u_mean + c_mean) ** 2 + delta**2) / (2 * delta)
    l3 = jnp.select(cond3, [jnp.abs(u_mean + c_mean), num], num)

    lambda_A = jnp.array(
        [
            [l1, jnp.zeros_like(u_mean), jnp.zeros_like(u_mean)],
            [jnp.zeros_like(u_mean), l2, jnp.zeros_like(u_mean)],
            [jnp.zeros_like(u_mean), jnp.zeros_like(u_mean), l3],
        ]
    ).transpose(2, 0, 1)

    UC_R = U_to_Uc(U_R, jnp.insert(S, -1, S[-1], axis=0))
    UC_L = U_to_Uc(U_L, jnp.insert(S, 0, S[0], axis=0))

    # print(f"U_L: {U_L[:,ind]}")
    # print(f"H_L: {H_L[ind]}")
    # print(f"U_R: {U_R[:,ind]}")
    # print(f"u_roe**2: {0.5 * u_mean[ind]**2}")
    # print(f"H_roe: {H_mean[ind]}")
    # print(f"rho_roe: {rho_mean[ind]}")
    # print(A_R[ind,:,:])
    # print(A_L[ind,:,:])
    # print(lambda_A[ind,:,:])

    return vmap(A_abs)(A_R, lambda_A, A_L, UC_R.T, UC_L.T)


@jit
def dUC_dt(U_now,S, S_all, dx, dS_dx):
    # Add ghost cells
    input_ghost = get_input_ghost_cells(U_now)
    output_ghost = get_output_ghost_cells(U_now)

    U_L = jnp.concatenate([input_ghost, U_now], axis=1)
    U_R = jnp.concatenate([U_now, output_ghost], axis=1)
    U_all = jnp.concatenate([U_L, output_ghost], axis=1)

    Q_nodes = get_sources(U_now, dS_dx)
    F_nodes = get_fluxes(U_all, S_all)

    roe = roe_scheme(U_L, U_R,S).T
    F_faces = (F_nodes[:, 1:] + F_nodes[:, :-1]) / 2 - 0.5 * roe

    # print(f"F_nodes: {F_nodes[:,ind]}")
    # print(f"F_nodes: {F_nodes[:,ind+1]}")
    # print(f"Roe:     {roe[:,ind]}")
    # print(f"Q_nodes: {Q_nodes[:,ind]}")
    # print(f"F:       {(F_faces[:,ind+1] - F_faces[:,ind])/dx[ind]}")

    return Q_nodes - (F_faces[:, 1:] - F_faces[:, :-1]) / dx


@jit
def timestep(U, dt,S, S_all , dx, dS_dx):
    UC0_curr = U_to_Uc(U, S)

    # Runge Kutta
    for rki in [0.1084, 0.2602, 0.5052, 1.0]:
        # print(rki)
        # print("$$$$$$$$$$$$$$$$$$$$$\n")
        UC = UC0_curr + dt * rki * dUC_dt(U,S, S_all, dx, dS_dx)
        U = Uc_to_U(UC, S)
        # print(f"UC0:     {UC0_curr[:,ind]}")
        # print(f"UC:      {UC[:,ind]}")
        # print(f"U:       {U[:,ind]}")
        # print("$$$$$$$$$$$$$$$$$$$$$\n")
    diff = U - Uc_to_U(UC0_curr,S)
    return U,  diff


@jit
def check_nan(U,i):
    if jnp.isnan(U).any():
        # for k in range(U.shape[1]):
        # print(k)
        # print(U[:,k])
        print(f"%%%%%%%%%%%%%%%%")
        print(f"NAN at iteration {i}")
        print(f"Ui: {U[:, -1]}")
        print(f"%%%%%%%%\n")
        raise StopIteration

@jit
def calc_cfl(U,dt,dx):
    u = U[1]
    H = total_enthalpy_from_U(U)
    c = jnp.sqrt((gamma - 1) * (H - 0.5 * u**2))
    return jnp.max(jnp.abs(u) + c)*dt/dx
