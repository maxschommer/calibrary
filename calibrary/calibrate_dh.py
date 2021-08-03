import time
from typing import List
from numba import njit
import numpy as np
import jax
import jax.numpy as jnp
from jax import grad, hessian, jit

from scipy.optimize import minimize


@njit(cache=True)
def dh_to_trans_mats(dh_params: np.ndarray):
    theta = dh_params[:, 0]
    a = dh_params[:, 1]
    alpha = dh_params[:, 2]
    d = dh_params[:, 3]
    res_mats = np.zeros((dh_params.shape[0], 4, 4))

    # Fill row 0
    res_mats[:, 0, 0] = np.cos(theta)
    res_mats[:, 0, 1] = -np.sin(theta) * np.cos(theta)
    res_mats[:, 0, 2] = np.sin(theta) * np.sin(alpha)
    res_mats[:, 0, 3] = a * np.cos(theta)

    # Fill row 1
    res_mats[:, 1, 0] = np.sin(theta)
    res_mats[:, 1, 1] = np.cos(theta) * np.cos(alpha)
    res_mats[:, 1, 2] = -np.cos(theta) * np.sin(alpha)
    res_mats[:, 1, 3] = a * np.sin(theta)

    # Fill row 2
    res_mats[:, 2, 1] = np.sin(alpha)
    res_mats[:, 2, 2] = np.cos(alpha)
    res_mats[:, 2, 3] = d

    # Identity row 3
    res_mats[:, 3, 3] = 1

    return res_mats


@jit
def dh_to_trans_mats_jax(dh_params: jnp.ndarray):
    theta = dh_params[:, 0]
    a = dh_params[:, 1]
    alpha = dh_params[:, 2]
    d = dh_params[:, 3]
    res_mats = jnp.zeros((dh_params.shape[0], 4, 4))

    # Fill row 0
    res_mats = jax.ops.index_update(
        res_mats, jax.ops.index[:, 0, 0], jnp.cos(theta))
    res_mats = jax.ops.index_update(
        res_mats, jax.ops.index[:, 0, 1], -jnp.sin(theta) * jnp.cos(theta))
    res_mats = jax.ops.index_update(
        res_mats, jax.ops.index[:, 0, 2], jnp.sin(theta) * jnp.sin(alpha))
    res_mats = jax.ops.index_update(
        res_mats, jax.ops.index[:, 0, 3], a * jnp.cos(theta))

    # Fill row 1
    res_mats = jax.ops.index_update(
        res_mats, jax.ops.index[:, 1, 0], jnp.sin(theta))
    res_mats = jax.ops.index_update(
        res_mats, jax.ops.index[:, 1, 1], jnp.cos(theta) * jnp.cos(alpha))
    res_mats = jax.ops.index_update(
        res_mats, jax.ops.index[:, 1, 2], -jnp.cos(theta) * jnp.sin(alpha))
    res_mats = jax.ops.index_update(
        res_mats, jax.ops.index[:, 1, 3], a * jnp.sin(theta))

    # Fill row 2
    res_mats = jax.ops.index_update(
        res_mats, jax.ops.index[:, 2, 1], jnp.sin(alpha))
    res_mats = jax.ops.index_update(
        res_mats, jax.ops.index[:, 2, 2], jnp.cos(alpha))
    res_mats = jax.ops.index_update(res_mats, jax.ops.index[:, 2, 3], d)

    # Identity row 3

    res_mats = jax.ops.index_update(res_mats, jax.ops.index[:, 3, 3], 1)

    return res_mats


def arr_multiply_accumulate(arr: np.ndarray, axis=0):
    arr_roll = np.moveaxis(arr, axis, 0)
    res = np.zeros_like(arr_roll)
    for i, arr_slice in enumerate(arr_roll):
        if i != 0:
            res[i] = res[i - 1] @ arr_slice
        else:
            res[i] = arr_slice
    return np.moveaxis(res, 0, axis)


@njit(cache=True)
def arr_multiply_accumulate_numba(arr: np.ndarray):
    res = np.zeros_like(arr)
    for i, arr_slice in enumerate(arr):
        if i != 0:
            res[i] = res[i - 1] @ arr_slice
        else:
            res[i] = arr_slice
    return res


def arr_multiply_accumulate_jax(arr: jnp.ndarray, axis=0):
    arr_roll = jnp.moveaxis(arr, axis, 0)
    res: List[jnp.array] = []
    for i, arr_slice in enumerate(arr_roll):
        if i != 0:
            res.append(jnp.matmul(res[i - 1], arr_slice))
        else:
            res.append(arr_slice)
    return jnp.stack(res, axis)


@njit(cache=True)
def _optimize_dh_error(dh_params_proposed: np.ndarray,
                       end_effector_poses: np.ndarray,
                       joint_angles: np.ndarray):
    # Dh parameters come in as a flat array
    dh_params_proposed = np.reshape(dh_params_proposed, (-1, 4))
    error = []
    for i in range(len(joint_angles)):
        js = joint_angles[i]
        ee_pose = end_effector_poses[i]
        full_dh_params = dh_params_proposed.copy()
        full_dh_params[:, 0] += js
        ee = arr_multiply_accumulate_numba(
            dh_to_trans_mats(full_dh_params))[-1]
        error.append(np.sum((ee - ee_pose)**2))
    return np.sum(np.array(error))


def _optimize_dh_error_jax(dh_params_proposed: jnp.ndarray,
                           end_effector_poses: jnp.ndarray,
                           joint_angles: jnp.ndarray) -> float:
    # Dh parameters come in as a flat array
    dh_params_proposed = jnp.reshape(dh_params_proposed, (-1, 4))
    error = []
    for js, ee_pose in zip(joint_angles, end_effector_poses):
        full_dh_params = jax.ops.index_update(
            dh_params_proposed, jax.ops.index[:, 0], dh_params_proposed[:, 0] + js)

        joint_mats = dh_to_trans_mats_jax(full_dh_params)
        ee = arr_multiply_accumulate_jax(
            joint_mats, 0)[-1]
        error.append(jnp.sum((ee - ee_pose)**2))

    return sum(error)


def optimize_dh(joint_angles: np.ndarray,
                dh_params: np.ndarray,
                end_effector_poses: np.ndarray):

    def error_func(dh_proposed):
        return _optimize_dh_error(dh_proposed,
                                  end_effector_poses,
                                  joint_angles)

    res = minimize(error_func, dh_params.flatten())
    return res.x.reshape(dh_params.shape), res.fun
