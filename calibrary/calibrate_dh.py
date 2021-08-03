import time
from typing import List, Tuple
from numba import njit
import numpy as np

from scipy.optimize import minimize


@njit(cache=True)
def dh_to_trans_mats(dh_params: np.ndarray) -> np.ndarray:
    """Convert Denavit–Hartenberg parameters into relative transformation
    matrices.

    Args:
        dh_params (np.ndarray): An #Nx4 array of DH parameters, which are
            expected to have rows which correspond to the joints, and columns
            which correspond to:
                theta offsets: angle about previous z, from old x to new x
                a: length of the common normal
                alpha: angle about common normal, from old z axis to new z axis
                d: offset along previous z to the common normal

    Returns:
        np.ndarray: An #Nx4x4 array of transformation matrices.
    """
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


def arr_multiply_accumulate(arr: np.ndarray, axis: int = 0) -> np.ndarray:
    """Performs a matrix multiply accumulate along an axis.

    Args:
        arr (np.ndarray): An #Nx#Mx#M array to multiply the elements along the
            zeroth axis along and accumulate the result.
        axis (int): The axis to multiply and accumulate along.

    Returns:
        np.ndarray: An #Nx#Mx#M array of multiplied and accumulated matrices.
    """
    arr_roll = np.moveaxis(arr, axis, 0)
    res = np.zeros_like(arr_roll)
    for i, arr_slice in enumerate(arr_roll):
        if i != 0:
            res[i] = res[i - 1] @ arr_slice
        else:
            res[i] = arr_slice
    return np.moveaxis(res, 0, axis)


@njit(cache=True)
def arr_multiply_accumulate_numba(arr: np.ndarray) -> np.ndarray:
    """Performs a matrix multiply accumulate along the zeroth axis, accelerated
    by Numba.

    Args:
        arr (np.ndarray): An #Nx#Mx#M array to multiply the elements along the
            zeroth axis along and accumulate the result.

    Returns:
        np.ndarray: An #Nx#Mx#M array of multiplied and accumulated matrices.
    """
    res = np.zeros_like(arr)
    for i, arr_slice in enumerate(arr):
        if i != 0:
            res[i] = res[i - 1] @ arr_slice
        else:
            res[i] = arr_slice
    return res


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


def optimize_dh(joint_angles: np.ndarray,
                dh_params: np.ndarray,
                end_effector_poses: np.ndarray) -> Tuple[np.ndarray, float]:
    """Optimize a set of initial DH parameters given an observed set of end
    effector poses and the associated joint angles.

    Args:
        joint_angles (np.ndarray): An #Nx#M array of joint angles, where there
            are #N sample poses, and #M joints.
        dh_params (np.ndarray): An #Mx4 array of DH parameters representing the
            arm.
        end_effector_poses (np.ndarray): An #Nx4x4 array of transformation
            matrices representing the end effector poses (observed).

    Returns:
        Tuple[np.ndarray, float]: A tuple comprising:
            (np.ndarray): An #Mx4 array of optimized DH parameters
            (float): The resulting error function after optimization.
    """
    def error_func(dh_proposed):
        return _optimize_dh_error(dh_proposed,
                                  end_effector_poses,
                                  joint_angles)

    res = minimize(error_func, dh_params.flatten())
    return res.x.reshape(dh_params.shape), res.fun
