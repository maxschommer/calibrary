import unittest
import numpy as np

from calibrary.calibrate_dh import (
    dh_to_trans_mats, arr_multiply_accumulate, optimize_dh
)

theta_offsets = [0, 0, -np.pi / 2, 0, 0, np.pi]
a_vals = [64.2, 305, 0, 0, 0, 0]
alphas = [-(np.pi / 2), 0, np.pi / 2, -(np.pi / 2), np.pi / 2, 0]
d_vals = [169.77, 0, 0, -222.63, 0, -36.25]

dh_params = np.stack([theta_offsets, a_vals, alphas, d_vals], axis=1)


class TestCalibrateDh(unittest.TestCase):

    def test_dh_to_trans_mats(self):
        trans_mats = dh_to_trans_mats(dh_params)

        tmat_zero_target = np.array([
            [1.00000000e+00, -0.00000000e+00, -0.00000000e+00, 6.42000000e+01],
            [0.00000000e+00, 6.12323400e-17, 1.00000000e+00, 0.00000000e+00],
            [0.00000000e+00, -1.00000000e+00, 6.12323400e-17, 1.69770000e+02],
            [0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 1.00000000e+00]
        ])
        self.assertTrue(np.allclose(tmat_zero_target, trans_mats[0]))

    def test_optimize_dh(self):
        """Test that optimizing a set of DH parameters works.
        """
        dh_truth = dh_params.copy()
        dh_adj = dh_truth.copy()
        ang_rand_mag = 0.1
        pos_rand_mag = 2
        dh_adj[:, 0] += np.random.rand(dh_truth.shape[0]) * ang_rand_mag
        dh_adj[:, 1] += np.random.rand(dh_truth.shape[0]) * pos_rand_mag
        dh_adj[:, 2] += np.random.rand(dh_truth.shape[0]) * ang_rand_mag
        dh_adj[:, 3] += np.random.rand(dh_truth.shape[0]) * pos_rand_mag

        joint_angles_all = (2 * np.random.rand(1000, 6) - 1) * np.pi
        ee_poses = []
        for joint_angles in joint_angles_all:
            dh_full = dh_truth.copy()
            dh_full[:, 0] += joint_angles
            trans_mats = dh_to_trans_mats(dh_full)
            ee = arr_multiply_accumulate(trans_mats, 0)[-1]
            ee_poses.append(ee)
        ee_poses = np.stack(ee_poses, axis=0)
        res_dh, err = optimize_dh(joint_angles_all, dh_adj, ee_poses)
        self.assertTrue(err < 1)


if __name__ == "__main__":
    unittest.main()
