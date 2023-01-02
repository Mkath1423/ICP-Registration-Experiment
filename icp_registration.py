import copy

from typing import List

import numpy as np
import open3d as o3d

import open3d.cpu.pybind.t.pipelines.registration as treg

from open3d.cpu.pybind.t.geometry import PointCloud
from open3d.core import Tensor


def visualize_point_cloud(pcd):
    o3d.visualization.draw_geometries([pcd.to_legacy()],
                                      # zoom=0.1,
                                      # front=[1, 0, 0],
                                      # lookat=[0, 0, 1],
                                      # up=[0, 0, 1],
                                      # point_show_normal=False
                                      )


def draw_registration_result(source, target, transformation):
    source_temp = copy.deepcopy(source.to_legacy())
    target_temp = copy.deepcopy(target.to_legacy())
    source_temp.paint_uniform_color([1, 0.706, 0])
    target_temp.paint_uniform_color([0, 0.651, 0.929])
    source_temp.transform(transformation)
    o3d.visualization.draw_geometries([source_temp, target_temp])


def numpy_position_to_pointcloud(pointcloud):
    if not (pointcloud, np.ndarray):
        raise TypeError(f"Pointcloud must be a numpy array of shape [N, 3] not {type(pointcloud)}")

    n, dims = pointcloud.shape
    if not dims == 3:
        raise TypeError(f"Pointcloud must be a numpy array of shape [N, 3] not {pointcloud.shape}")

    return PointCloud(
        Tensor.from_numpy(pointcloud)
    )


def callback_after_iteration(loss_log_map, verbose):
    if verbose:
        print(
            "Iteration Index: {}, Scale Index: {}, Scale Iteration Index: {}, Fitness: {}, Inlier RMSE: {},".format(
                loss_log_map["iteration_index"].item(),
                loss_log_map["scale_index"].item(),
                loss_log_map["scale_iteration_index"].item(),
                loss_log_map["fitness"].item(),
                loss_log_map["inlier_rmse"].item()))


class ICPRegistration:
    def __init__(self,
                 init_source_to_target: np.ndarray = None,
                 max_correspondence_distances: List[float] = None,
                 voxel_sizes: List[float] = None,
                 criteria_list=None,
                 estimation=None):
        self.template = None

        # init_source_to_target
        if init_source_to_target is None:
            self.init_source_to_target = o3d.core.Tensor.from_numpy(np.identity(4))
        else:
            self.init_source_to_target = o3d.core.Tensor.from_numpy(init_source_to_target)

        # max_correspondence_distances
        if max_correspondence_distances is None:
            self.max_correspondence_distances = o3d.utility.DoubleVector([0.3, 0.14, 0.07])
        else:
            self.max_correspondence_distances = o3d.utility.DoubleVector(max_correspondence_distances)

        # voxel_sizes
        if voxel_sizes is None:
            self.voxel_sizes = o3d.utility.DoubleVector([0.1, 0.05, 0.025])
        else:
            self.voxel_sizes = o3d.utility.DoubleVector(voxel_sizes)

        # criteria_list
        if criteria_list is None:
            self.criteria_list = [
                treg.ICPConvergenceCriteria(relative_fitness=0.0001,
                                            relative_rmse=0.0001,
                                            max_iteration=20),
                treg.ICPConvergenceCriteria(0.00001, 0.00001, 15),
                treg.ICPConvergenceCriteria(0.000001, 0.000001, 10)
            ]
        else:
            self.criteria_list = [treg.ICPConvergenceCriteria(i[0], i[1], i[2]) for i in criteria_list]

        # estimation
        if estimation is None:
            self.estimation = treg.TransformationEstimationPointToPlane()
        else:
            self.estimation = estimation

    def set_template(self, pointcloud):
        if isinstance(pointcloud, PointCloud):
            self.template = pointcloud

        else:
            self.template = numpy_position_to_pointcloud(pointcloud)

    def __call__(self, pointcloud, verbose=True) -> np.ndarray:
        if self.template is None:
            raise RuntimeError("template not set. please call ICPRegistration.set_template(pointcloud)")

        if isinstance(pointcloud, PointCloud):
            # pointclouds should have their normals precomputed
            target = pointcloud
        else:
            target = numpy_position_to_pointcloud(pointcloud)
            target.estimate_normals()


        init_source_to_target = self.init_source_to_target.clone()

        registration_result = treg.multi_scale_icp(
            self.template, target,
            self.voxel_sizes,
            self.criteria_list,
            self.max_correspondence_distances,
            init_source_to_target,
            self.estimation,
            lambda loss_log_map : callback_after_iteration(loss_log_map, verbose)
        )

        return registration_result.transformation.numpy()


if __name__ == "__main__":
    # use premade pointclouds loaded from pcd files
    demo_icp_pcds = o3d.data.DemoICPPointClouds()
    source_pcd = o3d.t.io.read_point_cloud(demo_icp_pcds.paths[0])
    target_pcd = o3d.t.io.read_point_cloud(demo_icp_pcds.paths[1])

    icp = ICPRegistration()
    # set the temple with either a PointCloud or a np.ndarray [N, 3]
    icp.set_template(source_pcd)

    # call with a PointCloud or a np.ndarray [N, 3] to get the transform matrix (np.ndarray)
    transform = icp(target_pcd)

    print(transform)
    draw_registration_result(source_pcd.to_legacy(), target_pcd.to_legacy(), transform)
