import os
import os.path

import pickle

import numpy as np
import open3d as o3d

from matplotlib import pyplot as plt

from open3d.cpu.pybind.t.geometry import PointCloud, RGBDImage, Image
from open3d.core import Tensor

from icp_registration import ICPRegistration, draw_registration_result, visualize_point_cloud


def load_pkl(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"specified path does not exist: {path}")

    if not os.path.isfile(path):
        raise FileNotFoundError(f"specified path is not a file: {path}")

    with open(path, "rb") as f:
        file = pickle.load(f)

    return file


def plot_rgbd(rgbd):
    plt.subplot(1, 2, 1)
    plt.title('grayscale image')
    plt.imshow(rgbd.color)
    plt.subplot(1, 2, 2)
    plt.title('depth image')
    plt.imshow(rgbd.depth)
    plt.show()


def get_rgbd_image(data):
    img = Image((data["img"]).astype(np.uint16))
    dep = Image((data["dep"] * 255).astype(np.uint16))

    rgbd = RGBDImage(img, dep)

    return rgbd


def get_point_cloud(rgbd, intrinsics):
    pcd = PointCloud().create_from_rgbd_image(
        rgbd,
        intrinsics,
        with_normals=True
    )
    pcd.transform([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])
    return pcd


if __name__ == "__main__":

    camera_intrinsics = Tensor(
        o3d.camera.PinholeCameraIntrinsic(
            o3d.camera.PinholeCameraIntrinsicParameters.PrimeSenseDefault).intrinsic_matrix
    )

    source_path = r"realsense_can_pickles/frame_2023-01-01T18.31.43.697631.pkl"
    source_rgbd = get_rgbd_image(load_pkl(source_path))
    source = get_point_cloud(source_rgbd, camera_intrinsics)

    target_path = r"realsense_can_pickles/frame_2023-01-01T18.33.19.295918.pkl"
    target_rgbd = get_rgbd_image(load_pkl(target_path))
    target = get_point_cloud(target_rgbd, camera_intrinsics)

    plot_rgbd(source_rgbd)
    plot_rgbd(target_rgbd)

    icp = ICPRegistration()
    icp.set_template(source)

    transformation = icp(target)
    print(f"transform: {transformation}")

    draw_registration_result(source, target, transformation)

