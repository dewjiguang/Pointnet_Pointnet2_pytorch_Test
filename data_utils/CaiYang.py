import open3d as o3d
import numpy as np
from open3d.cpu.pybind.visualization import draw_geometries

# 均匀采样
def JunyunCaiyang(points,RGB,value,ids):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.colors = o3d.utility.Vector3dVector(RGB)
    pcd.normals = o3d.utility.Vector3dVector(np.concatenate((ids.reshape((ids.size,1)), np.zeros((ids.size, 2))), axis=1))
    pcd_new = o3d.geometry.PointCloud.uniform_down_sample(pcd, int(points.shape[0]/value))
    # 再转回数组
    xyz = np.asarray(pcd_new.points)
    rgb = np.asarray(pcd_new.colors)
    selected_point_idxs = np.asarray(pcd_new.normals)[:,0].astype(int)
    axis = (np.concatenate((xyz, rgb), axis=1))
    # 可视化
    # o3d.visualization.draw_geometries([pcd])
    return axis,selected_point_idxs



# TODO 其他采样方式