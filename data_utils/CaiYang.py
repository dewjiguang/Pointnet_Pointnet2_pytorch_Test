import open3d as o3d
import numpy as np
from open3d.cpu.pybind.visualization import draw_geometries
# points:所有的点 --num_point：4096个大小 --point_idxs：随机筛出的点的id（要在被筛出的这些点中取4096个）
# 均匀采样
def JunyunCaiyang(points,num_point,point_idxs):
    # 选中的点的id
    ids = np.random.choice(point_idxs, int(point_idxs.size / num_point) * num_point, replace=False)
    # 选中点的坐标
    points_select = points[ids, :][:, 0:3]
    # 选中点的RGB
    RGB = points[ids, :][:, 3:7]
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points_select)
    pcd.colors = o3d.utility.Vector3dVector(RGB)
    pcd.normals = o3d.utility.Vector3dVector(np.concatenate((ids.reshape((ids.size,1)), np.zeros((ids.size, 2))), axis=1))
    pcd = pcd.sample_points_uniformly(number_of_points=1)  # number_of_points参数为采样点数
    pcd_new = o3d.geometry.PointCloud.uniform_down_sample(pcd, int(points_select.shape[0]/num_point))
    # 再转回数组
    xyz = np.asarray(pcd_new.points)
    rgb = np.asarray(pcd_new.colors)
    selected_point_idxs = np.asarray(pcd_new.normals)[:,0].astype(int)
    axis = (np.concatenate((xyz, rgb), axis=1))
    # 可视化
    # o3d.visualization.draw_geometries([pcd])
    return axis,selected_point_idxs



# TODO 其他采样方式
# 体素下采样 由于体素下采样要求输入体素大小（单位米）所以无法保证输出为4096，且效果也不一定有多大提高，所以先暂且搁置
def TiSuCaiyang(points,num_point,point_idxs):
    # ids = np.random.choice(point_idxs, int(point_idxs.size / num_point) * num_point, replace=False)
    # points=points[ids, :][:, 0:3]
    # RGB=points[point_idxs, :][:, 3:7]
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points[point_idxs, :][:, 0:3])
    # pcd.colors = o3d.utility.Vector3dVector(RGB)
    pcd.normals = o3d.utility.Vector3dVector(np.concatenate((point_idxs.reshape((point_idxs.size,1)), np.zeros((point_idxs.size, 2))), axis=1))
    pcd_new = o3d.geometry.PointCloud.voxel_down_sample(pcd, 0.02)
    # 再转回数组
    # xyz = np.asarray(pcd_new.points)
    # rgb = np.asarray(pcd_new.colors)
    # axis = (np.concatenate((xyz, rgb), axis=1))
    selected_point_idxs = np.asarray(pcd_new.normals)[:, 0].astype(int)

    # 可视化
    # o3d.visualization.draw_geometries([pcd])
    # 随机取样成4096个
    selected_point_idxs = np.random.choice(selected_point_idxs, num_point, replace=True)
    selected_points = points[selected_point_idxs, :]
    return selected_points, selected_point_idxs


def vector_angle(x, y):
        Lx = np.sqrt(x.dot(x))
        Ly = (np.sum(y ** 2, axis=1)) ** (0.5)
        cos_angle = np.sum(x * y, axis=1) / (Lx * Ly)
        angle = np.arccos(cos_angle)
        angle2 = angle * 360 / 2 / np.pi
        return angle2

    # 曲率下采样
def QuLvCaiiyang(points,num_point,point_idxs):
        pcd = o3d.geometry.PointCloud()
        RGB = points[point_idxs, :][:, 3:7]
        pcd.points = o3d.utility.Vector3dVector((points[:,0:3]))
        pcd.colors = o3d.utility.Vector3dVector(RGB)
        pcd.normals = o3d.utility.Vector3dVector(np.concatenate((point_idxs.reshape((point_idxs.size, 1)), np.zeros((point_idxs.size, 2))), axis=1))

        knn_num = 10  # 自定义参数值(邻域点数)
        angle_thre = 30  # 自定义参数值(角度值)
        N = 5  # 自定义参数值(每N个点采样一次)
        C = 10  # 自定义参数值(采样均匀性>N)

        point = points[:,0:3]
        point_size = point.shape[0]
        tree = o3d.geometry.KDTreeFlann(pcd)
        o3d.geometry.PointCloud.estimate_normals(
        pcd, search_param=o3d.geometry.KDTreeSearchParamKNN(knn=knn_num))
        normal = np.asarray(pcd.normals)
        normal_angle = np.zeros((point_size))
        for i in range(point_size):
            [_, idx, dis] = tree.search_knn_vector_3d(point[i], knn_num + 1)
            current_normal = normal[i]
            knn_normal = normal[idx[1:]]
            normal_angle[i] = np.mean(vector_angle(current_normal, knn_normal))

        point_high = point[np.where(normal_angle >= angle_thre)]
        point_low = point[np.where(normal_angle < angle_thre)]
        pcd_high = o3d.geometry.PointCloud()
        pcd_high.points = o3d.utility.Vector3dVector(point_high)
        pcd_low = o3d.geometry.PointCloud()
        pcd_low.points = o3d.utility.Vector3dVector(point_low)
        pcd_high_down = o3d.geometry.PointCloud.uniform_down_sample(pcd_high, N)
        pcd_low_down = o3d.geometry.PointCloud.uniform_down_sample(pcd_low, C)
        pcd_new = o3d.geometry.PointCloud()
        pcd_new.points = o3d.utility.Vector3dVector(np.concatenate((np.asarray(pcd_high_down.points),
                                                                     np.asarray(pcd_low_down.points))))
        xyz = np.asarray(pcd_new.points)
        rgb = np.asarray(pcd_new.colors)
        selected_point_idxs = np.asarray(pcd_new.normals)[:, 0].astype(int)
        axis = (np.concatenate((xyz, rgb), axis=1))
        # 可视化
        # o3d.visualization.draw_geometries([pcd])
        return axis, selected_point_idxs

#FPS采样算法 有待优化，因为这里是对整个房间的点全部采样，慢的无法训练

import numpy as np
olderr = np.seterr(all='ignore')
def farthest_point_sample(points, num_point,point_idxs):
    N, D = points.shape
    xyz = points[:,:3]
    centroids = np.zeros((num_point,))
    distance = np.ones((N,)) * 1e10
    farthest = np.random.randint(0, N)
    for i in range(num_point):
        centroids[i] = farthest
        centroid = xyz[farthest, :]
        dist = np.sum((xyz - centroid) ** 2, -1)
        mask = dist < distance
        distance[mask] = dist[mask]
        farthest = np.argmax(distance, -1)
    point = points[centroids.astype(np.int32)]
    return point,centroids.astype(np.int32)
