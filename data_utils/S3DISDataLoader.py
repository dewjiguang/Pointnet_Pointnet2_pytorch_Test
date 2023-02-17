import os
import numpy as np
from tqdm import tqdm
from torch.utils.data import Dataset
from data_utils import CaiYang
type_size=5

class S3DISDataset(Dataset):
    def __init__(self, split='train', data_root='trainval_fullarea', num_point=4096, test_area=5, block_size=1.0, sample_rate=1, transform=None):
        super().__init__()
        self.num_point = num_point #4096 多少个点一组？
        self.block_size = block_size #1.0 采样密度？1为全采样？
        self.transform = transform
        # 将'data/stanford_indoor3d/'路径下的文件排序
        rooms = sorted(os.listdir(data_root))
        rooms = [room for room in rooms if 'Area_' in room]
        # rooms = [room for room in rooms if 'Area_5' not in room]
        # rooms = [room for room in rooms if 'Area_4' not in room]
        # rooms = [room for room in rooms if 'Area_3' not in room]
        # 构建测试集或训练集的数据的文件名（转换好的npy），放在room里，默认区域5为测试集
        if split == 'train':
            rooms_split = [room for room in rooms if not 'Area_{}'.format(test_area) in room]
        else:
            rooms_split = [room for room in rooms if 'Area_{}'.format(test_area) in room]

        self.room_points, self.room_labels = [], [] #每个房间的点云和标签
        self.room_coord_min, self.room_coord_max = [], [] #每个房间的最大值和最小值？
        num_point_all = [] #初始化每个房间点的总数的列表
        labelweights = np.zeros(type_size) #初始标签权重
# tqdm 可视化进度条,每个NPY为一个房间，将每个房间（97个）的point（xyzrgb），label，max，min，都存在self里
        for room_name in tqdm(rooms_split, total=len(rooms_split)):
            room_path = os.path.join(data_root, room_name)
            room_data = np.load(room_path)  # 一个小房间内的xyzrgbl, N*7
            # 将np.array的数组，取索引从0到6的[:, 0: 6]。[:, 6]：取索引为7的,所以l为label？
            points, labels = room_data[:, 0:6], room_data[:, 6]  # xyzrgb, N*6; l, N
            #没办法了，做个映射吧 一共0，1，2，6，12这五个标签。映射成01234
            # tmp=[]
            # for i in labels:
            #     if i == 6:
            #         tmp.append(np.float64(3.0))
            #         continue
            #     if i == 12:
            #         tmp.append(np.float64(4.0))
            #         continue
            #     else:
            #         tmp.append(i)
            # labels=np.array(tmp)
            # histogram为统计直方图，这里是统计单个房间的所有点占14分类的比例（所有点放入14个桶中），为后面加权计算做准备
            tmp, _ = np.histogram(labels, range(6)) # 13分类更改14-->6
            labelweights += tmp
            # 寻找该房间中所有xyz的最小值和最大值,把标签，点云数据，最大最小值，点的数量，都赋给了self？
            coord_min, coord_max = np.amin(points, axis=0)[:3], np.amax(points, axis=0)[:3]
            self.room_points.append(points), self.room_labels.append(labels)
            self.room_coord_min.append(coord_min), self.room_coord_max.append(coord_max)
            num_point_all.append(labels.size)
        # for循环结束，每个房间一个文件，目前统计了每个房间的点云（xyzrgb），标签，最大，最小值？都放在self里
        # labelweights为一个14个元素的数组，每个元素代表属于该分类的点的数量（所有文件）
        labelweights = labelweights.astype(np.float32)
        labelweights = labelweights / np.sum(labelweights)# 权重归一化
        #amax为求最大值，power为求次方例如np.power([[0,1],[2,3]],[[4,5],[6,7]])=[[0,1][64,3^5]]
        #所以这一步是最大权重除以每个权重的值，开三次方??不知道在干啥
        self.labelweights = np.power(np.amax(labelweights) / labelweights, 1 / 3.0)
        print(self.labelweights)
        # sample_prob 为每个房间的点的个数占所有的房间点数之和的比例。
        sample_prob = num_point_all / np.sum(num_point_all)
        # 所有房间所有点加起来*超参1（采样概率），再除以4096，这是在划分blocak？一个20760个
        num_iter = int(np.sum(num_point_all) * sample_rate / num_point)
        room_idxs = []
       # 这是在根据每个房间点占总点的比率，求每个房间占的block的个数？
        for index in range(len(rooms_split)):
            room_idxs.extend([index] * int(round(sample_prob[index] * num_iter)))
            #创造了一个room_idxs，例如第一个房间分到3个block，第二个房间分到5个block，第三个房间2个block
            # 该list为：[0,0,0,1,1,1,1,1,2,2]
        self.room_idxs = np.array(room_idxs)
        print("Totally {} samples in {} set.".format(len(self.room_idxs), split))
    # room_idxs为大小为block数量，内容为000112233这样的npy，room_points为npy数量*房间点数*6，room——lables为n*1
    def __getitem__(self, idx):
        room_idx = self.room_idxs[idx]
        points = self.room_points[room_idx]   # N * 6
        labels = self.room_labels[room_idx]   # N
        N_points = points.shape[0]  # N
        # points为选中房间的点云数*6，labels为选中房间的点云数*1
        while (True):
            center = points[np.random.choice(N_points)][:3] #在该房间中一个随机的XYZ
            block_min = center - [self.block_size / 2.0, self.block_size / 2.0, 0]
            block_max = center + [self.block_size / 2.0, self.block_size / 2.0, 0]
            # np.where（true or false,x,y）true返回x否则返回y
            # 这里是只有一个参数的用法np.where(np.array([2,4,6,8])>5) 输出array[6,8]
            # 这里是筛选所有X,Y在上述给定范围内的点 的id？
            point_idxs = np.where((points[:, 0] >= block_min[0]) & (points[:, 0] <= block_max[0]) & (points[:, 1] >= block_min[1]) & (points[:, 1] <= block_max[1]))[0]
            # 如果剩下的符合条件的点大于1024，则跳出，否则重新设置条件，重新筛选
            if point_idxs.size > 1024:
                break
        # 跳出循环了，筛出了point_idxs个点，使用np.random.choice进行随机采样replace=True为允许重复采样
        # TODO 改进采样算法
        if point_idxs.size >= self.num_point:
            # 原取样方法
            selected_point_idxs = np.random.choice(point_idxs, self.num_point, replace=False)
            selected_points = points[selected_point_idxs, :]
            # 改进后
            # selected_points,selected_point_idxs = CaiYang.JunyunCaiyang(points,self.num_point,point_idxs)
        else:
            # 通常不会走这条，后续看情况再改
            selected_point_idxs = np.random.choice(point_idxs, self.num_point, replace=True)
            selected_points = points[selected_point_idxs, :]
        # normalize selected_points为选中的4096个xyzrgb
          # num_point * 6
        # 返回一个特定形状的用0填充的数组，用于处理数据
        current_points = np.zeros((selected_points.shape[0], 9))  # num_point * 9
        current_points[:, 6] = selected_points[:, 0] / self.room_coord_max[room_idx][0]
        current_points[:, 7] = selected_points[:, 1] / self.room_coord_max[room_idx][1]
        current_points[:, 8] = selected_points[:, 2] / self.room_coord_max[room_idx][2]
        selected_points[:, 0] = selected_points[:, 0] - center[0]
        selected_points[:, 1] = selected_points[:, 1] - center[1]
        selected_points[:, 3:6] /= 255.0
        current_points[:, 0:6] = selected_points
        # 经过一系列的不知名操作，得到了N*9的current_points，
        # 其中前两个为XY减去一个中心值,第三个Z不变，456为RGB/256,
        # 789为对应的XYZ除以对应的XYZ最大值
        current_labels = labels[selected_point_idxs]
        if self.transform is not None:
            current_points, current_labels = self.transform(current_points, current_labels)
        return current_points, current_labels

    def __len__(self):
        return len(self.room_idxs)

class ScannetDatasetWholeScene():
    # prepare to give prediction on each points
    def __init__(self, root, block_points=4096, split='test', test_area=5, stride=0.5, block_size=1.0, padding=0.001):
        self.block_points = block_points
        self.block_size = block_size
        self.padding = padding
        self.root = root
        self.split = split
        self.stride = stride
        self.scene_points_num = []
        assert split in ['train', 'test']
        if self.split == 'train':
            self.file_list = [d for d in os.listdir(root) if d.find('Area_%d' % test_area) is -1]
        else:
            self.file_list = [d for d in os.listdir(root) if d.find('Area_%d' % test_area) is not -1]
        self.scene_points_list = []
        self.semantic_labels_list = []
        self.room_coord_min, self.room_coord_max = [], []
        for file in self.file_list:
            data = np.load(root + file)
            points = data[:, :3]
            self.scene_points_list.append(data[:, :6])
            self.semantic_labels_list.append(data[:, 6])
            coord_min, coord_max = np.amin(points, axis=0)[:3], np.amax(points, axis=0)[:3]
            self.room_coord_min.append(coord_min), self.room_coord_max.append(coord_max)
        assert len(self.scene_points_list) == len(self.semantic_labels_list)

        labelweights = np.zeros(type_size)
        for seg in self.semantic_labels_list:
            tmp, _ = np.histogram(seg, range(6))
            self.scene_points_num.append(seg.shape[0])
            labelweights += tmp
        labelweights = labelweights.astype(np.float32)
        labelweights = labelweights / np.sum(labelweights)
        self.labelweights = np.power(np.amax(labelweights) / labelweights, 1 / 3.0)

    def __getitem__(self, index):
        point_set_ini = self.scene_points_list[index]
        points = point_set_ini[:,:6]
        labels = self.semantic_labels_list[index]
        coord_min, coord_max = np.amin(points, axis=0)[:3], np.amax(points, axis=0)[:3]
        grid_x = int(np.ceil(float(coord_max[0] - coord_min[0] - self.block_size) / self.stride) + 1)
        grid_y = int(np.ceil(float(coord_max[1] - coord_min[1] - self.block_size) / self.stride) + 1)
        data_room, label_room, sample_weight, index_room = np.array([]), np.array([]), np.array([]),  np.array([])
        for index_y in range(0, grid_y):
            for index_x in range(0, grid_x):
                s_x = coord_min[0] + index_x * self.stride
                e_x = min(s_x + self.block_size, coord_max[0])
                s_x = e_x - self.block_size
                s_y = coord_min[1] + index_y * self.stride
                e_y = min(s_y + self.block_size, coord_max[1])
                s_y = e_y - self.block_size
                point_idxs = np.where(
                    (points[:, 0] >= s_x - self.padding) & (points[:, 0] <= e_x + self.padding) & (points[:, 1] >= s_y - self.padding) & (
                                points[:, 1] <= e_y + self.padding))[0]
                if point_idxs.size == 0:
                    continue
                num_batch = int(np.ceil(point_idxs.size / self.block_points))
                point_size = int(num_batch * self.block_points)
                replace = False if (point_size - point_idxs.size <= point_idxs.size) else True
                point_idxs_repeat = np.random.choice(point_idxs, point_size - point_idxs.size, replace=replace)
                point_idxs = np.concatenate((point_idxs, point_idxs_repeat))
                np.random.shuffle(point_idxs)
                data_batch = points[point_idxs, :]
                normlized_xyz = np.zeros((point_size, 3))
                normlized_xyz[:, 0] = data_batch[:, 0] / coord_max[0]
                normlized_xyz[:, 1] = data_batch[:, 1] / coord_max[1]
                normlized_xyz[:, 2] = data_batch[:, 2] / coord_max[2]
                data_batch[:, 0] = data_batch[:, 0] - (s_x + self.block_size / 2.0)
                data_batch[:, 1] = data_batch[:, 1] - (s_y + self.block_size / 2.0)
                data_batch[:, 3:6] /= 255.0
                data_batch = np.concatenate((data_batch, normlized_xyz), axis=1)
                label_batch = labels[point_idxs].astype(int)
                batch_weight = self.labelweights[label_batch]
                data_room = np.vstack([data_room, data_batch]) if data_room.size else data_batch
                label_room = np.hstack([label_room, label_batch]) if label_room.size else label_batch
                sample_weight = np.hstack([sample_weight, batch_weight]) if label_room.size else batch_weight
                index_room = np.hstack([index_room, point_idxs]) if index_room.size else point_idxs
        data_room = data_room.reshape((-1, self.block_points, data_room.shape[1]))
        label_room = label_room.reshape((-1, self.block_points))
        sample_weight = sample_weight.reshape((-1, self.block_points))
        index_room = index_room.reshape((-1, self.block_points))
        return data_room, label_room, sample_weight, index_room

    def __len__(self):
        return len(self.scene_points_list)

if __name__ == '__main__':
    data_root = '/data/yxu/PointNonLocal/data/stanford_indoor3d_myself/'
    num_point, test_area, block_size, sample_rate = 4096, 5, 1.0, 0.01
    point_data = S3DISDataset(split='train', data_root=data_root, num_point=num_point, test_area=test_area, block_size=block_size, sample_rate=sample_rate, transform=None)
    print('point data size:', point_data.__len__())
    print('point data 0 shape:', point_data.__getitem__(0)[0].shape)
    print('point label 0 shape:', point_data.__getitem__(0)[1].shape)
    import torch, time, random
    manual_seed = 123
    random.seed(manual_seed)
    np.random.seed(manual_seed)
    torch.manual_seed(manual_seed)
    torch.cuda.manual_seed_all(manual_seed)
    def worker_init_fn(worker_id):
        random.seed(manual_seed + worker_id)
    train_loader = torch.utils.data.DataLoader(point_data, batch_size=16, shuffle=True, num_workers=16, pin_memory=True, worker_init_fn=worker_init_fn)
    for idx in range(4):
        end = time.time()
        for i, (input, target) in enumerate(train_loader):
            print('time: {}/{}--{}'.format(i+1, len(train_loader), time.time() - end))
            end = time.time()