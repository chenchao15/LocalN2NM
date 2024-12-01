
import torch
import torch.nn.functional as F
import numpy as np
import os
from scipy.spatial import cKDTree
import trimesh
from deep_sdf.data import load_mesh
import point_cloud_utils as pcu
from gen_noise import noise_points_produce

local = True
load_noise = False

def process_data(data_dir, dataname):
    if os.path.exists(os.path.join(data_dir, dataname) + '.ply'):
        pointcloud = trimesh.load(os.path.join(data_dir, dataname) + '.ply').vertices
        pointcloud = np.asarray(pointcloud)
    elif os.path.exists(os.path.join(data_dir, dataname) + '.xyz'):
        pointcloud = np.load(os.path.join(data_dir, dataname)) + '.xyz'
    else:
        print('Only support .xyz or .ply data. Please make adjust your data.')
        exit()
    shape_scale = np.max([np.max(pointcloud[:,0])-np.min(pointcloud[:,0]),np.max(pointcloud[:,1])-np.min(pointcloud[:,1]),np.max(pointcloud[:,2])-np.min(pointcloud[:,2])])
    shape_center = [(np.max(pointcloud[:,0])+np.min(pointcloud[:,0]))/2, (np.max(pointcloud[:,1])+np.min(pointcloud[:,1]))/2, (np.max(pointcloud[:,2])+np.min(pointcloud[:,2]))/2]
    pointcloud = pointcloud - shape_center
    pointcloud = pointcloud / shape_scale

    POINT_NUM = pointcloud.shape[0] // 60
    POINT_NUM_GT = pointcloud.shape[0] // 60 * 60
    QUERY_EACH = 1000000//POINT_NUM_GT

    point_idx = np.random.choice(pointcloud.shape[0], POINT_NUM_GT, replace = False)
    pointcloud = pointcloud[point_idx,:]
    ptree = cKDTree(pointcloud)
    sigmas = []
    for p in np.array_split(pointcloud,100,axis=0):
        d = ptree.query(p,51)
        sigmas.append(d[0][:,-1])
    
    sigmas = np.concatenate(sigmas)
    sample = []
    sample_near = []

    for i in range(QUERY_EACH):
        scale = 0.25 * np.sqrt(POINT_NUM_GT // 20000)
        tt = pointcloud + scale*np.expand_dims(sigmas,-1) * np.random.normal(0.0, 1.0, size=pointcloud.shape)
        sample.append(tt)
        tt = tt.reshape(-1,POINT_NUM,3)

        sample_near_tmp = []
        for j in range(tt.shape[0]):
            nearest_idx = search_nearest_point(torch.tensor(tt[j]).float().cuda(), torch.tensor(pointcloud).float().cuda())
            nearest_points = pointcloud[nearest_idx]
            nearest_points = np.asarray(nearest_points).reshape(-1,3)
            sample_near_tmp.append(nearest_points)
        sample_near_tmp = np.asarray(sample_near_tmp)
        sample_near_tmp = sample_near_tmp.reshape(-1,3)
        sample_near.append(sample_near_tmp)
        
    sample = np.asarray(sample)
    sample_near = np.asarray(sample_near)

    np.savez(os.path.join(out_dir, dataname)+'.npz', sample = sample, point = pointcloud, sample_near = sample_near)


def process_data2(data_dir, objname, subsample=8000, supervision=False):
    highmesh = load_mesh(objname)
    points = highmesh.sample(40000)
    hv, hf = highmesh.vertices, highmesh.faces
    
    all_sample = []
    all_near = []
    for i in range(40):
        print(i)
        point_idx = np.random.choice(points.shape[0], subsample, replace = False)
        pointcloud = points[point_idx, :3]
        samples1 = pointcloud[:subsample // 2] + 1.0*np.expand_dims(0.047,-1) * np.random.normal(0.0, 1.0, size=(subsample // 2, 3))
        samples2 = pointcloud[:subsample // 2] + 0.5*np.expand_dims(0.047,-1) * np.random.normal(0.0, 1.0, size=(subsample // 2, 3))
        samples = np.concatenate([samples1, samples2], 0)
        if supervision:
            sdf, _, _ = pcu.signed_distance_to_mesh(samples, hv, hf)
            sdf = sdf.reshape([-1, 1])
            all_sample.append(samples)
            all_near.append(sdf)
        else:
            tree = cKDTree(pointcloud)
            dis, index = tree.query(samples)
            pointcloud = pointcloud[index]
            all_sample.append(samples)
            all_near.append(pointcloud)
    all_sample = np.array(all_sample)
    all_near = np.array(all_near)
    if supervision:
        np.savez(os.path.join(data_dir, 'sdf.npz'), sample=all_sample, sdf=all_near)
    else:
        np.savez(os.path.join(data_dir, 'samples.npz'), sample=all_sample, sample_near=all_near)



def process_data_local(data_dir, objname, base_sample=10000, supervision=False, outname=None, sample_radius=10000, sample_param=1.0, classname=None, density=1.0):
    if objname[-4:] == '.npy':
        pointcloud = np.load(objname)
    elif objname[-4:] == '.obj':
        mesh = load_mesh(objname, normalize=False)
        pointcloud = mesh.sample(40000)
    elif objname[-4:] == '.ply':
        pointcloud = trimesh.load(objname)
        pointcloud = pointcloud.vertices

    shape_scale = np.max([np.max(pointcloud[:,0])-np.min(pointcloud[:,0]),np.max(pointcloud[:,1])-np.min(pointcloud[:,1]),np.max(pointcloud[:,2])-np.min(pointcloud[:,2])])
    shape_center = [(np.max(pointcloud[:,0])+np.min(pointcloud[:,0]))/2, (np.max(pointcloud[:,1])+np.min(pointcloud[:,1]))/2, (np.max(pointcloud[:,2])+np.min(pointcloud[:,2]))/2]
    pointcloud = pointcloud - shape_center
    pointcloud = pointcloud / shape_scale
    points = pointcloud


    all_samples = []
    all_nears = []
    for i in range(420):
        point_idx = np.random.choice(points.shape[0], base_sample, replace=False)
        pointcloud = points[point_idx, :3]
        samples1 = pointcloud[:base_sample // 2] + 0.5 * sample_param*np.expand_dims(0.047,-1) * np.random.normal(0.0, 1.0, size=(base_sample // 2, 3))
        samples2 = pointcloud[:base_sample // 2] + 0.1 * sample_param *np.expand_dims(0.047,-1) * np.random.normal(0.0, 1.0, size=(base_sample // 2, 3))
        samples = np.concatenate([samples1, samples2], 0)
        tree = cKDTree(pointcloud)
        dis, index = tree.query(samples)
        pointcloud = pointcloud[index]
        select_point_idx = np.random.choice(pointcloud.shape[0], 1, replace=False)
        select_point = pointcloud[select_point_idx, :3]
        tree2 = cKDTree(pointcloud)
        dis2, index2 = tree2.query(select_point, k=sample_radius)
        final_samples = samples[index2]
        final_points = pointcloud[index2]
        all_samples.append(final_samples)
        all_nears.append(final_points)
    all_samples = np.asarray(all_samples)
    all_nears = np.asarray(all_nears)
    
    if outname is not None:
        np.savez(os.path.join(data_dir, outname), sample=all_samples, sample_near=all_nears)
    else:
        np.savez(os.path.join(data_dir, 'sample_locals.npz'), sample=all_samples, sample_near=all_nears) 


def search_nearest_point(point_batch, point_gt):
    num_point_batch, num_point_gt = point_batch.shape[0], point_gt.shape[0]
    point_batch = point_batch.unsqueeze(1).repeat(1, num_point_gt, 1)
    point_gt = point_gt.unsqueeze(0).repeat(num_point_batch, 1, 1)

    distances = torch.sqrt(torch.sum((point_batch-point_gt) ** 2, axis=-1) + 1e-12) 
    dis_idx = torch.argmin(distances, axis=1).detach().cpu().numpy()

    return dis_idx
    

class DatasetNP:
    def __init__(self, dataname, sample_radius=1000, sample_param=1.0, supervision=False, load_noise=False, noise_level=0.01, classname=None, datadir=None, density=1.0):
        super(DatasetNP, self).__init__()
        self.device = torch.device('cuda')
        # self.conf = conf
        self.noise_level = noise_level
        self.sample_radius=sample_radius
        self.density = density
        if classname == 'shapenet':
            self.data_dir = dataname.replace('model_normalized.obj', '')
            self.objname = 'model_normalized'
        else:
            self.data_dir = datadir
            self.objname = dataname.split('/')[-1]
            self.objname = self.objname.split('.')[0]

        self.np_data_name = 'samples.npz'
        self.supervision=supervision
        if supervision:
            self.np_data_name = 'sdf.npz'
        elif not supervision and local and not load_noise:
            self.np_data_name = self.objname + '_samples.npz'
        elif not supervision and local and load_noise:
            self.np_data_name = self.objname + '_noise_sample.npz'
        print(self.np_data_name)

        if load_noise:
            noise_dataname = os.path.join(self.data_dir, 'noise_' + str(self.noise_level) + '_' + str(self.density) + '_ununimal.ply')
            if not os.path.exists(noise_dataname):
                noise_points_produce(dataname, noise_dataname, self.noise_level, self.density)
            dataname = noise_dataname

        if not os.path.exists(os.path.join(self.data_dir, self.np_data_name)):
            print('Data existing. Loading data...')
        else:
            print('Data not found. Processing data...')
            if not local:
                process_data2(self.data_dir, dataname, supervision=supervision)
            else:
                process_data_local(self.data_dir, dataname, supervision=supervision, outname=self.np_data_name, sample_radius=sample_radius, sample_param=sample_param, classname=classname, density=self.density)

        load_data = np.load(os.path.join(self.data_dir, self.np_data_name))
        
        if supervision:
            self.sdf = np.asarray(load_data['sdf']).reshape(-1, 1)
        else:
            self.point = np.asarray(load_data['sample_near']).reshape(-1,3)
        self.sample = np.asarray(load_data['sample']).reshape(-1,3)
        if local:
            self.sample_points_num = self.sample.shape[0] #-1
        else:
            self.sample_points_num = self.sample.shape[0] - 1
        if not supervision:
            self.object_bbox_min = np.array([np.min(self.point[:,0]), np.min(self.point[:,1]), np.min(self.point[:,2])]) -0.05
            self.object_bbox_max = np.array([np.max(self.point[:,0]), np.max(self.point[:,1]), np.max(self.point[:,2])]) +0.05
            print('Data bounding box:',self.object_bbox_min,self.object_bbox_max)
    
        if supervision:
            self.sdf = torch.from_numpy(self.sdf).to(self.device).float()
        else:
            self.point = torch.from_numpy(self.point).to(self.device).float()
        self.sample = torch.from_numpy(self.sample).to(self.device).float()
        
        print('Load data: End')

    def np_train_data(self, batch_size):
        index_coarse = np.random.choice(10, 1)
        index_fine = np.random.choice(self.sample_points_num//10, batch_size, replace = False)
        index = index_fine * 10 + index_coarse
        if self.supervision:
            points = self.sdf[index]
        else:
            points = self.point[index]
        sample = self.sample[index]
        out = torch.cat([sample, points], 1)
        return out.to(self.device) 

    def np_train_local_data(self, batch_size):
        if batch_size != self.sample_radius:
            print('batch_size does not equal to!!!')
            exit()
        indexes = np.random.choice(self.sample_points_num, batch_size, replace=False)
        points = self.point[indexes]
        sample = self.sample[indexes]
        out = torch.cat([sample, points], 1)
        return out.to(self.device)
