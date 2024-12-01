#!/usr/bin/env python3
# Copyright 2004-present Facebook. All Rights Reserved.

import glob
import logging
import numpy as np
import os
import random
import trimesh
import torch
import torch.utils.data

import deep_sdf.workspace as ws
from scipy.spatial import cKDTree
import point_cloud_utils as pcu
from single_dataset import process_data

def load_mesh(path, normalize=True):
  
    # water_path = path.replace('.obj', '_water.obj')
    # if os.path.exists(water_path):
    #     return trimesh.load(water_path)   

    obj_mesh = trimesh.load(path)
    if isinstance(obj_mesh, trimesh.Scene):
        mesh = trimesh.util.concatenate(
            tuple(trimesh.Trimesh(vertices=g.vertices, faces=g.faces) for g in obj_mesh.geometry.values()))
    else:
        assert (isinstance(obj_mesh, trimesh.Trimesh))
        mesh = obj_mesh

    if normalize:
        bbox = mesh.bounding_box.bounds
        loc = (bbox[0] + bbox[1]) / 2
        scale = (bbox[1] - bbox[0]).max() / (1 - 0)

        mesh.apply_translation(-loc)
        mesh.apply_scale(1 / scale)


        angle = -90 / 180 * np.pi
        R = trimesh.transformations.rotation_matrix(angle, [0, 1, 0])
        mesh.apply_transform(R)
    # v, f = pcu.make_mesh_watertight(mesh.vertices,mesh.faces, resolution=80000)
    # mesh = trimesh.Trimesh(v, f)
    # mesh.export(water_path)
    return mesh


def get_instance_filenames(data_source, split, load_kdtree, test=False):
    npzfiles = []
    for dataset in split:
        for class_name in split[dataset]:
            low_path = f'/data/cc/deepsdf_dataset/low_sap_data/{class_name}'
            for instance_name in split[dataset][class_name]:
                low_name = os.path.join(low_path, instance_name, '1000_5.ply')
                # if not os.path.exists(low_name):
                #     continue
                instance_filename = os.path.join(
                    class_name, instance_name, 'models', 'model_normalized.obj'
                )
                
                data_path = os.path.join(data_source, instance_filename)
                '''if load_kdtree:
                    instance_filename = os.path.join(class_name, instance_name)
                    data_path = os.path.join(data_source, instance_filename, 'pointcloud.npz')
                    print(data_path)'''

                # only for true test:
                npzfiles += [instance_filename]               
                continue

                if not os.path.isfile(data_path):
                    continue

                if os.path.exists(data_path):
                    '''try:
                        if not load_kdtree:
                            cc = np.load(data_path)
                    except:
                        continue'''
                    npzfiles += [instance_filename]
                # elif os.path.exists(os.path.join(data_source, ws.sdf_samples_subdir, instance_filename2)):
                    # npzfiles += [instance_filename2]
    
    # npzfiles = npzfiles[1:2]
    # npzfiles = npzfiles[55:56]
    # npzfiles = npzfiles[56:57]
    '''if not test:
        for i in range(50):
            npzfiles.append(npzfiles[-1])
    '''
    # npzfiles = npzfiles[:53]
    newnpz = []
    for i in range(5):
        newnpz.extend(npzfiles)
    npzfiles = newnpz
    npzfiles = npzfiles[:60]
    # npzfiles = npzfiles[:1200]
    # npzfiles = npzfiles[:300]
    print('all obj lens: ', len(npzfiles))
    return npzfiles



class NoMeshFileError(RuntimeError):
    """Raised when a mesh file is not found in a shape directory"""

    pass


class MultipleMeshFileError(RuntimeError):
    """"Raised when a there a multiple mesh files in a shape directory"""

    pass


def find_mesh_in_directory(shape_dir):
    mesh_filenames = list(glob.iglob(shape_dir + "/**/*.obj")) + list(
        glob.iglob(shape_dir + "/*.obj")
    )
    if len(mesh_filenames) == 0:
        raise NoMeshFileError()
    elif len(mesh_filenames) > 1:
        raise MultipleMeshFileError()
    return mesh_filenames[0]


def remove_nans(tensor):
    tensor_nan = torch.isnan(tensor[:, 3])
    return tensor[~tensor_nan, :]


def read_sdf_samples_into_ram(filename):
    npz = np.load(filename)
    pos_tensor = torch.from_numpy(npz["pos"])
    neg_tensor = torch.from_numpy(npz["neg"])

    return [pos_tensor, neg_tensor]


def read_sdf_samples_into_kdtree(filename, level=0):
    # npz = np.loadtxt(filename)
    '''pc = np.load(filename)
    points = pc['points']
    normals = pc['normals']
    npz = np.concatenate([points, normals], 1)    

    splits = filename.split('/')
    class_name, obj_name = splits[-3], splits[-2]
    low_path = f'/data1/cc/deepsdf_dataset/low_sap_data/{class_name}/{obj_name}/1000_5.ply'
    mesh = trimesh.load(low_path)
    points, index = mesh.sample(20000, return_index=True)
    normals = mesh.face_normals[index]
    lownpz = np.concatenate([points, normals], 1)
    
    return [npz, cKDTree(npz[:, :3]), lownpz, cKDTree(lownpz[:, :3])]'''
    splits = filename.split('/')
    print(filename)
    class_name, obj_name = splits[4], splits[5]
    # low_path = f'/data/cc/deepsdf_dataset/low_sap_data/{class_name}/{obj_name}/1000_{level}.ply'
    mesh = None
    # mesh = trimesh.load(low_path)
    # v, f = pcu.make_mesh_watertight(highmesh.vertices,highmesh.faces, resolution=80000)
    try:
        highmesh = load_mesh(filename)
    except:
        highmesh = mesh
    # v, f = pcu.make_mesh_watertight(highmesh.vertices,highmesh.faces, resolution=80000)
    #  highmesh = trimesh.Trimesh(v, f)
    return [highmesh, highmesh]

def read_sdf_samples_into_n2n(filename, level=0):
    npz = np.load(filename)
    points = npz['sample_near']
    samples = npz['sample']
    return samples, points

def unpack_sdf_samples(filename, subsample=None):
    # print('111111adsdsd')
    npz = np.load(filename)
    if subsample is None:
        return npz
    pos_tensor = remove_nans(torch.from_numpy(npz["pos"]))
    neg_tensor = remove_nans(torch.from_numpy(npz["neg"]))
    
    # split the sample into half
    half = int(subsample / 2)

    random_pos = (torch.rand(half) * pos_tensor.shape[0]).long()
    random_neg = (torch.rand(half) * neg_tensor.shape[0]).long()

    sample_pos = torch.index_select(pos_tensor, 0, random_pos)
    sample_neg = torch.index_select(neg_tensor, 0, random_neg)
    
    samples = torch.cat([sample_pos, sample_neg], 0)
    
    return samples


def unpack_sdf_samples_from_kdtree2_old(data, subsample=None):
    points, normal, kdtree, lowpoints, lownormal, lowkdtree = data[0][:, :3], data[0][:, 3:], data[1], data[2][:, :3], data[2][:, 3:], data[3]
    point_idx = np.random.choice(points.shape[0], subsample, replace = False)
    pointcloud = points[point_idx, :3]
    samples1 = pointcloud[:subsample // 2] + 9.0*np.expand_dims(0.027,-1) * np.random.normal(0.0, 1.0, size=(subsample // 2, 3))
    samples2 = pointcloud[:subsample // 2] + 1.0*np.expand_dims(0.027,-1) * np.random.normal(0.0, 1.0, size=(subsample // 2, 3))
    samples = np.concatenate([samples1, samples2], 0)
    
    dis, idx = kdtree.query(samples, k=3)
    avg_normal = np.mean(normal[idx], axis=1)
    sdf = np.sum((samples - points[idx][:, 0]) * avg_normal, axis=-1)
    sdf = sdf[..., None]
    # samples = np.concatenate([samples[None], sdf[None]], 2)
    # print(np.max(samples), np.min(samples))
    # print('sampe:s ', samples.shape)
    
    dis2, idx2 = lowkdtree.query(samples, k=3)
    avg_normal2 = np.mean(lownormal[idx2], axis=1)
    sdf2 = np.sum((samples - lowpoints[idx2][:, 0]) * avg_normal2, axis=-1)
    sdf2 = sdf2[..., None]
    samples = np.concatenate([samples[None], sdf[None], sdf2[None]], 2)
    return torch.from_numpy(samples).float()


def unpack_sdf_samples_from_n2n2(data, subsample=None):
    highmesh, lowmesh = data[0], data[1]
    points = highmesh.sample(40000)
    hv, hf = highmesh.vertices, highmesh.faces
    v, f = lowmesh.vertices, lowmesh.faces

    point_idx = np.random.choice(points.shape[0], subsample, replace = False)
    pointcloud = points[point_idx, :3]
    samples1 = pointcloud[:subsample // 2] + 1.0*np.expand_dims(0.027,-1) * np.random.normal(0.0, 1.0, size=(subsample // 2, 3))
    samples2 = pointcloud[:subsample // 2] + 0.5*np.expand_dims(0.027,-1) * np.random.normal(0.0, 1.0, size=(subsample // 2, 3))
    samples = np.concatenate([samples1, samples2], 0)
    
    kree = cKDTree(pointcloud)
    dis, index = kree.query(samples)
    pointcloud = pointcloud[index]
    
    '''sdf, _, _ = pcu.signed_distance_to_mesh(samples, hv, hf)
    sdf = sdf.reshape([-1])
    sdf = sdf.reshape([-1, 1])
    '''
    sdf = samples[:, :1]
    results = np.concatenate([samples[None], pointcloud[None], sdf[None]], 2)
    return torch.from_numpy(results).float()
    

def unpack_sdf_samples_from_n2n(data, subsample=None):
    sample, sample_near = data[0], data[1]
    sample = sample.reshape(-1, 3)
    sample_near = sample_near.reshape(-1, 3)
    point_idx = np.random.choice(sample.shape[0], subsample, replace = False)
    points = sample_near[point_idx]
    sample = sample[point_idx]
    samples = torch.cat([points, sample], 1)
    return samples.float()



def unpack_sdf_samples_from_kdtree(data, subsample=None, supervision=False):
    highmesh = data[0]
    points = highmesh.sample(40000)
    hv, hf = highmesh.vertices, highmesh.faces
    # v, f = lowmesh.vertices, lowmesh.faces

    point_idx = np.random.choice(points.shape[0], subsample, replace = False)
    pointcloud = points[point_idx, :3]
    samples1 = pointcloud[:subsample // 2] + 8.0*np.expand_dims(0.027,-1) * np.random.normal(0.0, 1.0, size=(subsample // 2, 3))
    samples2 = pointcloud[:subsample // 2] + 0.2*np.expand_dims(0.027,-1) * np.random.normal(0.0, 1.0, size=(subsample // 2, 3))
    samples = np.concatenate([samples1, samples2], 0)

    if supervision:
        sdf, _, _ = pcu.signed_distance_to_mesh(samples, hv, hf)
        sdf = sdf.reshape([-1, 1])
        samples = np.concatenate([samples[None], sdf[None], sdf[None]], 2)
    else:
        tree = cKDTree(pointcloud)
        dis, index = tree.query(samples)
        pointcloud = pointcloud[index]
        samples = np.concatenate([samples[None], pointcloud[None]], 2)
    return torch.from_numpy(samples).float()


def unpack_sdf_samples_from_ram(data, subsample=None):
    # print('adsdsdsdsds')
    if subsample is None:
        return data
    pos_tensor = data[0]
    neg_tensor = data[1]

    # split the sample into half
    half = int(subsample / 2)

    pos_size = pos_tensor.shape[0]
    neg_size = neg_tensor.shape[0]

    pos_start_ind = random.randint(0, pos_size - half)
    sample_pos = pos_tensor[pos_start_ind : (pos_start_ind + half)]

    if neg_size <= half:
        random_neg = (torch.rand(half) * neg_tensor.shape[0]).long()
        sample_neg = torch.index_select(neg_tensor, 0, random_neg)
    else:
        neg_start_ind = random.randint(0, neg_size - half)
        sample_neg = neg_tensor[neg_start_ind : (neg_start_ind + half)]

    samples = torch.cat([sample_pos, sample_neg], 0)

    return samples


class SDFSamples(torch.utils.data.Dataset):
    def __init__(
        self,
        data_source,
        split,
        subsample,
        load_ram=False,
        print_filename=False,
        unsuper=False,
        load_n2n=False,
        num_files=1000000,
        test=False,
    ):
        self.subsample = subsample
      
        self.data_source = data_source
       
        self.npyfiles = get_instance_filenames(data_source, split, unsuper, test)

        if test:
            return

        logging.debug(
            "using "
            + str(len(self.npyfiles))
            + " shapes from data source "
            + data_source
        )

        self.load_ram = load_ram
        self.load_n2n = load_n2n
        if load_ram:
            self.loaded_data = []
            for f in self.npyfiles:
                filename = os.path.join(self.data_source, ws.sdf_samples_subdir, f)
                npz = np.load(filename)
                pos_tensor = remove_nans(torch.from_numpy(npz["pos"]))
                neg_tensor = remove_nans(torch.from_numpy(npz["neg"]))
                self.loaded_data.append(
                    [
                        pos_tensor[torch.randperm(pos_tensor.shape[0])],
                        neg_tensor[torch.randperm(neg_tensor.shape[0])],
                    ]
                )
        self.unsuper = unsuper
        
        if True:
            self.loaded_data = []
            print('loading dataset...')
            for f in self.npyfiles:
                # print(f'process : {f}')
                filename = os.path.join(self.data_source, f)
                classf, objf = f.split('/')[0], f.split('/')[1]
                try:
                    highmesh = load_mesh(filename, normalize=True)
                except:
                    pass
                # v, f = pcu.make_mesh_watertight(highmesh.vertices,highmesh.faces, resolution=80000)
                # highmesh = trimesh.Trimesh(v, f)
                # highmesh.export('cc.ply')
                # exit()
                for i in range(1):
                    lownpz_name = f'/data/cc/deepsdf_dataset/low_sap_data/{classf}/{objf}/1000_{i}.ply'
                    # lownpz_name = f'/data1/cc/deepsdf_dataset/low_sap_data/{classf}/{objf}/1000_0.ply'
                    # print(lownpz_name)
                    '''try:
                        mesh = trimesh.load(lownpz_name)
                    except:
                        a = 3'''
                    # lv, lf = pcu.make_mesh_watertight(mesh.vertices,highmesh.faces, resolution=80000)
                    # mesh = trimesh.Trimesh(lv, lf)
                    self.loaded_data.append([highmesh, highmesh])
        elif load_n2n:
            self.loaded_data = []
            print('loading datasets...')
            for f in self.npyfiles:
                print(f)
                filename = os.path.join(self.data_source, f)
                classf, objf = f.split('/')[0], f.split('/')[1]
                out = f'data/SRB/{objf}.npz'
                if not os.path.exists(out):
                    sample, sample_near = process_data(out, filename)
                else:
                    npz = np.load(out)
                    sample = npz['sample']
                    sample_near = npz['sample_near']
                self.loaded_data.append([torch.from_numpy(sample), torch.from_numpy(sample_near)])
        print('final datalen : ', len(self.loaded_data))
        
    def __len__(self):
        return len(self.npyfiles)

    def __getitem__(self, idx):
        if self.load_ram:
            return (
                unpack_sdf_samples_from_ram(self.loaded_data[idx], self.subsample),
                idx,
            )
        elif self.unsuper:
            return unpack_sdf_samples_from_kdtree(self.loaded_data[idx], self.subsample), idx
        else:
            return unpack_sdf_samples_from_kdtree(self.loaded_data[idx], self.subsample, supervision=True), idx
