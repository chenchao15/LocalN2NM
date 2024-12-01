#!/usr/bin/env python3
# Copyright 2004-present Facebook. All Rights Reserved.

import logging
import numpy as np
import plyfile
import skimage.measure
import time
import torch
import trimesh
from scipy.spatial import cKDTree

import deep_sdf.utils


def create_mesh(
    decoder, latent_vec, filename, N=256, max_batch=32 ** 3, offset=None, scale=None, types=0, level=0.0,
):
    start = time.time()
    ply_filename = filename
  
    print('create mesh ing ')    

    decoder.eval()

    # NOTE: the voxel_origin is actually the (bottom, left, down) corner, not the middle
    voxel_origin = [-1, -1, -1]
    voxel_size = 2.0 / (N - 1)

    overall_index = torch.arange(0, N ** 3, 1, out=torch.LongTensor())
    samples = torch.zeros(N ** 3, 4)

    # transform first 3 columns
    # to be the x, y, z index
    samples[:, 2] = overall_index % N
    samples[:, 1] = (overall_index.long() / N) % N
    samples[:, 0] = ((overall_index.long() / N) / N) % N

    # transform first 3 columns
    # to be the x, y, z coordinate
    samples[:, 0] = (samples[:, 0] * voxel_size) + voxel_origin[2]
    samples[:, 1] = (samples[:, 1] * voxel_size) + voxel_origin[1]
    samples[:, 2] = (samples[:, 2] * voxel_size) + voxel_origin[0]

    num_samples = N ** 3

    samples.requires_grad = False

    head = 0

    while head < num_samples:
        sample_subset = samples[head : min(head + max_batch, num_samples), 0:3].cuda()
        sdf1 = deep_sdf.utils.decode_sdf(decoder, latent_vec, sample_subset)
        sdf1 = sdf1.squeeze(1).detach().cpu()
        # sdf2 = sdf2.squeeze(1).detach().cpu()
        if types == 0:
            samples[head : min(head + max_batch, num_samples), 3] = sdf1
        else:
            samples[head : min(head + max_batch, num_samples), 3] = sdf1
        head += max_batch

    sdf_values = samples[:, 3]
    sdf_values = sdf_values.reshape(N, N, N)

    end = time.time()
    print("sampling takes: %f" % (end - start))
    
    if types == 0:
        ply_name = ply_filename + '_' + str(level) + '.ply'
    else:
        ply_name = ply_filename + '_high.ply'
    convert_sdf_samples_to_ply(
        sdf_values.data.cpu(),
        voxel_origin,
        voxel_size,
        ply_name,
        offset,
        scale,
        level,
    )

def unpack_sdf_samples_from_kdtree(samples, kdtree, points, normal):
    dis, idx = kdtree.query(samples, k=3)
    avg_normal = np.mean(normal[idx], axis=1)
    sdf = np.sum((samples - points[idx][:, 0]) * avg_normal, axis=-1)
    return sdf, sdf
    
def create_mesh2(
    decoder, latent_vec, filename, N=256, max_batch=32 ** 3, offset=None, scale=None, types=0
):
    class_name = '02691156'
    low_path = f'/data1/cc/deepsdf_dataset/low_sap_data/{class_name}/1021a0914a7207aff927ed529ad90a11/1000_5.ply'
    mesh = trimesh.load(low_path)
    lps, lps_index = mesh.sample(20000, return_index=True)
    nps = mesh.face_normals[lps_index]
    lownpz = np.concatenate([lps, nps], 1)
    kdtree = cKDTree(lownpz[:, :3])

    start = time.time()
    ply_filename = filename

    decoder.eval()

    # NOTE: the voxel_origin is actually the (bottom, left, down) corner, not the middle
    voxel_origin = [-1, -1, -1]
    voxel_size = 2.0 / (N - 1)

    overall_index = torch.arange(0, N ** 3, 1, out=torch.LongTensor())
    samples = torch.zeros(N ** 3, 4)

    # transform first 3 columns
    # to be the x, y, z index
    samples[:, 2] = overall_index % N
    samples[:, 1] = (overall_index.long() / N) % N
    samples[:, 0] = ((overall_index.long() / N) / N) % N

    # transform first 3 columns
    # to be the x, y, z coordinate
    samples[:, 0] = (samples[:, 0] * voxel_size) + voxel_origin[2]
    samples[:, 1] = (samples[:, 1] * voxel_size) + voxel_origin[1]
    samples[:, 2] = (samples[:, 2] * voxel_size) + voxel_origin[0]

    num_samples = N ** 3

    samples.requires_grad = False
    samples = samples.detach().cpu().numpy()
    head = 0

    while head < num_samples:
        sample_subset = samples[head : min(head + max_batch, num_samples), 0:3] # .cuda()
        # sdf1, sdf2 = deep_sdf.utils.decode_sdf(decoder, latent_vec, sample_subset)
        sdf1, sdf2 = unpack_sdf_samples_from_kdtree(sample_subset, kdtree, lps, nps)
        
        # sdf1 = sdf1.squeeze(1).detach().cpu()
        # sdf2 = sdf2.squeeze(1).detach().cpu()
        if types == 0:
            samples[head : min(head + max_batch, num_samples), 3] = sdf1
        else:
            samples[head : min(head + max_batch, num_samples), 3] = sdf2
        head += max_batch

    sdf_values = samples[:, 3]
    sdf_values = sdf_values.reshape(N, N, N)

    end = time.time()
    print("sampling takes: %f" % (end - start))
    
    if types == 0:
        ply_name = ply_filename + '_low.ply'
    else:
        ply_name = ply_filename + '_high.ply'
    convert_sdf_samples_to_ply(
        sdf_values, #.data.cpu(),
        voxel_origin,
        voxel_size,
        ply_name,
        offset,
        scale,
    )


def convert_sdf_samples_to_ply(
    pytorch_3d_sdf_tensor,
    voxel_grid_origin,
    voxel_size,
    ply_filename_out,
    offset=None,
    scale=None,
    level=0.0,
):
    """
    Convert sdf samples to .ply

    :param pytorch_3d_sdf_tensor: a torch.FloatTensor of shape (n,n,n)
    :voxel_grid_origin: a list of three floats: the bottom, left, down origin of the voxel grid
    :voxel_size: float, the size of the voxels
    :ply_filename_out: string, path of the filename to save to

    This function adapted from: https://github.com/RobotLocomotion/spartan
    """
    start_time = time.time()

    numpy_3d_sdf_tensor = pytorch_3d_sdf_tensor.numpy()

    # verts, faces, normals, values = skimage.measure.marching_cubes_lewiner(
    #     numpy_3d_sdf_tensor, level=0.0, spacing=[voxel_size] * 3
    # )
    print(np.min(numpy_3d_sdf_tensor))
    print(np.max(numpy_3d_sdf_tensor))
    minn = np.min(numpy_3d_sdf_tensor)
    maxn = np.max(numpy_3d_sdf_tensor)
    a = (minn + maxn) / 2.0
    
    try:
        verts, faces, _, _ = skimage.measure.marching_cubes(numpy_3d_sdf_tensor, level=level, spacing=[voxel_size] * 3)
    except:
        return 
    #  transform from voxel coordinates to camera coordinates
    # note x and y are flipped in the output of marching_cubes
    mesh_points = np.zeros_like(verts)
    mesh_points[:, 0] = voxel_grid_origin[0] + verts[:, 0]
    mesh_points[:, 1] = voxel_grid_origin[1] + verts[:, 1]
    mesh_points[:, 2] = voxel_grid_origin[2] + verts[:, 2]

    # apply additional offset and scale
    if scale is not None:
        mesh_points = mesh_points / scale
    if offset is not None:
        mesh_points = mesh_points - offset

    # try writing to the ply file

    num_verts = verts.shape[0]
    num_faces = faces.shape[0]

    verts_tuple = np.zeros((num_verts,), dtype=[("x", "f4"), ("y", "f4"), ("z", "f4")])

    for i in range(0, num_verts):
        verts_tuple[i] = tuple(mesh_points[i, :])

    faces_building = []
    for i in range(0, num_faces):
        faces_building.append(((faces[i, :].tolist(),)))
    faces_tuple = np.array(faces_building, dtype=[("vertex_indices", "i4", (3,))])

    el_verts = plyfile.PlyElement.describe(verts_tuple, "vertex")
    el_faces = plyfile.PlyElement.describe(faces_tuple, "face")

    ply_data = plyfile.PlyData([el_verts, el_faces])
    logging.debug("saving mesh to %s" % (ply_filename_out))
    ply_data.write(ply_filename_out)

    logging.debug(
        "converting to ply format and writing to file took {} s".format(
            time.time() - start_time
        )
    )
