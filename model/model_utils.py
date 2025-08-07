import torch
from pointnet2_ops import pointnet2_utils

def knn_index(k_nearest_neighbors, source_coord, target_coord):
    """
    Input:
        k_nearest_neighbors: max sample number in local region
        source_coord: all points, [B, C, M]
        target_coord: query points, [B, C, N]
    Return:
        group_idx: grouped points index, [B, M, k_nearest_neighbors]
    """
    sqrdists = square_distance(source_coord, target_coord)
    _, group_idx = torch.topk(sqrdists, k_nearest_neighbors, dim = -1, largest=False, sorted=False)
    return group_idx


def square_distance(source, target):
    """
    Calculate Euclid distance between each two points.
    src^T * dst = xn * xm + yn * ym + zn * zm
    sum(src^2, dim=-1) = xn*xn + yn*yn + zn*zn;
    sum(dst^2, dim=-1) = xm*xm + ym*ym + zm*zm;
    dist = (xn - xm)^2 + (yn - ym)^2 + (zn - zm)^2
         = sum(src**2,dim=-1)+sum(dst**2,dim=-1)-2*src^T*dst
    Input:
        src: source points, [B, N, C]
        dst: target points, [B, M, C]
    Output:
        dist: per-point square distance, [B, N, M]
    """
    B, _, SOURCE_NUM_POINTS = source.shape
    _, _, TARGET_NUM_POINTS = target.shape
    dist = -2 * torch.matmul(source.permute(0, 2, 1), target)
    dist += torch.sum(source ** 2, -2).view(B, SOURCE_NUM_POINTS, 1)
    dist += torch.sum(target ** 2, -2).view(B, 1, TARGET_NUM_POINTS)
    return dist  

def fps_downsample(points, sample_num):
    """
    Downsample the point cloud using FPS(Furthest Point Sampling).
    Input:
        - points: bs, [3 + feature_dim], N
        - sample_num: int, number of points to sample
    Output:
        - return: bs, [3 + feature_dim], sample_num
    """
    with torch.no_grad():
        coords = points[:, :3] # The first 3 channels are the coordinates
        xyz = coords.transpose(1, 2).contiguous() # b, n, 3
        fps_idx = pointnet2_utils.furthest_point_sample(xyz, sample_num)
        new_points = pointnet2_utils.gather_operation(points, fps_idx) # Gather the features. For each center point(fps_idx), get the feature of the k nearest neighbors
        return new_points