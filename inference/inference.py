import trimesh
from inference.Inferer import Inferer
from model.AdaPoinTr import AdaPoinTr
from dataset.dataset import DeepCADDataset

if __name__ == "__main__":
    model = AdaPoinTr()
    inferer = Inferer(model, model_path="runs/AdaPoinTr_DeepCAD_Aug14_09-21-49/best_model.pth")
    dataset = DeepCADDataset(
        data_root='/mnt/d/data/processed_deepcad',
        index_path='/mnt/d/data/DeepCAD/data_index/deduplicated_deepcad_testing_7_30.txt'
    )
    for i in range(len(dataset)):
        gt_brep_grid, gt_point_cloud, partial_point_cloud, cropped_point_cloud = dataset[i]
        coarse_point_cloud, rebuild_points = inferer.single_infer(point_cloud=partial_point_cloud)
        trimesh.PointCloud(partial_point_cloud.cpu().numpy()).export("partial.ply")
        trimesh.PointCloud(gt_point_cloud.cpu().numpy()).export("gt.ply")
        trimesh.PointCloud(coarse_point_cloud.reshape(-1, 3).cpu().numpy()).export("coarse.ply")
        trimesh.PointCloud(rebuild_points.reshape(-1, 3).cpu().numpy()).export("rebuild.ply")
        break
