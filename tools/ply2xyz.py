import argparse
import open3d as o3d
import numpy as np


def convert_ply_to_xyz(ply_file, xyz_file):
    """Convert PLY file to XYZ format using Open3D."""
    try:
        # Read PLY file using Open3D
        point_cloud = o3d.io.read_point_cloud(ply_file)
        
        # Get points as numpy array
        points = np.asarray(point_cloud.points)
        
        # Write XYZ file
        with open(xyz_file, 'w') as f:
            for point in points:
                x, y, z = point
                f.write(f"{x} {y} {z}\n")
        
        print(f"Converted {len(points)} points from {ply_file} to {xyz_file}")
        
    except Exception as e:
        print(f"Error converting file: {e}")
        return False
    
    return True


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ply_file", type=str, required=True)
    parser.add_argument("--xyz_file", type=str, required=True)
    args = parser.parse_args()

    ply_file = args.ply_file
    xyz_file = args.xyz_file
    
    convert_ply_to_xyz(ply_file, xyz_file)