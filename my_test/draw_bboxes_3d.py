import torch
from mmdet3d.visualization import Det3DLocalVisualizer
from mmdet3d.structures import LiDARInstance3DBoxes
import numpy as np

points = np.fromfile(
    'tests/data/kitti/training/velodyne/000000.bin', dtype=np.float32)
points = points.reshape(-1, 4)
visualizer = Det3DLocalVisualizer(
    save_dir="my_test", vis_backends=[dict(type='LocalVisBackend')])
# set point cloud in visualizer
visualizer.set_points(points,points_size = 2)
bboxes_3d = LiDARInstance3DBoxes(
    torch.tensor([[8.7314, -1.8559, -1.5997, 1.2000, 0.4800, 1.8900,
                   -1.5808]]))
# Draw 3D bboxes
visualizer.draw_bboxes_3d(bboxes_3d)
visualizer.show(save_path="my_test/draw_bboxes_3d.png")