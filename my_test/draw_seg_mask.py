import torch
from mmdet3d.visualization import Det3DLocalVisualizer
import numpy as np

points = np.fromfile('tests/data/s3dis/points/Area_1_office_2.bin', dtype=np.float32)
points = points.reshape(-1, 3)
visualizer = Det3DLocalVisualizer(save_dir="my_test", vis_backends=[dict(type='LocalVisBackend')])
visualizer.set_points(points,points_size = 2)
mask = np.random.rand(points.shape[0], 3)
points_with_mask = np.concatenate((points, mask), axis=-1)
# Draw 3D points with mask
visualizer.draw_seg_mask(points_with_mask)
visualizer.show(save_path="my_test/draw_seg_mask.png")