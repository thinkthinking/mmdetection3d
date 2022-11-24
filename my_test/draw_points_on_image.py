import mmcv
from mmengine import load
from mmdet3d.visualization import Det3DLocalVisualizer
import numpy as np
info_file = load('demo/data/kitti/000008.pkl')
points = np.fromfile('demo/data/kitti/000008.bin', dtype=np.float32)
points = points.reshape(-1, 4)[:, :3]
lidar2img = np.array(info_file['data_list'][0]['images']['CAM2']['lidar2img'], dtype=np.float32)

visualizer = Det3DLocalVisualizer(save_dir="my_test",vis_backends=[dict(type='LocalVisBackend')])
img = mmcv.imread('demo/data/kitti/000008.png')
img = mmcv.imconvert(img, 'bgr', 'rgb')
visualizer.set_image(img)
visualizer.draw_points_on_image(points, lidar2img)
visualizer.add_image('points_on_image', visualizer.get_image())
# visualizer.show()