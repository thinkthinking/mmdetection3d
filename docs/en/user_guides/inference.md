# Inference

## Introduction

We provide scripts for multi-modality/single-modality (LiDAR-based/vision-based), indoor/outdoor 3D detection and 3D semantic segmentation demos. The pre-trained models can be downloaded from [model zoo](https://github.com/open-mmlab/mmdetection3d/blob/master/docs/en/model_zoo.md/). We provide pre-processed sample data from KITTI, SUN RGB-D, nuScenes and ScanNet dataset. You can use any other data following our pre-processing steps.

## Testing

### 3D Detection

#### Single-modality demo

To test a 3D detector on point cloud data, simply run:

```shell
python demo/pcd_demo.py ${PCD_FILE} ${CONFIG_FILE} ${CHECKPOINT_FILE} [--device ${GPU_ID}] [--score-thr ${SCORE_THR}] [--out-dir ${OUT_DIR}] [--show]
```

The visualization results including a point cloud and predicted 3D bounding boxes will be saved in `${OUT_DIR}/PCD_NAME`, which you can open using [MeshLab](http://www.meshlab.net/). Note that if you set the flag `--show`, the prediction result will be displayed online using [Open3D](http://www.open3d.org/).

Example on KITTI data using [SECOND](https://github.com/open-mmlab/mmdetection3d/tree/master/configs/second) model:

```shell
python demo/pcd_demo.py \
demo/data/kitti/000008.bin \
configs/second/second_hv_secfpn_8xb6-80e_kitti-3d-car.py \
checkpoints/hv_pointpillars_secfpn_6x8_160e_kitti-3d-car_20220331_134606-d42d15ed.pth \
--save-name my_test/pcd_demo.png \
--show 
```

Example on SUN RGB-D data using [VoteNet](https://github.com/open-mmlab/mmdetection3d/tree/master/configs/votenet) model:

```shell
python demo/pcd_demo.py demo/data/sunrgbd/sunrgbd_000017.bin configs/votenet/votenet_8xb16_sunrgbd-3d.py checkpoints/votenet_8xb16_sunrgbd-3d_20200620_230238-4483c0c0.pth
```

Remember to convert the VoteNet checkpoint if you are using mmdetection3d version >= 0.6.0. See its [README](https://github.com/open-mmlab/mmdetection3d/blob/master/configs/votenet/README.md/) for detailed instructions on how to convert the checkpoint.

#### Multi-modality demo

To test a 3D detector on multi-modality data (typically point cloud and image), simply run:

```shell
python demo/multi_modality_demo.py ${PCD_FILE} ${IMAGE_FILE} ${ANNOTATION_FILE} ${CONFIG_FILE} ${CHECKPOINT_FILE} [--device ${GPU_ID}] [--score-thr ${SCORE_THR}] [--out-dir ${OUT_DIR}] [--show]
```

where the `ANNOTATION_FILE` should provide the 3D to 2D projection matrix. The visualization results including a point cloud, an image, predicted 3D bounding boxes and their projection on the image will be saved in `${OUT_DIR}/PCD_NAME`.

Example on KITTI data using [MVX-Net](https://github.com/open-mmlab/mmdetection3d/tree/master/configs/mvxnet) model:

```shell
python demo/multi_modality_demo.py demo/data/kitti/000008.bin \
demo/data/kitti/000008.png \
demo/data/kitti/000008.pkl \
configs/mvxnet/mvxnet_fpn_dv_second_secfpn_8xb2-80e_kitti-3d-3class.py \
checkpoints/dv_mvx-fpn_second_secfpn_adamw_2x8_80e_kitti-3d-3class_20210831_060805-83442923.pth \
--save-name my_test/demo_multi_modality.png \
--out-dir my_test \
--show
```

Example on SUN RGB-D data using [ImVoteNet](https://github.com/open-mmlab/mmdetection3d/tree/master/configs/imvotenet) model:

```shell
python demo/multi_modality_demo.py \
demo/data/sunrgbd/000017.bin \
demo/data/sunrgbd/000017.jpg \
demo/data/sunrgbd/000017_infos.pkl \
configs/imvotenet/imvotenet_faster-rcnn-r50_fpn_4xb2_sunrgbd-3d.py \
checkpoints/imvotenet_faster_rcnn_r50_fpn_2x4_sunrgbd-3d-10class_20210819_225618-62eba6ce.pth \
--save-name my_test/demo_multi_modality.png \
--out-dir my_test \
--show
```

### Monocular 3D Detection

To test a monocular 3D detector on image data, simply run:

```shell
python demo/mono_det_demo.py ${IMAGE_FILE} ${ANNOTATION_FILE} ${CONFIG_FILE} ${CHECKPOINT_FILE} [--device ${GPU_ID}] [--cam-type ${CAM_TYPE}] [--score-thr ${SCORE-THR}] [--out-dir ${OUT_DIR}] [--show]
```

where the `ANNOTATION_FILE` should provide the 3D to 2D projection matrix (camera intrinsic matrix), and `CAM_TYPE` should be specified according to dataset. For example, if you want to inference on the front camera image, the `CAM_TYPE` should be set as `CAM_2` for KITTI, and `CAM_FRONT` for nuScenes. By specifying `CAM_TYPE`, you can even infer on any camera images for datasets with multi-view cameras, such as nuScenes and Waymo. `SCORE-THR` is the 3D bbox threshold while visualization. The visualization results including an image and its predicted 3D bounding boxes projected on the image will be saved in `${OUT_DIR}/IMG_NAME`.

Example on nuScenes data using [FCOS3D](https://github.com/open-mmlab/mmdetection3d/tree/master/configs/fcos3d) model:

```shell
python demo/mono_det_demo.py \
demo/data/nuscenes/n015-2018-07-24-11-22-45+0800__CAM_BACK__1532402927637525.jpg \
demo/data/nuscenes/n015-2018-07-24-11-22-45+0800__CAM_BACK__1532402927637525_mono3d.pkl \
configs/fcos3d/fcos3d_r101-caffe-dcn_fpn_head-gn_8xb2-1x_nus-mono3d_finetune.py \
checkpoints/fcos3d_r101_caffe_fpn_gn-head_dcn_2x8_1x_nus-mono3d_20210715_235813-4bed5239.pth \
--out-dir my_test/demo_mono_det.png \
--show
```

Note that when visualizing results of monocular 3D detection for flipped images, the camera intrinsic matrix should also be modified accordingly. See more details and examples in PR [#744](https://github.com/open-mmlab/mmdetection3d/pull/744).

### 3D Segmentation

To test a 3D segmentor on point cloud data, simply run:

```shell
python demo/pcd_seg_demo.py ${PCD_FILE} ${CONFIG_FILE} ${CHECKPOINT_FILE} [--device ${GPU_ID}] [--out-dir ${OUT_DIR}] [--show]
```

The visualization results including a point cloud and its predicted 3D segmentation mask will be saved in `${OUT_DIR}/PCD_NAME`.

Example on ScanNet data using [PointNet++ (SSG)](https://github.com/open-mmlab/mmdetection3d/tree/master/configs/pointnet2) model:

```shell
python demo/pcd_seg_demo.py \
demo/data/scannet/scene0000_00.bin \
configs/pointnet2/pointnet2_msg_2xb16-cosine-250e_scannet-seg.py \
checkpoints/pointnet2_msg_16x2_cosine_250e_scannet_seg-3d-20class_20210514_144009-24477ab1.pth \
--out-dir demo_seg.png
```
