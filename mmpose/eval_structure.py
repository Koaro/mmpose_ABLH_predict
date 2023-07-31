# Copyright (c) OpenMMLab. All rights reserved.
import os
import warnings
from argparse import ArgumentParser


from mmpose.apis import (inference_top_down_pose_model, init_pose_model,
                         vis_pose_result)
from mmpose.datasets import DatasetInfo



def main():
    """Visualize the demo images.

    Require the json_file containing boxes.
    """
    parser = ArgumentParser()
    parser.add_argument('pose_config', help='Config file for detection')
    parser.add_argument('pose_checkpoint', help='Checkpoint file')
    parser.add_argument('--img-root', type=str, default='', help='Image root')
    parser.add_argument(
        '--json-file',
        type=str,
        default='',
        help='Json file containing image info.')
    parser.add_argument(
        '--show',
        action='store_true',
        default=False,
        help='whether to show img')
    parser.add_argument(
        '--out-img-root',
        type=str,
        default='',
        help='Root of the output img file. '
        'Default not saving the visualization images.')
    parser.add_argument(
        '--device', default='cuda:0', help='Device used for inference')
    parser.add_argument(
        '--kpt-thr', type=float, default=0.3, help='Keypoint score threshold')
    parser.add_argument(
        '--radius',
        type=int,
        default=4,
        help='Keypoint radius for visualization')
    parser.add_argument(
        '--thickness',
        type=int,
        default=1,
        help='Link thickness for visualization')

    args = parser.parse_args()


    # build the pose model from a config file and a checkpoint file
    pose_model = init_pose_model(
        args.pose_config, None, device=args.device.lower())
    
    config_name = args.pose_config.split('/')[-1][:-3]
    # print(config_name)

    with open(f'eval_{config_name}.txt', 'w') as file:
        # print(pose_model.eval())
        file.write(str(pose_model.eval()))

'''
python eval_structure.py configs/body/2d_kpt_sview_rgb_img/topdown_heatmap/atomos/hourglass_128_10daysAmonth_2feature_no_SHM.py
python eval_structure.py configs/body/2d_kpt_sview_rgb_img/topdown_heatmap/atomos/newclamp/hourglass_AfterInterpolation_6daysAmonth_2feature.py
'''

if __name__ == "__main__":
    main()

    