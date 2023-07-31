# Copyright (c) OpenMMLab. All rights reserved.
import os
import warnings
from argparse import ArgumentParser

import mmcv
from xtcocotools.coco import COCO

from mmpose.apis import (inference_top_down_pose_model, init_pose_model,
                         vis_pose_result)
from mmpose.datasets import DatasetInfo

import csv

from preprocess import ProcessRaw

def main():
    """Visualize the demo images.

    Require the json_file containing boxes.
    """
    parser = ArgumentParser()
    parser.add_argument('pose_config', help='Config file for detection')
    parser.add_argument('pose_checkpoint', help='Checkpoint file')
    parser.add_argument('--img-root', type=str, default='', help='Image root')
    parser.add_argument('--csv-root', type=str, default='', help='CSV root')
    parser.add_argument('--split', type=int, nargs='+', help='Split dataset [test, val, train]')
    parser.add_argument('--feature', nargs='+', default=[], help='Chose feature to predict [rw, snr, si]')
    parser.add_argument(
        '--json-file',
        type=str,
        default='',
        help='Json file containing image info.') ###
    parser.add_argument(
        '--show',
        action='store_true',
        default=False,
        help='whether to show img')
    parser.add_argument(
        '--out-root',
        type=str,
        default='',
        help='Root of the output file. '
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

    assert args.show or (args.out_root != '')


    task_num = 0
    if args.img_root != '' and args.csv_root == '':
        task_num = 1 # predict from img
    elif args.img_root == '' and args.csv_root != '':
        task_num = 2 # predict from csv

    assert task_num != 0

    if task_num == 1 : # img
        img_path = args.img_root
        annotation_path = args.json_file
    elif task_num == 2 :
        annotation_path = args.out_root # first make .json at out_root, then locate .json
        img_path = os.path.join(args.out_root, 'imgs/')
        ProcessRaw(args.csv_root, img_path, annotation_path, split=args.split, feature=args.feature)
        annotation_path = os.path.join(annotation_path, 'annotations.json')

    coco = COCO(annotation_path)
    # build the pose model from a config file and a checkpoint file
    pose_model = init_pose_model(
        args.pose_config, args.pose_checkpoint, device=args.device.lower())

    dataset = pose_model.cfg.data['test']['type']
    dataset_info = pose_model.cfg.data['test'].get('dataset_info', None)
    if dataset_info is None:
        warnings.warn(
            'Please set `dataset_info` in the config.'
            'Check https://github.com/open-mmlab/mmpose/pull/663 for details.',
            DeprecationWarning)
    else:
        dataset_info = DatasetInfo(dataset_info)

    img_keys = list(coco.imgs.keys())

    # optional
    return_heatmap = False

    # e.g. use ('backbone', ) to return backbone feature
    output_layer_names = None

    # 要輸出成csv的list
    pblh_list = []

    # process each image
    for i in mmcv.track_iter_progress(range(len(img_keys))):
        # get bounding box annotations
        image_id = img_keys[i]
        image = coco.loadImgs(image_id)[0]
        image_name = os.path.join(img_path, image['file_name'])
        ann_ids = coco.getAnnIds(image_id)

        # make person bounding boxes
        person_results = []
        for ann_id in ann_ids:
            person = {}
            ann = coco.anns[ann_id]
            # bbox format is 'xywh'
            person['bbox'] = ann['bbox']
            person_results.append(person)

        # test a single image, with a list of bboxes
        pose_results, returned_outputs = inference_top_down_pose_model(
            pose_model,
            image_name,
            person_results,
            bbox_thr=None,
            format='xywh',
            dataset=dataset,
            dataset_info=dataset_info,
            return_heatmap=return_heatmap,
            outputs=output_layer_names)

        date = image_name[-16:-4]
        date = f'{date[:4]}/{date[4:6]}/{date[6:8]} {date[8:10]}:{date[10:12]}'

        # print(f"\n #id: {image_id}")
        # for i, result in enumerate(pose_results):
        #     print( f" # RESULT[{i}]")
        #     for key in result:
        #         print(f" #[{key}]: {result[key]}")
        # print(f" #date: {date}")

        pblh_list.append({'time':date, 'PBLH':pose_results[0]['keypoints'][0][1]})

        isInterpolated = (pose_results[0]['bbox'][3] != 128) # check if the data is interpolated

        if args.out_root == '':
            out_file = None
        else:
            os.makedirs(args.out_root, exist_ok=True)
            out_file = os.path.join(args.out_root, f'vis_{i}.jpg')

        out_file = None ##### no img output

        vis_pose_result(
            pose_model,
            image_name,
            pose_results,
            dataset=dataset,
            dataset_info=dataset_info,
            kpt_score_thr=args.kpt_thr,
            radius=args.radius,
            thickness=args.thickness,
            show=args.show,
            out_file=out_file)

    csv_path = os.path.join(args.out_root, 'predict_PBLH.csv')
    with open(csv_path, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['date', 'pblh', 'level'])

        for pblh in pblh_list:
            level = pblh['PBLH']
            height = (level - 1) * 26 + 51
            time = pblh['time']

            writer.writerow([time, height, level])


if __name__ == '__main__':
    main()
