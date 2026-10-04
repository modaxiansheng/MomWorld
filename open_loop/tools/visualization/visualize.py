import os
import glob
import argparse
from tqdm import tqdm

import cv2
import numpy as np
from PIL import Image

import mmcv
from mmcv import Config
from mmdet import __version__ as mmdet_version
from mmdet.datasets import build_dataset
from projects.mmdet3d_plugin.datasets.builder import build_dataloader
from tools.visualization.bev_render import BEVRender
from tools.visualization.cam_render import CamRender

plot_choices = dict(
    draw_pred = True, # True: draw gt and pred; False: only draw gt
    det = True,
    track = True, # True: draw history tracked boxes
    motion = True,
    map = True,
    planning = True,
)
START = 0
END = 81
INTERVAL = 1


class Visualizer:
    def __init__(
        self,
        args,
        plot_choices,
    ):
        self.out_dir = args.out_dir
        self.combine_dir = os.path.join(self.out_dir, 'combine')
        os.makedirs(self.combine_dir, exist_ok=True)
        self.gpus = args.gpus
        # gpu_ids = ",".join(str(x) for x in self.gpus)
        # print(gpu_ids)
        # os.environ["CUDA_VISIBLE_DEVICES"] = gpu_ids
        cfg = Config.fromfile(args.config)
        self.dataset = build_dataset(cfg.data.val)
        self.results = mmcv.load(args.result_path)
        self.bev_render = BEVRender(plot_choices, self.out_dir)
        self.cam_render = CamRender(plot_choices, self.out_dir)

    def add_vis(self, index):
        data = self.dataset.get_data_info(index)
        result = self.results[index]['img_bbox']

        bev_gt_path, bev_pred_path = self.bev_render.render(data, result, index)
        cam_pred_path = self.cam_render.render(data, result, index)
        self.combine(bev_gt_path, bev_pred_path, cam_pred_path, index)
    def add_vis_long(self, index, stride=1, visible_count=3):
        max_len = min(
            len(self.dataset),
            len(self.results),
        )

        index0 = index
        index1 = index + stride
        index2 = index + 2 * stride

        if index2 >= max_len:
            return False

        data0 = self.dataset.get_data_info(index0)
        data1 = self.dataset.get_data_info(index1)
        data2 = self.dataset.get_data_info(index2)

        result0 = self.results[index0]["img_bbox"]
        result1 = self.results[index1]["img_bbox"]
        result2 = self.results[index2]["img_bbox"]

        bev_paths = []

        for count in range(1, visible_count + 1):
            bev_path = self.bev_render.render_long_final(
                data0,
                data1,
                data2,
                result0,
                result1,
                result2,
                index,
                visible_count=count,
                suffix=f"long_{count}",
            )
            bev_paths.append(bev_path)

        images = []
        for path in bev_paths:
            image = cv2.imread(path)
            if image is not None:
                images.append(image)

        if len(images) == 0:
            return False

        min_h = min(image.shape[0] for image in images)
        resized_images = []

        for image in images:
            scale = min_h / image.shape[0]
            new_w = int(image.shape[1] * scale)
            resized = cv2.resize(image, (new_w, min_h))
            resized_images.append(resized)

        gap = np.ones((min_h, 40, 3), dtype=np.uint8) * 255

        merge_list = []
        for i, image in enumerate(resized_images):
            if i > 0:
                merge_list.append(gap)
            merge_list.append(image)

        merge_image = np.concatenate(merge_list, axis=1)

        save_path = os.path.join(
            self.combine_dir,
            f"{str(index).zfill(4)}_long_three_panel.jpg",
        )

        cv2.imwrite(save_path, merge_image)

        return True


    def combine(self, bev_gt_path, bev_pred_path, cam_pred_path, index):
        bev_gt = cv2.imread(bev_gt_path)
        bev_image = cv2.imread(bev_pred_path)
        cam_image = cv2.imread(cam_pred_path)
        merge_image = cv2.hconcat([cam_image, bev_image, bev_gt])
        save_path = os.path.join(self.combine_dir, str(index).zfill(4) + '.jpg')
        cv2.imwrite(save_path, merge_image)

    def image2video(self, fps=12, downsample=4):
        imgs_path = glob.glob(os.path.join(self.combine_dir, '*.jpg'))
        imgs_path = sorted(imgs_path)
        img_array = []
        for img_path in tqdm(imgs_path):
            img = cv2.imread(img_path)
            height, width, channel = img.shape
            img = cv2.resize(img, (width//downsample, height //
                             downsample), interpolation=cv2.INTER_AREA)
            height, width, channel = img.shape
            size = (width, height)
            img_array.append(img)
        out_path = os.path.join(self.out_dir, 'video.mp4')
        out = cv2.VideoWriter(
            out_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, size)
        for i in range(len(img_array)):
            out.write(img_array[i])
        out.release()


def parse_args():
    parser = argparse.ArgumentParser(
        description='Visualize groundtruth and results')
    parser.add_argument('config', help='config file path')
    parser.add_argument('--result-path', 
        default=None,
        help='prediction result to visualize'
        'If submission file is not provided, only gt will be visualized')
    parser.add_argument(
        '--out-dir', 
        default='vis',
        help='directory where visualize results will be saved')
    parser.add_argument('--gpus', type=int, nargs='+', default=0, help='gpu ids')

    parser.add_argument(
        "--long-bev",
        action="store_true",
        help="draw t, t+1, t+2 final planning trajectories in one BEV",
    )

    parser.add_argument(
        "--long-stride",
        type=int,
        default=1,
        help="frame stride for long BEV visualization",
    )

    parser.add_argument(
        "--visible-count",
        type=int,
        default=3,
        help="number of trajectories to draw, choose from 1, 2, 3",
    )
    args = parser.parse_args()

    return args

def main():
    args = parse_args()
    visualizer = Visualizer(args, plot_choices)

    for idx in tqdm(range(START, END, INTERVAL)):
        if idx >= len(visualizer.results):
            break

        if args.long_bev:
            ok = visualizer.add_vis_long(
                idx,
                stride=args.long_stride,
                visible_count=args.visible_count,
            )

            if not ok:
                break
        else:
            visualizer.add_vis(idx)
    
    visualizer.image2video()

if __name__ == '__main__':
    main()