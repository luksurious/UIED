import argparse
import glob
import os
import time
from os.path import join as pjoin, join

import numpy as np
import cv2
from tqdm import tqdm
import matplotlib.pyplot as plt

import detect_compo.ip_region_proposal as ip


def resize_height_by_longest_edge(img_path, resize_length=800):
    org = cv2.imread(img_path)
    height, width = org.shape[:2]
    if height > width:
        return resize_length
    else:
        return int(resize_length * (height / width))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('input', type=str, help='Path to file to run it on')
    parser.add_argument('--output', type=str, default='data/out2')
    parser.add_argument('--mobile', action='store_true')
    parser.add_argument('--web', action='store_true')
    parser.add_argument('--show', action='store_true')
    parser.add_argument('--classify', action='store_true')
    parser.add_argument('--skip_ip', action='store_true')
    parser.add_argument('--skip_ocr', action='store_true')
    parser.add_argument('--skip_merge', action='store_true')
    args = parser.parse_args()

    key_params = {'min-grad': 4, 'ffl-block': 5, 'min-ele-area': 25, 'merge-contained-ele': True,
                  'max-word-inline-gap': 4, 'max-line-gap': 4}
    if args.mobile:
        key_params = {**key_params, 'min-grad': 4, 'ffl-block': 5, 'min-ele-area': 25, 'max-word-inline-gap': 6,
                      'max-line-gap': 1}
    elif args.web:
        key_params = {**key_params, 'min-grad': 3, 'ffl-block': 5, 'min-ele-area': 25, 'max-word-inline-gap': 4,
                      'max-line-gap': 4}

    # initialization
    input_img_root = args.input
    output_root = args.output

    # data = json.load(open('E:/Mulong/Datasets/rico/instances_test.json', 'r'))

    input_imgs = []
    for ext in ('*.gif', '*.png', '*.jpg'):
        input_imgs.extend(glob.glob(join(input_img_root, ext)))

    # input_imgs = sorted(input_imgs, key=lambda x: int(x.split('/')[-1][:-4]))  # sorted by index

    is_clf = args.classify
    is_ip = not args.skip_ip
    is_ocr = not args.skip_ocr
    is_merge = not args.skip_merge

    # Load deep learning models in advance

    print('Load models...')
    # Setup
    # Load deep learning models in advance
    ocr_model = None
    if is_ocr:
        import detect_text_east.ocr_east as ocr
        import detect_text_east.lib_east.eval as ocr_eval
        os.makedirs(pjoin(output_root, 'ocr'), exist_ok=True)
        ocr_model = ocr_eval.load()

    compo_classifier = None
    if is_ip:
        import detect_compo.ip_region_proposal as ip
        os.makedirs(pjoin(output_root, 'ip'), exist_ok=True)
        if is_clf:
            from cnn.CNN import CNN
            compo_classifier = {'Elements': CNN('Elements')}
            # compo_classifier['Image'] = CNN('Image')
            # compo_classifier['Noise'] = CNN('Noise')
    if is_merge:
        import merge
    print('... Done\n')

    run_times, run_clocks = [], []

    # set the range of target inputs' indices
    for idx, input_img in enumerate(tqdm(input_imgs)):
        start_time, start_clock = time.time(), time.clock()
        resized_height = resize_height_by_longest_edge(input_img)
        if is_ocr:
            ocr.east(input_img, output_root, ocr_model, key_params['max-word-inline-gap'],
                     resize_by_height=resized_height, show=args.show, batch=True)

        if is_ip:
            ip.compo_detection(input_img, output_root, key_params, batch=True,
                               classifier=compo_classifier, resize_by_height=resized_height, show=args.show)

        if is_merge:
            name = input_img.split('/')[-1][:-4]
            compo_path = pjoin(output_root, 'ip', str(name) + '.json')
            ocr_path = pjoin(output_root, 'ocr', str(name) + '.json')
            merge.incorporate(input_img, compo_path, ocr_path, output_root, params=key_params,
                              resize_by_height=resized_height, show=args.show, batch=True, out_suffix=str(idx) + name)

        run_times.append(time.time() - start_time)
        run_clocks.append(time.clock() - start_clock)

    print('Avg. runtime {:.2f}, avg. clock time {:.2f}'.format(np.mean(run_times), np.mean(run_clocks)))
    print('Median runtime {:.2f}, median clock time {:.2f}'.format(np.median(run_times), np.median(run_clocks)))
    plt.boxplot(run_times)
    plt.show()
