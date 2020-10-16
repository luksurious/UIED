import argparse
import time
from os.path import join as pjoin
import cv2
import os


def resize_height_by_longest_edge(img_path, resize_length=800):
    org = cv2.imread(img_path)
    height, width = org.shape[:2]
    if height > width:
        return resize_length
    else:
        return int(resize_length * (height / width))


if __name__ == '__main__':
    '''
        ele:min-grad: gradient threshold to produce binary map         
        ele:ffl-block: fill-flood threshold
        ele:min-ele-area: minimum area for selected elements 
        ele:merge-contained-ele: if True, merge elements contained in others
        text:max-word-inline-gap: words with smaller distance than the gap are counted as a line
        text:max-line-gap: lines with smaller distance than the gap are counted as a paragraph

        Tips:
        1. Larger *min-grad* produces fine-grained binary-map while prone to over-segment element to small pieces
        2. Smaller *min-ele-area* leaves tiny elements while prone to produce noises
        3. If not *merge-contained-ele*, the elements inside others will be recognized, while prone to produce noises
        4. The *max-word-inline-gap* and *max-line-gap* should be dependent on the input image size and resolution

        mobile: {'min-grad':4, 'ffl-block':5, 'min-ele-area':25, 'max-word-inline-gap':6, 'max-line-gap':1}
        web   : {'min-grad':3, 'ffl-block':5, 'min-ele-area':25, 'max-word-inline-gap':4, 'max-line-gap':4}
    '''

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

    # set input image path
    input_path_img = args.input
    output_root = args.output

    resized_height = resize_height_by_longest_edge(input_path_img)

    is_clf = args.classify
    is_ip = not args.skip_ip
    is_ocr = not args.skip_ocr
    is_merge = not args.skip_merge

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
        os.makedirs(pjoin(output_root, 'ip'), exist_ok=True)
        if is_clf:
            from cnn.CNN import CNN
            compo_classifier = {'Elements': CNN('Elements')}
            # compo_classifier['Image'] = CNN('Image')
            # compo_classifier['Noise'] = CNN('Noise')
    print('... Done\n')

    start_time, start_clock = time.time(), time.clock()

    if is_ocr:
        import detect_text_east.ocr_east as ocr
        print("Run OCR")
        ocr.east(input_path_img, output_root, ocr_model, key_params['max-word-inline-gap'],
                 resize_by_height=resized_height, show=args.show)

    if is_ip:
        import detect_compo.ip_region_proposal as ip
        print("Run IP")
        ip.compo_detection(input_path_img, output_root, key_params,
                           classifier=compo_classifier, resize_by_height=resized_height, show=args.show)

    if is_merge:
        import merge
        print("Run Merge")
        name = input_path_img.split('/')[-1][:-4]
        compo_path = pjoin(output_root, 'ip', str(name) + '.json')
        ocr_path = pjoin(output_root, 'ocr', str(name) + '.json')
        merge.incorporate(input_path_img, compo_path, ocr_path, output_root, params=key_params,
                          resize_by_height=resized_height, show=args.show)

    print('Total time elapsed {:.2f}s, clock time {:.2f}'.format(time.time() - start_time, time.clock() - start_clock))
