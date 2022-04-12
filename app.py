from tool.utils import *
from tool.torch_utils import *
from tool.darknet2pytorch import Darknet
import torch
import argparse

import cv2
import glob

"""hyper parameters"""
use_cuda = True

def detect_cv2(cfgfile, weightfile, imgfile):
    m = Darknet(cfgfile)
    m.print_network()
    m.load_weights(weightfile)
    print('Loading weights from %s... Done!' % (weightfile))

    if use_cuda:
        m.cuda()

    namesfile = 'detection/coco.names'
    class_names = load_class_names(namesfile)


    name = imgfile + '*.jpg'
    name = imgfile
    inputfiles = glob.glob(name)
    print('input files length', len(inputfiles))
    for i in inputfiles:
        print(i)
        img = cv2.imread(i)
        sized = cv2.resize(img, (m.width, m.height))
        sized = cv2.cvtColor(sized, cv2.COLOR_BGR2RGB)

        for i in range(2):
            start = time.time()
            boxes = do_detect(m, sized, 0.2, 0.6, use_cuda)
            finish = time.time()
            if i == 1:
                print('%s: Predicted in %f seconds.' % (imgfile, (finish - start)))

        plot_boxes_cv2(img, boxes[0], savename='predictions.jpg', class_names=class_names)

        exit(0)

def get_args():
    parser = argparse.ArgumentParser('Test your image or video by trained model.')
    parser.add_argument('-cfgfile', type=str, default='yolov4.cfg',
                        help='path of cfg file', dest='cfgfile')
    parser.add_argument('-weightfile', type=str,
                        default='yolov4.weights',
                        help='path of trained model.', dest='weightfile')
    parser.add_argument('-imgfile', type=str, required=True,
                        help='path of your image file.', dest='imgfile')
    parser.add_argument('-torch', type=bool, default=False,
                        help='use torch weights')
    args = parser.parse_args()

    return args


if __name__ == '__main__':
    args = get_args()
    detect_cv2(args.cfgfile, args.weightfile, args.imgfile)
