from tool.utils import *
from tool.torch_utils import *
from tool.darknet2pytorch import Darknet
import torch
import argparse

import cv2
import glob
import time

import logging
from waggle.plugin import Plugin
from waggle.data.vision import Camera

TOPIC_TEMPLATE = "env.count.yolo"

def detect_cv2(args):
    logging.basicConfig(
        level=logging.DEBUG if args.debug else logging.INFO,
        format='%(asctime)s %(message)s',
        datefmt='%Y/%m/%d %H:%M:%S')


    m = Darknet(args.cfgfile)
    m.load_weights(args.weightfile)
    print('Loading weights Done!')

    use_cuda = torch.cuda.is_available()
    if use_cuda:
        m.cuda()

    namesfile = 'detection/coco.names'
    class_names = load_class_names(namesfile)

    sampling_countdown = -1
    if args.sampling_interval >= 0:
        logging.info("sampling enabled -- occurs every %sth inferencing", args.sampling_interval)
        sampling_countdown = args.sampling_interval

    #files = [args.imgfile]
    camera = Camera(args.stream)
    pre = {}
    with Plugin() as plugin:
        while True:
            for sample in camera.stream():
            #for i in files:

                do_sampling = False
                if sampling_countdown > 0:
                    sampling_countdown -= 1
                elif sampling_countdown == 0:
                    do_sampling = True
                    sampling_countdown = args.sampling_interval


                image = sample.data
                height = image.shape[0]
                width = image.shape[1]
                timestamp = sample.timestamp

                #image = cv2.imread(i)
                #timestamp = time.time()

                sized = cv2.resize(image, (m.width, m.height))
                sized = cv2.cvtColor(sized, cv2.COLOR_BGR2RGB)

                start = time.time()
                boxes = do_detect(m, sized, args.confidence_level, 0.6, use_cuda)
                finish = time.time()

                image, found = plot_boxes_cv2(image, boxes[0], class_names=class_names)


                detection_stats = 'found objects: '
                for object_found, count in found.items():
                    detection_stats += f'{object_found} [{count}] '
                    plugin.publish(f'{TOPIC_TEMPLATE}.{object_found}', count, timestamp=timestamp)
                    print(f'{TOPIC_TEMPLATE}.{object_found}', count)
                logging.info(detection_stats)





                if do_sampling and found != {} and found != pre:
                    sample.data = image
                    sample.save(f'sample_{timestamp}.jpg')
                    plugin.upload_file(f'sample_{timestamp}.jpg')
                    logging.info("uploaded sample")
                pre = found

                if args.interval > 0:
                    time.sleep(args.interval)



                if args.continuous == False:
                    exit(0)
                if args.interval > 0:
                    time.sleep(args.interval)


def get_args():
    parser = argparse.ArgumentParser('Test your image or video by trained model.')
    parser.add_argument('-cfgfile', type=str, default='yolov4.cfg',
                        help='path of cfg file', dest='cfgfile')
    parser.add_argument('-weightfile', type=str,
                        default='yolov4.weights',
                        help='path of trained model.', dest='weightfile')
    parser.add_argument('-imgfile', type=str,
                        help='path of your image file.', dest='imgfile')


    parser.add_argument(
        '-stream', dest='stream',
        action='store', default="camera",
        help='ID or name of a stream, e.g. sample')
    parser.add_argument(
        '-confidence-level', dest='confidence_level',
        action='store', default=0.4,
        help='Confidence level [0. - 1.] to filter out result')
    parser.add_argument(
        '-interval', dest='interval',
        action='store', default=0, type=int,
        help='Inference interval in seconds')
    parser.add_argument(
        '-sampling-interval', dest='sampling_interval',
        action='store', default=-1, type=int,
        help='Sampling interval between inferencing')
    parser.add_argument(
        '-debug', dest='debug',
        action='store_true', default=False,
        help='Debug flag')
    parser.add_argument(
        '-continuous', dest='continuous',
        action='store_true', default=False,
        help='Flag to run this plugin forever')

    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = get_args()
    detect_cv2(args)
