"""
This code is generated by Ridvan Salih KUZU @DLR
LAST EDITED:  01.06.2021
ABOUT SCRIPT: Sometimes there may be a need for evaluating the output of single input image or outputs of images in a directory. This script provides this option

"""

import argparse
from general.post_processing import threshold_mask
import albumentations as A
import os
from general.custom_data_generator import InstanceGenerator
from PIL import Image
import numpy as np
from pathlib import Path
from tensorflow.keras.models import load_model

parser = argparse.ArgumentParser(description='Segmentation')

parser.add_argument('--cuda', default='0', type=str, help=' cuda devices (default: 0)')
parser.add_argument('-i','--input-dir', default='dataset/supervised/image', type=str, help='path to the input image folder or image file')
parser.add_argument('-l','--label-dir', default='None', type=str, help='path to the input labels folder or image file')
parser.add_argument('-o','--out-dir', default='dataset/supervised/predicted_supervised', type=str, help='path to the output image folder or image file')
parser.add_argument('-w','--weights', default='modeldir/model_best_lr_1e-05_wd_0.001_ba_10.tf', type=str, help='Reference Model Directory (default: model)')
parser.add_argument('-s', '--self-supervised', dest='self_supervised', action='store_true', help='self-supervised learning')

args = parser.parse_args()

def main():
    '''
       THIS IS THE MAIN FUNCTION FOR EVALUATION CALLS.
      '''

    arguments=args.__str__()[args.__str__().find("(")+1:args.__str__().find(")")].split(',')
    for argument in arguments: print("ARGUMENT PASSED: {}".format(argument.strip()))

    os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda

    INPUT_SHAPE = [128, 128, 3]
    OUTPUT_SHAPE = [128, 128, 1]

    TEST_TRANSFORMATION = A.Compose([
        A.Resize(INPUT_SHAPE[0], INPUT_SHAPE[1]),
        A.Normalize(mean=0, std=0.5),
    ])

    if args.label_dir =='None': args.label_dir = None

    generator=InstanceGenerator(TEST_TRANSFORMATION,args.input_dir,args.label_dir)
    model = load_model(args.weights)

    evaluate_model(model, generator)


def evaluate_model(model, generator):
    '''
    THIS FUNCTION RUNS MODEL EVALUATION.
    :param model: pretrained model
    :param generators: data loader generator
    :return: returns nothing
    '''

    print('EVALUATION SESSION STARTED!')
    model.summary()

    for X, image_name, label_name in generator:
        to_display = {}
        pred_mask = model.predict(X).squeeze()
        if not args.self_supervised:
            pred_mask=threshold_mask(pred_mask)
        to_display['original'] = Image.open(image_name).resize(pred_mask.shape)
        if label_name is not None:
            to_display['label'] = Image.open(label_name).resize(pred_mask.shape)
        to_display['predicted'] = Image.fromarray((pred_mask * 255).astype(np.uint8))

        #display_results(to_display)
        save_predicted(image_name,to_display['predicted'])


def save_predicted(input_image_name,predicted_image):
    '''
    THIS FUNCTION SAVES THE PREDICTED DATA.
    :param input_image_name: name of the input image to be evaluated
    :param predicted_image: predicted image as an output of evaluation
    :return: returns nothing
    '''
    if not os.path.exists(args.out_dir):
        os.makedirs(args.out_dir)

    if Path(args.out_dir).is_dir():
        out_file = args.out_dir + '/' + Path(input_image_name).stem + '.png'
    else:
        out_file = args.out_dir

    predicted_image.save(out_file)


if __name__ == '__main__':
    main()