"""
This code is generated by Ridvan Salih KUZU @DLR
LAST EDITED:  01.06.2021
ABOUT SCRIPT:
It is a main script for training a segmentation system for a given database.
"""
from general.custom_data_generator import TrainValidTestGenerator
import argparse
from general.model_zoo import *
from tensorflow.keras.models import load_model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping
import albumentations as A
import csv
from math import ceil
from general.utils import print_history
from general.custom_losses import NTXent
from general.custom_metrics import mean_iou
import warnings
import logging
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # FATAL
logging.getLogger('tensorflow').setLevel(logging.FATAL)






parser = argparse.ArgumentParser(description='Segmentation')

parser.add_argument('--model-type', default=6, type=int, metavar='MT', help='0: MobileUNet, 1: UNet, 2: ResUNet, 3:AttentionUNet with ResNet, 4:AttentionUNet with EfficientNet-B0, 5:AttentionUNet with EfficientNet-B7, 6:Attention UNet without backbone')
parser.add_argument('--start-epoch', default=0, type=int, metavar='SE', help='start epoch (default: 0)')
parser.add_argument('--num-epochs', default=180, type=int, metavar='NE', help='number of epochs to train (default: 180)')
parser.add_argument('--num-workers', default=2, type=int, metavar='NW', help='number of workers in training (default: 2)')
parser.add_argument('--num-augment', default=1, type=int, metavar='NA', help='number of augment to train (default: 1)')
parser.add_argument('--num-classes', default=2, type=int, metavar='NC', help='number of classes (default: 2)')
parser.add_argument('--batch-size', default=32, type=int, metavar='BS', help='number of batch size (default: 32)')
parser.add_argument('--learning-rate', default=0.001, type=float, metavar='LR', help='learning rate (default: 0.001)')
parser.add_argument('--weight-file', default='None', type=str, help='weight Directory (default: modeldir)')
parser.add_argument('-s', '--self-supervised', dest='self_supervised', action='store_true', help='self supervised learning with SimCLR')
parser.add_argument('-r', '--self-reconstruct', dest='self_reconstruct', action='store_true', help='self supervised learning with constructing original images')
parser.add_argument('--temperature', default=0.1, type=float, metavar='TE', help='learning temperature (default: 1)')
parser.add_argument('--loss1', default=1, type=float, metavar='L1', help='first loss weight in self supervised learning (default: 1)')
parser.add_argument('--loss2', default=1, type=float, metavar='L2', help='second loss weight in self supervised learning (default: 1)')

parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true', help='evaluate the model (it requires the wights path to be given')
parser.add_argument('--cuda', default='0', type=str, help=' cuda devices (default: 0)')

parser.add_argument('--data-dir', default='../dataset/supervised/image/', type=str, help='path to the image directory')
parser.add_argument('--label-dir', default='../dataset/supervised/mask/', type=str, help='path to the mask directory')
parser.add_argument('--pair-csv', default='None', type=str, help='path to the pairs CSV')
parser.add_argument('--out-dir', default='../modeldir/', type=str, help='Out Directory (default: modeldir)')
parser.add_argument('--log-file', default='performance-logs.csv', type=str, help='path to log dir (default: logdir)')

args = parser.parse_args()

def main():
    '''
     THIS IS THE MAIN FUNCTION FOR TRAINING AND EVALUATION CALLS.
    '''

    # PRINT ALL ARGUMENTS AND VALUES AT THE BEGINNING
    arguments=args.__str__()[args.__str__().find("(")+1:args.__str__().find(")")].split(',')
    for argument in arguments: print("ARGUMENT PASSED: {}".format(argument.strip()))

    # SET NUMBER OF GPU
    os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda
    if len(args.cuda)>1:
        print("MULTI-GPU TRAINING")
        multi_gpu=True
    else:
        print("SINGLE-GPU TRAINING")
        multi_gpu=False

    if not os.path.exists(args.out_dir):
        os.makedirs(args.out_dir)

    # SET INPUT and OUTPUT SHAPES
    # BE SURE THAT INPUT SIZE is 512x512 in the final configuration if the trained model will be used in annotation_window.py
    INPUT_SHAPE = [512, 512, 3]
    if args.self_supervised and not args.self_reconstruct:
        OUTPUT_SHAPE = [512, 512, 1]
    else:
        OUTPUT_SHAPE = [512, 512, 1]

    # SET AUGMENTATION FOR TRAINING DATA
    TRAIN_TRANSFORMATION = A.Compose([
        A.Resize(INPUT_SHAPE[0], INPUT_SHAPE[1]),
        A.Normalize(mean=0, std=0.5),
        A.RandomRotate90(),
        A.Rotate(),
        A.RandomResizedCrop(INPUT_SHAPE[0], INPUT_SHAPE[1], ratio=(0.98, 1.02)),
        A.Flip(),
        A.ShiftScaleRotate(rotate_limit=0, shift_limit_x=0.02, shift_limit_y=0.02),
        A.RandomBrightnessContrast(0.05,0.05),
    ])

    # SET AUGMENTATION FOR VALIDATION DATA
    VALID_TRANSFORMATION = A.Compose([
        A.Resize(INPUT_SHAPE[0], INPUT_SHAPE[1]),
        A.Normalize(mean=0, std=0.5),
        A.RandomRotate90(),
        A.Flip(),
    ])

    # SET AUGMENTATION FOR TEST DATA
    TEST_TRANSFORMATION = A.Compose([
        A.Resize(INPUT_SHAPE[0], INPUT_SHAPE[1]),
        A.Normalize(mean=0, std=0.5),
    ])

    # SET TRAINING, VALIDATION, TEST SPLIT
    if args.self_supervised:
        print('The model will be built for self-supervised learning')
        VALID_TEST_SPLIT = (0.1, 0.1) # 80-10-10 PERCENT SPLIT FOR TRAINING, VALIDATION AND TEST
    else:
        print('The model will be built to make segmentation on the original inputs')
        VALID_TEST_SPLIT = (32, 32) # 32 SAMPLES FOR VALIDATION AND TEST

    if args.pair_csv == 'None':
        args.pair_csv=None

    # PREPARING THE DATA GENERATORS FOR TRAINING, VALIDATION AND TEST
    generators=TrainValidTestGenerator(args.data_dir, args.label_dir, args.pair_csv,
                                       workers=args.num_workers,
                                       multi_gpu=multi_gpu,
                                       batch_size=args.batch_size,
                                       augment_time=args.num_augment,
                                       image_shape=INPUT_SHAPE,
                                       mask_shape=OUTPUT_SHAPE,
                                       is_shuffle=True,
                                       self_supervised=args.self_supervised,
                                       self_reconstruct=args.self_reconstruct,
                                       valid_test_split=VALID_TEST_SPLIT,
                                       transformation=(TRAIN_TRANSFORMATION, VALID_TRANSFORMATION, TEST_TRANSFORMATION))

    # DETERMINE GPU DISTRIBUTION STRATEGY
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        if multi_gpu:
            strategy = tf.distribute.MirroredStrategy()
            with strategy.scope():
                train_and_eval(INPUT_SHAPE,generators)
        else: train_and_eval(INPUT_SHAPE,generators)


def train_and_eval(input_shape,generators):
    '''
     THIS FUNCTION PREPARES TRAINING AND EVALUATION CALLS.
     :param input_shape: (width,height,channel) shape of the input images
     :param generators: data loader generator
     :return: returns nothing
    '''

    if args.evaluate: # IF MODEL EVALUATION IS CALLED, ONLY EVALUATE THE MODEL PERFORMANCE
        evaluate_model(load_model(args.weight_file, compile=True), generators, logging=False)

    else: # IF MODEL TRAINING IS CALLED, BOTH TRAIN AND EVALUATE THE MODEL PERFORMANCE
        if args.self_supervised: # IF ANY OF SELF-SUPERVISED LEARNING APPROACH IS CALLED
            model = get_model(args.model_type, input_shape, is_freeze=True)
            if not args.self_reconstruct: # IF SIMCLR BASED SELF-SUPERVISED LEARNING IS CALLED, PREPARE CNN-MLP HEADER FOR TRAINING
                model = get_sim_clr_model(model, input_shape)
            if args.weight_file.endswith('.tf'): # IF ANY OF SELF-SUPERVISED LEARNING IS CALLED WITH PRETRAINED WEIGHTS, LOAD THE WEIGHTS
                temp = load_model(args.weight_file, compile=False)
                model.set_weights(temp.get_weights())
        else: # IF STANDARD SUPERVISED LEARNING IS CALLED
            model = get_model(args.model_type, input_shape, is_freeze=False)
            if args.weight_file.endswith('.tf'): # IF SUPERVISED LEARNING IS CALLED WITH PRETRAINED WEIGHTS, LOAD THE WEIGHTS
                temp = load_model(args.weight_file, compile=False)
                if not args.self_reconstruct:   # IF SUPERVISED LEARNING IS CALLED WITH PRETRAINED WEIGHTS OF SELF-SUPERVISED LEARNING BASED SIMCLR APPROACH
                    model.set_weights(temp.base_model.get_weights())
                else: # IF SUPERVISED LEARNING IS CALLED WITH PRETRAINED WEIGHTS OF SELF-SUPERVISED LEARNING BASED ON RECONSTRUCTIVE APPROACH
                    model.set_weights(temp.get_weights())

        train_model(model, generators, warmup=True)
        train_model(model, generators, warmup=False)

        model.load_weights(args.out_dir + '/model_best_lr_{}_te_{}_l1_{}_l2_{}_ty_{}_ba_{}.h5'.format(args.learning_rate,args.temperature,args.loss1,args.loss2, args.model_type,args.batch_size))
        evaluate_model(model, generators)

        # THE FINAL MODEL AT THE END OF TRAINING AND EVALUATION WILL BE SAVED AS TF FORMATTED FILE
        model.save(args.out_dir + '/model_best_lr_{}_te_{}_l1_{}_l2_{}_ty_{}_ba_{}.tf'.format(args.learning_rate,args.temperature,args.loss1,args.loss2, args.model_type,args.batch_size))
        os.remove(args.out_dir +  '/model_best_lr_{}_te_{}_l1_{}_l2_{}_ty_{}_ba_{}.h5'.format(args.learning_rate,args.temperature,args.loss1,args.loss2, args.model_type,args.batch_size))


def compile_model(model,warmup=True):
    '''
         THIS FUNCTION COMPILE THE MODEL.
         :param model: selected model for training and validation
         :param warmup: define if it is a warmup call or not
         :return: returns model and num_epochs
        '''
    if warmup:
        # IF WARMUP IS CALLED, MAKE A SHORT TRAINING WITH VERY SMALL STEPS
        print('WARM-UP SESSION STARTED!')
        learning_rate = args.learning_rate / 100
        num_epochs = ceil(args.num_epochs / 6)
        if not args.self_supervised:
            # TODO: COMPARE THE PERFORMANCE FOR TRAINABLE AND NON-TRAINABLE LAYERS IN STANDARD SUPERVISED LEARNING
            for idx in range(len(model.layers) // 2): model.layers[idx].trainable = True
    else:
        print('TRAINING SESSION STARTED!')
        learning_rate = args.learning_rate
        num_epochs = args.num_epochs
        if not args.self_supervised:
            # IN STANDARD SUPERVISED LEARNING, SET PREVOUSLY MANIPULATED LAYERS AS TRAINABLE AGAIN
            for idx in range(len(model.layers) // 2): model.layers[idx].trainable = True

    model.summary()

    # DETERMINE THE LOSS FUNCTION, OPTIMIZER AND METRICS FOR ALL KIND OF TRAINING TYPES
    if args.self_supervised:
        if args.self_reconstruct:
            loss_function = tf.keras.losses.MeanAbsoluteError()
            optimizer = Adam(learning_rate=learning_rate)
            loss_weights = 1
            metrics = [tf.keras.metrics.MeanAbsoluteError()]
        else:
            loss_function = {'contrastive': NTXent(args.temperature), 'binary': tf.keras.losses.BinaryCrossentropy()}
            loss_weights = {'contrastive': args.loss1, 'binary': args.loss2}
            # optimizer = LARSOptimizer(
            #    learning_rate,
            #    momentum=0.9,
            #    weight_decay=1e-4,
            #    exclude_from_weight_decay=['batch_normalization', 'bias'])
            optimizer = Adam(learning_rate=learning_rate)
            metrics = ["accuracy"]
    else:
        loss_function = tf.keras.losses.MeanAbsoluteError()
        optimizer = Adam(learning_rate=learning_rate)
        loss_weights=1
        metrics = [mean_iou(),"accuracy" ]

    # COMPILE THE MODEL WITH THE CALLBACKS
    model.compile(optimizer=optimizer, loss=loss_function, loss_weights=loss_weights, metrics=metrics)
    return model,num_epochs


def train_model(model, generators, warmup=True, logging=True):
    '''
     THIS FUNCTION COMPILE AND RUN TRAINING.
     :param model: selected model for training and validation
     :param generators: data loader generator
     :param warmup: define if it is a warmup call or not
     :return: returns nothing
    '''
    model,num_epochs=compile_model(model,warmup)
    callbacks = [
        ReduceLROnPlateau(verbose=1),
        EarlyStopping(patience=40),
        ModelCheckpoint(args.out_dir + '/model_best_lr_{}_te_{}_l1_{}_l2_{}_ty_{}_ba_{}.h5'.format(args.learning_rate,args.temperature,args.loss1,args.loss2, args.model_type, args.batch_size),monitor='val_loss', verbose=1, save_best_only=True, save_weights_only=False),
        #TensorBoard(log_dir="logs/lr_{}_wd_{}_ba_{}".format(args.learning_rate, args.weight_decay, args.batch_size), histogram_freq=0, write_graph=True),
    ]

    # RUN THE MODEL
    history = model.fit(
                        generators.get_generator('train'),
                        steps_per_epoch=generators.get_steps('train'),
                        epochs=num_epochs,
                        workers=args.num_workers,
                        callbacks=callbacks,
                        use_multiprocessing=generators.multi_gpu,
                        shuffle=False,
                        validation_data=generators.get_generator('valid'),
                        validation_steps=generators.get_steps('valid')
                        )

    # PERFORMANCE PLOT FILE NAMES
    if logging:
        loss_log = args.out_dir + '/loss_lr_{}_te_{}_l1_{}_l2_{}_ty_{}_ba_{}.png'.format(args.learning_rate, args.temperature,args.loss1,args.loss2, args.model_type,args.batch_size)
        accuracy_log = args.out_dir + '/accuracy_lr_{}_te_{}_l1_{}_l2_{}_ty_{}_ba_{}.png'.format(args.learning_rate, args.temperature,args.loss1,args.loss2, args.model_type, args.batch_size)
        iou_log = args.out_dir + '/iou_lr_{}_te_{}_l1_{}_l2_{}_ty_{}_ba_{}.png'.format(args.learning_rate, args.temperature,args.loss1,args.loss2, args.model_type,args.batch_size)

        # PLOT PERFORMANCE GRAPHS FOR EACH KIND OF SUPERVISED LEARNING
        if args.self_supervised:
            if args.self_reconstruct:
                print_history(history, 'loss', loss_log)
            else:
                print_history(history, 'contrastive_loss', loss_log)
                print_history(history, 'binary_accuracy', accuracy_log)
        else:
            print_history(history, 'loss', loss_log)
            print_history(history, 'accuracy', accuracy_log)
            print_history(history, 'iou', iou_log)


def evaluate_model(model, generators, logging=True):
    '''
     THIS FUNCTION COMPILE AND RUN TRAINING.
     :param model: selected model for evaluation
     :param generators: data loader generator
     :param logging: define if the results will be logged or not
     :return: returns nothing
    '''

    print('EVALUATION SESSION STARTED!')
    model, _ = compile_model(model, False)

    tr=model.evaluate(generators.get_generator('train'), steps=generators.get_steps('train'))
    tr_loss, tr_acc = tr[0] , tr[-1]

    val=model.evaluate(generators.get_generator('valid'), steps=generators.get_steps('valid'))
    val_loss, val_acc = val[0] , val[-1]

    te = model.evaluate(generators.get_generator('test'),steps=generators.get_steps('test'))
    test_loss, test_acc = te[0] , te[-1]

    if not args.self_supervised:
        tr_iou = tr[1]
        val_iou = val[1]
        test_iou = te[1]

    if logging:
        if not args.self_supervised:
            header = ['model_type', 'learning_rate', 'batch_size', 'train_loss', 'valid_loss', 'test_loss', 'train_acc','valid_acc', 'test_acc','train_iou','valid_iou', 'test_iou']
            info = [args.model_type, args.learning_rate, args.batch_size, tr_loss, val_loss, test_loss, tr_acc, val_acc, test_acc, tr_iou,val_iou,test_iou]
        else:
            if args.self_reconstruct:
                header = ['model_type', 'temperature', 'learning_rate', 'batch_size', 'train_loss', 'valid_loss','test_loss']
                info = [args.model_type, args.temperature, args.learning_rate, args.batch_size, tr_loss, val_loss,test_loss]

            else:
                header = ['model_type', 'temperature', 'loss1', 'loss2', 'learning_rate', 'batch_size', 'train_loss', 'valid_loss', 'test_loss', 'train_acc','valid_acc', 'test_acc']
                info = [args.model_type, args.temperature, args.loss1, args.loss2, args.learning_rate, args.batch_size, tr_loss, val_loss, test_loss, tr_acc, val_acc, test_acc]

        if not os.path.exists(args.out_dir+'/'+args.log_file):
            with open(args.out_dir+'/'+args.log_file, 'w') as file:
                logger = csv.writer(file)
                logger.writerow(header)
                logger.writerow(info)
        else:
            with open(args.out_dir+'/'+args.log_file, 'a') as file:
                logger = csv.writer(file)
                logger.writerow(info)


if __name__ == '__main__':
    main()