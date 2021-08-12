#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 10 13:28:00 2019
"""

import argparse
import os
import sys
import keras
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.metrics import roc_curve
import json
import pickle
import pandas as pd
import tensorflow as tf
import numpy as np
from transforms import *
from DenseNet.DenseNet3D_wBN import *
import keras.optimizers as OPTS
from DenseNet.losses import categorical_focal_loss, binary_focal_loss
import time
from keras.utils.generic_utils import get_custom_objects

def get_readable_ctime():
    return time.strftime("%Y-%m-%d_%H:%M:%S")

def auc_roc(y_true, y_pred):
    # any tensorflow metric
    value, update_op = tf.contrib.metrics.streaming_auc(y_pred, y_true)

    # find all variables created for this matric
    metric_vars = [i for i in tf.local_variables() if 'auc_roc' in i.name.split('/')[1]]
    # Add metric variables to GLOBAL_VARIABLES collecion.
    # They will be initialized for new session.
    for v in metric_vars:
        tf.add_to_collection(tf.GraphKeys.GLOBAL_VARIABLES, v)
    # force to update metric values
    with tf.control_dependencies([update_op]):
        value = tf.identity(value)
        return value

def model_with_weights(model, weights, skip_mismatch):
    """ Load weights for model.

    Args
        model         : The model to load weights for.
        weights       : The weights to load.
        skip_mismatch : If True, skips layers whose shape of weights doesn't match with the model.
    """
    if weights is not None:
        # loads the weights of the model from a hdf5 file (created by save_weights)
        model.load_weights(weights, by_name=True, skip_mismatch=skip_mismatch)
    return model


def create_models(args, p, freeze_backbone=False):  # >>> can use variadic auguments to have more model config options
    """ 
    *Args
        backbone : A function to call to create a densenet model with a given backbone.
        classes        : The number of classes to train.
        weights            : The weights to load into the model.
        multi_gpu          : The number of GPUs to use for training.
        freeze_backbone    : If True, disables learning for the backbone.

    Returns
        training_model   : The training model. If multi_gpu=0, this is identical to model.
    """

    modifier = freeze_model if freeze_backbone else None

    # Keras recommends initialising a multi-gpu model on the CPU to ease weight sharing, and to prevent OOM errors.
    # optionally wrap in a parallel model
    in_shape = tuple([int(x) for x in args.input_shape.lstrip('(').rstrip(')').split(',')])
    if args.generate_model:
        ## generate a densenet model from DenseNet3D
        print('will generate a model from DenseNet3D')
        model = DenseNet3D(in_shape, depth=args.model_depth, nb_dense_block=args.nb_block, growth_rate=args.growth_rate,
                      nb_filter=-1, nb_layers_per_block=-1,
                      bottleneck=True, reduction=args.reduction,
                      dropout_rate=0, weight_decay=1e-5,
                      subsample_initial_block=False, #True,
                      include_top=True,
                      input_tensor=None,
                      pooling='max', classes=args.classes, activation='sigmoid', pi=p)

    else:
        if args.backbone == 'densenet121':
            model = DenseNet3DImageNet121(in_shape, classes = args.classes, pooling = 'avg')

        if args.backbone == 'densenet169':
            model = DenseNet3DImageNet169(in_shape, classes = args.classes, pooling = 'avg')

    if args.multi_gpu_force:
        from keras.utils import multi_gpu_model
        with tf.device('/cpu:0'):
            if args.weights is not None:
                model = model_with_weights(model, weights=args.weights, skip_mismatch=True)
        training_model = multi_gpu_model(model, gpus=args.multi_gpu)

    else:
        if args.weights is not None:
            model = model_with_weights(model, weights=args.weights, skip_mismatch=True)
        training_model = model

    # learning rate (lr)?
    #optm = OPTS.Adam(lr=1e-6)
    #optm = OPTS.RMSprop(lr=0.001, rho=0.9, epsilon=1e-6)

    optm = OPTS.sgd(lr=2e-6, momentum = 0.9, decay=1e-5, clipvalue = 0.5, nesterov=True)  # adam or modified adam??
    # compile model

    training_model.compile(loss=[binary_focal_loss(alpha= args.alpha, gamma=args.gamma)], optimizer = optm, metrics = ['accuracy', auc_roc])
    #training_model.compile(loss= 'binary_crossentropy', optimizer=optm, metrics = ['accuracy'])

    return training_model


class Generator(keras.utils.Sequence):
    def __init__(
        self,
		path,
        list_IDs, #entire train/ validation dataset IDs
        df,  # patch info DataFrame
        dim,      #image patch dimensions (H,W,D)
        n_channels,  # number of channels (image modalities)
        com_args,  # other common arguments
        transform_generator = False, #True for image augmentatio
        batch_size = 16,
        shuffle = True,
        transform_parameters = None,  #if transform_generator is True, transform_parameters required
    ):
        """ Initialize Generator object.
        Args
            transform_generator    : A boolin factor determining to activate randomly transformation of images.
            batch_size             : The size of the batches to generate.
            group_method           : Determines how images are grouped together (defaults to 'none', one of ('none', 'entire').
            shuffle_groups         : If True, shuffles the groups each epoch.
            transform_parameters   : The transform parameters used for data augmentation.
        """

        self.list_IDs = list_IDs
        self.df = df
        self.dim = dim
        self.channels = n_channels
        self.transform_generator = transform_generator
        self.batch_size = int(batch_size)
        self.shuffle = shuffle
        self.transform_parameters = transform_parameters
        self.folder_path = path
        self.data_folder = '_'.join([str(x) for x in self.dim])
        self.on_epoch_end()
        self.com_args = com_args
        super(Generator, self).__init__()


    def on_epoch_end(self):
        #updates group indexes after each epoch
        self.indexes = np.arange(len(self.list_IDs))
        if self.shuffle:
            np.random.shuffle(self.indexes)

    def __len__(self):
        return int(np.ceil(len(self.list_IDs)/float(self.batch_size)))


    def __getitem__(self, index):
        #Genereate indexes of the batch
        batch_indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]
        #Find list of IDs in the batch
        IDs_in_batch = [self.list_IDs[idx] for idx in batch_indexes]
        # Generate data
        return self.__data_generation(IDs_in_batch)


    def __data_generation(self, ID_list):
        # Generates data containing batch_size samples --> X: (n_samples, *dim, n_channels)
        X = np.empty((len(ID_list), *self.dim, self.channels))
        y = np.empty(len(ID_list), dtype = int)

        if self.transform_generator:
            for idx, ID in enumerate(ID_list):
                # random_transform_generator:  channel first
                X_load = []
                for f_path in self.folder_path:
                    X_tmp = np.load(os.path.join(f_path, self.data_folder, ID))
                    if X_load == []:
                        X_load = X_tmp
                    else:
                        if len(X_tmp.shape) < len(X_load.shape):
                            X_load = np.append(X_load, X_tmp.reshape(*X_tmp.shape, -1), axis = -1)
                        elif len(X_tmp.shape) > len(X_load.shape):
                            assert len(X_load.shape) >= len(X_tmp.shape), print("Alarm! DICOM file is after Ktrans file")
                            X_load = np.append(X_load.reshape(*X_load.shape, -1), X_tmp, axis = -1)
                        else:
                            X_load = np.append(X_load, X_tmp, axis = -1)
                load_x_patch = np.moveaxis(X_load, -1, 0) # move the ch_axis from the last to the first
                X_channels = np.moveaxis(random_transform_generator(load_x_patch,  r_axis=1, c_axis=2, s_axis=3, ch_axis=0, trans_pars = self.transform_parameters,), 0, -1)
                X[idx,] = X_channels#[:,:,:,:2] #X_channels
                y[idx] = int(self.df[self.df['File name'] == ID]['label']) ## from labels extracted from pd.DataFrame


        else:
            for idx, ID in enumerate(ID_list):
                X_load = []
                for f_path in self.folder_path:
                    X_tmp = np.load(os.path.join(f_path, self.data_folder, ID))
                    if X_load == []:
                        X_load = X_tmp
                    else:
                        if len(X_tmp.shape) < len(X_load.shape):
                            X_load = np.append(X_load, X_tmp.reshape(*X_tmp.shape, -1), axis = -1)
                        elif len(X_tmp.shape) > len(X_load.shape):
                            X_load = np.append(X_load.reshape(*X_load.shape, -1), X_tmp, axis = -1)
                        else:
                            X_load = np.append(X_load, X_tmp, axis = -1)

                X[idx,] = X_load#[:,:,:,:2] #X_load
                y[idx] = int(self.df[self.df['File name'] == ID]['label']) ## from labels extracted from pd.DataFrame

        assert 'n_classes' in self.com_args.keys(), 'common_args does not include "n_classes"'
        #categorical = np.zeros((n, self.com_args['n_classes']), dtype=dtype)
        if len(ID_list) < self.batch_size:
            print('batch size = {}'.format(len(ID_list)))

        ## 09/16/19 adding'z' ##############

        return X, y
       

### create generators
def create_generators(args, preprocess_image_IDs_path, preprocess_ktrans_IDs_path, n_folds = 5):
    """ Create generators for training and validation.
        Args:   parseargs object containing configuration for generators.
             including -- batch_size, n_classess, input_shape, random_transform, config
    """

    common_args = {
        'batch_size'       : args.batch_size,
        'n_classes'   : args.classes,
        'input_shape'   : [int(x) for x in args.input_shape.split(',')],
        'config' : args.config,
    }

    # ensure preprocess_image_IDs_path is a .pkl file
    assert preprocess_image_IDs_path.endswith('.pkl'), 'dicom patch file type is incorrect.'
    dicom_folder_path, __ = os.path.split(preprocess_image_IDs_path)

    if preprocess_ktrans_IDs_path:
        assert preprocess_ktrans_IDs_path.endswith('.pkl'), 'ktran patch file type is incorrect.'
        ktrans_folder_path, __ = os.path.split(preprocess_ktrans_IDs_path)
        folder_path = [dicom_folder_path, ktrans_folder_path]
    else:
        folder_path = [dicom_folder_path]
    # create random transform parameter setting for augmenting training data
    # e.g. trans_pars ={'rot_rg':15, 'shift_rgs':(4, 4), 'sheer':10, 'zoom': (0.8, 1.1), 'brightness': (0.8, 1.2), 'flip': (0.02, 0.3)}
    if args.random_transform:
        trans_gen = True
        with open(common_args['config'], 'r') as f:
            config = json.load(f)
        trans_pars = {
            'rot_rg': config['rot_range'],
            'shift_rgs': config['shift_range'],
            'sheer': config['sheer_degree'],
            'zoom': config['zoom_factor'],
            #'brightness': config['bright_range'],
            'flip': config['flip_prob'],
			}
    #data_folder = '_'.join([str(x) for x in common_args['input_shape']])
    id_file_list = pickle.load(open(preprocess_image_IDs_path, 'rb'))
    if preprocess_ktrans_IDs_path:
        ktrans_id_file_list = pickle.load(open(preprocess_ktrans_IDs_path, 'rb'))
        ID_file_list = pd.merge(id_file_list, ktrans_id_file_list, on=['File name', 'Patch_name', 'label'], how='inner')
    else:
        ID_file_list = id_file_list
    ##### due to imbalance data, the following lines aim to randomly select IDs from the major class
    red_factor =  sum(ID_file_list.label)/len(ID_file_list)
    na_ratio = len(ID_file_list[ID_file_list.Zone == 'NA'])/len(ID_file_list)
    print('positive label ratio =', red_factor)
    print('total number of ID files =', len(ID_file_list))
    print('NA ratio =', na_ratio)
    ID_files = []
    for idx, patch_row in ID_file_list.iterrows():
        if 'f' in patch_row.Patch_name:   #################### <-- need adjust to change the total number of image patches
            ID_files.append(patch_row)

        elif np.random.rand(1) >= (1. - red_factor):
            ID_files.append(patch_row)

    #    else:
    #        ID_files.append(patch_row)
    ID_files = pd.DataFrame(ID_files).reset_index(drop=True)
    used_red_factor = sum(ID_files['label'])/len(ID_files['label'])
    print('#### positive label ratio in training =', used_red_factor, 'labels =', sum(ID_files['label']), 'total images = ', len(ID_files['label']))


    #########################################################
    #train_list_IDs, val_list_IDs = train_test_split(list(ID_files['File name']), train_size=0.8, random_state=args.seed)   # <- way to split train and validation
    kfold = KFold(n_folds, shuffle=True, random_state=args.seed)
    Train_generator, Validation_generator, Predict_generator = list(), list(), list()
    for train_list_IDXs, val_list_IDXs in kfold.split(list(ID_files['File name'])):
        # for only fid images are in validation set
        train_list_IDs = list(ID_files['File name'][train_list_IDXs])
        val_list_IDs =list(ID_files['File name'][val_list_IDXs])
        val_pred_ids = val_list_IDs #[]
        #### below 3 lines is the version of finding fids in validation set ####
        #for ID_name in val_list_IDs:
        #    if 'fid' in ID_name and 'shift' not in ID_name:
        #        val_pred_ids.append(ID_name)
        ########################################################################
        #### below 3 lines is the version of finding fids in WHOLE dataset  ####
        #for ID_name in list(ID_file_list['File name']):
        #    if 'fid' in ID_name:
        #        val_pred_ids.append(ID_name)
        ########################################################################
        print('total number of train image sets" ', len(train_list_IDs))
        print('total number of validation image sets" ', len(val_list_IDs))
        print('total number of predicted validation image sets" ', len(val_pred_ids))

        #if preprocess_ktrans_IDs_path:
            #channels = common_args['input_shape'][-1] + 1
        #else:
        channels = common_args['input_shape'][-1]

        train_generator = Generator(
	        folder_path,
            train_list_IDs,
            df = ID_file_list,
            dim = common_args['input_shape'][:-1],
            n_channels = channels,  #common_args['input_shape'][-1],
            transform_generator = trans_gen,
            batch_size = common_args['batch_size'],
            shuffle = True,
            transform_parameters = trans_pars,
            com_args = common_args)

        validation_generator = Generator(
		    folder_path,
            val_list_IDs,
            df = ID_file_list,
            dim = common_args['input_shape'][:-1],
            n_channels = channels, #common_args['input_shape'][-1],
            transform_generator = False,
            batch_size = common_args['batch_size'],
            shuffle = True,
            transform_parameters = None,
            com_args = common_args)

        predict_generator = Generator(
            folder_path,
            val_pred_ids,
            df = ID_file_list,
            dim = common_args['input_shape'][:-1],
            n_channels = channels, #common_args['input_shape'][-1],
            transform_generator = False,
            batch_size = 1, #common_args['batch_size'],
            shuffle = False,
            transform_parameters = None,
            com_args = common_args)
        '''
        else:
            raise ValueError('Invalid data type received: {}'.format(args.dataset_type))
        '''
        Train_generator.append(train_generator)
        Validation_generator.append(validation_generator)
        Predict_generator.append(predict_generator)

    #return train_generator, validation_generator, predict_generator
    return Train_generator, Validation_generator, Predict_generator
#####################################################


def parse_args(args):
    # parse the auguments
    parser = argparse.ArgumentParser(description='Training script for training a DenseNet network.')
    parser.add_argument('dataset_path', type=str, help='Path to dataset directory.')

    group = parser.add_mutually_exclusive_group()
    group.add_argument('--snapshot',          help='Resume training from a snapshot.')#, action='store_true')
    group.add_argument('--imagenet-weights',  help='Initialize the model with pretrained imagenet weights. This is the default behaviour.', action='store_const', const=False, default=False)
    group.add_argument('--weights',           help='Initialize the model with weights from a file.')
    group.add_argument('--no-weights',        help='Don\'t initialize the model with any weights.', dest='imagenet_weights', action='store_const', const=True)

    model_sel = parser.add_mutually_exclusive_group() #required =  True)
    model_sel.add_argument('--backbone',      help='Backbone model used by DenseNet.', choices=['densenet121', 'densenet169', 'densenet'])
    model_sel.add_argument('--generate_model',help='Generate a user-defined DenseNet model', action='store_true')

    parser.add_argument('--growth_rate',      help='number of filters increasing for each block in the model', type=int, default=16)
    parser.add_argument('--reduction',        help='inverse of compression factor', type=float, default=0.25)
    parser.add_argument('-e', '--seed',       help='seed number of random state', type=int, default=17)
    parser.add_argument('-P', '--pi', help='top layer bias initialized value', type=float, default=0.01)
    parser.add_argument('-a', '--alpha',      help='Paramter alpha in focal loss function', type=float, default=0.25)
    parser.add_argument('-g', '--gamma',      help='Parameter gamma in focal loss function', type=float, default=2)
    parser.add_argument('-s', '--input_shape',help='Input shape with channels_last dim ordering. It should have exactly 4 inputs dims')
    parser.add_argument('-d', '--model_depth',help='Number or layers in the DenseNet.', type=int, default=40)
    parser.add_argument('-b', '--nb_block',   help='Number of dense blocks', type=int, default=2)
    parser.add_argument('--classes',          help='Number of classes for the task of classification', type=int, default=1)
    parser.add_argument('--batch-size',       help='Size of the batches.', default=1, type=int)
    parser.add_argument('--gpu',              help='Id of the GPU to use (as reported by nvidia-smi).')
    parser.add_argument('--multi-gpu',        help='Number of GPUs to use for parallel processing.', type=int, default=0)
    parser.add_argument('--multi-gpu-force',  help='Extra flag needed to enable (experimental) multi-gpu support.', action='store_true')
    parser.add_argument('--epochs',           help='Number of epochs to train.', type=int, default=10)
    parser.add_argument('--steps',            help='Number of steps per epoch.', type=int, default=248)
    parser.add_argument('--lr',               help='Learning rate.', type=float, default=1e-3)
    parser.add_argument('--snapshot-path',    help='Path to store snapshots of models during training (defaults to \'./snapshots\')', default='./snapshots')
    parser.add_argument('--tensorboard-dir',  help='Log directory for Tensorboard output', default='./logs')
    parser.add_argument('--no-snapshots',     help='Disable saving snapshots.', dest='snapshots', action='store_false')
    #parser.add_argument('--no-evaluation',    help='Disable per epoch evaluation.', dest='evaluation', action='store_false')
    parser.add_argument('--freeze-backbone',  help='Freeze training of backbone layers.', action='store_true')
    parser.add_argument('--random-transform', help='Randomly transform image and annotations.', action='store_true')
    #parser.add_argument('--image-min-side',   help='Rescale the image so the smallest side is min_side.', type=int, default=800)
    #parser.add_argument('--image-max-side',   help='Rescale the image if the largest side is larger than max_side.', type=int, default=1333)
    parser.add_argument('--config',           help='Path to a configuration parameters .ini file.')
    #parser.add_argument('--weighted-average', help='Compute the mAP using the weighted average of precisions among classes.', action='store_true')
    parser.add_argument('--compute-val-loss', help='Compute validation loss during training', dest='compute_val_loss', action='store_true')

    # Fit generator arguments
    parser.add_argument('--multiprocessing',  help='Use multiprocessing in fit_generator.', action='store_true')
    parser.add_argument('--workers',          help='Number of generator workers.', type=int, default=-1)
    parser.add_argument('--max-queue-size',   help='Queue length for multiprocessing workers in fit_generator.', type=int, default=4)

    return check_args(parser.parse_args(args))


def check_args(parsed_args):
    """ Function to check for inherent contradictions within parsed arguments.
    For example, batch_size < num_gpus
    Intended to raise errors prior to backend initialisation.
    Args
        parsed_args: parser.parse_args()
    Returns
        parsed_args
    """
    if parsed_args.generate_model:
        if parsed_args.input_shape is None:
            raise ValueError(
            "input shape must be specified as 4-element tuple of channel_last dim ordering")

    if parsed_args.classes < 1:
        raise ValueError(
                "input number of classes has to be greater than or equal to one for the task of classification")

    if parsed_args.multi_gpu > 1 and parsed_args.batch_size < parsed_args.multi_gpu:
        raise ValueError(
            "Batch size ({}) must be equal to or higher than the number of GPUs ({})".format(parsed_args.batch_size,
                                                                                             parsed_args.multi_gpu))

    if parsed_args.multi_gpu > 1 and parsed_args.snapshot:
        raise ValueError(
            "Multi GPU training ({}) and resuming from snapshots ({}) is not supported.".format(parsed_args.multi_gpu,
                                                                                                parsed_args.snapshot))

    if parsed_args.multi_gpu > 1 and not parsed_args.multi_gpu_force:
        raise ValueError("Multi-GPU support is experimental, use at own risk! Run with --multi-gpu-force if you wish to continue.")

    return parsed_args

#####################################################

class RedirectModel(keras.callbacks.Callback):
    """Callback which wraps another callback, but executed on a different model.
    ```python
    model = keras.models.load_model('model.h5')
    model_checkpoint = ModelCheckpoint(filepath='snapshot.h5')
    parallel_model = multi_gpu_model(model, gpus=2)
    parallel_model.fit(X_train, Y_train, callbacks=[RedirectModel(model_checkpoint, model)])
    ```
    Args
        callback : callback to wrap.
        model    : model to use when executing callbacks.
    """

    def __init__(self,
                 callback,
                 model):
        super(RedirectModel, self).__init__()

        self.callback = callback
        self.redirect_model = model

    def on_epoch_begin(self, epoch, logs=None):
        self.callback.on_epoch_begin(epoch, logs=logs)

    def on_epoch_end(self, epoch, logs=None):
        self.callback.on_epoch_end(epoch, logs=logs)

    def on_batch_begin(self, batch, logs=None):
        self.callback.on_batch_begin(batch, logs=logs)

    def on_batch_end(self, batch, logs=None):
        self.callback.on_batch_end(batch, logs=logs)

    def on_train_begin(self, logs=None):
        # overwrite the model with our custom model
        self.callback.set_model(self.redirect_model)

        self.callback.on_train_begin(logs=logs)

    def on_train_end(self, logs=None):
        self.callback.on_train_end(logs=logs)


def makedirs(path):
    # Intended behavior: try to create the directory
    # passes if the directory existing; fails otherwise
    try:
        os.makedirs(path)
    except OSError:
        if not os.path.isdir(path):
          raise

def scheduler(epoch, lrate):
    if epoch % 100 == 0:
        lrate *= 0.5
    return lrate


def create_callbacks(model, args): # prediction_model, validation_generator, args):
    """ Creates the callbacks to use during training.

    Args
        model: The model used for training.
        prediction_model: The model that should be used for validation.
        validation_generator: The generator for creating validation data.
        args: parseargs args object.

    Returns:
        A list of callbacks used for training.
    """
    callbacks = []
    # tensorboard call back
    tensorboard_callback = None

    if args.tensorboard_dir:
        tensorboard_callback = keras.callbacks.TensorBoard(
            log_dir                = args.tensorboard_dir,
            histogram_freq         = 0,
            batch_size             = args.batch_size,
            write_graph            = True,
            write_grads            = False,
            write_images           = False,
            embeddings_freq        = 0,
            embeddings_layer_names = None,
            embeddings_metadata    = None
        )
        callbacks.append(tensorboard_callback)

    # save the model (checkpoint callback)
    if args.snapshots:
        # ensure directory created first; otherwise h5py will error after epoch.
        makedirs(args.snapshot_path+"_CV")
        checkpoint = keras.callbacks.ModelCheckpoint(
            os.path.join(
                args.snapshot_path+"_CV",
                '{backbone}_{{epoch:02d}}_{{val_loss:.3f}}.h5'.format(backbone=args.backbone)
            ),
            monitor='val_loss', #'val_auc_roc',
            verbose=1,
            save_best_only=True,

            mode='min'
        )
        checkpoint = RedirectModel(checkpoint, model)
        callbacks.append(checkpoint)

    callbacks.append(keras.callbacks.ReduceLROnPlateau(
        monitor  = 'val_loss',
        factor   = 0.5,
        patience = 6,
        verbose  = 1,
        mode     = 'auto',
        min_delta  = 0.00001,
        cooldown = 0,
        min_lr   = 1e-6
    ))


    callbacks.append(keras.callbacks.CSVLogger(
        filename = './csv_logs/training_log_'+get_readable_ctime()+'.csv',
        append = True
    ))


    callbacks.append(keras.callbacks.LearningRateScheduler(
        scheduler
    ))


    callbacks.append(keras.callbacks.EarlyStopping(
        monitor = 'val_loss',
        min_delta = 1e-6,
        patience = 20,
        verbose = 1,
        mode = 'auto',
        restore_best_weights = True
    ))

    ## self defination callback function
    #callbacks.append(fid_evaluation)

    return callbacks


def check_keras_version(min_ver):
  #minimum_keras_version = 2, 2, 5

  detected = keras.__version__
  required = '.'.join(map(str, min_ver))
  try:
    assert(tuple(map(int, detected.split('.'))) >= min_ver), 'You are using keras version {}. The minimum required version is {}.'.format(detected, required)
  except AssertionError as e:
    print(e, file=sys.stderr)
    #sys.exit(1)


def get_session():
    """ Construct a modified tf session.
    """
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    return tf.Session(config=config)

#####################################################
## for test purpose
def main(img_patch_file_path, img_ktrans_patch_file_path = None, args = None):

    # parse arguments
    if args is None:
        args = sys.argv[1:]
    args = parse_args(args)
    print(args)

    # create object that stores backbone information
    #backbone = models.backbone(args.backbone)

    # make sure keras is the minimum required version
    check_keras_version((2,2,4))

    # optionally choose specific GPU
    if args.gpu:
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    keras.backend.tensorflow_backend.set_session(get_session())

    # optionally load config parameters
    if args.config:
        args.config = args.config #read_config_file(args.config) # <<<  nothing in config;  need to do later

    # create the generators  << final step before doing fitting
    #for n-fold cross-valiation
    Train_generator, Validation_generator, Pred_generator = create_generators(args, img_patch_file_path, img_ktrans_patch_file_path)
    
    Models = list()
    for idx, (train_generator, validation_generator, pred_generator) in enumerate(zip(Train_generator, Validation_generator, Pred_generator)):
        # create the model
        if args.snapshot is not None:
            model_path = os.path.join(args.snapshot_path, args.snapshot.split(',')[idx].strip())
            print('Loading the model from {}, which will take a while ...'.format(model_path))
            get_custom_objects().update({'auc_roc':auc_roc})
            
            model = keras.models.load_model(os.path.join(args.snapshot_path, args.snapshot.split(',')[idx].strip()), custom_objects={'binary_focal_loss_fixed':binary_focal_loss()}) # need to specify the .hdf5 file path, backbone_name = args.backbone)
            optm = OPTS.sgd(lr=5e-5, momentum=0.9, decay=1e-6, clipvalue=0.5, nesterov=True)
            model.compile(loss=[binary_focal_loss(alpha=args.alpha, gamma=args.gamma)], optimizer=optm, metrics=['accuracy', auc_roc])
            ######################
        elif weights is None:
            print('Creating model; this may take a while...')
            model = create_models(args = args, p = args.pi)  ## <-  need to work on "create_models(backbone, num_classes, weights = None, multi_gpu=0, freeze_backbone=False)
            model.summary()

        # create the callbacks
        callbacks = create_callbacks(model, args)

        if not args.compute_val_loss:
            valdation_generator = None   # <-- not sure if it is necessary

        print('train generator length:', len(train_generator))
        print('train generator 1:', train_generator[0][0].shape)
        print('validation generator length:', len(validation_generator))
        print('validation generator 1:', validation_generator[0][0].shape)
        print('pred generator length:', len(pred_generator))
        print('pred generator length:', pred_generator[0][0].shape)

        #start training
        
        model.fit_generator(
            generator = train_generator,
            steps_per_epoch = len(train_generator), #args.steps,
            epochs = args.epochs,
            verbose = 1,
            callbacks = callbacks,
            validation_data = validation_generator,
            validation_steps = len(validation_generator),
            max_queue_size=args.max_queue_size,
        )
        
        acc = []
        PRED_X, Y = [], []
        for IDX, (x_item, y_item) in enumerate(pred_generator):
            pred_x = model.predict(x_item)
            if (pred_x[0][0] >= 0.5) == y_item[0]:
                acc.append(1)
            PRED_X.extend(pred_x[0])  ### block it later
            Y.extend(y_item)    ### block it later
        #y_pred = tf.convert_to_tensor(PRED_X)
        y_pred = np.array(PRED_X)
        y_true = np.array(Y)
        print('y_pred:', y_pred.shape)
        print('y_true:', y_true.shape)

        ###### remove below later #################
        #AUC_value, AUC_update_op = tf.contrib.metrics.streaming_auc(y_pred, y_true)
        AUC_value, AUC_update_op = tf.metrics.auc(y_true, y_pred)
        fpr, tpr, thresholds = roc_curve(y_true, y_pred)
        roc = pd.DataFrame({'fpr':fpr, 'tpr':tpr, 'thr':thresholds})
        roc.to_csv(args.snapshot_path[2:]+'_f'+str(idx)+'.csv')

        ###### remove above later ################
        print('total correct counts: {}; accuracy: {}'.format(sum(acc), sum(acc)/(IDX+1)))
        Models.append(model)


if __name__ == '__main__':
   
    with open(sys.argv[1]) as json_file:    # config_file
        config_file = json.load(json_file)
    img_patch_file_path = os.path.join(os.getcwd(), 'data/img_patches_zone', sys.argv[2])
    if sys.argv[4] == 'W/ktrans':
        img_ktrans_patch_file_path = os.path.join(os.getcwd(), 'data/img_ktrans_patches_zone', sys.argv[5])
    else:
        img_ktrans_patch_file_path = None

    with open(sys.argv[3]) as args_file:   # args_file
        args = json.load(args_file)
    
    main(img_patch_file_path, img_ktrans_patch_file_path, args)

######################################################
    #img_patch_file_path = os.path.join(os.getcwd(), './data/img_patches/48_48_5_img_patches.pkl')
    #args = ['.', '--generate_model', '--classes=2', '--model_depth=88', '--input_shape=96, 96, 7, 3', '--gpu=gpu0', '--multi-gpu=1', '--batch-size=32', '--compute-val-loss', '--multiprocessing', '--config=config_file.json', '--random-transform']
    #args = ['.', '--generate_model', '--classes=2', '--model_depth=64', '--input_shape=48, 48, 5, 3', '--epochs=50', '--steps=218', '--gpu=gpu0', '--multi-gpu=1', '--batch-size=72', '--compute-val-loss', '--multiprocessing', '--config=config_file.json', '--random-transform']

    ### if continues
    #args = ['.', '--snapshot=densenet_05_0.01.h5', '--backbone=densenet', '--classes=2', '--model_depth=88', '--input_shape=96, 96, 7, 3', '--gpu=gpu0', '--multi-gpu=1', '--batch-size=32', '--steps=248', '--compute-val-loss', '--multiprocessing', '--config=config_file.json', '--random-transform']
    #args = ['.', '--snapshot=densenet_02_0.07.h5', '--backbone=densenet', '--classes=2', '--model_depth=88', '--input_shape=64, 64, 5, 3', '--gpu=gpu0', '--multi-gpu=1', '--batch-size=32', '--batch-size=32', '--compute-val-loss', '--multiprocessing', '--config=config_file.json', '--random-transform']
######################################################  
# python train_ver5_bin_red0_gr16cv.py config_file.json 42_42_1_img_patches.pkl args_list.json W/ktrans 42_42_1_img_ktrans_patches.pkl
