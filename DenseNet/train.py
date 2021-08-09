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
import json
import pickle
import tensorflow as tf
import numpy as np


'''
### check webpage:  https://stanford.edu/~shervine/blog/keras-how-to-generate-data-on-the-fly
#### Create a Generator class ####
class Generator(keras.utils.Sequence):
    def __init__(self, list_IDs, batch_size = 32):
        self.batch_size = batch_size
        self.list_IDs
        self.shuffle = shuffle
        self.on_epoch_end()
        
    def on_epoch_end(self):
        #updates indexes after each epoch
        self.indexes = np.arange(len(self.list_IDs))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)
            
    def __getitem__(self, index):
'''

class Generator(keras.utils.Sequence):
    def __init__(
        self, 
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
        #X = np.empty((self.batch_size, *self.dim, self.channels))   # channel last
        #y = np.empty((self.batch_size), dtype = int)

        if self.transform_generator:
            for idx, ID in enumerate(ID_list):
                # random_transform_generator:  channel first    
                load_x_patch = np.moveaxis(np.load(os.path.join(self.data_folder, ID)), -1, 0) # move the ch_axis from the last to the first
                X[idx,] = np.moveaxis(random_transform_generator(load_x_patch,  r_axis=1, c_axis=2, s_axis=3, ch_axis=0, trans_pars = self.transform_parameters,), 0, -1)
                y[idx] = int(self.df[self.df['File name'] == ID]['label']) ## from labels extracted from pd.DataFrame
                
        else:
            for idx, ID in enumerate(ID_list):
                X[idx,] = np.load(os.path.join(self.data_folder,ID))
                y[idx] = int(self.df[self.df['File name'] == ID]['label']) ## from labels extracted from pd.DataFrame
                

        assert 'n_classes' in self.com_args.keys(), 'common_args does not include "n_classes"'
        #categorical = np.zeros((n, self.com_args['n_classes']), dtype=dtype)
        print(ID_list)
        print(y)
        return X, keras.utils.to_categorical(y, num_classes = self.com_args['n_classes'])
     

'''
  data_gen = tf.keras.preprocessing.image.ImageDataGenerator(
    rotation_range=90,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest')
    
    
  model.fit_generator(data_gen.flow(trainX, trainY, batch_size=..))
'''

### create generators
def create_generators(args, preprocess_image_IDs_path):
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
    assert preprocess_image_IDs_path.endswith('.pkl'), 'File type is incorrect.'
    
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
            'brightness': config['bright_range'],
            'flip': config['flip_prob'],
            }
    #data_folder = '_'.join([str(x) for x in common_args['input_shape']])
    ID_file_list = pickle.load(open(preprocess_image_IDs_path, 'rb'))
    train_list_IDs, val_list_IDs = train_test_split(list(ID_file_list['File name']), train_size=0.8, random_state=42)   # <- way to split train and validation
    
    train_generator = Generator(
        train_list_IDs, 
        df = ID_file_list,
        dim = common_args['input_shape'][:-1],
        n_channels = common_args['input_shape'][-1],
        transform_generator = trans_gen, 
        batch_size = common_args['batch_size'],
        shuffle = True, 
        transform_parameters = trans_pars, 
        com_args = common_args)

    validation_generator = Generator(
        val_list_IDs,
        df = ID_file_list,
        dim = common_args['input_shape'][:-1],
        n_channels = common_args['input_shape'][-1],
        transform_generator = False, 
        batch_size = common_args['batch_size'],
        shuffle = True, 
        transform_parameters = None, 
        com_args = common_args)
    '''
    else:
        raise ValueError('Invalid data type received: {}'.format(args.dataset_type))
    '''
    return train_generator, validation_generator

#####################################################
# need to be integrated later (maybe in __main__)


### test create_generator and Generator
config_file = {
    'rot_range': 15,
    'shift_range': (4,4),
    'sheer_degree': 10,
    'zoom_factor': (0.8, 1.1),
    'bright_range': (0.8, 1.2),
    'flip_prob': (0.02, 0.3)
    }

with open('config_file.json','w') as f:
  json.dump(config_file, f)
  

img_patch_file_path = os.path.join(os.getcwd(), '96_96_7_img_patches.pkl')

######################################################

def parse_args(args):
    # parse the auguments
    parser = argparse.ArgumentParser(description='Training script for training a DenseNet network.')
    parser.add_argument('dataset_path', type=str, help='Path to dataset directory.')
    
    group = parser.add_mutually_exclusive_group()
    group.add_argument('--snapshot',          help='Resume training from a snapshot.')
    group.add_argument('--imagenet-weights',  help='Initialize the model with pretrained imagenet weights. This is the default behaviour.', action='store_const', const=False, default=False)
    group.add_argument('--weights',           help='Initialize the model with weights from a file.')
    group.add_argument('--no-weights',        help='Don\'t initialize the model with any weights.', dest='imagenet_weights', action='store_const', const=True)

    model_sel = parser.add_mutually_exclusive_group() #required =  True)
    model_sel.add_argument('--backbone',      help='Backbone model used by DenseNet.', choices=['densenet121', 'densenet169'])
    model_sel.add_argument('--generate_model',help='Generate a user-defined DenseNet model', action='store_true')
    
    parser.add_argument('-s', '--input_shape',help='Input shape with channels_last dim ordering. It should have exactly 4 inputs dims')
    parser.add_argument('-d', '--model_depth',help='Number or layers in the DenseNet.', type=int, default=40)
    parser.add_argument('--classes',          help='Number of classes for the task of classification', type=int, default=1)
    parser.add_argument('--batch-size',       help='Size of the batches.', default=1, type=int)
    parser.add_argument('--gpu',              help='Id of the GPU to use (as reported by nvidia-smi).')
    parser.add_argument('--multi-gpu',        help='Number of GPUs to use for parallel processing.', type=int, default=0)
    parser.add_argument('--multi-gpu-force',  help='Extra flag needed to enable (experimental) multi-gpu support.', action='store_true')
    parser.add_argument('--epochs',           help='Number of epochs to train.', type=int, default=50)
    parser.add_argument('--steps',            help='Number of steps per epoch.', type=int, default=200)
    parser.add_argument('--lr',               help='Learning rate.', type=float, default=1e-5)
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
    parser.add_argument('--max-queue-size',   help='Queue length for multiprocessing workers in fit_generator.', type=int, default=10)
    
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
# need to be integrated later (maybe in __main__)

args = ['.', '--generate_model', '--classes=2', '--model_depth=88', '--input_shape=96, 96, 7, 3', '--gpu=gpu0', '--multi-gpu=1', '--batch-size=64', '--compute-val-loss', '--multiprocessing', '--config=config_file.json', '--random-transform']
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
        
    '''    
    ### it seems unncessary to keep the evalutaion callback 
    ### evaluation should be done in .fit_generator and/ or .evaluate_generator
    # evaluation callback
    if args.evaluation and validation_generator:
        if args.dataset_type == 'coco':
            from ..callbacks.coco import CocoEval

            # use prediction model for evaluation
            evaluation = CocoEval(validation_generator, tensorboard=tensorboard_callback)
        else:
            evaluation = Evaluate(validation_generator, tensorboard=tensorboard_callback, weighted_average=args.weighted_average)
        evaluation = RedirectModel(evaluation, prediction_model)
        callbacks.append(evaluation)
    '''
    
    # save the model (checkpoint callback)
    if args.snapshots:
        # ensure directory created first; otherwise h5py will error after epoch.
        makedirs(args.snapshot_path)
        checkpoint = keras.callbacks.ModelCheckpoint(
            os.path.join(
                args.snapshot_path,
                '{backbone}_{{epoch:02d}}_{{val_loss:.2f}}.h5'.format(backbone=args.backbone)
            ),
            monitor='val_loss',
            verbose=1,
            save_best_only=False,

            mode='min'
        )
        checkpoint = RedirectModel(checkpoint, model)
        callbacks.append(checkpoint)

    callbacks.append(keras.callbacks.ReduceLROnPlateau(
        monitor  = 'val_loss',
        factor   = 0.5,
        patience = 4,
        verbose  = 1,
        mode     = 'auto',
        min_delta  = 0.0001,
        cooldown = 0,
        min_lr   = 0.0001
    ))
    
    callbacks.append(keras.callbacks.EarlyStopping(
        monitor = 'val_loss', 
        min_delta = 1e-6, 
        patience = 6, 
        verbose = 1, 
        mode = 'auto',
        restore_best_weights = True
    ))

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



