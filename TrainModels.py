from datetime import datetime
import logging
import numpy as np
import pathlib as pl
import random
import re
from typing import Union, Tuple

import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID" 
os.environ["CUDA_VISIBLE_DEVICES"] = ""

# MobileNet(s)
from keras.applications import MobileNet, MobileNetV2, MobileNetV3Small, MobileNetV3Large
# ResNet(s)
from keras.applications import ResNet50, ResNet101, ResNet152, ResNet50V2, ResNet101V2, ResNet152V2
# Inception
from keras.applications import InceptionV3, InceptionResNetV2
from keras.applications import Xception
# VGG
from keras.applications import VGG16, VGG19
# https://keras.io/api/applications/

from keras.optimizers import Adam
from keras import Input
from keras.layers import Dense
from keras.models import Model

from keras.preprocessing.image import ImageDataGenerator


class TrainModels():
    # local/private variables
    paths = None
    _path_to_data = None
    _path_to_models = None

    _data_generator = None
    _random_seed = None
    _file_extension = 'jpg'
    __split_names = {'training': 'Trn', 'validation': 'Val', 'test': 'Tst'}
    
    _batch_size = None
    n_files = None
    n_classes = 0
    img_size = None
    color_mode = 'rgb'
    epochs = None

    model_pretrained = False
    model_name = None

    def __init__(self, path_to_data: Union[pl.Path, str], 
                 epochs: int,
                 random_seed: int = 42,
                 path_to_save_models: Union[str, pl.Path] = None
                 ) -> None:
        # set local variables
        self._path_to_data = self.set_path_to_data(path_to_data)
        self._path_to_models = pl.Path(path_to_save_models)
        self.epochs = epochs
        self._random_seed = random_seed
        self.save_model = save_model

        random.seed(self._random_seed)
        # turn logging on
        logging.basicConfig(filename='log.out', level=logging.INFO)


    def set_path_to_data(self, path_to_data: Union[pl.Path, str]) -> bool:
        # TODO: check for correct folder structure!
        self._path_to_data = pl.Path(path_to_data)
        # self.paths = dict()  # FIXME: initialize as dict?
        for el in self.__split_names.items():
            self.paths[el[0]] = self._path_to_data.joinpath(el[1])
        # TODO get n_classes
        return True

    def set_data_generator_instance(self) -> bool:
        self._data_generator = ImageDataGenerator(rescale=1. / 255,
                                                  width_shift_range=[-0.2, 0.2],
                                                  height_shift_range=[-0.2, 0.2],
                                                  rotation_range=10,
                                                  vertical_flip=True,
                                                  horizontal_flip=True,
                                                  brightness_range=[0.2, 2.0],
                                                  preprocessing_function=__add_noise,
                                                  zoom_range=[0.9, 1.1],
                                                  )
        # nested local function
        def __add_noise(img: np.ndarray) -> np.ndarray:
            VARIABILITY = 50
            deviation = VARIABILITY * random.random()
            noise = np.random.normal(0, deviation, img.shape)
            img += noise
            np.clip(img, 0., 255.)
            return img
        return True
    
    def get_data_generator(self, key: str):
        # TODO: no augmentation + shuffle for test data!
        return self._data_generator.flow_from_directory(self.paths[key], 
                                                        target_size=self.img_size, 
                                                        color_mode=self.color_mode,
                                                        class_mode='categorical', 
                                                        batch_size=self.batch_size[key],
                                                        shuffle=True, 
                                                        seed=self._random_seed
                                                        )

    def _set_batch_sizes(self) -> bool:
        self._batch_size = dict()
        for ky in self.__split_names.keys():
            files = self.paths[ky].glob('*' +  self._file_extension)
            self.n_files[ky] = len(files)
            # set batch sizes
            self._batch_size[ky] = self.__determine_batch_size(self.n_files[ky])

        def __determine_batch_size(n_files: int) -> int:
            batch_size = -1
            remainder = 9999
            # find most suitable batch size #TODO: ensure that no invalid numbers can be selected
            for sz in range(30, 9, -1):
                tmp_remainder = n_files % sz
                if tmp_remainder < remainder:
                    remainder = tmp_remainder
                    batch_size = sz
                    if remainder == 0:
                        break
            return batch_size
        return True

    def get_steps_per_epoch(self, key: str) -> int:
        return np.floor(self.n_files[key] / self._batch_size[key])

    # ----- Models
    def _compile_model(self) -> bool:
        self.model.compile(optimizer=Adam(),
                           loss='categorical_crossentropy',
                           metrics=['categorical_accuracy']
                           )
        return True

    def fit(self):
        self.model.fit(self.get_data_generator('training'),
                       steps_per_epoch=self.get_steps_per_epoch('training'), 
                       epochs=self.epochs,
                       verbose=True)
        return self.model

    def analyze(self, model_name: str, ) -> Model:
        # set model
        self.set_model(model_name)
        # train model
        self.fit()
        # save model
        if self._path_to_models:
            self.model.save(self._path_to_models.joinpath(f'{model_name}.h5'))
        return self.model

    def __add_new_model_head(self) -> Model:
        base_model = self.model
        model_head = Dense(2, activation='softmax')(base_model.output)
        self.model = Model(inputs=base_model.input, outputs=model_head)
        return self.model
    
    def _general_model_parameters(self) -> dict:
        args = {'classes': self.n_classes}
        # input shape
        if re.match('RGB', self.color_mode, re.IGNORECASE):
            args['input_shape'] = self.img_size + (3,)
        else:  # color_mode == grayscale
            args['input_shape'] = self.img_size + (1,)
        
        # pretrained weights
        if self.model_pretrained:
            args['weights'] = 'ImageNet'
            if self.n_classes != 1000:
                args['include_top'] = False
                args['pooling'] = 'avg'
        else:
            args['weights'] = None
        
        return args

    def set_model(self, model_name: str) -> Model:
        args = self._general_model_parameters()
        
        if re.match('Inception((V3)|(ResNetV2))', model_name):
            # InceptionV3, InceptionResNetV2

            if self.model_name == 'InceptionV3':
                self.model = InceptionV3(*args)
        elif re.match('MobileNet(V[2,3])?((Small)|(Large))?', model_name):
            # MobileNet, MobileNetV2, MobileNetV3Small, MobileNetV3Large

            if self.model_name == 'MobileNet':
                self.model = MobileNet(*args)
            elif self.model_name == 'MobileNetV2':
                self.model = MobileNetV2(*args)
            elif self.model_name == 'MobileNetV3Small':
                self.model = MobileNetV3Small(*args)
            elif self.model_name == 'MobileNetV3Large':
                self.model = MobileNetV3Large(*args)
        elif re.match('ResNet[50,101,152](V2)?', self.model_name):
            # ResNet50, ResNet101, ResNet152, ResNet50V2, ResNet101V2, ResNet152V2

            if self.model_name == 'ResNet50':
                self.model = ResNet50(*args)
            elif self.model_name == 'ResNet101':
                self.model = ResNet101(*args)
            elif self.model_name == 'ResNet152':
                self.model = ResNet152(*args)
            elif self.model_name == 'ResNet50V2':
                self.model = ResNet50V2(*args)
            elif self.model_name == 'ResNet101V2':
                self.model = ResNet101V2(*args)
            elif self.model_name == 'ResNet152V2':
                self.model = ResNet152V2(*args)
        elif re.match('VGG[16,19]', model_name):
            # VGG16, VGG19

            if self.model_name == 'VGG16':
                self.model = VGG16(*args)
            elif self.model_name == 'VGG19':
                self.model = VGG19(*args)
        elif re.match('Xception', model_name):
            # Xception

            self.model = Xception(*args)
        else:
            raise ValueError(f'Unknown model architecture: {model_name}')
        

        if re.search('pretrain(ed)?', model_name, re.IGNORECASE):
            self.__add_new_model_head()
            self.model_name += '_pretrained'

        # compile model
        self._compile_model()
        return self.model



if __name__ == '__main__':
    path_to_data_folder = pl.Path('Data')
    path_to_save_models = pl.Path('Models')
    
    models_to_analyze = ['MobileNet', 'MobileNetV2', 'InceptionV3']

    train = TrainModels(path_to_data_folder, 
                        epochs=2,
                        save_model=False,
                        verbose=True
                        )
    for mdl in models_to_analyze:
        train.analyze(mdl)

