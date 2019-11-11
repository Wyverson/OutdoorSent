import cv2, getopt, os, sys
import numpy as np

from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix

import tensorflow as tf
from keras import layers as L
from keras.applications import VGG16
from keras.applications import ResNet50
from keras.applications import InceptionV3
from keras.applications import Xception
from keras.applications import DenseNet169
from keras.utils import Sequence
from keras.utils import to_categorical
from keras.callbacks import LearningRateScheduler
from keras.callbacks import ModelCheckpoint
from keras.models import Model


class DataGenerator(Sequence):
    def __init__(self, img_files,
                 labels,
                 att_dir=None,
                 img_size=None,
                 batch_size=32,
                 shuffle=True,
                 n_classes=3):

        self.img_files = img_files
        self.att_dir = att_dir
        self.img_size = img_size
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.labels = labels
        self.n_classes = n_classes
        self.on_epoch_end()

    def __len__(self):
        return int((len(self.img_files)-1)/self.batch_size+1)

    def __getitem__(self, index):
        if index == self.__len__():
            indexes = self.indexes[index*self.batch_size:]
        else:
            indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]
        img_files_temp = [self.img_files[k] for k in indexes]
        if self.labels is not None:
            labels_temp = [self.labels[k] for k in indexes]
            X, y = self.__data_generation(img_files_temp, labels_temp)
            return X, y
        else:
            X = self.__data_generation(img_files_temp, img_files_temp)
            return X

    def on_epoch_end(self):
        self.indexes = np.arange(len(self.img_files))
        if self.shuffle:
            np.random.shuffle(self.indexes)

    def crop_resize(self, img_file):
        img = cv2.imread(img_file, cv2.IMREAD_COLOR)
        y, x, c = img.shape
        if y < x:
            aux = int((x-y)/2)
            img = img[:,aux:aux+y]
        elif x < y:
            aux = int((y-x)/2)
            img = img[aux:aux+x,:]
        return np.float32(cv2.resize(img, (self.img_size, self.img_size)))/255.

    def __data_generation(self, img_files_temp, labels_temp):
        X_img = []
        X_att = []
        y = []
        for lbl, img_file in zip(labels_temp, img_files_temp):
            img = self.crop_resize(img_file)
            X_img.append(img)
            img_name = ((img_file.split('/')[-1]).rsplit('.', 1)[-2])
            if self.att_dir is not None:
                a = self.att_dir+'/'+img_name+'.txt'
                att = np.loadtxt(a)
                X_att.append(att)
            y.append(lbl)
        if self.att_dir is not None:
            X = [np.asarray(X_img), np.asarray(X_att)]
        else:
            X = np.asarray(X_img)
        if self.labels is None:
            return X
        else:
            return X, to_categorical(y, num_classes=self.n_classes)


class OutdoorSent:
    __known_models = ['inception', 'resnet', 'vgg',
                      'xception', 'densenet', 'robust']

    def __init__(self, model, attributes=None, n_atts=102, n_classes=3):
        self.n_atts = n_atts
        self.n_classes = n_classes
        if '.h5' in model:
                ######################
                # Load trained model #
                ######################
            m = model.split('/')[-1].split('_')
            assert(m[0] in self.__known_models)
            if m[1] == 'T':
                assert(attributes is not None)
            self.model_name = m[0]
            self.attributes = attributes
            self.setSize()
            self.model = self.loadModel()
            self.model.load_weights(model)
        else:
            assert(model in self.__known_models)
            self.model_name = model
            self.attributes = attributes
            self.setSize()
            self.model = self.loadModel()
        self.model.compile(optimizer='adam',
                           loss='categorical_crossentropy',
                           metrics=['accuracy'])

    def setSize(self):
        if self.model_name == 'inception':
            self.img_size = 299
        else:
            self.img_size = 224

    def loadModel(self):
        input_tensor = L.Input(shape=(self.img_size, self.img_size, 3),
                               name='images')
        if self.model_name == 'robust':
            return self.robustModel()
        elif self.model_name == 'vgg':
                # Very Deep Convolutional Networks
                #  for Large-Scale Image Recognition
                # https://arxiv.org/abs/1409.1556
            base_model = VGG16(weights='imagenet',
                               include_top=False,
                               input_tensor=input_tensor)
            x = L.Flatten()(base_model.output)
        elif self.model_name == 'resnet':
                # Deep Residual Learning for Image Recognition
                # https://arxiv.org/abs/1512.03385
            base_model = ResNet50(weights='imagenet',
                                  include_top=False,
                                  input_tensor=input_tensor)
        elif self.model_name == 'inception':
                # Rethinking the Inception Architecture for Computer Vision
                # https://arxiv.org/abs/1512.00567
            base_model = InceptionV3(weights='imagenet',
                                     include_top=False,
                                     input_tensor=input_tensor)
        elif self.model_name == 'xception':
                # Xception: Deep Learning with Depthwise Separable Convolutions
                # https://arxiv.org/abs/1610.02357
            base_model = Xception(weights='imagenet',
                                  include_top=False,
                                  input_tensor=input_tensor)
        elif self.model_name == 'densenet':
                # Densely Connected Convolutional Networks
                # https://arxiv.org/abs/1608.06993
            base_model = DenseNet169(weights='imagenet',
                                     include_top=False,
                                     input_tensor=input_tensor)

        if self.model_name != 'vgg':
            x = L.GlobalAveragePooling2D()(base_model.output)

        if self.attributes is not None:
            att_tensor = L.Input(shape=(self.n_atts,), name='attributes')
            x = L.Concatenate()([x, att_tensor])

        x = L.Dense(2048, activation='relu')(x)
        x = L.Dropout(rate=0.2)(x)
        x = L.Dense(1024, activation='relu')(x)
        x = L.Dropout(rate=0.2)(x)
        x = L.Dense(24, activation='relu')(x)

        predictions = L.Dense(self.n_classes, activation='softmax')(x)

        if self.attributes is not None:
            model = Model(inputs=[input_tensor, att_tensor],
                          outputs=predictions)
        else:
            model = Model(inputs=[input_tensor], outputs=predictions)
        return model

    def robustModel(self):
            # Robust Image Sentiment Analysis Using
            #  Progressively Trained and Domain Transferred Deep Networks
            # https://arxiv.org/abs/1509.06041
        input_tensor = L.Input(shape=(self.img_size, self.img_size, 3))

        model = L.Conv2D(filters=96,
                         kernel_size=(11,11),
                         strides=4,
                         activation='relu')(input_tensor)
        model = L.Lambda(lambda a: tf.nn.lrn(input=a))(model)
        model = L.MaxPooling2D(pool_size=(3, 3), strides=2)(model)

        model = L.Conv2D(filters=256,
                         kernel_size=(5, 5),
                         strides=2,
                         activation='relu')(model)
        model = L.Lambda(lambda a: tf.nn.lrn(input=a))(model)
        model = L.MaxPooling2D(pool_size=(6, 6), strides=6)(model)

        model = L.Flatten()(model)
        if self.attributes is not None:
            att_tensor = L.Input(shape=(self.n_atts,))
            model = L.Concatenate()([model, att_tensor])

        model = L.Dense(units=1024, activation='relu')(model)
        model = L.Dropout(rate=0.2)(model)
        model = L.Dense(units=1024, activation='relu')(model)
        model = L.Dropout(rate=0.2)(model)
        model = L.Dense(units=24, activation='relu')(model)

        predictions = L.Dense(units=self.n_classes, activation='softmax')(model)

        if self.attributes is not None:
            return Model(inputs=[input_tensor, att_tensor], outputs=predictions)
        else:
            return Model(inputs=[input_tensor], outputs=predictions)


    def loadDataset(self, images, classify=False):
        if classify:
            X_imgs = []
            y = None
            if os.path.isfile(images):
                with open(images) as f:
                    for line in f:
                        img_path = line.split()[0]
                        X_imgs.append(img_path)
            elif os.path.isdir(images):
                import glob
                X_imgs = glob.glob('{}/*.*'.format(images))
            return np.asarray(X_imgs), y
        else:
            X_imgs = []
            y = []
            assert(os.path.isfile(images))
            with open(images) as f:
                for line in f:
                    img_path, lbl = line.split()
                    X_imgs.append(img_path)
                    y.append(int(lbl))
            return np.asarray(X_imgs), np.asarray(y)

    def __weights(self, y):
        c = np.zeros(self.n_classes)
        for i in range(self.n_classes):
            c[i] = np.sum(np.where(y==i, 1., 0.))

        class_weight = {}
        for i in range(self.n_classes):
            class_weight[i] = min(c)/c[i]
        print('Class Weights:', class_weight)
        return class_weight

    def trainModel(self, X_imgs, y, epochs=20, validation_split=0.2, k=5):
        def fit_model(X_t, y_t, v='', epochs=20):
            def lr_scheduler(epoch):
                if epoch >= 0.7*epochs:
                    return 1e-6
                elif epoch >= 0.4*epochs:
                    return 1e-5
                else:
                  return 1e-4
            lr_decay = LearningRateScheduler(lr_scheduler)
            X_train, X_val, y_train, y_val = train_test_split(X_t, y_t,
                                                              test_size = 0.2,
                                                              stratify = y_t,
                                                              random_state = 42)
            class_weight = self.__weights(y_train)
            train_datagen = DataGenerator(X_train, y_train,
                                          att_dir = self.attributes,
                                          img_size = self.img_size,
                                          batch_size = 32,
                                          n_classes=self.n_classes)
            val_datagen = DataGenerator(X_val, y_val,
                                        att_dir = self.attributes,
                                        img_size = self.img_size,
                                        batch_size = 32,
                                        shuffle = False,
                                        n_classes=self.n_classes)

            name = 'Weights/{}_{}_{}.h5'.format(self.model_name,
                    ('F' if self.attributes is None else 'T'), v)
            checkpoint = ModelCheckpoint(name,
                                         monitor = 'val_acc',
                                         verbose = 1,
                                         save_best_only = True,
                                         save_weights_only = True,
                                         mode = 'max')

            self.model.fit_generator(train_datagen,
                                epochs = epochs,
                                validation_data = val_datagen,
                                callbacks=[lr_decay, checkpoint],
                                class_weight = class_weight)
            return name

        if not os.path.isdir('Weights'):
            try:
                os.mkdir('Weights')
            except:
                print("Creation of directory 'Weights' failed!")
                exit(1)
        assert(k > 0 and k < 10)

        if k > 1:
            kf = StratifiedKFold(n_splits=k)
            self.model.save_weights('temp_weights.h5')
            for i, (tr, te) in enumerate(kf.split(X_imgs, y)):
                if i != 0:
                    self.model.load_weights('temp_weights.h5')
                name = fit_model(X_imgs[tr], y[tr], i, epochs)

                self.model.load_weights(name)
                p_name = 'preds_{}_{}.txt'.format(self.model_name,('F' if self.attributes is None else 'T'))
                self.classify(X_imgs[te], y[te], p_name)

        elif k == 1:
            fit_model(X_imgs, y, epochs)


    def classify(self, X, y_true, save=False):
        test_datagen = DataGenerator(X, y_true,
                                     att_dir = self.attributes,
                                     img_size = self.img_size,
                                     batch_size = 1,
                                     shuffle = False,
                                     n_classes=self.n_classes)
        pred = self.model.predict_generator(test_datagen, verbose=0)
        y_pred = np.argmax(pred, axis = 1)
        if y_true is not None:
            print(classification_report(y_true, y_pred))
            print(accuracy_score(y_true, y_pred))
            print(confusion_matrix(y_true, y_pred))
        if save:
            if self.n_classes == 3:
                classes = ['Negative', 'Neutral', 'Positive']
            elif self.n_classes == 2:
                classes = ['Negative', 'Positive']
            else:
                print("Can't predict more than 3 classes")
                return
            with open(save, 'a+') as f:
                for img, p, c in zip(X, pred, y_pred):
                    f.write('{} {} {}\n'.format(img, p, classes[c]))


if __name__ == '__main__':
    try:
        opts, args = getopt.getopt(sys.argv[1:],
                'm:a:t:c:e:k:n:')
    except:
        sys.exit(1)

    model_name = None
    attributes = None
    images = None
    k = 5
    train = False
    classi = False
    evalu = False
    n_atts = 102

    for opt, arg in opts:
        if opt == '-m':
            model_name = arg
        elif opt == '-a':
            attributes = arg
        elif opt == '-t':
            images = arg
            train = True
        elif opt == '-k':
            k = int(arg)
        elif opt == '-e':
            images = arg
            evalu = True
        elif opt == '-n':
            n_atts = int(arg)
        elif opt == '-c':
            images = arg
            classi = True

    if (train and evalu) or (train and classi) or (classi and evalu):
        print("Please use only one the folowwing options: '-t', '-c' or '-e'")
        exit(0)

    outSent = OutdoorSent(model_name, attributes, n_atts=n_atts, n_classes=3)
    X, y = outSent.loadDataset(images, classi)
    if train:
        outSent.trainModel(X, y, k=k)
    elif evalu or classi:
        name = 'preds_{}_{}_{}.txt'.format(outSent.model_name, ('F' if attributes is None else 'T'), images.split('.')[0])
        outSent.classify(X, y, name)
