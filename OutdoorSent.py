
import cv2, sys, getopt
import numpy as np
from keras.models import load_model, Model
from PIL import Image
from places365 import SunAttributes

import tensorflow as tf
from keras import layers as L
from keras.applications import VGG16, ResNet50, InceptionV3, imagenet_utils
from keras.optimizers import Adam
from keras.callbacks import LearningRateScheduler, ModelCheckpoint
from keras import utils


class OutdoorSent:
	X_train = None
	y_train = None
	X_test = None
	y_test = None

	def __init__(this, model='inception_T_1.h5', use_attributes=True):
		if '.h5' in model:
			m = model.split('_')
			assert(m[0] in ['inception', 'resnet', 'vgg', 'robust'])
			this.model_name = m[0]
			this.use_attributes = True if m[1] == 'T' else False
			this.model = load_model(model)
			this.setSize()
		else:
			assert(model in ['inception', 'resnet', 'vgg', 'robust'])
			this.model_name = model
			this.use_attributes = use_attributes
			this.setSize()
			this.model = this.loadModel()
		if this.use_attributes:
			this.sun = SunAttributes()
		this.model.compile(optimizer='adam',
						   loss='categorical_crossentropy',
						   metrics=['accuracy'])

	def setSize(this):
		if this.model_name == 'inception':
			this.img_size = 299
		elif this.model_name in ['vgg', 'resnet']:
			this.img_size = 227
		else:
			this.img_size = 224


	def loadModel(this):
		input_tensor = L.Input(shape=(this.img_size, this.img_size, 3),
							   name='images')
		if this.model_name == 'vgg':
			base_model = VGG16(weights='imagenet',
							   include_top=False,
							   input_tensor=input_tensor)
			x = L.Flatten()(base_model.output)
		elif this.model_name == 'resnet':
			base_model = ResNet50(weights='imagenet',
								  include_top=False,
								  input_tensor=input_tensor)
			x = L.Flatten()(base_model.output)
		elif this.model_name == 'inception':
			base_model = InceptionV3(weights='imagenet',
								include_top=False,
								input_tensor=input_tensor)
			x = L.GlobalAveragePooling2D()(base_model.output)
		elif this.model_name == 'robust':
			return this.robustModel()

		if this.use_attributes:
			att_tensor = L.Input(shape=(102,), name='attributes')
			x = L.Concatenate()([x, att_tensor])

		x = L.Dense(4096, activation='relu')(x)
		x = L.Dense(4096, activation='relu')(x)
		x = L.Dense(24, activation='relu')(x)

		predictions = L.Dense(3, activation='softmax')(x)

		if this.use_attributes:
			model = Model(inputs=[input_tensor, att_tensor],
						  outputs=predictions)
		else:
			model = Model(inputs=[input_tensor], outputs=predictions)
		return model

	def robustModel(this):
		input_tensor = L.Input(shape=(this.img_size, this.img_size, 3))

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
		if this.use_attributes:
			att_tensor = L.Input(shape=(102,))
			model = L.Concatenate()([model, att_tensor])

		model = L.Dense(units=1024, activation='relu')(model)
		model = L.Dense(units=1024, activation='relu')(model)
		model = L.Dense(units=24, activation='relu')(model)

		predictions = L.Dense(units=3, activation='softmax')(model)

		if this.use_attributes:
			return Model(inputs=[input_tensor, att_tensor], outputs=predictions)
		else:
			return Model(inputs=[input_tensor], outputs=predictions)

	def cropResize(this, img):
		y, x, c = img.shape
		if y < x:
			aux = int((x-y)/2)
			img = img[:,aux:aux+y]
		elif x < y:
			aux = int((y-x)/2)
			img = img[aux:aux+x,:]
		return cv2.resize(img, (this.img_size, this.img_size))

	def loadAttributes(this, att_file):
		att = np.loadtxt(att_file)
		att = list(att)
		return att

	def loadImage(this, img_path, att_path=None):
		img = Image.open(img_path)
		att = None
		if this.use_attributes:
			if att_path == None:
				att = this.sun.attributes(img)
			else:
				a = ((img_path.split('/')[-1]).rsplit('.', 1)[-2])+'.sun'
				a = att_path+'/'+a
				att = list(np.loadtxt(a))
		img = this.cropResize(np.array(img))
		img = np.float64(img)/255
		return img, att

	def loadDataset(this, images, atts=None, train=False, classify=True):
		X = []
		attributes = []
		y = []
		if '.txt' not in images:
			img, att = this.loadImage(images)
			X.append(img)
			if this.use_attributes:
				attributes.append(att)
		else:
			with open(images) as f:
				for line in f:
					if not classify:
						img_path, lbl = line.split('\t')
						y.append(int(lbl))
					else:
						img_path = line.rstrip('/n/r').split('\t')[0]
					img, att = this.loadImage(img_path, atts)
					if this.use_attributes:
						attributes.append(att)
					X.append(img)
		if classify:
			train = False
		else:
			y = utils.to_categorical(y)
		print(images, 'loaded')
		if this.use_attributes:
			if train:
				this.X_train = [np.array(X), np.array(attributes)]
				this.y_train = y
			else:
				this.X_test = [np.array(X), np.array(attributes)]
				this.y_test = y
		else:
			if train:
				this.X_train, this.y_train = np.array(X), y
			else:
				this.X_test, this.y_test = np.array(X), y

	def trainModel(this, epochs=30, k=1):
		assert(this.X_train is not None and this.y_train is not None)
		assert(this.X_test is not None and this.y_test is not None)
		def lr_scheduler(epoch):
			if epoch >= 0.7*epochs:
				return 1e-6
			elif epoch >= 0.4*epochs:
				return 1e-5
			else:
				return 1e-4
		lr_decay = LearningRateScheduler(lr_scheduler)

		name = 'models/'+this.model_name
		name += '_'+('T' if this.use_attributes else 'F')
		name += '_'+str(k)+'.h5'
		checkpoint = ModelCheckpoint(name,
									 monitor='val_acc',
									 verbose=1,
									 save_best_only=True,
									 mode='max')
		print('Training Model')
		this.model.fit(x=this.X_train,
				  y=this.y_train,
				  batch_size=64,
				  epochs=epochs,
				  callbacks=[lr_decay, checkpoint],
				  validation_data=(this.X_test, this.y_test))

	def classify(this, images, attributes=None):
		this.loadDataset(images, attributes, False, True)
		pred = this.model.predict(x=this.X_test, batch_size=1)
		return pred

if __name__ == '__main__':
	try:
		opts, args = getopt.getopt(sys.argv[1:],'m:a:c:k:tu')
	except getopt.GetoptError:
		print('-m <model name> -a <attributes folder> -c <images> -k <fold> -t <train_file> <validation file>')
		sys.exit(1)

	model_name = 'inception_T_1.h5'
	attributes = None
	train = False
	images = None
	train_file = None
	val_file = None
	classify = False
	use_att = False
	k = '1'

	for opt, arg in opts:
		if opt == '-m':
			model_name = arg
		elif opt == '-u':
			use_att = True
		elif opt == '-k':
			k = arg
		elif opt == '-a':
			attributes = arg
		elif opt == '-c':
			classify = True
			images = arg
		elif opt == '-t':
			train = True
			assert(len(args) == 2)
			train_file = args[0]
			val_file = args[1]
		elif opt == '-p':
			print_model = True

	outSent = OutdoorSent(model_name, use_att)
	if train:
		outSent.loadDataset(train_file, attributes, True, False)
		outSent.loadDataset(val_file, attributes, False, False)
		outSent.trainModel(epochs=25, k=k)
	if classify:
		preds = outSent.classify(images, attributes)
		print(preds)