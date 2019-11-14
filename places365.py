# PlacesCNN to predict the scene category, attribute, and class activation map in a single pass
# by Bolei Zhou, sep 2, 2017

# ATTENTION requires python 3.6.x


import torch
from torch.autograd import Variable as V
import torchvision.models as models
from torchvision import transforms as trn
from torch.nn import functional as F
import os, sys, glob
import numpy as np
import cv2
from PIL import Image


def save_attributes(out, att):
	np.savetxt(out, att)

def classify(image, out):
	try:
		img = Image.open(image)
	except:
		print('Could not load image: ', image)
		sys.exit(0)
	if len(np.shape(img)) == 2:
		img = img.convert('RGB')
	if np.shape(img)[2] != 3:
		return -1

	def load_labels():
		# prepare all the labels
		# scene category relevant
		file_name_category = 'categories_places365.txt'
		if not os.access(file_name_category, os.W_OK):
			synset_url = 'https://raw.githubusercontent.com/csailvision/places365/master/categories_places365.txt'
			os.system('wget ' + synset_url)
		classes = list()
		with open(file_name_category) as class_file:
			for line in class_file:
				classes.append(line.strip().split(' ')[0][3:])
		classes = tuple(classes)

		# indoor and outdoor relevant
		file_name_IO = 'IO_places365.txt'
		if not os.access(file_name_IO, os.W_OK):
			synset_url = 'https://raw.githubusercontent.com/csailvision/places365/master/IO_places365.txt'
			os.system('wget ' + synset_url)
		with open(file_name_IO) as f:
			lines = f.readlines()
			labels_IO = []
			for line in lines:
				items = line.rstrip().split()
				labels_IO.append(int(items[-1]) -1) # 0 is indoor, 1 is outdoor
		labels_IO = np.array(labels_IO)

		# scene attribute relevant
		file_name_attribute = 'labels_sunattribute.txt'
		if not os.access(file_name_attribute, os.W_OK):
			synset_url = 'https://raw.githubusercontent.com/csailvision/places365/master/labels_sunattribute.txt'
			os.system('wget ' + synset_url)
		with open(file_name_attribute) as f:
			lines = f.readlines()
			labels_attribute = [item.rstrip() for item in lines]
		file_name_W = 'W_sceneattribute_wideresnet18.npy'
		if not os.access(file_name_W, os.W_OK):
			synset_url = 'http://places2.csail.mit.edu/models_places365/W_sceneattribute_wideresnet18.npy'
			os.system('wget ' + synset_url)
		W_attribute = np.load(file_name_W)

		return classes, labels_IO, labels_attribute, W_attribute

	def hook_feature(module, input, output):
		features_blobs.append(np.squeeze(output.data.cpu().numpy()))

	def returnTF():
	# load the image transformer
		tf = trn.Compose([
			trn.Scale((224,224)),
			trn.ToTensor(),
			trn.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
		])
		return tf


	def load_model():
		# this model has a last conv feature map as 14x14
		model_file = 'whole_wideresnet18_places365.pth.tar'
		if not os.access(model_file, os.W_OK):
			os.system('wget http://places2.csail.mit.edu/models_places365/' + model_file)
			os.system('wget https://raw.githubusercontent.com/csailvision/places365/master/wideresnet.py')

		from functools import partial
		import pickle
		pickle.load = partial(pickle.load, encoding="latin1")
		pickle.Unpickler = partial(pickle.Unpickler, encoding="latin1")
		model = torch.load(model_file, map_location=lambda storage, loc: storage, pickle_module=pickle)

		#model = torch.load(model_file, map_location=lambda storage, loc: storage) # allow cpu
		model.eval()
		# hook the feature extractor
		features_names = ['layer4','avgpool'] # this is the last conv layer of the resnet
		for name in features_names:
			model._modules.get(name).register_forward_hook(hook_feature)
		return model

	# load the labels
	classes, labels_IO, labels_attribute, W_attribute = load_labels()

	# load the model
	features_blobs = []
	model = load_model()

	# load the transformer
	tf = returnTF() # image transformer

	# get the softmax weight
	params = list(model.parameters())
	weight_softmax = params[-2].data.numpy()
	weight_softmax[weight_softmax<0] = 0

	input_img = V(tf(img).unsqueeze(0), volatile=True)

	# forward pass
	logit = model.forward(input_img)
	h_x = F.softmax(logit).data.squeeze()
	probs, idx = h_x.sort(0, True)

	# output the IO prediction
	io_image = np.mean(labels_IO[idx[:10].numpy()]) # vote for the indoor or outdoor

	# output the scene attributes
	responses_attribute = W_attribute.dot(features_blobs[1])

	save_attributes(out, responses_attribute)
	return 0 if io_image < 0.5 else 1



if len(sys.argv) < 2:
	print("Usage: python3 places365.py Images")
	sys.exit(0)

IN = sys.argv[1]
if not os.path.isdir('SUN'):
	os.mkdir('SUN')


if os.path.isdir(IN):
	for img in glob.glob(IN+'/*.jpg'):
		io_img = classify(img, "SUN/"+img.split('/')[-1].replace('.jpg', '.txt'))
		print(img, io_img)
else:
	with open(IN) as f:
		for line in f:
			img = line.split()[0]
			classify(img, "SUN/"+img.split('/')[-1].replace('.jpg', '.txt'))
			print(img, io_img)