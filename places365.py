import torch, os, sys
import numpy as np
from torch.autograd import Variable as V
from torchvision import transforms as trn
from PIL import Image

class SunAttributes:
	def __init__(this):
		this.features_blobs = []
		file_name_W = 'W_sceneattribute_wideresnet18.npy'
		if not os.access(file_name_W, os.W_OK):
			synset_url = 'http://places2.csail.mit.edu/models_places365/W_sceneattribute_wideresnet18.npy'
			os.system('wget ' + synset_url)
		this.W_attribute = np.load(file_name_W)
		model_file = 'wideresnet18_places365.pth.tar'
		if not os.access(model_file, os.W_OK):
			os.system('wget http://places2.csail.mit.edu/models_places365/' + model_file)
			os.system('wget https://raw.githubusercontent.com/csailvision/places365/master/wideresnet.py')
		import wideresnet
		this.model = wideresnet.resnet18(num_classes=365)
		checkpoint = torch.load(model_file, map_location=lambda storage, loc: storage)
		state_dict = {str.replace(k,'module.',''): v for k,v in checkpoint['state_dict'].items()}
		this.model.load_state_dict(state_dict)
		this.model.eval()
		features_names = ['layer4','avgpool']
		for name in features_names:
			this.model._modules.get(name).register_forward_hook(this.__hook_feature)
		this.tf = trn.Compose([
			trn.Resize((224,224)),
			trn.ToTensor(),
			trn.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
		])

	def __hook_feature(this, module, input, output):
		this.features_blobs.append(np.squeeze(output.data.cpu().numpy()))

	def attributes(this, image):
		this.features_blobs = []
		input_img = V(this.tf(image).unsqueeze(0))
		logit = this.model.forward(input_img)
		response_attributes = this.W_attribute.dot(this.features_blobs[1])
		return response_attributes

if __name__ == '__main__':
	img = Image.open(sys.argv[1])
	SA = SunAttributes()
	atts = SA.attributes(img)
	print(atts)