import glob, os, sys
import numpy as np

yolo = 'YOLO'
sun = 'SUN'

if len(sys.argv) >= 3:
	entrada = sys.argv[1]
	saida = sys.argv[2]
else:
	print('Usage: python3 joinAtts.py in_file/directory out_directory [yolo_directory sun_directory]')
if len(sys.argv) == 5:
	yolo = sys.argv[3]
	sun = sys.argv[4]

if not os.path.isdir(saida):
	os.system('mkdir {}'.format(saida))

if os.path.isdir(entrada):
	for img in glob.glob(entrada+'/*.*'):
		f = img.split('/')[-1].rsplit('.', 1)[0]
		f += '.txt'
		a = np.loadtxt('{}/{}'.format(yolo,f))
		b = np.loadtxt('{}/{}'.format(sun, f))
		c = np.concatenate((a,b))
		np.savetxt(saida+'/'+f, c)
elif os.path.isfile(entrada):
	with open(entrada) as g:
		for line in g:
			img = line.split()[0]
			f = img.split('/')[-1].rsplit('.', 1)[0]
			f += '.txt'
			a = np.loadtxt('{}/{}'.format(yolo, f))
			b = np.loadtxt('{}/{}'.format(sun, f))
			c = np.concatenate((a,b))
			np.savetxt(saida+'/'+f, c)

