import sys

# this file is expected to be in {caffe_root}/examples
caffe_root = '/home/vagrant/software/caffe/build/install/'
sys.path.insert(0, caffe_root + 'python')

import os
import cv2
import caffe
import telebot
import numpy as np

class CVCBot(object):
	def __init__(self, key_fname, root_path, caffe_args):
		# load the TOKEN from a given file
		self.key = self.load_key(key_fname)
		# set project path
		self.root_path = root_path
		# create a new Telegram Bot object
		self.bot = telebot.TeleBot(self.key)
		# create caffe net
		self.caffe_net = CVCaffee(**caffe_args)

		# Handle '/start' and '/help'
		@self.bot.message_handler(commands=['help', 'start'])
		def send_welcome(message):
		    self.bot.reply_to(message, """\
			Hi there, I am the CVCBot.
			I am here to do some computer vision stuff. Just send me an \
			image and let the magic flow!\
			""")

		# Handle a simple text message
		@self.bot.message_handler(func=lambda message: True)
		def echo_message(message):
		    self.bot.reply_to(message, "Just send me an image!!")

		# Handles all sent documents and audio files
		@self.bot.message_handler(content_types=['photo'])
		def handle_photo(message):

			def download_image(file_info):
				# download the image
				downloaded_file = self.bot.download_file(file_info.file_path)
				# convert to opencv format
				img = file_to_arr(downloaded_file)
				# save image to system
				filename = '%s.png' % file_info.file_id
				path_file = os.path.join(self.root_path, 'tmp', filename)
				cv2.imwrite(path_file, img)
				return path_file

			def file_to_arr(file):
				# url to array
				arr = np.asarray(bytearray(file), dtype=np.uint8)
				img = cv2.imdecode(arr,-1) # 'load it as it is'
				return img

			print '[INFO] Image received from: ' + str(message.from_user)

			# send info to user
			self.bot.reply_to(message, \
				'**** Wait until your image is being precessed ****')

			file_info = None
			for p in message.photo:
				file_info = self.bot.get_file(p.file_id)

			# download the image and save it to the system
			img_path = download_image(file_info)
			# process photo
			prediction = self.caffe_net.predict(img_path)
			
			# send result to user
			self.bot.reply_to(message, prediction)

	def load_key(self, key_fname):
		with open(key_fname, 'r') as f:
		    return f.readlines()[0]
	
	def run(self):
		self.bot.polling()


class CVCaffee(object):
	def __init__(self, proto_txt, caffe_model, labels):
		# load caffe model
		self.net = caffe.Net(proto_txt, caffe_model, caffe.TEST)
		# load labels
		self.labels = np.loadtxt(labels, str, delimiter='\t')
		# transformer to preprocess the input data
		self.transformer = self.init_transformer()

		# set net to batch size of 1
		# googlenet were trained with images of soze 224x224
		self.net.blobs['data'].reshape(1, 3, 224, 224)

	def init_transformer(self):
		# input preprocessing: 'data' is the name of the input blob == net.inputs[0]
		transformer = caffe.io.Transformer({'data': self.net.blobs['data'].data.shape})
		transformer.set_transpose('data', (2,0,1))
		# the reference model operates on images in [0,255] range instead of [0,1]
		transformer.set_raw_scale('data', 255)
		# the reference model has channels in BGR order instead of RGB
		transformer.set_channel_swap('data', (2,1,0))
		return transformer


	def predict(self, img_path):
		# set image array as input
		self.net.blobs['data'].data[...] = self.transformer.preprocess('data', \
			caffe.io.load_image(img_path))
		out = self.net.forward()
		print('Predicted class is #{}.'.format(out['prob'][0].argmax()))

		# sort top k predictions from softmax output
		top_k = self.net.blobs['prob'].data[0].flatten().argsort()[-1:-6:-1]
		print self.labels[top_k]
		return self.labels[top_k][0].split(' ', 1)[1]

if __name__ == '__main__':

	project_path, filename = os.path.split(os.path.realpath(__file__))
	models_dir = os.path.join(project_path, 'data/models/googlenet')
	key_filename = os.path.join(project_path, 'data/key.txt')

	caffe_args = {
		'proto_txt': os.path.join(models_dir, 'deploy.prototxt'),
		'caffe_model': os.path.join(models_dir, 'bvlc_googlenet.caffemodel'),
		'labels': os.path.join(models_dir, 'synset_words.txt')
	}

	cvc_bot = CVCBot(key_filename, project_path, caffe_args)
	cvc_bot.run()
