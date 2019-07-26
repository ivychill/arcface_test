from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from scipy import misc
import sys
import os
import  pickle
import argparse
#import tensorflow as tf
import numpy as np
import mxnet as mx
import random
import cv2
import sklearn
from sklearn.decomposition import PCA
from time import sleep
from easydict import EasyDict as edict
# sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))
from mtcnn_detector import MtcnnDetector
import face_image
import face_preprocess



def do_flip(data):
  for idx in range(data.shape[0]):
    data[idx,:,:] = np.fliplr(data[idx,:,:])

def get_model(ctx, image_size, model_str, layer):
  _vec = model_str.split(',')
  assert len(_vec)==2
  prefix = _vec[0]
  epoch = int(_vec[1])
  print('---------*** Model  loading ***-----------')
  # sym, arg_params, aux_params = mx.model.load_checkpoint(prefix, epoch)
  model_path=os.path.join(os.path.dirname(__file__), '..', 'model')
  # # transform model to npy
  print(model_path)
  with open(os.path.join(model_path,'sym.pkl'), 'rb') as a_:                     # open file with write-mode
    # picklestring = pickle.dump(sym, a_)
    sym=pickle.load(a_)
  with open(os.path.join(model_path,'arg_params.pkl'), 'rb') as b_:                     # open file with write-mode
    # picklestring = pickle.dump(arg_params, b_)
    arg_params=pickle.load(b_)
  with open(os.path.join(model_path,'aux_params.pkl'), 'rb') as c_:                     # open file with write-mode
    # picklestring = pickle.dump(aux_params, c_)
    aux_params=pickle.load(c_)
  #end

  all_layers = sym.get_internals()
  sym = all_layers[layer+'_output']
  # print('sym',type(sym))
  # print('arg_params',type(arg_params))
  # print('aux_params',type(aux_params))
  model = mx.mod.Module(symbol=sym, context=ctx, label_names = None)
  #model.bind(data_shapes=[('data', (args.batch_size, 3, image_size[0], image_size[1]))], label_shapes=[('softmax_label', (args.batch_size,))])
  # print(image_size[0])
  model.bind(data_shapes=[('data', (1, 3, image_size[0], image_size[1]))])
  # print(arg_params,aux_params)
  model.set_params(arg_params, aux_params)
  return model

class FaceModel:
  def __init__(self, args):
    self.args = args
    ctx = mx.gpu(args.gpu)
    _vec = args.image_size.split(',')
    assert len(_vec)==2
    image_size = (int(_vec[0]), int(_vec[1]))
    self.model = None
    self.ga_model = None
    if len(args.model)>0:
      self.model = get_model(ctx, image_size, args.model, 'fc1')
    if len(args.ga_model)>0:
      self.ga_model = get_model(ctx, image_size, args.ga_model, 'fc1')

    # self.threshold = args.threshold
    self.det_minsize = 50
    self.det_threshold = [0.6,0.7,0.8]
    #self.det_factor = 0.9
    self.image_size = image_size
    mtcnn_path = os.path.join(os.path.dirname(__file__), '..','mtcnn-model')
    if args.det==0:
      # detector = MtcnnDetector(model_folder=mtcnn_path, ctx=ctx, num_worker=1, accurate_landmark = True, threshold=self.det_threshold)
      with open(os.path.join(mtcnn_path,'mtcnn_0.pkl'), 'rb') as d_:                     # open file with write-mode
        # picklestring = pickle.dump(detector, d_)
        detector=pickle.load(d_)
    else:
      # detector = MtcnnDetector(model_folder=mtcnn_path, ctx=ctx, num_worker=1, accurate_landmark = True, threshold=[0.0,0.0,0.2])
      with open(os.path.join(mtcnn_path,'mtcnn_1.pkl'), 'rb') as e_:                     # open file with write-mode
        # picklestring = pickle.dump(detector, e_)
        detector=pickle.load(e_)
    self.detector = detector


  def get_input(self, face_img):
    ret = self.detector.detect_face(face_img, det_type = self.args.det)
    if ret is None:
      return None
    bbox, points = ret
    if bbox.shape[0]==0:
      return None
    bbox = bbox[0,0:4]
    points = points[0,:].reshape((2,5)).T
    # print('bbox',bbox)
    # print('points',points)
    nimg = face_preprocess.preprocess(face_img, bbox, points, image_size='112,112')
    nimg = cv2.cvtColor(nimg, cv2.COLOR_BGR2RGB)
    aligned = np.transpose(nimg, (2,0,1))
    return aligned

  def get_feature(self, aligned):
    input_blob = np.expand_dims(aligned, axis=0)
    data = mx.nd.array(input_blob)
    db = mx.io.DataBatch(data=(data,))
    self.model.forward(db, is_train=False)
    embedding = self.model.get_outputs()[0].asnumpy()
    embedding = sklearn.preprocessing.normalize(embedding).flatten()
    return embedding

  def get_ga(self, aligned):
    input_blob = np.expand_dims(aligned, axis=0)
    data = mx.nd.array(input_blob)
    db = mx.io.DataBatch(data=(data,))
    self.ga_model.forward(db, is_train=False)
    ret = self.ga_model.get_outputs()[0].asnumpy()
    g = ret[:,0:2].flatten()
    gender = np.argmax(g)
    a = ret[:,2:202].reshape( (100,2) )
    a = np.argmax(a, axis=1)
    age = int(sum(a))

    return gender, age

