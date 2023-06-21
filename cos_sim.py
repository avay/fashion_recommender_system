from flask import Flask, render_template
from flask import request
import random
import os
#os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import numpy as np
import pandas as pd
import pickle
import tensorflow
from tensorflow.keras.preprocessing import image
from tensorflow.keras.layers import GlobalMaxPooling2D
from tensorflow.keras.applications.resnet50 import ResNet50,preprocess_input
from sklearn.neighbors import NearestNeighbors
from numpy.linalg import norm
from sklearn.metrics.pairwise import pairwise_distances


feature_list = np.array(pickle.load(open('embeddings.pkl','rb')))
filenames = pickle.load(open('filenames.pkl','rb'))

print("started cosine operation")
cos_sim = 1-pairwise_distances(feature_list,metric='cosine')
print("ended cosine operation")

pickle.dump(cos_sim,open('cos_sim.pkl','wb'))
