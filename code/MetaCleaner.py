import tensorflow as tf
from keras.models import Sequential
from keras.layers import Conv1D, MaxPooling2D, Flatten, Dense,Dropout,Bidirectional,Embedding,LSTM,MaxPooling1D,Activation,GlobalAveragePooling1D,GlobalMaxPooling1D,BatchNormalization,Attention,Conv2D
from tensorflow.keras import Model
from tensorflow.keras.layers import LSTM
from transformers.models.bert.modeling_tf_bert import TFBertLayer,TFBertPooler
from transformers import TFBertEmbeddings,BertConfig,BertTokenizer,AdamWeightDecay
import numpy as np
import random
from sklearn.metrics import *
from transformer import EncoderLayer
class MetaCleaner(Model):
    def __init__(self,):
        super(MetaCleaner, self,).__init__()
        self.gp=GlobalAveragePooling1D()
        self.mp=GlobalMaxPooling1D()
        self.relu=Activation('relu')
        self.softmax=tf.keras.layers.Dense(2,activation='softmax')

        self.embedding_matrix = np.load('./embedding_matrix.npy')
        self.emb=Embedding(65, 100, weights=[self.embedding_matrix], trainable=True)
        self.conv1=Conv1D(filters = 256,kernel_size = 3)
        self.conv2=Conv1D(filters = 256,kernel_size = 7)
        self.conv3=Conv1D(filters = 256,kernel_size = 11)
        self.conv4=Conv1D(filters = 256,kernel_size = 15)
        self.bert=EncoderLayer(num_heads=2,d_model=256)

        self.lineLayers=tf.keras.layers.Dense(2,activation='softmax')

    def get_angles(self, pos, i, d_model):
        angle_rates = 1 / np.power(10000, (2 * (i//2)) / np.float32(d_model))
        return pos * angle_rates
    def positional_encoding(self,position, d_model):
        angle_rads = self.get_angles(np.arange(position)[:, np.newaxis],
                                np.arange(d_model)[np.newaxis, :],
                                d_model)
        angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])
        angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])
        pos_encoding = angle_rads[np.newaxis, ...]
        return tf.cast(pos_encoding, dtype=tf.float32)
    
    def getGp(self,inputs):
        inputs=self.emb(inputs)
        out1=self.conv1(inputs)
        out2=self.conv2(inputs)
        out3=self.conv3(inputs)
        out4=self.conv4(inputs)
        out1=self.gp(out1)
        out2=self.gp(out2)
        out3=self.gp(out3)
        out4=self.gp(out4)
        out1=out1[:,tf.newaxis,:]
        out2=out2[:,tf.newaxis,:]
        out3=out3[:,tf.newaxis,:]
        out4=out4[:,tf.newaxis,:]
        out=tf.concat((out1,out2,out3,out4),axis=1)
        return out
    
    def forward(self, text):
        text=self.getGp(text)
        preNoise=self.bert(text)
        cleanText=text-preNoise
        cleanText=tf.keras.layers.Flatten()(cleanText)
        prediction=self.lineLayers(cleanText)
        return prediction
    
    def train_step(self, text ,Noisetext):
        text=self.getGp(text)
        Noisetext=self.getGp(Noisetext)
        noise=text-Noisetext
        preNoise=self.bert(Noisetext)
        cleanText=Noisetext-preNoise
        cleanText=tf.keras.layers.Flatten()(cleanText)
        prediction=self.lineLayers(cleanText)
        return prediction,noise,preNoise

    def call(self, text ,Noisetext=None,training=False):
        if training:
            return self.train_step(text ,Noisetext)
        else:
            return self.forward(text)