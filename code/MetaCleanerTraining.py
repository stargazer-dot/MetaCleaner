from gensim.models import word2vec
import pandas as pd
from keras.models import Sequential
import tensorflow as tf
import keras
import numpy as np

import os
from transformers import AdamWeightDecay
from itertools import product
from tqdm import tqdm
from MetaCleaner import MetaCleaner
def generate_kmers(k):
    nucleotides = ['A', 'C', 'G', 'T']
    kmers = [''.join(p) for p in product(nucleotides, repeat=k)]
    return kmers
def getKmers(sequence):
    size=3
    sequence=' '.join([sequence[x:x+size] for x in range(len(sequence) - size + 1)])
    return sequence
dic=np.load('./dict.npy',allow_pickle=True)
vocab=list(dic.item().keys())
voc={}
for i in range(1,len(vocab)+1):
    voc[i]=vocab[i-1]
print(voc)
ids_from_chars = tf.keras.layers.StringLookup(
    vocabulary=list(vocab))
chars_from_ids = tf.keras.layers.StringLookup(
    vocabulary=ids_from_chars.get_vocabulary(), invert=True, mask_token=None)

vocab_size=len(ids_from_chars.get_vocabulary())

print(vocab_size)
model=MetaCleaner()
optimizer = AdamWeightDecay(learning_rate=1e-3, weight_decay_rate=0.01)
loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False)
mse = tf.keras.losses.MeanSquaredError()
train_loss = tf.keras.metrics.Mean(name='train_loss')
train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='train_accuracy')
test_loss = tf.keras.metrics.Mean(name='test_loss')
test_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='test_accuracy')

def Mask(sequence):
    rate=30
    sequence_length = len(sequence)
    num_otherW = int((rate * sequence_length)/100)
    mask = tf.random.shuffle(tf.concat([tf.random.uniform((num_otherW,), minval=1, maxval=64, dtype=tf.int32),tf.fill([sequence_length-num_otherW], (0))], axis=0))
    MaskSequence=(mask+tf.cast(sequence, tf.int32))%vocab_size
    return MaskSequence

import random
def cutDna(sequence):
    sequence=sequence.split()
    length=random.randint(min(len(sequence),100),len(sequence))
    begin=random.randint(0,len(sequence)-length)
    sequence=sequence[begin:begin+length]
    return ' '.join(sequence)

def train_step(seq, maskseq, labels):
  with tf.device('/GPU:0'):
    with tf.GradientTape() as tape:
        prediction,noise,preNoise=model(seq,maskseq,training=True)
        loss = loss_object(labels, prediction)+mse(noise,preNoise)
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    train_loss(loss)
    train_accuracy(labels, prediction)

def test_step_noise(data):
    text,labels = data['text'],data['labels']
    text=ids_from_chars(tf.strings.split(text)).numpy()
    text = tf.keras.preprocessing.sequence.pad_sequences(text, padding='post')
    prediction=model(text,training=False)
    t_loss = loss_object(labels, prediction)
    test_loss(t_loss)
    prediction=tf.argmax(prediction,axis=-1)
    return tf.math.confusion_matrix(labels,prediction,2)

def test_step_clinical(data):
    text,reverse,labels = data['text'],data['reverse'],data['labels']
    text=[x.numpy().decode('utf-8') for x in text]
    reverse=[x.numpy().decode('utf-8') for x in reverse]
    for i in range(len(text)):
        text[i]= getKmers(text[i])
    for i in range(len(reverse)):
        reverse[i]= getKmers(reverse[i])
    text=ids_from_chars(tf.strings.split(text)).numpy()
    text = tf.keras.preprocessing.sequence.pad_sequences(text, padding='post')

    prediction=model(text,training=False)
    t_loss = loss_object(labels, prediction)
    test_loss(t_loss)
    prediction=tf.argmax(prediction,axis=-1)
    return tf.math.confusion_matrix(labels,prediction,2)


dummy_input = tf.zeros((32, 96))
output = model(dummy_input,training=False) 
if os.path.exists('./MetaCleaner.h5'):
    model.load_weights('./MetaCleaner.h5')

EPOCHS = 30
BATCH_SIZE = 1024
BUFFER_SIZE = 10000
train_ds = tf.data.experimental.make_csv_dataset(
    './phage400.csv',
    batch_size=BATCH_SIZE,
    column_names=['seq','reverse','labels'],
    na_value="?", 
    num_epochs=1,
    ignore_errors=True, 
    column_defaults=[tf.string,tf.string,tf.int32],
    shuffle=BUFFER_SIZE
)
val_ds = tf.data.experimental.make_csv_dataset(
    'val.csv',
    batch_size=2048,
    column_names=['text','labels'],
    num_epochs=1,
    ignore_errors=True, 
    column_defaults=[tf.string,tf.int32],
)

def train():
    for epoch in range(EPOCHS):
        train_loss.reset_states()
        train_accuracy.reset_states()
        test_loss.reset_states()
        for feature in tqdm(train_ds):
            text = ids_from_chars(tf.strings.split(feature['seq']))
            maskText=[x.numpy().decode('utf-8') for x in feature['seq']]
            for i in range(len(maskText)):
                maskText[i]= cutDna(maskText[i])
            maskText = ids_from_chars(tf.strings.split(maskText))
            maskText=tf.data.Dataset.from_tensor_slices(maskText)
            maskText=list(maskText.map(Mask).as_numpy_iterator())
            maskText = tf.keras.utils.pad_sequences(maskText, padding="post",maxlen=400)

            text=tf.data.Dataset.from_tensor_slices(text)
            text = tf.keras.utils.pad_sequences(text, padding="post",maxlen=400)

            train_step(text,maskText,feature['labels'])
        model.save_weights('./MetaCleaner.h5')
        cm=np.zeros((2,2))
        i=0
        for feature in tqdm(val_ds):
            cm+=test_step_noise(feature)
        acc=(cm[0][0]+cm[1][1])/np.sum(cm)
        print(
        f'Epoch {epoch + 1}, '
        f'Loss: {train_loss.result()}, '
        f'acc: {acc * 100}, '
        )

def test_noise():
        for i in range(1,21):
            cm=np.zeros((2,2))
            num=i/100
            test_ds = tf.data.experimental.make_csv_dataset(
                str(num)+'.csv',
                batch_size=2048,
                column_names=['text','labels'],
                num_epochs=1,
                ignore_errors=True, 
                column_defaults=[tf.string,tf.int32],
            )

            for feature in test_ds:
                    cm+=test_step_noise(feature)
            for label in range(2):
                recall=cm[label][label]/np.sum(cm[label,:])
                precision=cm[label][label]/np.sum(cm[:,label])
                f1=2*recall*precision/(recall+precision)
                print(
                    f'{recall:.4f}'
                    f'{precision:.4f}'
                    f'{f1:.4f}'
                )
            
            acc=(cm[0][0]+cm[1][1])/np.sum(cm)
            print(f'Confusion matrix: {cm},')
            print(
                f'Accuracy: {acc:.4f}'
            )

def test_clinical_ds():
        cm=np.zeros((2,2))
        clinical_ds = tf.data.experimental.make_csv_dataset(
            './real.csv',
            batch_size=2048,
            column_names=['text','reverse','labels'],
            num_epochs=1,
            ignore_errors=True, 
            column_defaults=[tf.string,tf.string,tf.int32],
        
        )
        for feature in tqdm(clinical_ds):
            cm+=test_step_clinical(feature)
        for label in range(2):
            recall=cm[label][label]/np.sum(cm[label,:])
            precision=cm[label][label]/np.sum(cm[:,label])
            f1=2*recall*precision/(recall+precision)
            print(
                    f'{recall:.4f}'
                    f'{precision:.4f}'
                    f'{f1:.4f}'
                )
        acc=(cm[0][0]+cm[1][1])/np.sum(cm)
        print(f'Confusion matrix: {cm},')
        print(
        f'Accuracy: {acc * 100}, '
        )

test_noise()
test_clinical_ds()
train()