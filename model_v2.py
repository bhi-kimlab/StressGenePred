# -*- coding: utf-8 -*-
import tensorflow as tf
import numpy as np
import pandas as pd
from TSData import TSData, TSTools
import os,sys
from sklearn import datasets, linear_model
import time


epoch_count = 30000
use_batch = True
batch_size = 40
adam_gradient = 0.01
epoch_alert = 50



class Predictor(object):
    def __init__(self, _input, _output):
        dim = _input.shape
        self.input = tf.placeholder(dtype='float', shape=[None,None])
        self.output = tf.placeholder(dtype='float', shape=[None,None])
        self.w = tf.Variable(tf.fill( (1,dim[1]), 30.0 ))
        self.b = tf.Variable(tf.fill( (1,dim[1]), -0.3 ))
        self.saver = tf.train.Saver()

    def predict(self):
        p = tf.sigmoid(tf.multiply((self.input+self.b),self.w))
        return p / tf.reshape(tf.reduce_sum(p,axis=1),(-1,1))

    def loss(self):
        # KL divergence calculate for CMCL
        def CMCL_loss_v2():
            beta = 0.01  # parameter for confidence; high beta, low confidence (high entropy)
            p_y = self.predict()
            return -beta * tf.log(p_y)

        # 1. loss reduction (in case of major one)
        # 2. KL divergence (in case of minor one)
        label = self.output
        loss_e = label * (self.output-self.predict())**2
        KL_label = 1.0 - label
        loss_kl = KL_label * CMCL_loss_v2()
        loss = loss_e + loss_kl
        return loss

    def learn(self):
        return tf.train.AdamOptimizer(adam_gradient).minimize(self.loss())

    def save(self, sess, path="output/net_predict_v2.ckpt"):
        self.saver.save(sess, path)

    def restore(self, sess, path="output/net_predict_v2.ckpt"):
        self.saver.restore(sess, path)


class Generator(object):
    def __init__(self, datas):
        microarray_size, label_size = datas.size()
        self.trait_count = microarray_size         # columns (gene count)
        self.label_count = label_size              # columns (label dimension)
        self.datas = datas

        # make placeholder
        self.features = tf.placeholder(dtype='float', shape=[None,self.trait_count])
        self.label = tf.placeholder(dtype='float', shape=[None,self.label_count])
        self.f_input = tf.placeholder(dtype='float', shape=[None,self.trait_count])

        # main weights for each genes (projection)
        self.weight = tf.Variable(tf.random_normal( (self.trait_count,self.label_count) ))

        # constant for fixed variable
        self.ones = tf.ones( [self.trait_count,] )

        self.weight_lambda = 1    # unweight panelty --> panelty on no-prediction
        self.rate = 0.001
        self.saver = tf.train.Saver()

    def loss(self):
        weight = self.weight
        #weight = tf.nn.dropout( self.weight, 0.2 )
        features = tf.matmul(self.label, tf.transpose(weight))
        loss = tf.nn.sigmoid_cross_entropy_with_logits(labels=self.features,logits=features)
        # too big weight is okay? (TODO - add weight to loss?)
        return loss

    def learn(self):
        loss = self.loss()
        loss = tf.reduce_mean(loss) + tf.reduce_mean(tf.square(self.weight))*0.01

        # additional loss: prediction loss
        out = tf.matmul(self.features, self.weight)
        loss_predict = tf.nn.softmax_cross_entropy_with_logits(labels=self.label,logits=out)
        #loss += loss_predict

        return tf.train.AdamOptimizer(self.rate).minimize( loss )

    def predict(self):
        weight = tf.sigmoid(self.weight)
        # average weight in column, to reduce weight of gene with multiplicate-feature.
        # set weight to each gene
        features = self.features / tf.square(tf.reduce_sum(weight, axis=1))
        #features = self.features
        #weight = weight / tf.reshape(tf.reduce_sum(weight, axis=1), (-1,1))     # TODO: should square?
        #weight = tf.square(weight)
        out = tf.matmul(features, weight)
        out = out / tf.reshape(tf.reduce_sum(out, axis=1), (-1,1))
        return out

    def save(self, sess, path="output/net_v2.ckpt"):
        self.saver.save(sess, path)

    def restore(self, sess, path="output/net_v2.ckpt"):
        self.saver.restore(sess, path)

    # (DEPRECIATED)
    def save_calculated(save, sess, features, path):
        # save calculated(predicted) result
        labels_calc = sess.run(self.predict(), feed_dict={
            self.features: features})
        labels_encoded = self.datas.labels
        #n_values = np.max(labels_encoded) + 1
        #labels_encoded = np.eye(n_values)[labels_encoded]
        #label_out = np.stack( (labels_calc, self.datas.labels), axis=1)
        print labels_calc.shape
        print labels_encoded.shape
        label_out = np.concatenate( (labels_calc, labels_encoded), axis=1)
        cols = []
        for n in ["predict","real"]:
            for n2 in self.datas.label_names:
                cols.append("%s_%s" % (n,n2))
        df_label = pd.DataFrame(data=label_out, index=self.datas.sample_names, columns=cols)
        df_label.to_csv("output/net_predict_v2.csv")
    
# --------------------------



learner = Generator(datas)
op_learn = learner.learn()
op_decode = learner.decode()
op_loss = learner.loss()

learner_pred = Predictor(_input, _output)
op_learn_pred = learner_pred.learn()



def init(sess):
    init = tf.global_variables_initializer()
    sess.run(init)



def learn(sess):
    print 'learning generative ...'
    for i in range(epoch_count):
        # adaptive epoch rate?
        if (i < 100):
            learner.rate = 0.01
        elif (i < 1000):
            learner.rate = 0.001

        if (use_batch):
            batch_microarrays, batch_labels = datas.batch(batch_size)
        else:
            batch_microarrays, batch_labels = dd

        sess.run(op_learn, feed_dict={
            learner.features: batch_microarrays,
            learner.label: batch_labels
            })

        if (i%epoch_alert == 0):
            loss = sess.run(op_loss, feed_dict={
                learner.features: batch_microarrays,
                learner.label: batch_labels
                })
            print 'epoch: %d, loss: %f' % (i, np.mean(loss))

    print 'learning predictor ...'
    df = sess.run(op_decode, feed_dict={
        learner.features: batch_microarrays,
        learner.label: batch_labels
        })
    _input = df.iloc[:,0:4]
    _output = df.iloc[:,4:8]
    for _ in range(5000):
        sess.run(op_learn_pred, feed_dict={
            learner_pred.input: _input,
            learner_pred.output: _output
            })
        learner.save(sess, df)


def save(sess):
    learner.save(sess)

def restore(sess):
    learner.restore(sess)


def get_result(sess):
    # save weight of generator
    weight = sess.run(self.weight)
    weight = 1/(1+np.exp(-weight))          # recalculate weight to sigmoid (Easy-to-interpret)
    weight = np.reshape(weight, (-1, 8))    # gather up / down at one row
    net = weight
    #loss = sess.run(self.loss(), )
    #net = np.concatenate( (weight,bias), axis=1 )


    df = pd.DataFrame(data=net, index=idx, columns=cols)
    # save weight
    w = sess.run(self.w)
    b = sess.run(self.b)
    cols = ['heat','salt','drou','cold']
    df = pd.DataFrame(data=np.concatenate((w,b),axis=0), index=['weight','b'], columns=cols)
    df.to_csv("output/net_predict_weight_v2.csv")



    # save weight of prediction
    p = self.predict()
    labels_calc = sess.run(p/tf.reshape(tf.reduce_sum(p,axis=1),(-1,1)), feed_dict={
        self.input: self._input
        })
    print df
    print self._input
    print labels_calc
    print sess.run(self.loss(), feed_dict={
        self.input: self._input,
        self.output: self._output
        })
    #print np.multiply(self._input, ([[2,2,2,2],])) + np.tile(np.array([[1,1,1,1],]), [92,1])
    label_out = np.concatenate( (labels_calc, self._output), axis=1)
    df_label = pd.DataFrame(data=label_out, index=_df.index, columns=_df.columns)

    # add one more column: entropy...
    col_entropy = -np.sum(labels_calc * np.log(labels_calc), axis=1)
    df_label['entropy'] = col_entropy

    df_label.to_csv("output/net_predict_v2_filter.csv")

    return {
        'generator': df_generator,
        'predictor': df_predictor
        }
