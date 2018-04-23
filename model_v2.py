# -*- coding: utf-8 -*-
import tensorflow as tf
import numpy as np
import pandas as pd
from TSData import TSData, TSTools
import os,sys
from sklearn import datasets, linear_model
import time


epoch_count = 30000
epoch_pred_count = 5000
use_batch = True
batch_size = 40
learning_rate = 0.001
epoch_alert = 50
model_name = "model_v2"


class Predictor(object):
    def __init__(self, n_label):
        self.init(n_label)

    def init(self, n_label):
        self.input = tf.placeholder(dtype='float', shape=[None,None])
        self.output = tf.placeholder(dtype='float', shape=[None,None])
        self.w = tf.Variable(tf.fill( (1,n_label), 30.0 ))
        self.b = tf.Variable(tf.fill( (1,n_label), -0.3 ))

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
        return tf.train.AdamOptimizer(learning_rate).minimize(self.loss())


class Generator(object):
    def __init__(self, n_features, n_label):
        self.init(n_features, n_label)

    def init(self, n_features, n_label):
        # make placeholder
        self.features = tf.placeholder(dtype='float', shape=[None,n_features])
        self.label = tf.placeholder(dtype='float', shape=[None,n_label])

        # main weights for each genes (projection)
        self.weight = tf.Variable(tf.random_normal( (n_features, n_label) ))

        self.weight_lambda = 1    # unweight panelty --> panelty on no-prediction
        self.rate = learning_rate

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
        ## average weight in column, to reduce weight of gene with multiplicate-feature.
        weight = tf.sigmoid(self.weight)
        #weight = weight / tf.reshape(tf.reduce_sum(weight, axis=1), (-1,1)) 
        #weight = tf.square(weight)

        ## feature input
        features = self.features
        #features = self.features / tf.square(tf.reduce_sum(weight, axis=1))

        out = tf.matmul(features, weight)
        out = out / tf.reshape(tf.reduce_sum(out, axis=1), (-1,1))  # normalize output
        return out

    # (DEPRECIATED)
    def save_calculated(save, sess, features, path="output/net_predict_v2.csv"):
        # save calculated(predicted) result
        labels_calc = sess.run(self.predict(), feed_dict={
            self.features: features})
        # design matrix (real result)
        labels_encoded = self.datas.labels
        print labels_calc.shape
        print labels_encoded.shape
        label_out = np.concatenate( (labels_calc, labels_encoded), axis=1)
        cols = []
        for n in ["predict","real"]:
            for n2 in self.datas.label_names:
                cols.append("%s_%s" % (n,n2))
        df_label = pd.DataFrame(data=label_out, index=self.datas.sample_names, columns=cols)
        df_label.to_csv(path)
    
# --------------------------


cnt_feature = int(os.environ['CNT_FEATURE'])
cnt_label = int(os.environ['CNT_LABEL'])

learner = Generator(100,4)
op_learn = learner.learn()
op_decode = learner.predict()
op_loss = learner.loss()

learner_pred = Predictor(4)
op_learn_pred = learner_pred.learn()



def init(sess):
    init = tf.global_variables_initializer()
    sess.run(init)



# no return
def learn(sess, df_expr, df_label):
    print 'learning generative ...'
    for i in range(epoch_count):
        # adaptive epoch rate?
        if (i < 100):
            learner.rate = learning_rate*10
        elif (i < 1000):
            learner.rate = learning_rate

        if (use_batch):
            batch_microarrays, batch_labels = TSTools.batch(batch_size, zip(df_expr, df_label))
        else:
            batch_microarrays, batch_labels = df_expr, df_label

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
    mat_gen = sess.run(op_decode, feed_dict={
        learner.features: df_expr
        })

    print 'learning predictor ...'
    for _ in range(epoch_pred_count):
        sess.run(op_learn_pred, feed_dict={
            learner_pred.input: mat_gen,
            learner_pred.output: df_label
            })
        learner.save(sess, df)


# returns 'numpy' matrix
def predict(sess, df_expr):
    mat_gen = sess.run(learner.predict(), feed_dict={
        learner.input: df_expr
        })
    mat_pred = sess.run(learner_pred.predict(), feed_dict={
        learner_pred.input: mat_gen
        })
    return mat_pred




# return readable result (dataframe)
def get_result(sess, df_exp, df_label):
    # save weight of generator
    weight = sess.run(learner.weight)
    weight = 1/(1+np.exp(-weight))          # recalculate weight to sigmoid (Easy-to-interpret)
    df_generator = pd.DataFrame(data=weight, index=df_exp.index, columns=df_label.index)


    # save weight
    w = sess.run(learner_pred.w)
    b = sess.run(learner_pred.b)
    df_predictor = pd.DataFrame(data=np.concatenate((w,b),axis=0),
            index=['weight','b'],
            columns=df_label.index)


    return {
        'generator': df_generator,
        'predictor': df_predictor
        }
