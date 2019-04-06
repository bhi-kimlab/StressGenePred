# -*- coding: utf-8 -*-
import tensorflow as tf
import numpy as np
import pandas as pd
from TSData import TSData, TSTools
import os,sys
from sklearn import datasets, linear_model
import time




class Predictor(object):
    def __init__(self, n_label, rate):
        self.input = tf.placeholder(dtype='float', shape=[None,None])       # ( 40000, 4 )
        self.output = tf.placeholder(dtype='float', shape=[None,None])      # ( werwer )
        self.w = tf.Variable(tf.fill( (1,n_label), 20.0 ))                  # ( 4 )
        self.b = tf.Variable(tf.fill( (1,n_label), -0.2 ))
        # parameter for confidence; high beta, low confidence (high entropy)
        self.beta = 0.01
        self.rate = rate

    # set effective start value (for sigmoid)
    def assign_b(self, sess, v):
        sess.run(tf.assign(self.b, v))

    def predict(self):
        # average by weight (divide by expressing feature count)
        #f_count = tf.reduce_sum(self.input)
        p = tf.sigmoid(tf.multiply(self.input + self.b, self.w))
        return p
        #return p / tf.reshape(tf.reduce_sum(p,axis=1),(-1,1))               # normalization(sum to 1) by row

    def assign_bias(self, b):
        return tf.assign(self.b, b)

    # use CMCL loss
    def loss(self):
        loss_softmax = False    # use softmax instead of CMCL

        if (loss_softmax):
            pred = self.predict()
            return tf.nn.softmax_cross_entropy_with_logits(labels=self.output, logits=pred)
        else:
            # KL divergence calculate for CMCL
            def CMCL_loss_v2():
                p_y = self.predict()
                return -self.beta * tf.log(p_y)

            # to calculate loss, make predict result normalized in row
            pred = self.predict()
            pred = pred / tf.reshape(tf.reduce_sum(pred, axis=1), (-1,1))

            # 1. loss reduction (in case of major one)
            # 2. KL divergence (in case of minor one)
            label = self.output
            loss_e = label * (self.output-pred)**2
            KL_label = 1.0 - label
            loss_kl = KL_label * CMCL_loss_v2()
            loss = tf.reduce_sum(loss_e + loss_kl)
            return loss

    def learn(self):
        return tf.train.AdamOptimizer(self.rate).minimize(self.loss())
    def learn_b(self):
        return tf.train.AdamOptimizer(self.rate).minimize(self.loss(), var_list=[self.b,])


class Generator(object):
    def __init__(self, n_features, n_label, rate):
        self.n_features = n_features
        self.n_label = n_label

        # make placeholder
        self.features = tf.placeholder(dtype='float', shape=[None,n_features])
        self.label = tf.placeholder(dtype='float', shape=[None,n_label])

        # main weights for each genes (projection)
        self.weight = tf.Variable(tf.random_normal( (n_features, n_label) ))

        self.rate = rate

    def loss(self):
        weight = self.weight
        features = tf.matmul(self.label, tf.transpose(weight))  # pred_features
        loss_pred_square = (tf.sigmoid(features) - self.features) ** 2
        loss_pred_sigmoid = tf.nn.sigmoid_cross_entropy_with_logits(labels=self.features,logits=features)
        #loss_pred_softmax = tf.nn.softmax_cross_entropy_with_logits(labels=self.label,logits=features)
        loss_weight = tf.square(self.weight)
        # integrated loss (relate to other stresses)
        # TODO: reduce it's dimension, if updown is true?
        updown = True
        if (updown):
            weight_feature = tf.reshape(self.weight, (self.n_features/2,-1))
        else:
            weight_feature = self.weight
        loss_integrate = tf.reduce_sum(tf.sigmoid(weight_feature), axis=1)**2
        return tf.reduce_mean(loss_pred_sigmoid) + tf.reduce_mean(loss_integrate)*0.06

    def learn(self):
        loss = self.loss()
        return tf.train.AdamOptimizer(self.rate).minimize( loss )

    def predict(self):
        # calculate out probability
        # (out : reverse calculated; [sample x label])
        weight = tf.sigmoid(self.weight)
        out = tf.matmul(self.features, weight)
        # divide by sum of feature
        out = out / tf.reshape(tf.reduce_sum(self.features, axis=1) + 0.000001, (-1,1))
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

class Model(object):
    def __init__(self):
        self.epoch_count = 30000
        self.epoch_pred_count = 1000
        self.use_batch = True
        self.batch_size = 40
        self.learning_rate = 0.001
        self.epoch_alert = 50
        self.model_name = "model_v2"
        self.only_predictor = False


    def init(self, cnt_feature, cnt_label):
        # ??
        if (self.use_batch):
            cnt_sample = self.batch_size
        else:
            cnt_sample = 0

        self.learner = Generator(cnt_feature,cnt_label,self.learning_rate)
        self.op_learn = self.learner.learn()
        self.op_decode = self.learner.predict()
        self.op_loss = self.learner.loss()

        self.learner_pred = Predictor(cnt_label,self.learning_rate*10)
        self.op_learn_pred = self.learner_pred.learn()
        self.op_learn_pred_b = self.learner_pred.learn_b()

        self.saver = tf.train.Saver()


    def save(self, sess, path):
        self.saver.save(sess, path)

    def restore(self, sess, path):
        self.saver.restore(sess, path)

    def init_sess(self, sess):
        init = tf.global_variables_initializer()
        sess.run(init)



    # no return
    def learn(self, sess, df_expr, df_label):
        learner = self.learner
        learner_pred = self.learner_pred
        learning_rate = self.learning_rate
        batch_size = self.batch_size
        use_batch = self.use_batch
        op_learn = self.op_learn
        op_decode = self.op_decode
        op_loss = self.op_loss

        print 'learning generative ...'
        if (self.only_predictor):
            print 'skip learning predictor due to option'
        else:
            for i in range(self.epoch_count):
                # adaptive epoch rate?
                if (i < 100):
                    learner.rate = learning_rate*10
                elif (i < 1000):
                    learner.rate = learning_rate

                if (use_batch):
                    batch_microarrays, batch_labels = TSTools.batch(batch_size, (df_expr.values, df_label.values), dim=0)
                else:
                    batch_microarrays, batch_labels = df_expr, df_label

                sess.run(op_learn, feed_dict={
                    learner.features: batch_microarrays,
                    learner.label: batch_labels
                    })

                if (i % self.epoch_alert == 0):
                    loss = sess.run(op_loss, feed_dict={
                        learner.features: batch_microarrays,
                        learner.label: batch_labels
                        })
                    print 'epoch: %d, loss: %f' % (i, np.mean(loss))

        mat_gen = sess.run(op_decode, feed_dict={
            learner.features: df_expr
            })

        # for debugging
        print np.hstack( (mat_gen, df_label) )

        print np.mean(mat_gen, axis=0)
        sess.run(learner_pred.assign_bias(-np.reshape(np.mean(mat_gen, axis=0), (1,-1)) ))

        print 'learning predictor ...'
        """
        for _ in range(100):
            sess.run(self.op_learn_pred_b, feed_dict={
                learner_pred.input: mat_gen,
                learner_pred.output: df_label
                })
        """
        for _ in range(self.epoch_pred_count):
            sess.run(self.op_learn_pred, feed_dict={
                learner_pred.input: mat_gen,
                learner_pred.output: df_label
                })


    # returns 'numpy' matrix
    def predict(self, sess, df_expr):
        learner = self.learner
        learner_pred = self.learner_pred


        mat_gen = sess.run(learner.predict(), feed_dict={
            learner.features: df_expr
            })
        mat_pred = sess.run(learner_pred.predict(), feed_dict={
            learner_pred.input: mat_gen
            })
        return mat_pred




    # return readable result (dataframe)
    def get_result(self, sess, df_expr, df_label):
        learner = self.learner
        learner_pred = self.learner_pred

        # save weight of generator
        weight = sess.run(learner.weight)
        weight = 1/(1+np.exp(-weight))          # recalculate weight to sigmoid (Easy-to-interpret)
        df_generator = pd.DataFrame(data=weight, index=df_expr.index, columns=df_label.columns)


        # save weight
        w = sess.run(learner_pred.w)
        b = sess.run(learner_pred.b)
        df_predictor = pd.DataFrame(data=np.concatenate((w,b),axis=0),
                index=['weight','b'],
                columns=df_label.columns)
        #beta = sess.run(learner_pred.beta)
        #df_predictor['beta'] = [beta,0]


        return {
            'generator': df_generator,
            'predictor': df_predictor
            }
