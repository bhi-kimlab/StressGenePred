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
        self.input = tf.placeholder(dtype='float', shape=[None,None])
        self.output = tf.placeholder(dtype='float', shape=[None,None])
        self.w = tf.Variable(tf.fill( (1,n_label), 10.0 ))
        self.b = tf.Variable(tf.fill( (1,n_label), -0.3 ))
        # parameter for confidence; high beta, low confidence (high entropy)
        self.beta = 0.005 #tf.Variable(0.01)
        self.rate = rate
        self.target_max_entropy = 1.2

    def predict(self):
        p = tf.sigmoid(tf.multiply((self.input+self.b),self.w))
        #return p
        return p / tf.reshape(tf.reduce_sum(p,axis=1),(-1,1))


    def loss(self):
        # KL divergence calculate for CMCL
        def CMCL_loss_v2():
            
            p_y = self.predict()
            return -self.beta * tf.log(p_y)

        # 1. loss reduction (in case of major one)
        # 2. KL divergence (in case of minor one)
        label = self.output
        pred = self.predict()
        loss_e = label * (self.output-pred)**2
        KL_label = 1.0 - label
        loss_kl = KL_label * CMCL_loss_v2()
        """
        # 3. target entropy
        entropy = -tf.reduce_sum(pred * tf.log(pred), axis=1)
        loss_entropy = tf.nn.relu(tf.reduce_max(entropy)-self.target_max_entropy)
        """
        loss_entropy = 0
        loss = tf.reduce_sum(loss_e + loss_kl) + loss_entropy
        return loss

    def learn(self):
        return tf.train.AdamOptimizer(self.rate).minimize(self.loss())


class Generator(object):
    def __init__(self, n_features, n_label, rate):
        # make placeholder
        self.features = tf.placeholder(dtype='float', shape=[None,n_features])
        self.label = tf.placeholder(dtype='float', shape=[None,n_label])

        # main weights for each genes (projection)
        self.weight = tf.Variable(tf.random_normal( (n_features, n_label) ))

        self.rate = rate

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
        #weight = self.weight
        weight = tf.sigmoid(self.weight)
        #weight = weight / tf.reshape(tf.reduce_sum(weight, axis=1), (-1,1)) 
        #weight = tf.square(weight)

        ## feature input
        features = self.features
        #features = self.features / tf.square(tf.reduce_sum(weight, axis=1))

        out = tf.matmul(features, weight)
        #out = out / tf.reshape(tf.reduce_sum(out, axis=1) + 0.001, (-1,1))  # normalize output + pseudovalue   --> DON'T!
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
        self.epoch_pred_count = 5000
        self.use_batch = True
        self.batch_size = 40
        self.learning_rate = 0.001
        self.epoch_alert = 50
        self.model_name = "model_v2"


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
        op_learn_pred = self.op_learn_pred

        print 'learning generative ...'
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

        print 'learning predictor ...'
        for _ in range(self.epoch_pred_count):
            sess.run(op_learn_pred, feed_dict={
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
