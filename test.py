from TSData import TSData, TSTools
import numpy as np
import pandas as pd
import tensorflow as tf
import argparse
from model_v21 import Model


# append real prediction & entropy information
# to pred dataframe from Learner_Predictor
def format_pred(df_pred, df_true):
    # rename dataframe and concatenate
    df_true = df_true.copy()
    df_true.columns = [x+'_true' for x in df_true.columns.tolist()]
    df_r = pd.concat( (df_pred, df_true), axis=1)

    # add one more column: entropy...
    col_entropy = -np.sum(df_pred * np.log(df_pred), axis=1)
    df_r['entropy'] = col_entropy

    return df_r


def main():
    parser = argparse.ArgumentParser()
    # inputs
    parser.add_argument("file_expr",
            help="sample-expression matrix csv file")
    parser.add_argument("file_label",
            help="sample-label matrix csv file")
    parser.add_argument("--load", type=str, default="output/model_v2",
            help="load result from tensorflow-restorable form")
    parser.add_argument("--save", type=str, default="output/model_v2_predict",
            help="save test result in csv file")
    parser.add_argument("--raw", action='store_true',
            help="don't use predict(filter) model, only use generative model.")
    # parameters
    args = parser.parse_args()

    # load datas
    df_expr = pd.read_csv(args.file_expr, index_col=0)
    df_label = pd.read_csv(args.file_label, index_col=0)
    """
    if (args.label_index):
        labels = args.label_index.split(',')
        df_label = df_label.reindex(index=labels, fill_value=0)
    """

    # now set environments and import
    model = Model()
    model.init(df_expr.shape[0], df_label.shape[1])

    with tf.Session() as sess:
        model.init_sess(sess)
        # first restore model
        model.restore(sess, args.load+'.ckpt')
        # do prediction
        if (args.raw):
            mat_pred = sess.run(model.learner.predict(),
                feed_dict={model.learner.features: df_expr.transpose()})
        else:
            mat_pred = model.predict(sess, df_expr.transpose())

        # format pred dataframe
        df_pred = pd.DataFrame(mat_pred, columns=df_label.columns, index=df_expr.columns)
        df_pred = format_pred(df_pred, df_label)
        # save result
        df_pred.to_csv(args.save+'.csv')

        # report learning result
        pred_label_rowmax = np.reshape(np.max(mat_pred,axis=1), (-1,1))
        pred_label_logic = (mat_pred == pred_label_rowmax)*1
        pred_logic = np.all(np.equal(pred_label_logic, df_label), axis=1)
        pred_logic_true = np.sum(pred_logic*1)
        pred_logic_false = np.sum(np.logical_not(pred_logic)*1)
        print 'Learning result: True %d, False %d' % (pred_logic_true, pred_logic_false)


if __name__=='__main__':
    main()
