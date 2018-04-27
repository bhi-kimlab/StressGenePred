import pandas as pd
import numpy as np
import sys,os,argparse
import tensorflow as tf
from model_v21 import Model


# TODO
# df: weight dataframe
def reduce_df_dim(df_weight):
    # reduce index and generate new dataframe with up/down column
    cols_orig = df_weight.columns.tolist()
    col_cnt = df_weight.shape[1]
    mat_weight = np.reshape(df_weight.values, (-1, col_cnt*2))    # gather up / down at one row
    idx = [ x.split('_',1)[0] for x in df_weight.index.tolist()[0::2] ]
    cols = []
    for n in ["up", "down"]:
        for n2 in df_weight.columns.tolist():
            cols.append("%s_%s" % (n,n2))
    df = pd.DataFrame(mat_weight, columns=cols, index=idx)

    # regenerate max value to new stress value
    for l in cols_orig:
        is_up_deg = (df["up_%s"%l] > df["down_%s"%l])
        new_col = is_up_deg * df["up_%s"%l] + np.logical_not(is_up_deg) * df["down_%s"%l]
        df[ l ] = new_col             # non-scaled related to other stresses

    return df


# in case of updown data,
# MUST process reduce_df_dim first to run this
def stress_reduce_to_others(df, labels):
    col_sum = np.sum([df["%s_ns"%l] for k in labels],axis=0)
    for l in self.datas.label_names:
        df[ l ] = df[ "%s_ns" % l ] * df[ "%s_ns" % l ] / col_sum   # scaled related to other stresses
    return df




def main():
    parser = argparse.ArgumentParser()
    # inputs
    parser.add_argument("file_expr",
            help="sample-expression matrix csv file")
    parser.add_argument("file_label",
            help="sample-label matrix csv file")
    parser.add_argument("--reduce_updown", action="store_true",
            help="do reduce_max by 2 (when input data is separated updown)")
    parser.add_argument("--reopti_stress", action="store_true",
            help="do weight div by sum of row")
    parser.add_argument("--save", type=str, default="output/model_v2",
            help="save result in tensorflow-restorable form")
    parser.add_argument("--save_csv", type=str, default="output/model_v2",
            help="save result in readable form (csv)")
    # parameters
    parser.add_argument("--epoch_count", type=int, default=20000)
    parser.add_argument("--batch_size", type=int, default=40)
    parser.add_argument("--rate", type=float, default=0.01)
    args = parser.parse_args()

    # load & arrange files first
    df_expr = pd.read_csv(args.file_expr, index_col=0)
    df_label = pd.read_csv(args.file_label, index_col=0)
    """
    if (args.label_index):
        labels = args.label_index.split(',')
        df_label = df_label.reindex(columns=labels, fill_value=0)
    """

    # now set environments and import
    model = Model()
    model.epoch_count = args.epoch_count
    model.use_batch = args.batch_size > 0
    model.batch_size = args.batch_size
    model.learning_rate = args.rate
    model.init(df_expr.shape[0], df_label.shape[1])

    with tf.Session() as sess:
        # do learning
        model.init_sess(sess)
        model.learn(sess, df_expr.transpose(), df_label)

        # save result if necessary
        if (args.save):
            model.save(sess, args.save+'.ckpt')
        if (args.save_csv):
            r = model.get_result(sess, df_expr, df_label)   # returns dict
            if (args.reduce_updown):
                r['generator'] = reduce_df_dim(r['generator'])
            if (args.reopti_stress):
                r['generator'] = stress_reduce_to_others(r['generator'], df_label.index.tolist())
            for k,df in r.items():
                df.to_csv("%s_%s.csv" % (args.save_csv, k))


if __name__=='__main__':
    main()
