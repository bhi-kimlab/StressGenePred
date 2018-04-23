import pandas as pd
import sys,os,argparse
import tensorflow as tf


# TODO
# df: weight dataframe
def reduce_df_dim(df_weight):
    # reduce index and generate new dataframe with up/down column
    col_cnt = df_weight.shape[1]
    mat_weight = np.reshape(df_weight, (-1, col_cnt*2))    # gather up / down at one row
    idx = [ x.split('_',1)[0] for x in self.datas.index[0::2] ]
    cols = []
    for n in ["up", "down"]:
        for n2 in df_weight.columns.tolist():
            cols.append("%s_%s" % (n,n2))
    df = pd.DataFrame(mat_weight, columns=cols, index=idx)

    # regenerate max value to new stress value
    for l in cols:
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
    parser.add_argument("--label_index", type=str,
            help="do when label reindexing is necessary (separated by comma)")
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
    df_expr = pd.DataFrame(args.file_expr, index_col=0)
    df_label = pd.DataFrame(args.file_label, index_col=0)
    if (args.label_index):
        labels = args.label_index.split(',')
        df_label = df_label.reindex(index=labels, fill_value=0)

    # now set environments and import
    os.environ['CNT_FEATURE'] = str(df_expr.shape[0])
    os.environ['CNT_SAMPLE'] = str(df_expr.shape[1])
    os.environ['CNT_LABEL'] = str(df_label.shape[0])
    import model_v2 as model

    saver = tf.train.Saver()
    with tf.Session() as sess:
        # do learning
        model.epoch_count = args.epoch_count
        model.use_batch = args.batch_size > 0
        model.batch_size = args.batch_size
        model.learning_rate = args.rate
        model.init(sess)
        model.learn(sess)

        # save result if necessary
        if (args.save):
            saver.save(sess, args.save+'.ckpt')
        if (args.save_csv):
            r = model_v2.get_result(sess)   # returns dict
            if (args.reduce_updown):
                r['generative'] = reduce_df_dim(r['generative'])
            if (args.reopti_stress):
                r['generative'] = stress_reduce_to_others(r['generative'], df_label.index.tolist())
            for k,df in r.items():
                df.to_csv("%s_%s.csv" % (args.save_csv, k))


if __name__=='__main__':
    main()
