from TSData import TSData, TSTools
import model_v2
import tensorflow


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

    with tf.Session() as sess:
        # load datas
        df_expr = pd.DataFrame(args.file_expr, index_col=0)
        df_label = pd.DataFrame(args.file_label, index_col=0)
        # first restore model
        model_v2.restore(sess, args.load)
        # do prediction
        if (args.raw):
            df_pred = sess.run(model_v2.learner.predict(),
                feed_dict={model_v2.learner.features: df_expr})
        else:
            mat_pred = model_v2.predict(sess, df_expr)
        # format pred dataframe
        df_pred = pd.DataFrame(mat_pred, columns=df_label.index, index=df_expr.columns)
        df_pred = format_pred(df_pred, df_label)
        # save result
        df_pred.to_csv(args.save+'.csv')


if __name__=='__main__':
    main()
