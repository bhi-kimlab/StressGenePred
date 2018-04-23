from TSData import TSData, TSTools
import model_v2
import tensorflow

# TODO: generate prediction-test code here.
model_rel = model_v2.SimplePredictor()
model_pred = model_v2.SimplePredictor()


def main():
    parser = argparse.ArgumentParser()
    # inputs
    parser.add_argument("file_expr",
            help="sample-expression matrix csv file")
    parser.add_argument("file_label",
            help="sample-label matrix csv file")
    parser.add_argument("--load", type=str, default="model_v2",
            help="load result from tensorflow-restorable form")
    parser.add_argument("--save", type=str, default="model_v2_test",
            help="save test result in csv file")
    # parameters
    args = parser.parse_args()

with tf.Session() as sess:
    # load datas
    df_expr = pd.DataFrame(args.file_expr, index_col=0)
    df_label = pd.DataFrame(args.file_label, index_col=0)
    # first restore model
    model_v2.restore()
    # then do prediction
    _input = tsl
    _pred_sig = sess.run(model_rel.predict(), feed_dict={
        model_rel.input:_input
        })
    _pred_sig_filtered = sess.run(model_pred.predict(), feed_dict={
        model_pred.input: _pred_sig
        })
    # TODO save result with formatted dataframe ...
    print _pred_sig


if __name__=='__main__':
    main()
