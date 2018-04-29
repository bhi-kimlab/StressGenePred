# adjust multiple pvalues
# (by bonferroni method?)
import sys,os
import argparse
import pandas as pd
import numpy as np
#import statsmodels.stats.multitest as multi

# returns pandas.Series (pseudovalue)
def pval_chi(df):
    return pd.Series(-2*np.sum(np.log(df+0.00001), axis=1), index=df.index)

def main():
    parser=argparse.ArgumentParser(usage='''[pvalue df: sampleid x geneid] [label df: sampleid x stress]''')
    parser.add_argument('dfpath', type=str)
    parser.add_argument('--label_file', type=str)
    '''
    parser.add_argument('--method', type=str,
            choices=["holm", "hochberg", "hommel", "bonferroni", "BH", "BY", "fdr", "none"],
            default="bonferroni")
    '''
    parser.add_argument('-o', '--output', type=str)
    args = parser.parse_args()

    df = pd.read_csv(args.dfpath, index_col=0)

    # split for each label 
    # if not given, then do it for all columns
    if (not args.label_file):
        df_r = pd.DataFrame(pval_chi(df), index=df.index, columns=['all',])
    else:
        df_label = pd.read_csv(args.label_file, index_col=0)
        # fit order of samples
        df = df.reindex(columns=df_label.index)
        df_r = pd.DataFrame()
        for c in df_label.columns.tolist():
            # get specific samples
            b_samples = df_label[df_label[c]==1]
            df_cond = df[b_samples.index]
            df_r[c] = pval_chi(df_cond)

    # save result
    df_r.to_csv( args.output )



if (__name__ == '__main__'):
    main()
