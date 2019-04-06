import numpy as np
import pandas as pd
from TSData import TSData, TSTools
from sklearn.decomposition import PCA
import scipy.stats
import os, sys, argparse


def tsd_generate(fp):
    TSTools.ConvertCSVtoTSDs(fp)

def tsd_readmatrix(tsl):
    TSTools.FillGenematrix(tsl, sep='\t')

def tsd_compile(tsl, toolname, outfn, args):
    pvalue = args.pvalue
    logtrans = args.logtrans
    updown = args.updown
    israw = args.raw
    reindex_fp = args.reindex_file
    reindex_df_fp = args.reindex_df

    # is analyze toolname is gexp analysis?
    is_gexp = False
    # TODO RAW_ALL, RAW_AVG, ...?
    if (toolname == "label"):
        r = TSTools.modelmatrix(tsl.get_labels())
        if (args.label_index):
            r = r.reindex(columns= args.label_index.split(','),fill_value=0)
    elif (toolname == "timelabel"):
        r = TSTools.modelmatrix(tsl.get_labels_per_time())
    elif (toolname == "DESeq"):
        TSTools.activateR()
        r = TSTools.do_DESeq(tsl)
        is_gexp = True
    elif (toolname == "FC"):
        r = TSTools.do_DEG_FC(tsl, 0.8)
        is_gexp = True
    elif (toolname == "limma"):
        TSTools.activateR()
        r = TSTools.do_limma(tsl, pval_mode='P.Value')
        is_gexp = True
    elif (toolname == "limma_adjp"):
        TSTools.activateR()
        r = TSTools.do_limma(tsl, pval_mode='adj.P.Val')
        is_gexp = True
    elif (toolname == "timepoint"):
        r = tsl.get_timepoints()
    elif (toolname == "timeDEG"):
        r = TSTools.do_DEG_per_time(tsl)
        is_gexp = True

    # these processes are only works when data is gene-expression data.
    if (is_gexp):
        if (israw):
            for k,v in r.items():
                # change filename if dict
                outfn_s = outfn.rsplit('.',2)
                if (len(outfn_s)==1):
                    outfn_s.append('')
                outfn_n = '.'.join((outfn_s[0], k, outfn_s[1], 'csv'))
                v.to_csv(outfn_n)

        pval = r['pvalue']
        tval = r['tvalue']
        # pvalue to logical value (from threshold)
        if (pvalue > 0):
            b_true = pval <= pvalue
            b_false = pval > pvalue
            print np.sum(b_true)
            if (args.ensure_count or args.max_count):
                # check for each column
                # is there are enough p-value marked genes.
                # if not, reset custom p-value for such column.
                for c in pval.columns.tolist():
                    pval_mark_cnt = np.sum(b_true[c])
                    if (args.ensure_count and pval_mark_cnt < args.ensure_count):
                        new_pvalue = sorted(pval[c].tolist())[args.ensure_count]
                        b_true[c] = (pval[c] <= new_pvalue)
                        b_false[c] = (pval[c] > new_pvalue)
                        print 'Column %s pvalue had been reseted to %.3f, due to valid element count was %d.' % (c,new_pvalue,pval_mark_cnt)
                    elif (args.max_count and pval_mark_cnt > args.max_count):
                        new_pvalue = sorted(pval[c].tolist())[args.max_count]
                        b_true[c] = (pval[c] <= new_pvalue)
                        b_false[c] = (pval[c] > new_pvalue)
                        print 'Column %s pvalue had been reseted to %.3f, due to valid element count was %d.' % (c,new_pvalue,pval_mark_cnt)
            pval[b_true] = 1
            pval[b_false] = 0
        # pvalue log transform
        if (args.logtrans):
            pval = np.log(pval)

        # apply tvalue sign to pvalue
        pval[tval < 0] *= -1        

        # do reindex in case if necessary
        # if no file had been specified, then use tsl[0].df.index
        if (reindex_fp):
            with open(reindex_fp,'r') as f:
                idx = f.read().strip().split('\n')
            pval = pval.reindex(index=idx, fill_value=0)
        elif (reindex_df_fp):
            idx = pd.read_csv(reindex_df_fp, index_col=0).index
            pval = pval.reindex(index=idx, fill_value=0)

        # separate up/down signal
        if (args.updown):         # split sign
            pval_n = np.empty((pval.shape[0]*2, pval.shape[1]), dtype=np.int32)
            pval_n[0::2] = np.clip(pval, 0, 1)
            pval_n[1::2] = -np.clip(pval, -1, 0)
            n_idx = []
            for i in pval.index:
                n_idx.append(i+'_up')
                n_idx.append(i+'_down')
            df_pval_n = pd.DataFrame(pval_n, columns=pval.columns, index=n_idx)
            pval = df_pval_n
            print 'pval...?'
        r = pval

    print outfn
    r.to_csv(outfn)

# batch utility
def batch(np_arr, batch_size, batch_count):
    r = []
    batch_indexes = np.random.choice(batch_count, size=batch_size)
    for np in np_arr:
        r.append(np[batch_indexes])
    return r

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("task", type=str,
            choices=["tsd", "read", "compile", "gen_genelist"],
            help="tsd : generate tsd file from csv\nread : read matrix of tsd files\ncompile : generate compiled dataset\ngen_genelist : generate genelist file from tsd file")
    parser.add_argument("-f", "--file",
            help="target file")
    parser.add_argument("-d", "--folder",
            help="target folder (may include target files)")
    parser.add_argument("--filters",
            help="set filter on loading files (ex: Species:A,MinRepCnt:2)")
    parser.add_argument("--updown", action='store_true',
            help="use sign separation(up/down) when compile")
    parser.add_argument("--pvalue", type=float, default=0,
            help="used p-value for compiling (optional; saved in raw value if not specified)")
    parser.add_argument("-t", "--target", type=str,
            help="target path (used for compling)")
    parser.add_argument("--logtrans", action='store_true',
            help="do p-value log transform when compile")
    parser.add_argument("--raw", action='store_true',
            help="save all data when compile (pvalue and tvalue)")
    parser.add_argument("--ensure_count", type=int,
            help="ensures p-value cutoff elements at least given number (by descending pvalue)")
    parser.add_argument("--max_count", type=int,
            help="ensures p-value cutoff elements at most given number (by descending pvalue)")
    parser.add_argument("--reindex_file", type=str,
            help="reindexing file, separated with \N.")
    parser.add_argument("--reindex_df", type=str,
            help="reindexing file, which has format of *.csv (DataFrame)")
    parser.add_argument("--label_index", type=str,
            help="do reindexing on compiling labels (separate with comma)")
    parser.add_argument("--tool", type=str, action="append",
            choices=["limma","limma_adjp","DESeq","timeDEG","timepoint","timelabel","FC"],
            help="used tool/method for compiling (multiple tools available)")
    args = parser.parse_args()

    # prepare for arguments
    print 'preparing files ...'
    tsl = TSTools.TSLoader()
    if (args.filters):
        for f in args.filters.split(','):
            k,v = f.split(':',2)
            tsl.addfilter(k,v)
    if (args.folder):
        tsl.loadpath(args.folder)
    if (args.file):
        tsl.append(TSData.load(args.file))
    print 'total %d timeseries files' % len(tsl)       

    # start task
    if (args.task == "tsd"):
        tsd_generate(args.file)
    elif (args.task == "read"):
        tsd_readmatrix(tsl)
    elif (args.task == "gen_genelist"):
        with open(args.target+'.txt','w') as f:
            f.write('\n'.join(tsl[0].df.index.tolist()))
    elif (args.task == "compile"):
        # each task is composed with 'small-tasks'
        # column: series-timepoint or series
        # row: differs from task (design matrix: stress(label), gexp analyze: genenames)
        tasks = {
            'limma': ['label', 'limma'],
            'limma_adjp': ['label', 'limma_adjp'],
            'DESeq': ['label', 'DESeq'],
            'FC': ['label', 'FC'],
            'timeDEG': ['timelabel', 'timepoint', 'timeDEG']
            }
        for toolname in args.tool:
            for t in tasks[toolname]:
                print 'compling %s...' % t
                tsd_compile(tsl, t, os.path.join(args.target+'.'+t+'.csv'), args)

if __name__=='__main__':
    main()
