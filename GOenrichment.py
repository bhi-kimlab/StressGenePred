#!/usr/bin/env python
import sys
import argparse
import scipy.stats as stats
import pandas as pd
import numpy as np



class GOenrichment(object):
    def __init__(self):
        #self.GO = []
        self.background_genes = set()
        self.cut_pvalue = 0
        self.cut_cnt = 0
        self.max_trait_cut = 0
        self.asc = True

    def load(self, path):
        dic_gene2trait={}
        dic_trait2gene={}
        dic_trait2ratio={}
        IF=open(path,'r')
        for line in IF:
            trait,genes=line.rstrip('\n').split('\t')
            lst_gene = genes.split(',')
            for gene in lst_gene:
                if len(self.background_genes) > 0 and gene not in self.background_genes:
                    continue
                if trait not in dic_trait2gene:
                    dic_trait2gene[trait]=set()
                if gene not in dic_gene2trait:
                    dic_gene2trait[gene]=set()
                dic_gene2trait[gene].add(trait)
                dic_trait2gene[trait].add(gene)

        # check threshold for trait
        if (self.max_trait_cut > 0):
            genes_to_removed = []
            for g,t in dic_gene2trait.items():
                if (len(t) > self.max_trait_cut):
                    genes_to_removed.append(g)
            print '%d genes will be removed by max_trait_cut option.' % len(genes_to_removed)
            for g in genes_to_removed:
                traits = dic_gene2trait[g]
                for t in traits:
                    dic_trait2gene[t].remove(g)
                del dic_gene2trait[g]

        # finalize
        for trait in dic_trait2gene.keys():
            dic_trait2ratio[trait]=float(len(dic_trait2gene[trait]))/len(dic_gene2trait)
        self.dic_gene2trait = dic_gene2trait
        self.dic_trait2gene = dic_trait2gene
        self.dic_trait2ratio = dic_trait2ratio

    def load_backgroundgenes(self, path):
        IF=open(path,'r')
        for line in IF:
            gene=line.rstrip('\n').split('\t',1)[0]
            self.background_genes.add(gene)

    def calculate_by_name(self, lst_gene, method='fisher'):
        dic_gene2trait = self.dic_gene2trait
        dic_trait2gene = self.dic_trait2gene
        dic_trait2ratio = self.dic_trait2ratio

        dic_trait2count={}
        set_tested_gene=set()
        for gid in lst_gene:
            if gid not in dic_gene2trait:
                continue
            for trait in dic_gene2trait[gid]:
                if not trait in dic_trait2count:
                    dic_trait2count[trait]=set()
                dic_trait2count[trait].add(gid)
            set_tested_gene.add(gid)
        rows = []
        for trait, set_gene in dic_trait2count.items():
            occured_in_tested, total_tested, occured_in_background, total_background = len(set_gene), len(set_tested_gene), len(dic_trait2gene[trait]), len(dic_gene2trait)
            if occured_in_tested == 0:
                pval=1.0
            else:
                if method == 'binomial':
                    pval=1.0-stats.binom.cdf(occured_in_tested-1, total_tested, dic_trait2ratio[trait])	# ovccured_in_test-1 means p(X>=n) i.e. contain
                elif method == 'fisher':
                    oddratio,pval=stats.fisher_exact([[occured_in_tested, total_tested-occured_in_tested], [occured_in_background-occured_in_tested, total_background-total_tested-occured_in_background+occured_in_tested]], alternative='greater')	# 2X2 fisher's exact test
            # only add geneset if necessary: [ ','.join(set_gene) ]
            rows.append([trait, pval, occured_in_tested, total_tested, occured_in_background, total_background])

        df_out = pd.DataFrame(rows, columns=['setid','pval','#occured_in_tested','#total_tested','#occured_in_background','#total_background'])

        # sort by pvalue column
        df_out.sort_values(by=['pval'], inplace=True)
        return df_out

    # method: fisher or binomial
    def calculate(self, lst_gene, method='fisher'):
        # filter out lst_genes
        # and change to value to genename
        lst_gene = lst_gene.sort_values(ascending=self.asc)
        if (self.cut_pvalue):
            lst_gene_new = []
            for i in lst_gene:
                if (i > self.cut_pvalue):
                    break
                lst_gene_new.append(i)
            lst_gene = pd.Series(lst_gene_new)
        if (self.cut_cnt):
            lst_gene = lst_gene[:self.cut_cnt]
        lst_gene = lst_gene.index.tolist()
        return self.calculate_by_name(lst_gene, method)




def main():
    parser=argparse.ArgumentParser(
            usage='''(path)GOenrichment (path)df_pvalue_table --column_name --count_cut or --pvalue_cut''')

    parser.add_argument('dfpath', metavar='str', help='dfpath including pvalue file')
    parser.add_argument('--trait2genes', metavar='str', default='GOBPname2gene.arabidopsis.txt', help='gene2subtype file')
    parser.add_argument('--column_name', metavar='str', help='column name including pvalue(separate by comma) (all columns are used if not specified)')
    parser.add_argument('-o', '--output', metavar='str', help='output to print result')
    parser.add_argument('--label_file', metavar='str', help='label to selectively calculate GOTerm')
    parser.add_argument('--max_trait_cut', type=int, default=0, help='use when filtering specific-reponsive gene by setting threshold for maximum trait count')
    parser.add_argument('--count_cut', type=int, default=0)
    parser.add_argument('--pvalue_cut', type=float, default=0)
    parser.add_argument('--descending', action='store_true')
    parser.add_argument('--method', required=False, metavar='[fisher|binomial]', default='fisher', help='method for statistical test')
    args = parser.parse_args()

    df = pd.read_csv(args.dfpath, index_col=0)

    # gather gene list to check
    if (not args.column_name):
        cols = df.columns.tolist()
    else:
        cols = args.column_name.split(',')

    # load trait2genes file and calculate GSEA for 'each' columns
    go = GOenrichment()
    go.asc = not args.descending
    go.cut_cnt = args.count_cut
    go.pvalue_cut = args.pvalue_cut
    go.max_trait_cut = args.max_trait_cut
    go.load(args.trait2genes)

    # output result file
    if args.output == 'stdout':
        OF=sys.stdout
    else:
        OF=open(args.output,'w')

    for c in cols:
        df_cond = df[c]
        OF.write(c+'\n')
        go.calculate(df_cond, args.method).to_csv(OF)
        OF.write('\n')


def main_old():
    parser=argparse.ArgumentParser(
            usage='''\
    %(prog)s [options] gene2subtype -trait2genes trait2genes 

    example: %(prog)s final.gene2phase.cold.txt -trait2genes GOBPname2gene.arabidopsis.txt -pcut 0.05 -topK None -o outfile.txt
    ''')

    parser.add_argument('gene2subtype', metavar='str', help='gene2subtype file')
    parser.add_argument('-trait2genes', required=False, metavar='str', default='GOBPname2gene.arabidopsis.txt', help='trait2genes file')
    parser.add_argument('-backgroundGenes', required=False, metavar='str', default='None', help='allgenes in first column file')
    parser.add_argument('-method', required=False, metavar='[fisher|binomial]', default='fisher', help='method for statistical test')
    parser.add_argument('-pcut', required=False, type=float, metavar='N', default=1.0, help='pvalue cutoff')
    parser.add_argument('-topK', required=False, type=int, metavar='N', default=None, help='show top K result')
    parser.add_argument('-o', dest='outfile', required=False, metavar='str', default='stdout', help='outfile')
    args=parser.parse_args()

    if args.outfile == 'stdout':
        OF=sys.stdout
    else:
        OF=open(args.outfile,'w')

    go = GOenrichment()
    if (args.backgroundGenes):
        go.load_backgroundgenes(args.backgroundGenes)
    go.load(args.trait2genes)

    IF=open(args.gene2subtype,'r')
    dic_subtype2gene={}
    for line in IF:
        s=line.rstrip().split('\t')
        if len(s) == 1:
            gene,subtype=s[0],'None'
        else:
            gene,subtype=s[0:2]
        if subtype not in dic_subtype2gene:
            dic_subtype2gene[subtype]=[]
        dic_subtype2gene[subtype].append(gene)
    for subtype, lst_gene in sorted(dic_subtype2gene.items(),key=lambda x:float(x[0]) if x[0].isdigit() else x[0]):
        df_cond = pd.Series(lst_gene)
        go.calculate(lst_gene, args.method).to_csv(OF)



if __name__=='__main__':
    main()
