#!/bin/bash

# automatically 
# args: [path-to-output]

PARAMS="--tool limma --tool --raw --ensure_count 1000 --pvalue 0.05"
EPOCH=1000
DO_DATASET=0

# make path
mkdir -p $1/output
set -e

if [ $DO_DATASET -eq 1 ]; then
# dataset generate
python dataset.py compile -d data/ $PARAMS -t $1/dataset_learn_005 --reindex_file dataset/index.txt  --updown --label_index "heat,salt,drought,cold" --filters "Species:Arabidopsis Thaliana,MinRepCnt:2"
python dataset.py compile -d data_test/ $PARAMS -t $1/dataset_test_005 --reindex_file dataset/index.txt  --updown --label_index "heat,salt,drought,cold"
fi

# learn

python learn.py $1/dataset_learn_005.limma.csv  $1/dataset_learn_005.label.csv  --reduce_updown --save $1/output/model --save_csv $1/output/params --epoch_count $EPOCH --batch_size 0

# test-learning model

python test.py $1/dataset_learn_005.limma.csv $1/dataset_learn_005.label.csv --load $1/output/model --save $1/output/predict_learn
python test.py $1/dataset_test_005.limma.csv $1/dataset_test_005.label.csv --load $1/output/model --save $1/output/predict_test

# test-GSEA

python GOenrichment.py $1/output/params_generator.csv --trait2genes GOBPname2gene.arabidopsis.txt --column_name "heat,salt,drought,cold" --count_cut 500 -o $1/output/GSEA_learn_top500.csv --descending

# test-GSEA on raw one

python pvalue_multiple.py $1/dataset_learn_005.pvalue.limma.csv --label_file $1/dataset_learn_005.label.csv -o $1/output/params_pvalue_raw.csv
python GOenrichment.py $1/output/params_pvalue_raw.csv --trait2genes GOBPname2gene.arabidopsis.txt --column_name "heat,salt,drought,cold" --count_cut 500 -o $1/output/GSEA_raw_top500.csv --descending
