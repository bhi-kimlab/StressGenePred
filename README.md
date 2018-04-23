#### Stress Predictor based on Machine Learning


### Learning

With prepared dataset,

```
python learn.py dataset/dataset_learn_005.limma.csv dataset/dataset_learn_005.label.csv --updown --save output/model_v2 --save_csv output/model_v2 --label_index "heat,salt,drought,cold"
```

### Testing prediction /w learnt parameter

```
python test.py dataset/dataset_test_005.limma.csv dataset/dataset_test_005.label.csv --save output/model_v2_predict --label_index "heat,salt,drought,cold
```

### Testing GSEA(Gene set enrichment analysis) score /w learnt parameter

```
python test_gsea.py dataset/parameter_gen.csv --gsea data/GSEA.csv
```

### How to generate dataset?

Just prepare your expression matrix (CEL file processed) and metadata (may write TSD header or generate from CSV). then use these commands:

```
python ../dataset.py tsd -f cold.csv
python ../dataset.py tsd -f heat.csv
python ../dataset.py read -d ./

# without any processing, just output raw p-value
python ../dataset.py compile -d ./ --tool limma -t dataset_test_005
# processing p-value /w threshold .05, (signed)
python ../dataset.py compile -d ./ --tool limma -t dataset_test_005 --pvalue 0.05
# processing p-value /w threshold .05, with reindexing & separating up/down signal (unsigned)
python ../dataset.py compile -d ./ --tool limma -t dataset_test_005 --pvalue 0.05 --updown --reindex_df ../dataprocess/ttest_pval.csv
```

Then DEG infomation will be stored at `dataset_test.csv` matrix.
You can use `--filter` to select TSData.
