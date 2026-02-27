# Script that borrows lncrnapy's repo's functions to evaluate classification metrics based on the prediction csvs.
# This requires running lncrnapy's classify function on the GENCODE47 and GENCODE49 test sets, which outputs the csvs in the g47_lncRNABERT_results and g49_lncRNABERT_results folders, respectively.

import pandas as pd
from lncrnapy.data import Data
from lncrnapy.evaluate import lncRNA_classification_report

# Load the predictions and true labels, then print the classification report.

pred = pd.read_csv('g47_lncRNABERT_results/g47_lncRNABERT_results.csv')['class']

true = Data(['data/split_gencode_47/pc_test.fa',
             'data/split_gencode_47/lnc_test.fa']).df['label']

print(lncRNA_classification_report(
    true, pred, 'lncRNA-BERT (3-mer)', 'GENCODE47 (Test)'
))

# Repeat for GENCODE49

pred = pd.read_csv('g49_lncRNABERT_results/g49_lncRNABERT_results.csv')['class']

true = Data(['data/split_gencode_49/pc_test.fa',
             'data/split_gencode_49/lnc_test.fa']).df['label']

print(lncRNA_classification_report(
    true, pred, 'lncRNA-BERT (3-mer)', 'GENCODE49 (Test)'
))