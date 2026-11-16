# triframe vs β-LNC — equivalence report
folds: [0, 1, 2, 3, 4]  max_pairs: all  batch_size: 1  atol_logit: 1e-05
OVERALL: PASS

## 1.4 probing / pattern (Layer 1)
   part  passed   fold  n_tokens  n_missing   max_gap_r2  max_gap_align_raw  max_gap_align_resid
probing    True fold_0      20.0        0.0 8.326673e-17       9.020562e-17         8.673617e-17
probing    True fold_1      20.0        0.0 9.714451e-17       9.367507e-17         9.367507e-17
probing    True fold_2      20.0        0.0 9.714451e-17       9.020562e-17         9.020562e-17
probing    True fold_3      20.0        0.0 8.326673e-17       9.367507e-17         9.020562e-17
probing    True fold_4      20.0        0.0 8.326673e-17       9.887924e-17         9.020562e-17

## 1.2 ablation (Layer 1)
    part  passed fold  n_rows  max_gap_acc   max_gap_f1  max_gap_baseline  n_nonzero_ref
ablation    True    0    40.0 9.855398e-17 9.055799e-17      0.000000e+00           39.0
ablation    True    1    40.0 9.280771e-17 9.801188e-17      1.110223e-16           40.0
ablation    True    2    40.0 9.562663e-17 9.107298e-17      1.110223e-16           40.0
ablation    True    3    40.0 9.340402e-17 9.714451e-17      1.110223e-16           40.0
ablation    True    4    40.0 9.584347e-17 9.280771e-17      0.000000e+00           40.0

## 2.1 Fréchet (Layer 2)
   part  passed  n_subgroups  n_fail  max_absdiff    n_lnc  n_mrna
frechet    True         20.0     0.0 1.389822e-11 153108.0 87424.0