# triframe vs β-LNC — equivalence report
folds: [0, 1, 2, 3, 4]  max_pairs: all  batch_size: 1  atol_logit: 1e-05
OVERALL: PASS

    part  fold  passed  groups  pairs  max_dlogit  noise_floor  iia_groups  iia_exact  iia_unexplained     stat_gap  steps_ref  steps_triframe same_tokens  max_iia_gap      sym_gap  n_sel  n_eval pairs_prefix_matches_3_1
patching     0    True    96.0  19200    0.000004     0.000002        96.0       96.0              0.0 1.620501e-07        NaN             NaN         NaN          NaN          NaN    NaN     NaN                      NaN
  greedy     0    True     NaN   1000         NaN          NaN         NaN        NaN              NaN          NaN       10.0            10.0        True 9.714451e-17 2.086163e-07  500.0   500.0                     True
patching     1    True    96.0  19200    0.000002     0.000001        96.0       96.0              0.0 1.598708e-07        NaN             NaN         NaN          NaN          NaN    NaN     NaN                      NaN
  greedy     1    True     NaN   1000         NaN          NaN         NaN        NaN              NaN          NaN       10.0            10.0        True 9.714451e-17 1.788139e-07  500.0   500.0                     True
patching     2    True    96.0  19200    0.000004     0.000002        96.0       96.0              0.0 8.238479e-08        NaN             NaN         NaN          NaN          NaN    NaN     NaN                      NaN
  greedy     2    True     NaN   1000         NaN          NaN         NaN        NaN              NaN          NaN        9.0             9.0        True 9.367507e-17 1.788139e-07  500.0   500.0                     True
patching     3    True    96.0  19200    0.000005     0.000003        96.0       96.0              0.0 1.800898e-07        NaN             NaN         NaN          NaN          NaN    NaN     NaN                      NaN
  greedy     3    True     NaN   1000         NaN          NaN         NaN        NaN              NaN          NaN       10.0            10.0        True 8.326673e-17 2.980232e-07  500.0   500.0                     True
patching     4    True    96.0  19200    0.000005     0.000003        96.0       96.0              0.0 2.781441e-07        NaN             NaN         NaN          NaN          NaN    NaN     NaN                      NaN
  greedy     4    True     NaN   1000         NaN          NaN         NaN        NaN              NaN          NaN       10.0            10.0        True 9.367507e-17 2.980232e-07  500.0   500.0                     True