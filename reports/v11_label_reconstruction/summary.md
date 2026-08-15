# V11 Target Reconstruction

- Sector rows: **4832**
- Reconstructed daily dates: **604**
- Merchant rows compared: **541**

- baseline_inc: MAE=0.000528, RMSE=0.000632, corr=1.00000000, max_abs=0.001564
- baseline_prev: MAE=2377.661595, RMSE=3105.834336, corr=0.93659906, max_abs=11629.151964
- future_excl_7: MAE=0.006329, RMSE=0.007853, corr=1.00000000, max_abs=0.023956
- future_incl_7: MAE=69475.140325, RMSE=88195.026892, corr=0.83535788, max_abs=271159.162039
- ratio_base_inc_fut_excl_7: MAE=0.000000, RMSE=0.000000, corr=1.00000000, max_abs=0.000000
- ratio_base_inc_fut_incl_7: MAE=0.064595, RMSE=0.081943, corr=0.85988438, max_abs=0.246134
- ratio_base_prev_fut_excl_7: MAE=0.015753, RMSE=0.021059, corr=0.99092486, max_abs=0.080618
- ratio_base_prev_fut_incl_7: MAE=0.075012, RMSE=0.095556, corr=0.81467384, max_abs=0.337225
