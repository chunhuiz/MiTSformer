mkdir -p ./result_imputation_logs

python -u run_imputation.py --data_name 'ETTh1'  --mask_rate 0.125 |tee ./result_imputation_logs/ETTh1_125.log
python -u run_imputation.py --data_name 'ETTh1'  --mask_rate 0.25 |tee ./result_imputation_logs/ETTh1_250.log
python -u run_imputation.py --data_name 'ETTh1'  --mask_rate 0.375 |tee ./result_imputation_logs/ETTh1_375.log
python -u run_imputation.py --data_name 'ETTh1'  --mask_rate 0.50 |tee ./result_imputation_logs/ETTh1500.log


python -u run_imputation.py --data_name 'ETTh2'  --mask_rate 0.125 |tee ./result_imputation_logs/ETTh2_125.log
python -u run_imputation.py --data_name 'ETTh2'  --mask_rate 0.25 |tee ./result_imputation_logs/ETTh2_250.log
python -u run_imputation.py --data_name 'ETTh2'  --mask_rate 0.375 |tee ./result_imputation_logs/ETTh2_375.log
python -u run_imputation.py --data_name 'ETTh2'  --mask_rate 0.50 |tee ./result_imputation_logs/ETTh2500.log




python -u run_imputation.py --data_name 'ETTm1'  --mask_rate 0.125 |tee ./result_imputation_logs/ETTm1_125.log
python -u run_imputation.py --data_name 'ETTm1'  --mask_rate 0.25 |tee ./result_imputation_logs/ETTm1_250.log
python -u run_imputation.py --data_name 'ETTm1'  --mask_rate 0.375 |tee ./result_imputation_logs/ETTm1_375.log
python -u run_imputation.py --data_name 'ETTm1'  --mask_rate 0.50 |tee ./result_imputation_logs/ETTm1500.log



python -u run_imputation.py --data_name 'ETTm2'  --mask_rate 0.125 |tee ./result_imputation_logs/ETTm2_125.log
python -u run_imputation.py --data_name 'ETTm2'  --mask_rate 0.25 |tee ./result_imputation_logs/ETTm2_250.log
python -u run_imputation.py --data_name 'ETTm2'  --mask_rate 0.375 |tee ./result_imputation_logs/ETTm2_375.log
python -u run_imputation.py --data_name 'ETTm2'  --mask_rate 0.50 |tee ./result_imputation_logs/ETTm2500.log




python -u run_imputation.py --data_name 'Electricity'  --mask_rate 0.125 |tee ./result_imputation_logs/Electricity_125.log
python -u run_imputation.py --data_name 'Electricity'  --mask_rate 0.25 |tee ./result_imputation_logs/Electricity_250.log
python -u run_imputation.py --data_name 'Electricity'  --mask_rate 0.375 |tee ./result_imputation_logs/Electricity_375.log
python -u run_imputation.py --data_name 'Electricity'  --mask_rate 0.50 |tee ./result_imputation_logs/Electricity500.log


python -u run_imputation.py --data_name 'Weather'  --mask_rate 0.25 |tee ./result_imputation_logs/Weather_250.log
python -u run_imputation.py --data_name 'Weather'  --mask_rate 0.125 |tee ./result_imputation_logs/Weather_125.log
python -u run_imputation.py --data_name 'Weather'  --mask_rate 0.375 |tee ./result_imputation_logs/Weather_375.log
python -u run_imputation.py --data_name 'Weather'  --mask_rate 0.50 |tee ./result_imputation_logs/Weathe500.log






