mkdir -p ./result_long_term_forecast_logs

python -u run_long_term_forecasting.py --data_name 'ETTh1' --pred_len 96 |tee ./result_long_term_forecast_logs/ETTh1_96.log
python -u run_long_term_forecasting.py --data_name 'ETTh1' --pred_len 192 |tee ./result_long_term_forecast_logs/ETTh1_192.log
python -u run_long_term_forecasting.py --data_name 'ETTh1' --pred_len 336 |tee ./result_long_term_forecast_logs/ETTh1_336.log
python -u run_long_term_forecasting.py --data_name 'ETTh1' --pred_len 720 |tee ./result_long_term_forecast_logs/ETTh1_720.log



python -u run_long_term_forecasting.py --data_name 'ETTm1' --pred_len 96 |tee ./result_long_term_forecast_logs/ETTm1_96.log
python -u run_long_term_forecasting.py --data_name 'ETTm1' --pred_len 192 |tee ./result_long_term_forecast_logs/ETTm1_192.log
python -u run_long_term_forecasting.py --data_name 'ETTm1' --pred_len 336 |tee ./result_long_term_forecast_logs/ETTm1_336.log
python -u run_long_term_forecasting.py --data_name 'ETTm1' --pred_len 720 |tee ./result_long_term_forecast_logs/ETTm1_720.log


python -u run_long_term_forecasting.py --data_name 'ETTh2' --pred_len 96 |tee ./result_long_term_forecast_logs/ETTh2_96.log
python -u run_long_term_forecasting.py --data_name 'ETTh2' --pred_len 192 |tee ./result_long_term_forecast_logs/ETTh2_192.log
python -u run_long_term_forecasting.py --data_name 'ETTh2' --pred_len 336 |tee ./result_long_term_forecast_logs/ETTh2_336.log
python -u run_long_term_forecasting.py --data_name 'ETTh2' --pred_len 720 |tee ./result_long_term_forecast_logs/ETTh2_720.log



python -u run_long_term_forecasting.py --data_name 'ETTm2' --pred_len 96 |tee ./result_long_term_forecast_logs/ETTm2_96.log
python -u run_long_term_forecasting.py --data_name 'ETTm2' --pred_len 192 |tee ./result_long_term_forecast_logs/ETTm2_192.log
python -u run_long_term_forecasting.py --data_name 'ETTm2' --pred_len 336 |tee ./result_long_term_forecast_logs/ETTm2_336.log
python -u run_long_term_forecasting.py --data_name 'ETTm2' --pred_len 720 |tee ./result_long_term_forecast_logs/ETTm2_720.log

python -u run_long_term_forecasting.py --data_name 'ILI' --seq_len 36 --T 36 --label_len 18 --pred_len 24 |tee ./result_long_term_forecast_logs/ILI_24.log
python -u run_long_term_forecasting.py --data_name 'ILI' --seq_len 36 --T 36 --label_len 18 --pred_len 36 |tee ./result_long_term_forecast_logs/ILI_36.log
python -u run_long_term_forecasting.py --data_name 'ILI' --seq_len 36 --T 36 --label_len 18 --pred_len 48 |tee ./result_long_term_forecast_logs/ILI_48.log
python -u run_long_term_forecasting.py --data_name 'ILI'   --seq_len 36 --T 36 --label_len 18 --pred_len 60 |tee ./result_long_term_forecast_logs/ILI_60.log


python -u run_long_term_forecasting.py --data_name 'Exchange' --pred_len 96 |tee ./result_long_term_forecast_logs/Exchange_96.log
python -u run_long_term_forecasting.py --data_name 'Exchange' --pred_len 192 |tee ./result_long_term_forecast_logs/Exchange_192.log
python -u run_long_term_forecasting.py --data_name 'Exchange' --pred_len 336 |tee ./result_long_term_forecast_logs/Exchange_336.log
python -u run_long_term_forecasting.py --data_name 'Exchange'   --pred_len 720 |tee ./result_long_term_forecast_logs/Exchange_720.log


python -u run_long_term_forecasting.py --data_name 'Weather' --pred_len 96 |tee ./result_long_term_forecast_logs/Weather_96.log
python -u run_long_term_forecasting.py --data_name 'Weather' --pred_len 192 |tee ./result_long_term_forecast_logs/Weather_192.log
python -u run_long_term_forecasting.py --data_name 'Weather' --pred_len 336 |tee ./result_long_term_forecast_logs/Weather_336.log
python -u run_long_term_forecasting.py --data_name 'Weather'   --pred_len 720 |tee ./result_long_term_forecast_logs/Weather_720.log



python -u run_long_term_forecasting.py --data_name 'Electricity' --pred_len 96 |tee ./result_long_term_forecast_logs/Electricity_96.log
python -u run_long_term_forecasting.py --data_name 'Electricity' --pred_len 192 |tee ./result_long_term_forecast_logs/Electricity_192.log
python -u run_long_term_forecasting.py --data_name 'Electricity' --pred_len 336 |tee ./result_long_term_forecast_logs/Electricity_336.log
python -u run_long_term_forecasting.py --data_name 'Electricity'   --pred_len 720 |tee ./result_long_term_forecast_logs/Electricity_720.log



python -u run_long_term_forecasting.py --data_name 'Traffic' --pred_len 96 |tee ./result_long_term_forecast_logs/Traffic_96.log
python -u run_long_term_forecasting.py --data_name 'Traffic' --pred_len 192 |tee ./result_long_term_forecast_logs/Traffic_192.log
python -u run_long_term_forecasting.py --data_name 'Traffic' --pred_len 336 |tee ./result_long_term_forecast_logs/Traffic_336.log
python -u run_long_term_forecasting.py --data_name 'Traffic'   --pred_len 720 |tee ./result_long_term_forecast_logs/Traffic_720.log
