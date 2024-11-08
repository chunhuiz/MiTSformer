mkdir -p ./result_ad_logs
python -u run_anomaly_detection.py --data_name 'SMD' --data 'SMD'   |tee ./result_ad_logs/SMD.log
python -u run_anomaly_detection.py --data_name 'SMAP' --data 'SMAP'  |tee ./result_ad_logs/SMAP.log
python -u run_anomaly_detection.py --data_name 'MSL' --data 'MSL'  |tee ./result_ad_logs/MSL.log
python -u run_anomaly_detection.py --data_name 'PSM' --data 'PSM'   |tee ./result_ad_logs/PSM.log
python -u run_anomaly_detection.py --data_name 'SWAT' --data 'SWAT'  |tee ./result_ad_logs/SWAT.log


