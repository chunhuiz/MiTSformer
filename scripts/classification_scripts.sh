mkdir -p ./result_run_classification_logs
python -u run_classification.py --data_name 'EthanolConcentration'  |tee ./result_run_classification_logs/thanolConcentration.log
python -u run_classification.py --data_name 'SelfRegulationSCP1'   |tee ./result_run_classification_logs/SelfRegulationSCP1.log
python -u run_classification.py --data_name 'Heartbeat'  |tee ./result_run_classification_logs/Heartbeat.log
python -u run_classification.py --data_name 'SpokenArabicDigits'  |tee ./result_run_classification_logs/SpokenArabicDigits.log
python -u run_classification.py --data_name 'Handwriting'  |tee ./result_run_classification_logs/Handwriting.log
python -u run_classification.py --data_name 'SelfRegulationSCP2'  |tee ./result_run_classification_logs/SelfRegulationSCP2.log
python -u run_classification.py --data_name 'JapaneseVowels'  |tee ./result_run_classification_logs/JapaneseVowels.log
python -u run_classification.py --data_name 'UWaveGestureLibrary' |tee ./result_run_classification_logs/UWaveGestureLibrary.log
python -u run_classification.py --data_name 'FaceDetection'  |tee ./result_run_classification_logs/FaceDetection.log
python -u run_classification.py --data_name 'PEMS-SF' --batch_size 16  |tee ./result_run_classification_logs/PEMS-SF.log





