mkdir -p ./result_ext_reg_logs
python -u run_extrinsic_regression.py --data_name 'HouseholdPowerConsumption1' |tee ./result_ext_reg_logs/house1.log
python -u run_extrinsic_regression.py --data_name 'HouseholdPowerConsumption2'  |tee ./result_ext_reg_logs/house2.log
python -u run_extrinsic_regression.py --data_name 'AppliancesEnergy'  |tee ./result_ext_reg_logs/AppliancesEnergy.log

python -u run_extrinsic_regression.py --data_name 'PPGDalia'  |tee ./result_ext_reg_logs/PPGDalia.log
python -u run_extrinsic_regression.py --data_name 'AustraliaRainfall'  |tee ./result_ext_reg_logs/AustraliaRainfall.log

python -u run_extrinsic_regression.py --data_name 'BeijingPM25Quality'  |tee ./result_ext_reg_logs/BeijingPM25Quality.log
python -u run_extrinsic_regression.py --data_name 'LiveFuelMoistureContent'  |tee ./result_ext_reg_logs/LiveFuelMoistureContent.log
python -u run_extrinsic_regression.py --data_name 'BenzeneConcentration' |tee ./result_ext_reg_logs/BenzeneConcentration.log

python -u run_extrinsic_regression.py --data_name 'IEEEPPG'  |tee ./result_ext_reg_logs/IEEEPPG.log
python -u run_extrinsic_regression.py --data_name 'BeijingPM10Quality'  |tee ./result_ext_reg_logs/BeijingPM10Quality.log


