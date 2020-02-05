#!/bin/bash
n=$(( 2 ** 5 ))
for ((i=0; i<n; i++)); do
    switches=()
    for j in {1..5}; do
	m=$(( ($i/(2 ** ($j-1)))%2 ))
	echo "i $i, j $j, m $m"
        if [ $m -eq 1 ]; then switches+=(1); else switches+=(0); fi;
    done;
    if [ ${switches[0]} -eq 1 ]; then logscale='True'; else logscale='False'; fi;	
    if [ ${switches[1]} -eq 1 ]; then quadratic_scale='True'; else quadratic_scale='False'; fi;	
    if [ ${switches[2]} -eq 1 ]; then remove_rigid='True'; else remove_rigid='False'; fi;	
    if [ ${switches[3]} -eq 1 ]; then Jw='1.0'; else Jw='0.'; fi;	
    if [ ${switches[4]} -eq 1 ]; then Hw='1.0'; else Hw='0.'; fi;	
    command=$( IFS=$','; echo "python -m src.run --run_local True --lr 3e-4 --max_train_steps 10000 --results_dir /efs_nmor/results --data_name bV_10_hmc_macro_benchmark --experiment_name ablation_${switches[*]} --log_scale $logscale --quadratic_scale $quadratic_scale --remove_rigid $remove_rigid --J_weight $Jw --H_weight $Hw --batch_size 512 --train_size 55000 --val_size 5000 --ffn_layer_sizes [128,128,128] --max_collectors 0 --max_evaluators 0" )
    echo $command
    logfile=$"log_${switches[*]}.txt"
    logfile=$(echo "${logfile// /}")
    echo $logfile
    $command &> $logfile &
    sleep 30
    # sleep 1
done;
