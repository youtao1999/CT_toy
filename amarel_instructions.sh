nohup run_tmi_fine.sh 8 'pproj' 0.75 0.25 0.05 40 3000 20 > run_tmi_fine_L8.log 2>&1 &
nohup run_tmi_fine.sh 12 'pproj' 0.75 0.25 0.05 40 3000 40 > run_tmi_fine_L12.log 2>&1 &
nohup run_tmi_fine.sh 16 'pproj' 0.75 0.25 0.05 40 3000 100 > run_tmi_fine_L16.log 2>&1 &
nohup run_tmi_fine.sh 20 'pproj' 0.75 0.25 0.05 40 3000 100 > run_tmi_fine_L20.log 2>&1 &

scp -r ty296@amarel:/scratch/ty296/tmi_fine_L*_pproj0.750_* .

