nohup run_sv_fine.sh 8 'pproj' 0.0 0.45 0.1 20 20 20 > run_sv_fine_L8.log 2>&1 &
nohup run_sv_fine.sh 12 'pproj' 0.0 0.45 0.1 20 2000 40 > run_sv_fine_L12.log 2>&1 &
nohup run_sv_fine.sh 16 'pproj' 0.0 0.45 0.1 20 2000 100 > run_sv_fine_L16.log 2>&1 &
nohup run_sv_fine.sh 20 'pproj' 0.0 0.45 0.1 20 2000 100 > run_sv_fine_L20.log 2>&1 &

scp -r ty296@amarel:/scratch/ty296/sv_fine_L*_pproj0.000_pc0.45 .

nohup run_sv_compare_fine.sh 8 'pctrl' 0.4 0.6 0.1 40 2000 20 > run_sv_compare_fine_L8.log 2>&1 &
nohup run_sv_compare_fine.sh 12 'pctrl' 0.4 0.6 0.1 40 2000 40 > run_sv_compare_fine_L12.log 2>&1 &
nohup run_sv_compare_fine.sh 16 'pctrl' 0.4 0.6 0.1 40 2000 100 > run_sv_compare_fine_L16.log 2>&1 &
nohup run_sv_compare_fine.sh 20 'pctrl' 0.4 0.6 0.1 40 2000 100 > run_sv_compare_fine_L20.log 2>&1 &

scp -r ty296@amarel:/scratch/ty296/sv_comparison_L*_pctrl0.400_pc0.800 .

# run run_sv.sh
nohup run_sv.sh 8 'pctrl' 0.0 "0.35:0.45:20" 2000 20 --comparison > run_sv_L8.log 2>&1 &
nohup run_sv.sh 12 'pctrl' 0.0 "0.35:0.45:20" 2000 40 --comparison > run_sv_L12.log 2>&1 &
nohup run_sv.sh 16 'pctrl' 0.0 "0.35:0.45:20" 2000 100 --comparison > run_sv_L16.log 2>&1 &
nohup run_sv.sh 20 'pctrl' 0.0 "0.55:0.65:20" 2000 200 --comparison > run_sv_L20.log 2>&1 &

scp -r ty296@amarel:/scratch/ty296/sv_comparison_L20_pctrl0.000_p0.550-0.650 .

# clearing out all the chunk files
find . -type f -name "*chunk*" -delete