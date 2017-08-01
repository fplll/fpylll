python -u bench_bkz2_otf.py .70 > bkz2_otf_p70_noGHbound.txt &
python -u bench_bkz2_otf.py .30 > bkz2_otf_p30_noGHbound.txt &
python -u bench_bkz2_otf.py .15 > bkz2_otf_p15_noGHbound.txt &
python -u bench_bkz2_otf.py .07 > bkz2_otf_p07_noGHbound.txt &

sleep 2

tail -f bkz2_otf_p*.txt
