python -u bench_bkz2_otf_subsol.py > bkz2_otf_subsol.txt &
python -u bench_bkz2_otf.py > bkz2_otf.txt &
python -u bench_bkz2.py > bkz2.txt &
sleep 2

tail -f bkz2*.txt
