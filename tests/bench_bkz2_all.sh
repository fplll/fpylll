python -u bench_bkz2_otf.py .70 > bkz2_otf_p70_noGHbAtAll.txt &
python -u bench_bkz2_otf.py .30 > bkz2_otf_p30_noGHbAtAll.txt &
python -u bench_bkz2_otf.py .15 > bkz2_otf_p15_noGHbAtAll.txt &
python -u bench_bkz2_otf.py .07 > bkz2_otf_p07_noGHbAtAll.txt &

sleep 2

tail -f bkz2_otf_p*.txt
