python -u bench_svpchallenge.py .5 4 > core0.txt &
python -u bench_svpchallenge.py .6 4 > core1.txt &
python -u bench_svpchallenge.py .7 4 > core2.txt &
python -u bench_svpchallenge.py .8 4 > core3.txt &

sleep 2

tail -f core*.txt | grep SUMMARY

