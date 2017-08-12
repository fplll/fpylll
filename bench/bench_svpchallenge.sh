python -u bench_svpchallenge.py .50 4 > core00.log &
python -u bench_svpchallenge.py .55 4 > core01.log &
python -u bench_svpchallenge.py .60 4 > core02.log &
python -u bench_svpchallenge.py .65 4 > core03.log &
python -u bench_svpchallenge.py .70 4 > core04.log &
python -u bench_svpchallenge.py .75 4 > core05.log &
python -u bench_svpchallenge.py .80 4 > core06.log &

python -u bench_svpchallenge.py .50 4 > core10.log &
python -u bench_svpchallenge.py .55 4 > core11.log &
python -u bench_svpchallenge.py .60 4 > core12.log &
python -u bench_svpchallenge.py .65 4 > core13.log &
python -u bench_svpchallenge.py .70 4 > core14.log &
python -u bench_svpchallenge.py .75 4 > core15.log &
python -u bench_svpchallenge.py .80 4 > core16.log &

python -u bench_svpchallenge.py .50 4 > core20.log &
python -u bench_svpchallenge.py .55 4 > core21.log &
python -u bench_svpchallenge.py .60 4 > core22.log &
python -u bench_svpchallenge.py .65 4 > core23.log &
python -u bench_svpchallenge.py .70 4 > core24.log &
python -u bench_svpchallenge.py .75 4 > core25.log &
python -u bench_svpchallenge.py .80 4 > core26.log &

python -u bench_svpchallenge.py .50 4 > core30.log &
python -u bench_svpchallenge.py .55 4 > core31.log &
python -u bench_svpchallenge.py .60 4 > core32.log &
python -u bench_svpchallenge.py .65 4 > core33.log &
python -u bench_svpchallenge.py .70 4 > core34.log &
python -u bench_svpchallenge.py .75 4 > core35.log &
python -u bench_svpchallenge.py .80 4 > core36.log &

sleep 2

tail -f core*.log | grep SUMMARY

