python -u bench_svpchallenge.py 34 32 > core00.log &
python -u bench_svpchallenge.py 38 32 > core01.log &
python -u bench_svpchallenge.py 42 32 > core02.log &
python -u bench_svpchallenge.py 46 32 > core03.log &
python -u bench_svpchallenge.py 48 32 > core04.log &

python -u bench_svpchallenge.py 50 32 > core05.log &
python -u bench_svpchallenge.py 52 32 > core06.log &
python -u bench_svpchallenge.py 54 32 > core07.log &
python -u bench_svpchallenge.py 56 32 > core08.log &
python -u bench_svpchallenge.py 58 32 > core09.log &

python -u bench_svpchallenge.py 60 32 > core10.log &
python -u bench_svpchallenge.py 62 32 > core11.log &
python -u bench_svpchallenge.py 66 32 > core12.log &
python -u bench_svpchallenge.py 70 32 > core13.log &

python -u bench_svpchallenge.py 34 32 > core20.log &
python -u bench_svpchallenge.py 38 32 > core21.log &
python -u bench_svpchallenge.py 42 32 > core22.log &
python -u bench_svpchallenge.py 46 32 > core23.log &
python -u bench_svpchallenge.py 48 32 > core24.log &

python -u bench_svpchallenge.py 50 32 > core25.log &
python -u bench_svpchallenge.py 52 32 > core26.log &
python -u bench_svpchallenge.py 54 32 > core27.log &
python -u bench_svpchallenge.py 56 32 > core28.log &
python -u bench_svpchallenge.py 58 32 > core29.log &

python -u bench_svpchallenge.py 60 32 > core20.log &
python -u bench_svpchallenge.py 62 32 > core21.log &
python -u bench_svpchallenge.py 66 32 > core22.log &
python -u bench_svpchallenge.py 70 32 > core23.log &



sleep 2

tail -f core*.log | grep SUMMARY

