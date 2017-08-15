python -u bench_svpchallenge.py 44 32 > core00.log &
python -u bench_svpchallenge.py 46 32 > core01.log &
python -u bench_svpchallenge.py 48 32 > core02.log &
python -u bench_svpchallenge.py 50 32 > core03.log &
python -u bench_svpchallenge.py 52 32 > core04.log &
python -u bench_svpchallenge.py 54 32 > core05.log &
python -u bench_svpchallenge.py 56 32 > core06.log &
python -u bench_svpchallenge.py 58 32 > core07.log &
python -u bench_svpchallenge.py 60 32 > core08.log &
python -u bench_svpchallenge.py 62 32 > core09.log &
python -u bench_svpchallenge.py 64 32 > core0A.log &
python -u bench_svpchallenge.py 76 32 > core0B.log &
python -u bench_svpchallenge.py 78 32 > core0C.log &
python -u bench_svpchallenge.py 80 32 > core0D.log &

python -u bench_svpchallenge.py 44 32 > core10.log &
python -u bench_svpchallenge.py 46 32 > core11.log &
python -u bench_svpchallenge.py 48 32 > core12.log &
python -u bench_svpchallenge.py 50 32 > core13.log &
python -u bench_svpchallenge.py 52 32 > core14.log &
python -u bench_svpchallenge.py 54 32 > core15.log &
python -u bench_svpchallenge.py 56 32 > core16.log &
python -u bench_svpchallenge.py 58 32 > core17.log &
python -u bench_svpchallenge.py 60 32 > core18.log &
python -u bench_svpchallenge.py 62 32 > core19.log &
python -u bench_svpchallenge.py 64 32 > core1A.log &
python -u bench_svpchallenge.py 76 32 > core1B.log &
python -u bench_svpchallenge.py 78 32 > core1C.log &
python -u bench_svpchallenge.py 80 32 > core1D.log &

sleep 2

tail -f core*.log | grep SUMMARY

