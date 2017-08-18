python -u svpchallenge_par2.py 30 1 > core00.log &
python -u svpchallenge_par2.py 30 1 > core01.log &
python -u svpchallenge_par2.py 30 1 > core02.log &
python -u svpchallenge_par2.py 30 1 > core03.log &
python -u svpchallenge_par2.py 30 1 > core04.log &
python -u svpchallenge_par2.py 30 1 > core05.log &
python -u svpchallenge_par2.py 30 1 > core06.log &
python -u svpchallenge_par2.py 30 1 > core07.log &
python -u svpchallenge_par2.py 30 1 > core08.log &
python -u svpchallenge_par2.py 30 1 > core09.log &
python -u svpchallenge_par2.py 30 1 > core0A.log &
python -u svpchallenge_par2.py 30 1 > core0B.log &
python -u svpchallenge_par2.py 30 1 > core0C.log &
python -u svpchallenge_par2.py 30 1 > core0D.log &

python -u svpchallenge_par2.py 30 1 > core10.log &
python -u svpchallenge_par2.py 30 1 > core11.log &
python -u svpchallenge_par2.py 30 1 > core12.log &
python -u svpchallenge_par2.py 30 1 > core13.log &
python -u svpchallenge_par2.py 30 1 > core14.log &
python -u svpchallenge_par2.py 30 1 > core15.log &
python -u svpchallenge_par2.py 30 1 > core16.log &
python -u svpchallenge_par2.py 30 1 > core17.log &
python -u svpchallenge_par2.py 30 1 > core18.log &
python -u svpchallenge_par2.py 30 1 > core19.log &
python -u svpchallenge_par2.py 30 1 > core1A.log &
python -u svpchallenge_par2.py 30 1 > core1B.log &
python -u svpchallenge_par2.py 30 1 > core1C.log &
python -u svpchallenge_par2.py 30 1 > core1D.log &

sleep 2

tail -f core*.log | grep SUMMARY

