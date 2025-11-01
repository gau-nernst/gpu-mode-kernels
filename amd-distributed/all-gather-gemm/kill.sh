set -e
set -x
# rm -rf ~/.triton/cache
pids=$(ps aux | grep "create_shemem.py" | grep -v grep | awk '{print $2}')

export HETER_NO_LOCAL=1
for pid in $pids; do
  echo "Killing process $pid..."
  kill -9 $pid
done

pids=$(ps aux | grep "make_share" | grep -v grep | awk '{print $2}')

export HETER_NO_LOCAL=1
for pid in $pids; do
  echo "Killing process $pid..."
  kill -9 $pid
done

pids=$(ps aux | grep "multiprocessing" | grep -v grep | awk '{print $2}')

export HETER_NO_LOCAL=1
for pid in $pids; do
  echo "Killing process $pid..."
  kill -9 $pid
done

rm -rf *.pkl

rm -rf finish.txt
rm -rf *.bin
