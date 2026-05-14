#!/bin/bash
# Monitor Phase B skip experiment progress
LOG_ROOT="$HOME/Fast-dLLM/v2/logs/skip_exp"

echo "============================================"
echo "Phase B Skip Experiment Monitor"
echo "Time: $(date)"
echo "============================================"

PID=$(cat "$LOG_ROOT/runlog.pid" 2>/dev/null)
if [ -n "$PID" ] && ps -p "$PID" > /dev/null 2>&1; then
    echo "✅ Process running: PID $PID"
    ETIME=$(ps -o etime= -p "$PID" | tr -d ' ')
    echo "   Elapsed: $ETIME"
else
    echo "❌ Process not running"
fi

echo
echo "GPU:"
nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total,temperature.gpu \
    --format=csv,noheader,nounits 2>/dev/null | awk -F', ' '{printf "  Util: %s%%   Mem: %s/%s MB   Temp: %s°C\n", $1, $2, $3, $4}'

echo
DONE=$(ls "$LOG_ROOT"/*/done.flag 2>/dev/null | wc -l)
TOTAL=17
echo "Settings done: $DONE / $TOTAL"

echo
echo "Per-setting progress:"
for dir in "$LOG_ROOT"/*/; do
    name=$(basename "$dir")
    if [ "$name" = "*" ]; then continue; fi
    n_samples=$(ls "$dir"/skip_stats_*.npz 2>/dev/null | wc -l)
    if [ -f "$dir/done.flag" ]; then
        if [ -f "$dir/summary.json" ]; then
            acc=$(python3 -c "import json; print(f\"{json.load(open('$dir/summary.json'))['accuracy']:.1f}%\")" 2>/dev/null || echo "?")
            echo "  ✅ $name  ($n_samples samples, acc=$acc)"
        else
            echo "  ✅ $name  ($n_samples samples)"
        fi
    else
        echo "  🔄 $name  ($n_samples / 100 samples)"
    fi
done
