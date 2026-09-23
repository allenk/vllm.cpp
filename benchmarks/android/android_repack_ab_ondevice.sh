#!/system/bin/sh
# Android repack A/B, interleaved, at the thread count this phone actually wants.
#
# WHY 6 THREADS AND NOT THE CONTRACT'S 4. The contract is "logical cores / 2",
# set on a homogeneous desktop. This SoC is 1+5+2 -- one X4 at 3.30 GHz, five
# A720 at 2.96-3.15, two A520 at 2.27 -- so "8 / 2 = 4" names a COUNT and not a
# SET. Swept on the device today:
#
#     threads   1      2      4      6      8
#     tok/s     2.53   3.94   5.14   5.43   4.81
#     cpu/wall  1.01   1.90   4.14   5.86   7.37
#
# cpu/wall tracks the flag at every level, so every leg engaged. The optimum is
# SIX, which is exactly the count of non-LITTLE cores, and EIGHT is 12% worse
# than six: the two A520 join the parallel region and then hold everyone at the
# barrier, because a fork-join is paced by its slowest thread.
#
# That 8-versus-6 ordering was confirmed interleaved (8/6/8/6: 4.94, 5.42, 4.56,
# 5.18) rather than left to a sweep whose last leg was also its hottest.
# Absolute values drift 4-8% across legs from heat; the RANKING does not move.
# => this A/B interleaves rather than running two blocks, for the same reason.
#
# GUARDS, each earned today: a 2026-09-19 server was still alive holding 3.3 GB;
# later an interrupted driver left THREE servers on one port with identical
# command lines, so the readiness probe answered from an orphan carrying none of
# the script's env vars. Refuse to start if any server exists; require exactly
# one per leg; read back that the kill engaged; trap so an interrupt leaves none.
#
# WHAT IT SHOULD REPRODUCE: docs/benchmarks/cpu-q8_0-repack.md section 3 reports
# scaling 1.18x off against 1.72x on for this chip. That run pinned clocks and
# this one does not, so ABSOLUTE tok/s will be lower here; the repack RATIO is
# the quantity to compare.
set -u
DEV=/data/local/tmp
BIN=$DEV/vllm-server
MODEL=$DEV/lm/Qwen3.5-2B-Q8_0.gguf
PORT=18090
THREADS=${THREADS:-6}
LEGS=${LEGS:-"1 0 1 0"}
OUT=$DEV/ab-results.txt
CUR=""

nservers() { ps -A -o NAME 2>/dev/null | grep -c '^vllm-server$'; }
portheld() { ss -ltn 2>/dev/null | grep -c ":$PORT " ; }
cleanup() { [ -n "$CUR" ] && kill -9 "$CUR" 2>/dev/null; exit 0; }
trap cleanup EXIT INT TERM

N=$(nservers); P=$(portheld)
if [ "$N" != "0" ] || [ "$P" != "0" ]; then
    echo "REFUSING TO START: $N server(s) running, port listeners=$P"
    ps -A -o PID,RSS,NAME 2>/dev/null | grep 'vllm-server$'
    exit 1
fi
: > $OUT

WORDS=""; i=0
while [ $i -lt 96 ]; do WORDS="$WORDS alpha bravo charlie delta"; i=$((i + 4)); done

probe() {
    U=$(cat /proc/sys/kernel/random/uuid 2>/dev/null || echo $$-$RANDOM)
    printf '{"model":"m","prompt":"%s %s","max_tokens":32,"temperature":0}' "$U" "$WORDS" > $DEV/.req.json
    curl -s -o /dev/null -w '%{time_total}' -m 1800 -H 'Content-Type: application/json' \
        --data @$DEV/.req.json "http://127.0.0.1:$PORT/v1/completions"
}

echo "ANDROID REPACK A/B (on-device, interleaved)  threads=$THREADS  Qwen3.5-2B-Q8_0"
free -m | sed -n 2p
echo
echo "repack  t1       t2       tok/s    cpu/wall  rssMB   RssAnonMB  RssFileMB"

for R in $LEGS; do
    VT_CPU_QUANT_REPACK=$R VLLM_CPP_CPU_THREADS=$THREADS \
        $BIN --model $MODEL --port $PORT --served-model-name m --device cpu \
        --max-model-len 4096 --num-blocks 512 > $DEV/ab-r$R.log 2>&1 &
    CUR=$!

    CODE=000; i=0
    while [ $i -lt 150 ]; do
        CODE=$(curl -s -o /dev/null -w '%{http_code}' -m 3 "http://127.0.0.1:$PORT/v1/models" 2>/dev/null)
        [ "$CODE" = "200" ] && break
        kill -0 $CUR 2>/dev/null || break
        sleep 2; i=$((i + 1))
    done
    N=$(nservers)
    if [ "$CODE" != "200" ] || [ "$N" != "1" ]; then
        echo "repack=$R   VOID (code=$CODE servers=$N)"
        tail -2 $DEV/ab-r$R.log
        kill -9 $CUR 2>/dev/null; CUR=""; sleep 3; continue
    fi

    probe > /dev/null 2>&1

    C0=$(awk '{print $14+$15}' /proc/$CUR/stat); W0=$(date +%s)
    T1=$(probe); T2=$(probe)
    W1=$(date +%s); C1=$(awk '{print $14+$15}' /proc/$CUR/stat)
    # RssAnon vs RssFile is how the tier is PROVEN to have engaged: the repacked
    # copy is anonymous, the borrowed mmap is file-backed. Throughput alone
    # cannot say which path ran.
    RSS=$(awk '/VmRSS/{print int($2/1024)}' /proc/$CUR/status)
    ANON=$(awk '/RssAnon/{print int($2/1024)}' /proc/$CUR/status)
    FILE=$(awk '/RssFile/{print int($2/1024)}' /proc/$CUR/status)

    RES=$(awk -v a="$T1" -v b="$T2" -v c0="$C0" -v c1="$C1" -v w0="$W0" -v w1="$W1" \
        'BEGIN{med=(a+b)/2; printf "%-8.2f %-8.2f", 32.0/med, ((c1-c0)/100.0)/((w1-w0)>0?(w1-w0):1)}')
    printf "%-7s %-8s %-8s %s  %-7s %-10s %s\n" "$R" "$T1" "$T2" "$RES" "$RSS" "$ANON" "$FILE"
    echo "$R $T1 $T2 $RES $RSS $ANON $FILE" >> $OUT

    kill -TERM $CUR 2>/dev/null
    i=0; while [ $i -lt 20 ]; do kill -0 $CUR 2>/dev/null || break; sleep 1; i=$((i + 1)); done
    kill -9 $CUR 2>/dev/null; sleep 3
    LEFT=$(nservers); [ "$LEFT" = "0" ] || echo "  WARNING: $LEFT server(s) still alive"
    CUR=""
done
echo ANDROID_AB_DONE
