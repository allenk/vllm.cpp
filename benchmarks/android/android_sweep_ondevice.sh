#!/system/bin/sh
# Thread sweep on a 1+5+2 phone, with the guards this run earned the hard way.
#
# THE QUESTION. The thread contract is "logical cores / 2", set on a homogeneous
# desktop. This SoC is not that: X4 at 3.30 GHz, five A720 at 2.96-3.15, two
# A520 at 2.27. "8 / 2 = 4" names a COUNT, not a SET. Pre-flight question 1 asks
# for the optimum measured ON THIS PLATFORM and it never has been.
#
# THREE GUARDS, one per way this run has already gone wrong today:
#
#  1. ORPHAN CHECK BEFORE ANYTHING. A vllm-server from 2026-09-19 was still
#     alive holding 3.3 GB, which left 1980 MB free against a 2012 MB model --
#     the repack-ON arm would have swapped and measured flash instead of the
#     kernel. Later, interrupting a driver left THREE servers alive at once, all
#     on port 18090 with identical command lines. The readiness probe then
#     answered from an ORPHAN that carried none of this script's env vars, so
#     the sweep would have reported a config it never set.
#     => refuse to start if any server exists or the port is held.
#
#  2. EXACTLY ONE SERVER PER LEG, verified after start. If a second appears the
#     leg is void, because the probe cannot say which one answered.
#
#  3. READ BACK THAT THE KILL ENGAGED, and trap EXIT/INT/TERM so an interrupted
#     run does not leave the orphan that caused guard 1.
#
# ON-DEVICE by necessity and by preference: `adb forward` registered but never
# created a host listener (`adb forward --list` showed the mapping while nothing
# listened), and device-side curl answered 200 immediately. Keeping the driver
# here also keeps the USB round-trip out of the timing.
#
# TIMING from `curl -w %{time_total}`; toybox date does not reliably carry %N.
# cpu/wall from /proc/<pid>/stat, because rule 2 of the checklist says that is
# the only acceptable proof a thread flag engaged.
set -u
DEV=/data/local/tmp
BIN=$DEV/vllm-server
MODEL=$DEV/lm/Qwen3.5-2B-Q8_0.gguf
PORT=18090
REPACK=${REPACK:-1}
LIST=${LIST:-"1 2 4 6 8"}
OUT=$DEV/sweep-results.txt
CUR=""

nservers() { ps -A -o NAME 2>/dev/null | grep -c '^vllm-server$'; }
portheld() { ss -ltn 2>/dev/null | grep -c ":$PORT " ; }

cleanup() {
    [ -n "$CUR" ] && kill -9 "$CUR" 2>/dev/null
    exit 0
}
trap cleanup EXIT INT TERM

# ---- GUARD 1 -------------------------------------------------------------
N=$(nservers); P=$(portheld)
if [ "$N" != "0" ] || [ "$P" != "0" ]; then
    echo "REFUSING TO START: $N server(s) already running, port listeners=$P"
    ps -A -o PID,RSS,NAME 2>/dev/null | grep 'vllm-server$'
    echo "Kill them by PID and re-run. A leftover server both steals CPU and answers the probe."
    exit 1
fi
: > $OUT

WORDS=""
i=0
while [ $i -lt 96 ]; do WORDS="$WORDS alpha bravo charlie delta"; i=$((i + 4)); done

probe() {
    N=$(cat /proc/sys/kernel/random/uuid 2>/dev/null || echo $$-$RANDOM)
    printf '{"model":"m","prompt":"%s %s","max_tokens":32,"temperature":0}' "$N" "$WORDS" > $DEV/.req.json
    curl -s -o /dev/null -w '%{time_total}' -m 1800 \
        -H 'Content-Type: application/json' \
        --data @$DEV/.req.json "http://127.0.0.1:$PORT/v1/completions"
}

echo "ANDROID THREAD SWEEP (on-device)  REPACK=$REPACK  Qwen3.5-2B-Q8_0  ~128 in / 32 out"
free -m | sed -n 2p
echo
echo "threads   t1       t2       tok/s    cpu/wall  rssMB"

for T in $LIST; do
    VT_CPU_QUANT_REPACK=$REPACK VLLM_CPP_CPU_THREADS=$T \
        $BIN --model $MODEL --port $PORT --served-model-name m --device cpu \
        --max-model-len 4096 --num-blocks 512 > $DEV/sweep-t$T.log 2>&1 &
    CUR=$!

    CODE=000; i=0
    while [ $i -lt 150 ]; do
        CODE=$(curl -s -o /dev/null -w '%{http_code}' -m 3 "http://127.0.0.1:$PORT/v1/models" 2>/dev/null)
        [ "$CODE" = "200" ] && break
        kill -0 $CUR 2>/dev/null || break
        sleep 2; i=$((i + 1))
    done

    # ---- GUARD 2 ---------------------------------------------------------
    N=$(nservers)
    if [ "$CODE" != "200" ] || [ "$N" != "1" ]; then
        echo "$T   VOID (code=$CODE servers=$N)"
        tail -2 $DEV/sweep-t$T.log
        kill -9 $CUR 2>/dev/null; CUR=""; sleep 3; continue
    fi

    probe > /dev/null 2>&1

    C0=$(awk '{print $14+$15}' /proc/$CUR/stat)
    W0=$(date +%s)
    T1=$(probe); T2=$(probe)
    W1=$(date +%s)
    C1=$(awk '{print $14+$15}' /proc/$CUR/stat)
    RSS=$(awk '/VmRSS/{print int($2/1024)}' /proc/$CUR/status)

    RES=$(awk -v a="$T1" -v b="$T2" -v c0="$C0" -v c1="$C1" -v w0="$W0" -v w1="$W1" \
        'BEGIN{med=(a+b)/2;
               printf "%-8.2f %-8.2f", 32.0/med, ((c1-c0)/100.0)/((w1-w0)>0?(w1-w0):1)}')
    printf "%-9s %-8s %-8s %s  %s\n" "$T" "$T1" "$T2" "$RES" "$RSS"
    echo "$T $T1 $T2 $RES $RSS" >> $OUT

    # ---- GUARD 3 ---------------------------------------------------------
    kill -TERM $CUR 2>/dev/null
    i=0
    while [ $i -lt 20 ]; do kill -0 $CUR 2>/dev/null || break; sleep 1; i=$((i + 1)); done
    kill -9 $CUR 2>/dev/null
    sleep 3
    LEFT=$(nservers)
    [ "$LEFT" = "0" ] || echo "  WARNING: $LEFT server(s) still alive after kill"
    CUR=""
done
echo ANDROID_SWEEP_DONE
