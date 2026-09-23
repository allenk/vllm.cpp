#!/system/bin/sh
# BF16 on the phone: ours against llama.cpp, ONE file, interleaved.
#
# WHY THIS MODEL AND THIS FILE. Qwen3-0.6B-BF16.gguf is already on the device
# and is the SAME file the Jetson legs used, so this fills the row the matrix
# has been missing -- one checkpoint, one dtype, three devices:
#
#     Qwen3-0.6B-BF16     ours     llama
#     Jetson Orin        18.72    30.93     measured
#     Android X4             ?        ?     <- this run
#     x86 Zen 5              ?        ?     still open
#
# It also isolates something the q8_0 legs cannot: repack serves q8_0 only, so a
# bf16 model exercises the plain CPU path with no tier to switch. And llama.cpp
# reads the identical file, so the axis is aligned by construction rather than
# by argument.
#
# ⚠️ 128 OUTPUT TOKENS, NOT 32, and that correction was earned today. Our rate
# is output tokens over TOTAL wall, so prefill sits in the denominator. Measured
# on this phone at ~128 prompt tokens, the same configuration reads:
#
#     max_tokens   32      128     256
#     tok/s       6.94    11.55   12.54
#
# At 32 the number is mostly prefill. The published section 3 figure of 9.97
# sits between the 32 and 128 readings, which is what a different output length
# looks like -- not a broken engine. Longer output also makes the two engines
# comparable on the part that dominates a real session.
#
# ⚠️ ABSOLUTES ARE NOT QUOTABLE ON THIS DEVICE. The same config read 5.32 inside
# a four-leg A/B and 6.94 run fresh -- about 25% drift from heat and back-to-back
# legs. Interleaving is what makes the RATIO survive that; block ordering would
# not. Clocks are already pinned (all eight cores on `performance` at their max
# frequency, verified, left that way by an earlier session).
#
# GUARDS: refuse to start if any server exists or the port is held; exactly one
# server per leg or the leg is void; read back that the kill engaged; trap so an
# interrupt leaves no orphan. Each of those answers a way this went wrong today,
# including three servers alive at once on one port with identical command lines.
set -u
DEV=/data/local/tmp
# Two containers, ONE checkpoint. Our GGUF loader's arch allowlist
# (model_loader.cpp kGgufArchArms) has eight rows and `qwen3` dense is not one,
# so the GGUF leg refuses to start -- verified here and on the Jetson, which is
# what makes it an engine capability boundary rather than one board's quirk.
# Our side therefore reads the safetensors the checkpoint was published as;
# llama.cpp keeps the BF16 GGUF converted from that same checkpoint.
# ⚠️ The container is then a second variable. On the Jetson its Vulkan tactic
# DID change with the container (GGUF took the scalar arm, safetensors the
# GEMV), caught only because a control leg logged the tactic. On CPU there is
# no equivalent log, so this comparison is a DEPLOYMENT one -- each engine on
# its own best path for its own container -- and not a kernel-for-kernel one.
OURS_MODEL=$DEV/lm/Qwen3-0.6B
LLAMA_MODEL=$DEV/lm/Qwen3-0.6B-BF16.gguf
OURS=$DEV/vllm-server
LLAMA=$DEV/llama2/llama-server
PORT=18091
THREADS=${THREADS:-6}
MAXTOK=${MAXTOK:-128}
LEGS=${LEGS:-"ours llama ours llama"}
OUT=$DEV/bf16-results.txt
CUR=""

nservers() { ps -A -o NAME 2>/dev/null | grep -cE '^(vllm-server|llama-server)$'; }
portheld() { ss -ltn 2>/dev/null | grep -c ":$PORT " ; }
cleanup() { [ -n "$CUR" ] && kill -9 "$CUR" 2>/dev/null; exit 0; }
trap cleanup EXIT INT TERM

N=$(nservers); P=$(portheld)
if [ "$N" != "0" ] || [ "$P" != "0" ]; then
    echo "REFUSING TO START: $N server(s) running, port listeners=$P"
    ps -A -o PID,RSS,NAME 2>/dev/null | grep -E '(vllm|llama)-server$'
    exit 1
fi
: > $OUT

W=""; i=0
while [ $i -lt 96 ]; do W="$W alpha bravo charlie delta"; i=$((i + 4)); done

probe() {
    U=$(cat /proc/sys/kernel/random/uuid 2>/dev/null || echo $$-$RANDOM)
    printf '{"model":"m","prompt":"%s %s","max_tokens":%s,"temperature":0}' "$U" "$W" "$MAXTOK" > $DEV/.b.json
    curl -s -o /dev/null -w '%{time_total}' -m 1800 -H 'Content-Type: application/json' \
        --data @$DEV/.b.json "http://127.0.0.1:$PORT/v1/completions"
}

echo "ANDROID BF16  Qwen3-0.6B (ours=safetensors, llama=GGUF)  threads=$THREADS  ~128 in / $MAXTOK out  interleaved"
free -m | sed -n 2p
echo
echo "engine   t1       t2       tok/s    cpu/wall  rssMB  AnonMB  FileMB"

for E in $LEGS; do
    if [ "$E" = "ours" ]; then
        VLLM_CPP_CPU_THREADS=$THREADS $OURS --model $OURS_MODEL --port $PORT \
            --served-model-name m --device cpu --max-model-len 4096 --num-blocks 512 \
            > $DEV/bf16-$E.log 2>&1 &
    else
        LD_LIBRARY_PATH=$DEV/llama2 $LLAMA -m $LLAMA_MODEL --port $PORT --host 127.0.0.1 \
            -c 4096 -np 1 -t $THREADS --alias m --no-webui > $DEV/bf16-$E.log 2>&1 &
    fi
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
        echo "$E   VOID (code=$CODE servers=$N)"
        tail -3 $DEV/bf16-$E.log
        kill -9 $CUR 2>/dev/null; CUR=""; sleep 3; continue
    fi

    probe > /dev/null 2>&1

    C0=$(awk '{print $14+$15}' /proc/$CUR/stat); W0=$(date +%s)
    T1=$(probe); T2=$(probe)
    W1=$(date +%s); C1=$(awk '{print $14+$15}' /proc/$CUR/stat)
    RSS=$(awk '/VmRSS/{print int($2/1024)}' /proc/$CUR/status)
    AN=$(awk '/RssAnon/{print int($2/1024)}' /proc/$CUR/status)
    FI=$(awk '/RssFile/{print int($2/1024)}' /proc/$CUR/status)

    RES=$(awk -v a="$T1" -v b="$T2" -v m="$MAXTOK" -v c0="$C0" -v c1="$C1" -v w0="$W0" -v w1="$W1" \
        'BEGIN{med=(a+b)/2; printf "%-8.2f %-8.2f", m/med, ((c1-c0)/100.0)/((w1-w0)>0?(w1-w0):1)}')
    printf "%-8s %-8s %-8s %s  %-6s %-7s %s\n" "$E" "$T1" "$T2" "$RES" "$RSS" "$AN" "$FI"
    echo "$E $T1 $T2 $RES $RSS $AN $FI" >> $OUT

    kill -TERM $CUR 2>/dev/null
    i=0; while [ $i -lt 20 ]; do kill -0 $CUR 2>/dev/null || break; sleep 1; i=$((i + 1)); done
    kill -9 $CUR 2>/dev/null; sleep 3
    LEFT=$(nservers); [ "$LEFT" = "0" ] || echo "  WARNING: $LEFT server(s) still alive"
    CUR=""
done
echo ANDROID_BF16_DONE
