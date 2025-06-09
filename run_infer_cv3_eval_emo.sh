SPK_LAB=utils/3D-Speaker
DNSMOS_LAB=utils/DNSMOS

export PYTHONPATH=${SPK_LAB}:${PYTHONPATH}

nj=1
asr_gpu=1

task=emotion_zeroshot
dumpdir=data/${task}
test_set="en zh" 

inference_dir="CV3-Eval"
. utils/parse_options.sh || exit 1;
inference_tag="${inference_dir}/${task}"
decode_dir=path/to/results/${task}

for lang in ${test_set}; do
    name_without_extension=$lang

    find ${decode_dir}/${name_without_extension}/wavs -name *.wav | awk -F '/' '{print $NF, $0}' | sed "s@\.wav @ @g" > ${decode_dir}/${name_without_extension}/wav.scp
    echo "Score Emotation for ${decode_dir}/${name_without_extension}"

    python scripts/eval_emotation.py ${decode_dir}/${name_without_extension}/wav.scp ${decode_dir}/${name_without_extension}/emo_result.txt >  ${decode_dir}/${name_without_extension}/emo_score.txt
done