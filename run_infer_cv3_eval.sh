SPK_LAB=utils/3D-Speaker
DNSMOS_LAB=utils/DNSMOS

export PYTHONPATH=${SPK_LAB}:${PYTHONPATH}

nj=1
asr_gpu=1

task=zero_shot
dumpdir=data/${task}
# test_set="zh en hard_zh hard_en ja ko" 
test_set="zh" 

# task=cross_lingual_zeroshot
# dumpdir=data/${task}
# # test_set="to_zh to_en to_hard_zh to_hard_en to_ja to_ko" 
# test_set="to_zh to_en" 

# task=emotion_zeroshot
# dumpdir=data/${task}
# test_set="en zh" 

# task=subjective_zeroshot
# dumpdir=data/${task}

# task=subjective_continue
# dumpdir=data/${task}
# test_set="emotion rhyme speed volume"

inference_dir="CV3-Eval"
. utils/parse_options.sh || exit 1;
inference_tag="${inference_dir}/${task}"
decode_dir=path/to/results/${task}

for lang in ${test_set}; do
    name_without_extension=$lang

    echo "Score WER for ${decode_dir}/${name_without_extension}"

    bash scripts/run_score_wer.sh ${dumpdir}/${name_without_extension}/text ${decode_dir}/${name_without_extension} ${name_without_extension} ${asr_gpu}

    find ${decode_dir}/${name_without_extension}/wavs -name *.wav | awk -F '/' '{print $NF, $0}' | sed "s@\.wav @ @g" > ${decode_dir}/${name_without_extension}/wav.scp
    
    echo "Score 3DSpeaker for ${decode_dir}/${name_without_extension}"
    python scripts/eval_speaker_similarity.py \
    --model_id damo/speech_eres2net_sv_en_voxceleb_16k \
    --local_model_dir ${SPK_LAB}/pretrained \
    --prompt_wavs ${dumpdir}/${name_without_extension}/prompt_wav.scp \
    --hyp_wavs ${decode_dir}/${name_without_extension}/wav.scp \
    --log_file ${decode_dir}/${name_without_extension}/spk_simi_scores.txt \
    --devices "0"

    echo "Score DNSMOS  for ${decode_dir}/${name_without_extension}"
    python ${DNSMOS_LAB}/dnsmos_local_wavscp.py -t ${decode_dir}/${name_without_extension}/wav.scp -e ${DNSMOS_LAB} -o ${decode_dir}/${name_without_extension}/mos.csv

    cat ${decode_dir}/${name_without_extension}/mos.csv | sed '1d' |awk -F ',' '{ sum += $NF; count++ } END { if (count > 0) print sum / count }'  > ${decode_dir}/${name_without_extension}/dnsmos_mean.txt
done
