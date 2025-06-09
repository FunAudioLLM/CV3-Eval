set -x

meta_lst=$1
output_dir=$2
lang=$3
num_job=$4

wav_wav_text=$output_dir/wav_res_ref_text
score_file=$output_dir/wav_res_ref_text.wer

workdir=$(cd $(dirname $0); cd ../; pwd)

python3 get_wav_res_ref_text.py $meta_lst $output_dir/wavs $wav_wav_text
# python3 prepare_ckpt.py

timestamp=$(date +%s)
thread_dir=${output_dir}/tmp/thread_metas_$timestamp/
out_dir=${thread_dir}/results/

mkdir -p $out_dir

num=`wc -l $wav_wav_text | awk -F' ' '{print $1}'`
num_per_thread=`expr $num / $num_job + 1`

split -l $num_per_thread --additional-suffix=.lst -d $wav_wav_text $thread_dir/thread-


num_job_minus_1=`expr $num_job - 1`
if [ ${num_job_minus_1} -ge 0 ];then
	for rank in $(seq 0 $((num_job - 1))); do
		sub_score_file=$out_dir/thread-0$rank.wer.out
		CUDA_VISIBLE_DEVICES=${rank} python3 run_wer.py $thread_dir/thread-0$rank.lst $sub_score_file $lang &
	done
fi
wait

# rm $wav_wav_text
# rm -f $out_dir/merge.out

cat $out_dir/thread-0*.wer.out >>  $out_dir/merge.out
python3 average_wer.py $out_dir/merge.out $score_file
