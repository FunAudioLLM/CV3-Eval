input_text=$(readlink -f "$1")
output_dir=$(readlink -f "$2")
lang=$3
ngpu=$4

cd utils

bash cal_wer.sh ${input_text} ${output_dir} $lang ${ngpu}

cd ../..
