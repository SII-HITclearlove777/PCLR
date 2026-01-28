#!/bin/bash

gpu_list="${CUDA_VISIBLE_DEVICES:-0}"
IFS=',' read -ra GPULIST <<< "$gpu_list"

CHUNKS=${#GPULIST[@]}

STAGE=$1

MODELPATH=$2

RESULT_DIR="../results/lrp/ImageNet"

for IDX in $(seq 0 $((CHUNKS-1))); do
    CUDA_VISIBLE_DEVICES=${GPULIST[$IDX]} python -m lrp.eval.model_vqa \
        --model-path $MODELPATH \
        --question-file /path/coin/json/Instructions-Origin/ImageNet/test.json \
        --image-folder /path/coin/images \
        --answers-file $RESULT_DIR/$STAGE/${CHUNKS}_${IDX}.jsonl \
        --num-chunks $CHUNKS \
        --chunk-idx $IDX \
        --temperature 0 \
        --conv-mode vicuna_v1 &
done

wait

output_file=$RESULT_DIR/$STAGE/merge.jsonl

# Clear out the output file if it exists.
> "$output_file"

# Loop through the indices and concatenate each file.
for IDX in $(seq 0 $((CHUNKS-1))); do
    cat $RESULT_DIR/$STAGE/${CHUNKS}_${IDX}.jsonl >> "$output_file"
done

python -m lrp.eval.eval_ImagetNet \
    --test-file /path/coin/images/Instructions-Origin/ImageNet/test.json \
    --result-file $output_file \
    --output-dir $RESULT_DIR/$STAGE \


