CUDA_VISIBLE_DEVICES=0,1,2,3 bash ./lrp_qwen/scripts/eval/1_eval_sqa.sh ScienceQA /path/lrp_model/lrp_qwen/checkpoints/1_ScienceQA_1
CUDA_VISIBLE_DEVICES=0,1,2,3 bash ./lrp_qwen/scripts/eval/2_eval_textqa.sh TextVQA /path/lrp_model/lrp_qwen/checkpoints/2_TextVQA_1
CUDA_VISIBLE_DEVICES=0,1,2,3 bash ./lrp_qwen/scripts/eval/3_eval_ImageNet.sh ImageNet /path/lrp_model/lrp_qwen/checkpoints/3_ImageNet_1
CUDA_VISIBLE_DEVICES=0,1,2,3 bash ./lrp_qwen/scripts/eval/4_eval_gqa.sh GQA /path/lrp_model/lrp_qwen/checkpoints/4_GQA_1
CUDA_VISIBLE_DEVICES=0,1,2,3 bash ./lrp_qwen/scripts/eval/5_eval_vizwiz.sh VizWiz /path/lrp_model/lrp_qwen/checkpoints/5_VizWiz_1
CUDA_VISIBLE_DEVICES=0,1,2,3 bash ./lrp_qwen/scripts/eval/6_eval_grounding.sh Grounding /path/lrp_model/lrp_qwen/checkpoints/6_Grounding_1
CUDA_VISIBLE_DEVICES=0,1,2,3 bash ./lrp_qwen/scripts/eval/7_eval_vqav2.sh VQAv2 /path/lrp_model/lrp_qwen/checkpoints/7_VQAv2_1
CUDA_VISIBLE_DEVICES=0,1,2,3 bash ./lrp_qwen/scripts/eval/8_eval_ocrvqa.sh OCRVQA /path/lrp_model/lrp_qwen/checkpoints/8_OCRVQA_1



CUDA_VISIBLE_DEVICES=0,1,2,3 bash ./lrp_qwen/scripts/eval/1_eval_sqa.sh OCRVQA /path/lrp_model/lrp_qwen/checkpoints/8_OCRVQA_1
CUDA_VISIBLE_DEVICES=0,1,2,3 bash ./lrp_qwen/scripts/eval/2_eval_textqa.sh OCRVQA /path/lrp_model/lrp_qwen/checkpoints/8_OCRVQA_1
CUDA_VISIBLE_DEVICES=0,1,2,3 bash ./lrp_qwen/scripts/eval/3_eval_ImageNet.sh OCRVQA /path/lrp_model/lrp_qwen/checkpoints/8_OCRVQA_1
CUDA_VISIBLE_DEVICES=0,1,2,3 bash ./lrp_qwen/scripts/eval/4_eval_gqa.sh OCRVQA /path/lrp_model/lrp_qwen/checkpoints/8_OCRVQA_1
CUDA_VISIBLE_DEVICES=0,1,2,3 bash ./lrp_qwen/scripts/eval/5_eval_vizwiz.sh OCRVQA /path/lrp_model/lrp_qwen/checkpoints/8_OCRVQA_1
CUDA_VISIBLE_DEVICES=0,1,2,3 bash ./lrp_qwen/scripts/eval/6_eval_grounding.sh OCRVQA /path/lrp_model/lrp_qwen/checkpoints/8_OCRVQA_1
CUDA_VISIBLE_DEVICES=0,1,2,3 bash ./lrp_qwen/scripts/eval/7_eval_vqav2.sh OCRVQA /path/lrp_model/lrp_qwen/checkpoints/8_OCRVQA_1