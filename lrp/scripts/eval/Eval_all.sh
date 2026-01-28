CUDA_VISIBLE_DEVICES=0,1,2,3 bash ./lrp/scripts/eval/1_eval_sqa.sh ScienceQA-7b /path/lrp_model/lrp/checkpoints/1_ScienceQA_1
CUDA_VISIBLE_DEVICES=0,1,2,3 bash ./lrp/scripts/eval/2_eval_textqa.sh TextVQA-7b /path/lrp_model/lrp/checkpoints/2_TextVQA_1
CUDA_VISIBLE_DEVICES=0,1,2,3 bash ./lrp/scripts/eval/3_eval_ImageNet.sh ImageNet-7b /path/lrp_model/lrp/checkpoints/3_ImageNet_1
CUDA_VISIBLE_DEVICES=0,1,2,3 bash ./lrp/scripts/eval/4_eval_gqa.sh GQA-7b /path/lrp_model/lrp/checkpoints/4_GQA_1
CUDA_VISIBLE_DEVICES=0,1,2,3 bash ./lrp/scripts/eval/5_eval_vizwiz.sh VizWiz-7b /path/lrp_model/lrp/checkpoints/5_VizWiz_1
CUDA_VISIBLE_DEVICES=0,1,2,3 bash ./lrp/scripts/eval/6_eval_grounding.sh Grounding-7b /path/lrp_model/lrp/checkpoints/6_Grounding_1
CUDA_VISIBLE_DEVICES=0,1,2,3 bash ./lrp/scripts/eval/7_eval_vqav2.sh VQAv2-7b /path/lrp_model/lrp/checkpoints/7_VQAv2_1
CUDA_VISIBLE_DEVICES=0,1,2,3 bash ./lrp/scripts/eval/8_eval_ocrvqa.sh OCRVQA-7b /path/lrp_model/lrp/checkpoints/8_OCRVQA_1



CUDA_VISIBLE_DEVICES=0,1,2,3 bash ./lrp/scripts/eval/1_eval_sqa.sh OCRVQA-7b /path/lrp_model/lrp/checkpoints/8_OCRVQA_1
CUDA_VISIBLE_DEVICES=0,1,2,3 bash ./lrp/scripts/eval/2_eval_textqa.sh OCRVQA-7b /path/lrp_model/lrp/checkpoints/8_OCRVQA_1
CUDA_VISIBLE_DEVICES=0,1,2,3 bash ./lrp/scripts/eval/3_eval_ImageNet.sh OCRVQA-7b /path/lrp_model/lrp/checkpoints/8_OCRVQA_1
CUDA_VISIBLE_DEVICES=0,1,2,3 bash ./lrp/scripts/eval/4_eval_gqa.sh OCRVQA-7b /path/lrp_model/lrp/checkpoints/8_OCRVQA_1
CUDA_VISIBLE_DEVICES=0,1,2,3 bash ./lrp/scripts/eval/5_eval_vizwiz.sh OCRVQA-7b /path/lrp_model/lrp/checkpoints/8_OCRVQA_1
CUDA_VISIBLE_DEVICES=0,1,2,3 bash ./lrp/scripts/eval/6_eval_grounding.sh OCRVQA-7b /path/lrp_model/lrp/checkpoints/8_OCRVQA_1
CUDA_VISIBLE_DEVICES=0,1,2,3 bash ./lrp/scripts/eval/7_eval_vqav2.sh OCRVQA-7b /path/lrp_model/lrp/checkpoints/8_OCRVQA_1