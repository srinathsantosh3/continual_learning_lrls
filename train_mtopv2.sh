#MULTIHEAD
#exp_name=$(uuidgen)
#python ./src/train_mtop_mmulti_head.py --all_langs en_de_fr_es_hi_th  --result_file $exp_name.csv --replay_freq 10 --epochs 10 \
#--replay  \
#--codeswitch_type pos \ 
#--UPOS_EVAL NOUN --CODE_SWITCH_PATH /home/epsilon/Workbenches/ML_VLM/continual_training/MTOP/dicts >$exp_name
#en_de_fr_es_hi_th

#ADAPTER
#exp_name=$(uuidgen)
#python ./src/train_mtop_adapter.py --all_langs en_de_fr_hi_es_th  --result_file $exp_name.csv --replay_freq 10 --epochs 10 \
#--replay_type  code_switch --replay  \
#--codeswitch_type pos \
#--UPOS_EVAL NOUN --CODE_SWITCH_PATH /home/epsilon/Workbenches/ML_VLM/continual_training/MTOP/dicts >$exp_name

#ADVABCED_ADAPTER


#to enable replay use  --replay   \
exp_name=$(uuidgen)
python ./src/train_mtop_adapterv2.py --all_langs ${10:- en_de_fr_hi_es_th}  --result_file $exp_name.csv \
--replay_freq 10 --epochs ${1:-10} \
--replay_type  ${2:-cs_memory}    \
--codeswitch_type ${3:-pos} \
--device_id ${4:-0} \
--memory_size ${5:-1000} \
--model_path /mnt/MIG_archive/Datasets/epsilon/datasets/checkpoints/mtop \
--exp_name $exp_name \
--task  \
--replay \
--lang_cs ${12:-en} \
--roberta_adapters \
--train_lang_adapter \
--train_from_scratch th_bn \
--replay_adapter ${6:-ccl} \
--code_switch_ratio ${7:-0.15} \
--model  ${8:-xlm-roberta-base}  \
--en_cs_percent 1 \
--batch_size ${11:-16} \
--UPOS_EVAL ${9:-NOUN} --CODE_SWITCH_PATH /home/epsilon/Workbenches/ML_VLM/continual_training/MTOP/dicts >$exp_name 2>&1
#-- task is important, do not ignore. If true enables language and task adaptersTODO: remove the flag from the code as it is mandatory.

#replay_type:  code_switch, memory, cs_memory, single_batch_memory, all_batches= all batches
#Codeswitch_type = pos, random
#model = xlm-roberta-base, bert-base-multilingual-cased

#exp_name=$(uuidgen)
#python ./src/train_mtop_mreplay.py --all_langs en_de_fr_hi_es_th  --result_file $exp_name.csv --replay_freq 10 --epochs 10 \
#--replay_type  code_switch --replay  \
#--codeswitch_type pos \
#--UPOS_EVAL NOUN --CODE_SWITCH_PATH /home/epsilon/Workbenches/ML_VLM/continual_training/MTOP/dicts >$exp_name
