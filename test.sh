# store weight in the dir ./weight like mvtec_prompt.pt, mvtec_adaptor.pt which are trained on mvtec dataset
# and visa_prompt.pt, visa_adaptor.pt which are trained on visa dataset
# zero-shot
data_dir=/data2/fqq 
CUDA_VISIBLE_DEVICES=0 python main.py --log_dir ./log/zero-shot --dataset visa --test_dataset mvtec --weight ./weight --data_dir $data_dir
CUDA_VISIBLE_DEVICES=0 python main.py --log_dir ./log/zero-shot --dataset mvtec --weight ./weight --data_dir $data_dir

# few-shot
for few_shot in 1 2 4
do
    for i in $(seq 1 5)
    do
        CUDA_VISIBLE_DEVICES=4 python main.py --log_dir ./log/few_shot/${few_shot}/mvtec --dataset mvtec --weight ./weight --test_dataset visa --fewshot ${few_shot} --seed -1 --data_dir $data_dir
        CUDA_VISIBLE_DEVICES=1 python main.py --log_dir ./log/few_shot/${few_shot}/visa --dataset visa --weight ./weight --test_dataset mvtec --fewshot ${few_shot} --seed -1 --data_dir $data_dir
    done
done

