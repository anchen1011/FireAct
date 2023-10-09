# Fine-tune


## Data

See [Fine-tune Data](https://github.com/anchen1011/FireAct/tree/main/data).


## Training

#### Llama/CodeLlama LoRA

Example of fine-tuning Llama-2-7b-chat-hf with HotpotQA trajectories:

(This command was tested on single RTX 4090 24GB GPU)

```
cd finetune/llama_lora
python finetune.py \
     --base_model meta-llama/Llama-2-7b-chat-hf \
     --data_path ../../data/finetune/alpaca_format/hotpotqa.json\
     --micro_batch_size 16 \
     --num_epochs 30 \
     --output_dir ../models/lora/[LORA NAME] \
     --val_set_size 0.01 \
     --cutoff_len 512 \
```


#### Llama/CodeLlama Full Model

Example of fine-tuning Llama-2-7b-chat-hf with HotpotQA trajectories:

(This command was tested on four A100 80GB GPUs)

```
cd finetune/llama_full
torchrun --nnodes 1 --nproc_per_node 4 finetune.py \
    --model_name_or_path meta-llama/Llama-2-7b-chat-hf \
    --data_path ../../data/finetune/alpaca_format/hotpotqa.json \
    --bf16 True \
    --output_dir ../models/full_models/[FULL MODEL NAME] \
    --num_train_epochs 30 \
    --per_device_train_batch_size 4 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 8 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 2000 \
    --save_total_limit 1 \
    --learning_rate 2e-3 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --fsdp "full_shard auto_wrap" \
    --fsdp_transformer_layer_cls_to_wrap 'LlamaDecoderLayer' \
    --tf32 True
```

#### OpenAI GPT-3.5

See [OpenAI Fine-tuning Guide](https://platform.openai.com/docs/guides/fine-tuning)

## References
1. Our Llama full model training code is based on [tatsu-lab/stanford_alpaca](https://github.com/tatsu-lab/stanford_alpaca)
2. Our Llama LoRA training code is based on [tloen/alpaca-lora](https://github.com/tloen/alpaca-lora)
3. Our GPT fine-tuning code is based on [anchen1011/chatgpt-finetune-ui](https://github.com/anchen1011/chatgpt-finetune-ui/)
