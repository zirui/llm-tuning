

# installation

```bash
pip install -r requirements.txt
```


# train

## GPT
```bash
model_path="your_in_path/gpt-neo-1.3B"
output_dir="your_out_path/gpt-neo-1.3B_out"
data_dir="your_data_dir"

deepspeed --num_gpus=1 minimal_trainer.py \
    --deepspeed ds_config_gpt.json \
    --model_name_or_path $model_path \
    --data_dir $data_dir \
    --do_train \
    --do_eval \
    --block_size 280 \
    --fp16 \
    --overwrite_cache \
    --evaluation_strategy="steps" \
    --output_dir $output_dir \
    --overwrite_output_dir \
    --num_train_epochs 1 \
    --eval_steps 200 \
    --gradient_accumulation_steps 1 \
    --per_device_train_batch_size 32 \
    --use_fast_tokenizer True \
    --learning_rate 5e-06 \
    --warmup_steps 5
```

## LLaMA

```bash
python train_llam.py \
    --data_path="path/to/your/data" \
    --micro_batch_size=8 \
    --batch_size=128 \
    --lr=3e-4 \
    --epochs=3 \
    --output_dir="lora-alpaca" \
    --model_pretrained_name="decapoda-research/llama-30b-hf"
```

# inference
# GPT
```
python infer.py
```

## LoRA
```
python infer_lora.py \
    --path_to_lora_adapters="tloen/alpaca-lora-7b" \
    --pretrained_model="decapoda-research/llama-7b-hf" \

```

# TODO

- [ ] T5 support
- [ ] GLM support