
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

device_map= {
    'gpt_neox.embed_in': 0,
    'gpt_neox.layers': 0,
    'gpt_neox.final_layer_norm': 0,
    'embed_out': 0
}

# model_path = "models/gpt-neo-1.3B_out"
model_path = "zirui3/gpt_1.4B_oa_instruct"
model = AutoModelForCausalLM.from_pretrained(model_path, device_map=device_map, torch_dtype=torch.float16, load_in_8bit=True )
tokenizer = AutoTokenizer.from_pretrained(model_path)

def generate_from_model(model, tokenizer):
    encoded_input = tokenizer(text, return_tensors='pt')
    output_sequences = model.generate(
                                    input_ids=encoded_input['input_ids'].cuda(0),
                                    do_sample=True,
                                    max_new_tokens=35,
                                    num_return_sequences=1,
                                    top_p=0.95,
                                    temperature=0.5,
                                    penalty_alpha=0.6,
                                    top_k=4,
                                    output_scores=True,
                                    return_dict_in_generate=True,
                                    repetition_penalty=1.03,
                                    eos_token_id=0,
                                    use_cache=True
                                  )
    gen_sequences = output_sequences.sequences[:, encoded_input['input_ids'].shape[-1]:]
    for sequence in gen_sequences:
        new_line=tokenizer.decode(sequence, skip_special_tokens=True)
        print(new_line)

#text = "User: Why is everyone so excited about AI chatbots these days?"
text = "User:implement quick sort with python"
generate_from_model(model, tokenizer)