import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

class Model:
    def __init__(self, name, max_tokens=256, temperature=0.7, padding_side="left", thinking=False):
        self.name = name
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.thinking = thinking

        self.tokenizer = AutoTokenizer.from_pretrained(name, trust_remote_code=True, padding_side=padding_side)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model = AutoModelForCausalLM.from_pretrained(
            name,
            trust_remote_code=True,
            torch_dtype="auto",
            device_map="auto"
        )

    def generate_response(self, prompt, apply_chat_template=True, n_completions=1):
        if apply_chat_template:
            text = self.tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt}],
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=self.thinking
            )
        else:
            text = prompt


        inputs = self.tokenizer(text, return_tensors="pt", padding=True).to(self.model.device)

        do_sample = self.temperature > 0
        generated_ids = self.model.generate(
            **inputs,
            max_new_tokens=self.max_tokens,
            temperature=self.temperature if do_sample else 1.0,
            do_sample=do_sample,
            num_return_sequences=n_completions,
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=self.tokenizer.eos_token_id
        )

        # input_length = inputs.input_ids.shape[1]
        # output_ids = generated_ids[:, input_length:]
        generated_texts = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=False)

        return generated_texts
    
    def batched_generate_response(self, prompts, apply_chat_template=True, batch_size=32, n_completions=1):
        if apply_chat_template:
            texts = [self.tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt}],
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=self.thinking
            ) for prompt in prompts]
        else:
            texts = prompts

        all_generated_texts = []
        for i in range(0, len(texts), batch_size):
            start_idx = i
            end_idx = min(i + batch_size, len(texts))
            batch_texts = texts[start_idx:end_idx]
            inputs = self.tokenizer(batch_texts, return_tensors="pt", padding=True).to(self.model.device)
            do_sample = self.temperature > 0
            generated_ids = self.model.generate(
                **inputs,
                max_new_tokens=self.max_tokens,
                temperature=self.temperature if do_sample else 1.0,
                do_sample=do_sample,
                num_return_sequences=n_completions,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id
            )
            generated_texts = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=False)
            generated_texts = [text.replace(self.tokenizer.pad_token, "") for text in generated_texts]
            all_generated_texts.extend(generated_texts) 
        return all_generated_texts
    
if __name__ == "__main__":
    model_tag = "Qwen/Qwen3-4B"
    model = Model(
        name=model_tag,
        max_tokens=2048,
        temperature=0,
        padding_side="left",
        thinking=False
    )