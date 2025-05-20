from transformers import GPT2LMHeadModel, GPT2Tokenizer
import torch

def load_model(model_path):
    """Load fine-tuned model"""
    tokenizer = GPT2Tokenizer.from_pretrained(model_path)
    model = GPT2LMHeadModel.from_pretrained(model_path)
    return model, tokenizer

def generate_text(prompt, model, tokenizer, max_length=100):
    """Generate text from prompt"""
    inputs = tokenizer.encode(prompt, return_tensors='pt')
    outputs = model.generate(inputs, max_length=max_length)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

if __name__ == "__main__":
    model, tokenizer = load_model('../models/gpt2-finetuned')
    print(generate_text("The future of AI will", model, tokenizer))
