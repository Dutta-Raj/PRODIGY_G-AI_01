import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer, Trainer, TrainingArguments
from datasets import load_dataset

def load_data(file_path):
    """Load and preprocess dataset"""
    dataset = load_dataset('text', data_files=file_path)
    return dataset

def train_model(dataset, config):
    """Fine-tune GPT-2 model"""
    tokenizer = GPT2Tokenizer.from_pretrained(config['model_name'])
    model = GPT2LMHeadModel.from_pretrained(config['model_name'])
    
    # Training setup
    training_args = TrainingArguments(
        output_dir=config['output_dir'],
        num_train_epochs=config['epochs'],
        per_device_train_batch_size=config['batch_size'],
        save_steps=config['save_steps']
    )
    
    # Train and save model
    trainer = Trainer(model=model, args=training_args, train_dataset=dataset)
    trainer.train()
    model.save_pretrained(config['output_dir'])
    return model

if __name__ == "__main__":
    config = {
        'model_name': 'gpt2',
        'epochs': 3,
        'batch_size': 4,
        'output_dir': '../models/gpt2-finetuned',
        'save_steps': 10_000
    }
    
    dataset = load_data('../data/raw/training_data.txt')
    train_model(dataset, config)
