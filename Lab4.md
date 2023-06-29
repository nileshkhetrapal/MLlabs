# Lab: Introduction to Hugging Face, Gradio, and Fine-Tuning a Model

In this updated lab, we'll introduce the Hugging Face Transformers library, Gradio, and the concept of tokenization. We'll also cover fine-tuning a pre-trained model. Our goal is to build and fine-tune a simple chatbot.

**Note**: This lab is intended to be run in Google Colab.

## Part 1: Introduction to Hugging Face Transformers

We'll start with installing the Hugging Face library and loading the DialoGPT model. 

```python
!pip install transformers

from transformers import AutoModelForCausalLM, AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained("microsoft/DialoGPT-medium")
model = AutoModelForCausalLM.from_pretrained("microsoft/DialoGPT-medium")
```

**Exercise 1.1**: What are some other models available in the Hugging Face model hub?

## Part 2: Introduction to Gradio

Gradio allows us to quickly create customizable UI components around our models.

```python
!pip install gradio
```

**Exercise 2.1**: Can you think of a use case for Gradio other than showcasing NLP models?

## Part 3: Introduction to Tokenization

Tokenization is the process of splitting text into individual 'tokens' or words.

**Exercise 3.1**: What are some challenges you might encounter when tokenizing text?

## Part 4: Fine-Tuning a Model

Fine-tuning involves slightly modifying the weights of a pre-trained model to make it perform better on a specific task. Let's fine-tune our DialoGPT model on a custom dataset. This can be any text data, but for simplicity, let's use a single repeated prompt-response pair:

```python
from transformers import TextDataset, DataCollatorForLanguageModeling

def fine_tune_model(model, tokenizer, prompt="Hello, how are you?", response="I'm good, thank you.", num_train_epochs=1):
    # This will create a dataset with the prompt-response pair repeated 100 times
    text = (prompt + "\n" + response + "\n") * 100
    with open("train.txt", "w") as f:
        f.write(text)
    
    train_dataset = TextDataset(
        tokenizer=tokenizer,
        file_path="train.txt",
        block_size=128)
    
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer, mlm=False,
    )

    from transformers import Trainer, TrainingArguments

    training_args = TrainingArguments(
        output_dir="./results",
        overwrite_output_dir=True,
        num_train_epochs=num_train_epochs,
        per_device_train_batch_size=1,
        save_steps=10_000,
        save_total_limit=2,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=train_dataset,
    )
    
    trainer.train()
    
fine_tune_model(model, tokenizer)
```

**Exercise 4.1**: How does fine-tuning work? What's happening under the hood when we fine-tune our model?

## Part 5: Building a Chatbot

Now that we have our fine-tuned model, let's use it to build a chatbot:

```python
def generate_response(input_text):
    input_text = input_text.strip()
    input_ids = tokenizer.encode(input_text + tokenizer.eos_token, return_tensors='pt')
    response_ids = model.generate(input_ids, max_length=1000, pad_token_id=tokenizer.eos_token_id)
    response_text = tokenizer.decode(response_ids[:, input_ids

.shape[-1]:][0], skip_special_tokens=True)
    return response_text.strip()

import gradio as gr
iface = gr.Interface(fn=generate_response, inputs="text", outputs="text")
iface.launch()
```

**Exercise 5.1**: How does the performance of the chatbot change after fine-tuning?

## Part 6: Conclusion and Further Reading

You've built a chatbot using Hugging Face and Gradio and learned about fine-tuning a model!

Here are some additional resources:

- [Hugging Face Fine-Tuning Guide](https://huggingface.co/transformers/training.html)
- [Practical Guide to Fine-Tuning Transformers](https://towardsdatascience.com/fine-tuning-transformers-723e6c2ea7ea)

**Exercise 6.1**: What are some potential issues with the way we fine-tuned our model? 

**Final Project**: Fine-tune a chatbot on a more complex dataset, such as the [Persona-Chat dataset](https://github.com/facebookresearch/ParlAI/tree/master/parlai/tasks/personachat). Try out different models from the Hugging Face model hub. Evaluate the chatbot's performance before and after fine-tuning and discuss the findings.
