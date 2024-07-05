





from transformers import GPT2Tokenizer, GPT2LMHeadModel
 
model_name = "gpt2"
model = GPT2LMHeadModel.from_pretrained(model_name)
tokenizer = GPT2Tokenizer.from_pretrained(model_name)



input_text = "这在哪"
inputs = tokenizer.encode(input_text, return_tensors="pt")
outputs = model.generate(inputs, max_length=100, num_return_sequences=1)
 
generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(generated_text)
