from transformers import AutoModelForCausalLM

model = AutoModelForCausalLM.from_pretrained('microsoft/phi-4')

print(model)