from transformers import AutoTokenizer, AutoModel

tokenizer = AutoTokenizer.from_pretrained("microsoft/trocr-small-handwritten")

model = AutoModel.from_pretrained("microsoft/trocr-small-handwritten")
print(model)
