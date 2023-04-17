import requests
from PIL import Image
from transformers import TrOCRProcessor, VisionEncoderDecoderModel

# load image from the IAM database
image = Image.open("examples/Red.png").convert("RGB")

print("[INFO] Load pretrained TrOCRProcessor")
processor = TrOCRProcessor.from_pretrained("microsoft/trocr-large-handwritten")
print("[INFO] Load pretrained VisionEncoderDecoderModel")
model = VisionEncoderDecoderModel.from_pretrained("microsoft/trocr-large-handwritten")

print("[INFO] Preprocess")
pixel_values = processor(images=image, return_tensors="pt").pixel_values
print("[INFO] Inference")
generated_ids = model.generate(pixel_values)
print("[INFO] Postprocess")
generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]

print(generated_text)
