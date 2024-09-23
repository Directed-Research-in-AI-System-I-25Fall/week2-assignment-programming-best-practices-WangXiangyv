from datasets import load_dataset, Dataset, DatasetDict
from transformers import pipeline

# Because of some connection error, I download the test split of mnist from huggingface manually.
dataset = Dataset.from_parquet("test-00000-of-00001.parquet", split='test')

images = dataset['image']
labels = dataset['label']

# For simpe, use huggingface pipeline directly.
image_classifier = pipeline("image-classification", model="microsoft/resnet-50", device="cuda")

preds = image_classifier(images)

preds = [p['label'] for p in preds[0]]

acc = 0

for i in range(len(preds)):
    if str(labels[i]) == preds[i]:
        acc += 1

print(f"acc: {acc/len(preds)}")

# Final acc is 0.0