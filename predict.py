import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import sys

# 加载模型和tokenizer
model_name = "./model"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)

# 分类标签
labels = ["Graphic", "Video", "Audio"]  # 根据您的数据集进行调整

def predict_issue(issue_description):
    inputs = tokenizer(issue_description, return_tensors="pt", truncation=True)
    outputs = model(**inputs)
    predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
    predicted_label = torch.argmax(predictions, dim=1).item()
    return labels[predicted_label]

if __name__ == "__main__":
    issue_description = sys.argv[1]
    category = predict_issue(issue_description)
    print(f"The issue category is: {category}")

