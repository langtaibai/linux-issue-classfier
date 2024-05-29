import pandas as pd
from datasets import Dataset
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification, Trainer, TrainingArguments
import torch

# 读取数据
data = pd.read_csv('data/linux_issues.csv')

# 将标签转换为数字
label_to_id = {label: idx for idx, label in enumerate(data['category'].unique())}
data['labels'] = data['category'].map(label_to_id)

# 将数据转换为Dataset对象
dataset = Dataset.from_pandas(data)

# 初始化tokenizer和模型
tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
model = DistilBertForSequenceClassification.from_pretrained('distilbert-base-uncased', num_labels=len(label_to_id))

# 数据预处理
def preprocess_function(examples):
    return tokenizer(examples['description'], truncation=True, padding='max_length', max_length=100)

tokenized_dataset = dataset.map(preprocess_function, batched=True)

# 设置格式
tokenized_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'labels'])

# 自定义Trainer类，覆盖compute_loss方法
class CustomTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits
        loss_fct = torch.nn.CrossEntropyLoss()
        loss = loss_fct(logits, labels)
        return (loss, outputs) if return_outputs else loss

# 训练参数
training_args = TrainingArguments(
    output_dir="./results",
    evaluation_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=10,
    per_device_eval_batch_size=10,
    num_train_epochs=3,
    weight_decay=0.01,
)

# 初始化Trainer
trainer = CustomTrainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
    eval_dataset=tokenized_dataset,
    tokenizer=tokenizer,
)

# 模型训练
trainer.train()

# 模型保存
model.save_pretrained("./model")
tokenizer.save_pretrained("./model")
