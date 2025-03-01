from transformers import (
    DataCollatorWithPadding,
    AutoTokenizer,
    AutoModelForSequenceClassification,
)
from datasets import *
from torch.utils.data import DataLoader
import torch

import evaluate


# 加载数据
data = load_dataset("csv", data_files="./ChnSentiCorp_htl_all.csv", split="train")
# 如果有空数据则返回None即丢弃空数据
data = data.filter(lambda x: x["review"] is not None)

# 划分数据集
data = data.train_test_split(test_size=0.1)

# tokensizer 定义
tokenizer = AutoTokenizer.from_pretrained("hfl/rbt3")


# 数据处理
def preprocess_function(examples):
    tokenizer_examples = tokenizer(examples["review"], truncation=True, max_length=128)
    tokenizer_examples["label"] = examples["label"]
    return tokenizer_examples


# 处理数据并且移除原数据列
data = data.map(
    preprocess_function, batched=True, remove_columns=data["train"].column_names
)

# 不懂
collator = DataCollatorWithPadding(tokenizer)

# 划分数据集
# 一个用来训练一个用来验证
trainloader = DataLoader(
    data["train"], batch_size=32, collate_fn=collator, shuffle=True
)
validestloader = DataLoader(
    data["test"], batch_size=32, collate_fn=collator, shuffle=True
)

model = AutoModelForSequenceClassification.from_pretrained("hfl/rbt3")
if torch.cuda.is_available():
    model = model.cuda()
# optimizer = adam(model.parameters(), lr=2e-5)
optimizer = torch.optim.Adam(model.parameters(), lr=2e-5)


# 评估相关
# 这里评估指标只选择了两个
# 其他的可参考： https://huggingface.co/tasks/text-classification
clf_metric = evaluate.combine(["accuracy", "f1"])


# 评估模型
def local_evaluate():
    model.eval()
    with torch.inference_mode():
        for batch in validestloader:
            if torch.cuda.is_available():
                batch = {k: v.cuda() for k, v in batch.items()}
            outputs = model(**batch)
            pred = torch.argmax(outputs.logits, dim=-1)
            clf_metric.add_batch(
                predictions=pred.long(), references=batch["labels"].long()
            )

    return clf_metric.compute()


# 训练模型
def train(epoch=3, log_step=100):
    global_step = 0
    for ep in range(epoch):
        model.train()
        for batch in trainloader:
            # 如果有英伟达的GPU并且支持CUDA的
            if torch.cuda.is_available():
                batch = {k: v.cuda() for k, v in batch.items()}
            optimizer.zero_grad()
            outputs = model(**batch)
            outputs.loss.backward()
            optimizer.step()
            # print(f"ep: {ep}, global_step: {global_step}, loss: {outputs.loss.item()}")
            if global_step % log_step == 0 or global_step == 0:
                print(
                    f"ep: {ep}, global_step: {global_step}, loss: {outputs.loss.item()}"
                )
            global_step += 1
        clf = local_evaluate()
        print(f"ep: {ep},{clf}")


# 开始训练
train()

# 测试模型
sen = "酒店很差"
id2_label = {0: "差评", 1: "好评"}
model.eval()
with torch.inference_mode():
    inputs = tokenizer(sen, return_tensors="pt")
    inputs = {k: v.cuda() for k, v in inputs.items()}
    logits = model(**inputs).logits
    pred = torch.argmax(logits, dim=-1)
    print(f"输出: {sen}\n模型预测结果: {id2_label[pred.item()]}")
