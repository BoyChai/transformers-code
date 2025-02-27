# 文本分类示例
# 导包
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import pandas as pd
from torch.utils.data import Dataset
from torch.utils.data import random_split
from torch.utils.data import DataLoader
import torch
import torch.optim as adam


# 创建DataSet
class MyDataset(Dataset):
    def __init__(self) -> None:
        super().__init__()
        self.data = pd.read_csv("./ChnSentiCorp_htl_all.csv")
        self.data = self.data.dropna()

    def __getitem__(self, index):
        return self.data.iloc[index]["review"], self.data.iloc[index]["label"]

    def __len__(self):
        return len(self.data)


dataset = MyDataset()
# 测试输出
# for i in range(5):
#     print(dataset[i])

# 划分数据集
# 将 dataset（包含所有酒店评论数据）随机分成两部分：
# trainset：90% 的数据，用于后续模型训练。
# validest：10% 的数据，用于验证模型效果。
trainset, validest = random_split(dataset, lengths=[0.9, 0.1])


# tokenizer
tokenizer = AutoTokenizer.from_pretrained("E:/AI/rbt3")


# 数据处理
def collate_func(batch):
    texts, labels = [], []
    for item in batch:
        # texts.append(item[0])
        texts.append(str(item[0]))  # 确保是字符串类型
        labels.append(item[1])
    inputs = tokenizer(
        texts,
        max_length=128,
        padding="max_length",
        truncation=True,
        return_tensors="pt",
    )
    inputs["labels"] = torch.tensor(labels)
    return inputs


# 创建DataLoader
# DataLoader是用来把数据转换为适合模型训练和评估的批量数据加载器。
# 这里的batch_size=32是每一批的数据大小，shuffle=True表示打乱数据顺序。
trainloader = DataLoader(trainset, batch_size=8, shuffle=True, collate_fn=collate_func)
validestloader = DataLoader(
    validest, batch_size=8, shuffle=True, collate_fn=collate_func
)
# 测试读取
# print(next(enumerate(trainloader))[1])

# 创建模型和优化器
model = AutoModelForSequenceClassification.from_pretrained("E:/AI/rbt3")
if torch.cuda.is_available():
    model = model.cuda()
# optimizer = adam(model.parameters(), lr=2e-5)
optimizer = torch.optim.Adam(model.parameters(), lr=2e-5)


# 评估模型
def evaluate():
    model.eval()
    acc_num = 0
    with torch.inference_mode():
        for batch in validestloader:
            if torch.cuda.is_available():
                batch = {k: v.cuda() for k, v in batch.items()}
            outputs = model(**batch)
            pred = torch.argmax(outputs.logits, dim=-1)
            acc_num += (pred == batch["labels"].long()).float().sum()
    return acc_num / len(validest)


# 训练模型
def train(epoch=3, log_step=100):
    global_step = 0
    for ep in range(epoch):
        print(f"epoch: {ep}")
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
        acc = evaluate()
        print(f"ep: {ep}, acc: {acc}")


# 开始训练
try:
    train()
except Exception as e:
    print(f"Error: {e}")
