# 导包
from datasets import *
from transformers import AutoTokenizer


# 加载数据集
# https://huggingface.co/datasets/madao33/new-title-chinese
def load():
    data = load_dataset("madao33/new-title-chinese")
    print(data)


# 加载数据任务子集
# https://huggingface.co/datasets/nyu-mll/glue
def load_subset():
    data = load_dataset("nyu-mll/glue", "cola")
    print(data)


# 加载一个任务集的子集
# https://huggingface.co/datasets/madao33/new-title-chinese
def load_subset_task():
    data = load_dataset("madao33/new-title-chinese", split="train")
    print(data)
    data = load_dataset("madao33/new-title-chinese", split="train[:50%]")
    print(data)
    data = load_dataset(
        "madao33/new-title-chinese", split=["train[50%:]", "train[:10%]"]
    )
    print(data)


# 数据集的查看
# https://huggingface.co/datasets/madao33/new-title-chinese
def dataset_info():
    data = load_dataset("madao33/new-title-chinese")
    # print(data["train"][0])
    print(data["train"]["title"][:5])


# 划分
# https://huggingface.co/datasets/madao33/new-title-chinese
def split_dataset():
    data = load_dataset("madao33/new-title-chinese")
    data = data["train"].train_test_split(test_size=0.2)
    print(data)


# 数据选取与过滤
# https://huggingface.co/datasets/madao33/new-title-chinese
def select_and_filter():
    data = load_dataset("madao33/new-title-chinese")
    data1 = data["train"].select([0, 1])
    data2 = data["train"].select([2, 3])
    data3 = data["train"].select(range(10))
    print(data3)
    # 只过滤在title中存在中国的数据
    data4 = data["train"].filter(lambda x: "中国" in x["title"])
    print(data4)


# 数据映射
# https://huggingface.co/datasets/madao33/new-title-chinese
def map_dataset():
    def add_prefix(x):
        x["title"] = "prefix: " + x["title"]

        return x

    data = load_dataset("madao33/new-title-chinese")
    data = data.map(add_prefix)
    print(data["train"][:5]["title"])


# 简单的数据处理
def simple_process():
    # https://huggingface.co/google-bert/bert-base-chinese
    # 一个中文的理解模型
    tokenizer = AutoTokenizer.from_pretrained("google-bert/bert-base-chinese")

    # 处理函数
    def process(example):
        model_inputs = tokenizer(example["content"], max_length=512, truncation=True)
        labels = tokenizer(example["title"], max_length=32, truncation=True)
        model_inputs["labels"] = labels["input_ids"]
        return model_inputs

    # 处理调用
    data = load_dataset("madao33/new-title-chinese")
    # process_data = data.map(process, batched=True, num_proc=4,remove_columns=["content", "title"])
    # process_data = data.map(process, batched=True, num_proc=4, remove_columns=data["train"].column_names)
    process_data = data.map(process, batched=True)
    print(process_data)


## 数据保存
def save_dataset():
    tokenizer = AutoTokenizer.from_pretrained("google-bert/bert-base-chinese")

    def process(example):
        model_inputs = tokenizer(example["content"], max_length=512, truncation=True)
        labels = tokenizer(example["title"], max_length=32, truncation=True)
        model_inputs["labels"] = labels["input_ids"]
        return model_inputs

    data = load_dataset("madao33/new-title-chinese")
    process_data = data.map(
        process, batched=True, remove_columns=data["train"].column_names
    )
    print("save dataset")
    print(process_data)
    process_data.save_to_disk("./local-new-title-chinese")


## 数据加载
def load_dataset_from_disk():
    data = load_from_disk("./local-new-title-chinese")
    print("load dataset")
    print(data)


## 本地数据加载
def load_local_csv():
    # data = load_dataset("csv", data_files="./ChnSentiCorp_htl_all.csv")
    # data = load_dataset("csv", data_files="./ChnSentiCorp_htl_all.csv", split="train")
    data = load_dataset("csv", data_dir="./", split="train")
    print(data)


if __name__ == "__main__":
    load_local_csv()
