from transformers import AutoConfig, AutoModel, AutoTokenizer

# 模型在线加载
# 参数force_download=True强制从网络下载
# model = AutoModel.from_pretrained("hfl/rbt3", force_download=True)
model_online = AutoModel.from_pretrained("hfl/rbt3")
# print(model_online)

# 离线加载
# git clone --depth 1 https://huggingface.co/hfl/rbt3
model_local = AutoModel.from_pretrained("E:/AI/rbt3")
# print(model_local)

# 加载模型参数
config = AutoConfig.from_pretrained("E:/AI/rbt3")
# 这里的输出就是基于他模型的一些配置，全部的配置需要参考BertConfig这个类
# print(config)

# 模型调用
sen = "弱小的我也有大梦想！"
tokenizer = AutoTokenizer.from_pretrained("E:/AI/rbt3")
# 使用pt返回的是torch的字典，不加是python的，这里需要用pt的
inputs = tokenizer(sen, return_tensors="pt")
# print(inputs)

# 下面是传入参数的形式，可以看到
# 这里的output_attentions=True会输出attention的权重
# 这里的参数是不会自动补全的，需要自己去从config和BertConfig中获取
model_attentions = AutoModel.from_pretrained("E:/AI/rbt3", output_attentions=True)
model = AutoModel.from_pretrained(
    "E:/AI/rbt3",
)
output = model(**inputs)
# 这里的输出会有很多很多的数值
# print(output)

# 带Model Head的模型
# 下面导入的这个是带有模型头的模型，可以直接进行文本分类任务
from transformers import AutoModelForSequenceClassification

clz_model = AutoModelForSequenceClassification.from_pretrained("E:/AI/rbt3")
output = clz_model(**inputs)
print(output)
