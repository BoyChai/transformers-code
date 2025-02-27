from transformers import *
import os

# 设置代理
os.environ["HTTP_PROXY"] = "http://127.0.0.1:2809"
os.environ["HTTPS_PROXY"] = "http://127.0.0.1:2809"

pipe = pipeline(
    # "text-classification", model="uer/roberta-base-finetuned-dianping-chinese"
    "text-classification"
)
print(pipe("你好你好"))
