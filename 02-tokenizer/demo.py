from transformers import AutoTokenizer

sen = "我是谁 你是谁 谁是你"

# 指定模型
tokenizer = AutoTokenizer.from_pretrained("uer/roberta-base-finetuned-dianping-chinese")

# 开始分词
tokens = tokenizer.tokenize(sen)
print(tokens)

# 将分词结果转换为id序列
input_ids = tokenizer.convert_tokens_to_ids(tokens)
print(input_ids)

# id序列转换回分词
tokens_new = tokenizer.convert_ids_to_tokens(input_ids)
print(tokens_new)


# 更便捷的分词
# 这个更便捷的分词，回在最前面和最后面加入一个标识，所以会比一步一步调用的多出俩id
# 通过add_special_tokens=False，可以关闭这个功能
# ids = tokenizer.encode(sen, add_special_tokens=False)
ids = tokenizer.encode(sen)
print(ids)

# 更便捷的解码
# 如果标记位是开启的，这里也会直接输出出来标记为
str_sen = tokenizer.decode(ids)
print(str_sen)

# 填充和截断
# 填充
ids = tokenizer.encode(sen, padding="max_length", max_length=15)
print(ids)

# 截断
ids = tokenizer.encode(sen, truncation=True, max_length=5)
print(ids)

# 上面是分词，下面是分句
inputs = tokenizer.encode_plus(sen, padding="max_length", max_length=15)
print(inputs)
# 另外一种调用方法时直接用tokenizer
inputs = tokenizer(sen, padding="max_length", max_length=15)
print(inputs)
