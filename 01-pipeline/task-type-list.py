from transformers.pipelines import SUPPORTED_TASKS


# print(SUPPORTED_TASKS.items())

for k, v in SUPPORTED_TASKS.items():
    print(k, v["type"])
