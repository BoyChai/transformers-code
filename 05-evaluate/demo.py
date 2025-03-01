import evaluate


# 查看支持的评估函数(list_evaluation_modules)
def list_evaluation_modules():
    # print(evaluate.list_evaluation_modules(include_community=False, with_details=True))
    print(evaluate.list_evaluation_modules())


# 加载评估函数(load)
def load():
    accuracy = evaluate.load("accuracy")
    print(accuracy)


# 查看评估函数说明(inputs_description)
def inputs_description():
    accuracy = evaluate.load("accuracy")
    print(accuracy.description)
    print(accuracy.inputs_description)


# 全局计算
def compute():
    accuracy = evaluate.load("accuracy")
    result = accuracy.compute(predictions=[0, 1, 2, 3], references=[0, 1, 2, 3])
    print(result)


# 迭代计算
def compute_iter():
    accuracy = evaluate.load("accuracy")
    # for ref, pred in zip([0, 1, 0, 1], [1, 0, 0, 1]):
    for ref, pred in zip([[0, 1], [0, 1]], [[1, 0], [0, 1]]):
        # accuracy.add(predictions=pred, references=ref)
        accuracy.add_batch(predictions=pred, references=ref)
    result = accuracy.compute()
    print(result)


# 多个评估指标计算
def compute_multiple():
    clf_metric = evaluate.combine(["accuracy", "f1", "recall", "precision"])
    result = clf_metric.compute(predictions=[0, 1, 0], references=[0, 1, 1])
    print(result)


# 雷达图
def rader():
    from evaluate.visualization import radar_plot
    import matplotlib.pyplot as plt

    data = [
        {"accuracy": 0.99, "precision": 0.8, "f1": 0.95, "latency_in_seconds": 33.6},
        {"accuracy": 0.98, "precision": 0.87, "f1": 0.91, "latency_in_seconds": 11.2},
        {"accuracy": 0.98, "precision": 0.78, "f1": 0.88, "latency_in_seconds": 87.6},
        {"accuracy": 0.88, "precision": 0.78, "f1": 0.81, "latency_in_seconds": 101.6},
    ]
    model_names = ["Model 1", "Model 2", "Model 3", "Model 4"]

    plot = radar_plot(data=data, model_names=model_names)
    plt.show()


rader()
