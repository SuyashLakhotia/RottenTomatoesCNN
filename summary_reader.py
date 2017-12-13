import tensorflow as tf

folder = ""
subfolder = ""
filename = ""
path = "runs/" + folder + "/summaries/" + subfolder + "/" + filename

accuracies = []
losses = []
for e in tf.train.summary_iterator(path):
    for v in e.summary.value:
        if v.tag == "accuracy_1":
            accuracies.append(v.simple_value)
        if v.tag == "loss_1":
            losses.append(v.simple_value)
