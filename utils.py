
def print_metrics(metrics, samples, phase):
    outputs = []
    for k in metrics.keys():
        outputs.append("{}: {:4f}".format(k, metrics[k] / samples))

    print("{}: {}".format(phase, ", ".join(outputs)))

