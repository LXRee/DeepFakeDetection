from itertools import islice


def chunks(data, SIZE=10000):
    """
    Divide JSON in chunks of SIZE
    :param data:
    :param SIZE:
    :return:
    """
    it = iter(data)
    for i in range(0, len(data), SIZE):
        yield {k: data[k] for k in islice(it, SIZE)}
