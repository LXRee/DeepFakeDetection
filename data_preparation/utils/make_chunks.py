from itertools import islice


def dict_chunks(data, SIZE=10000):
    """
    Divide JSON in chunks of size SIZE
    :param data:
    :param SIZE:
    :return:
    """
    it = iter(data)
    for i in range(0, len(data), SIZE):
        yield {k: data[k] for k in islice(it, SIZE)}


# Yield successive n-sized
# chunks from l.
def list_chunks(l, n):
    # looping till length l
    for i in range(0, len(l), n):
        yield l[i:i + n]
