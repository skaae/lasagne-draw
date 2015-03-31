import numpy as np
class BatchIterator(object):
    """
     Cyclic Iterators over batch indexes. Permutes and restarts at end
    """

    def __init__(self, batch_indices, batchsize, data, testing=False, process_func=None):
        if isinstance(batch_indices, int):
            self.n = batch_indices
            self.batchidx = np.arange(batch_indices)
        else:
            self.n = len(batch_indices)
            self.batchidx = np.array(batch_indices)

        self.batchsize = batchsize
        self.testing = testing
        if process_func is None:
            process_func = lambda x:x
        self.process_func = process_func

        if not isinstance(data, (list, tuple)):
            data = [data]

        self.data = data
        if not self.testing:
            self.createindices = lambda: np.random.permutation(self.n)
        else: # testing == true
            assert self.n % self.batchsize == 0, "for testing n must be multiple of batch size"
            self.createindices = lambda: range(self.n)

        self.perm = self.createindices()
        assert self.n > self.batchsize

    def __iter__(self):
        return self

    def __next__(self):
        return self.next()

    def _get_permuted_batches(self,n_batches):
        # return a list of permuted batch indeces
        batches = []
        for i in range(n_batches):

            # extend random permuation if shorter than batchsize
            if len(self.perm) <= self.batchsize:
                new_perm = self.createindices()
                self.perm = np.hstack([self.perm, new_perm])

            batches.append(self.perm[:self.batchsize])
            self.perm = self.perm[self.batchsize:]
        return batches

    def next(self):
        batch = self._get_permuted_batches(1)[0]   # extract a single batch
        data_batches = [self.process_func(data_n[batch]) for data_n in self.data]
        return data_batches





def threaded_generator(generator, num_cached=50):
    # this code is writte by jan Schluter
    # copied from https://github.com/benanne/Lasagne/issues/12
    import Queue
    queue = Queue.Queue(maxsize=num_cached)
    sentinel = object()  # guaranteed unique reference

    # define producer (putting items into queue)
    def producer():
        for item in generator:
            queue.put(item)
        queue.put(sentinel)

    # start producer (in a background thread)
    import threading
    thread = threading.Thread(target=producer)
    thread.daemon = True
    thread.start()

    # run as consumer (read items from queue, in current thread)
    item = queue.get()
    while item is not sentinel:
        yield item
        queue.task_done()
        item = queue.get()