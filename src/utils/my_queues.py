# no check in index_out_of_range
class FixedSizeQueue:
    def __init__(self, max_size: int):
        self.max_size = max_size
        self.queue = []

    def push(self, item):
        if len(self.queue) >= self.max_size:
            self.queue.pop(0)
        self.queue.append(item)

    def clear(self):
        self.queue = []

    def size(self):
        return len(self.queue)
    
    def empty(self):
        return len(self.queue) == 0
    
    def front(self):
        return self.queue[0]
    
    def back(self):
        return self.queue[-1]
