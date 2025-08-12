from time import time 

class Time:
    def __init__(self, name=None):
        self.name=name
        self.reset()
    
    def reset(self):
        print("Count has been reset!")
        self.start_time = time()
        self.count = 0
    
    def get(self):
        self.count += 1
        return time()-self.start_time
    
    def __str__(self):
        self.count += 1
        out = time() - self.start_time
        self.start_time=time()
        return "[DEBUG] name: {}, count: {}, time: {} seconds".format(self.name, self.count, out)
