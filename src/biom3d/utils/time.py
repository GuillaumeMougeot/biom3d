"""This submodule provide a class to easily manipulate time."""
from time import time
from typing import Optional 

class Time:
    """
    Simple utility class to measure elapsed time across multiple events.

    Useful for timing loops, function calls, or sections of code with optional naming
    and usage tracking (via a counter).

    :ivar str name: Name of the timer instance.
    :ivar float start_time: Timestamp when the timer was last reset.
    :ivar int count: Number of times `.get()` or `__str__()` has been called.
    """

    def __init__(self, name:Optional[str]=None):
        """
        Initialize time.

        Parameters
        ----------
        name : str, optional
            Optional name identifier for the timer (used in __str__ debug output).        
        """
        self.name=name
        self.reset()
    
    def reset(self)->None:
        """
        Reset the timer and the internal call counter.

        This will set the start time to the current time and reset the count to 0.
        """
        print("Count has been reset!")
        self.start_time = time()
        self.count = 0
    
    def get(self)->float:
        """
        Get the elapsed time in seconds since the last reset.

        Increments the internal call counter.

        Returns
        -------
        float
            Elapsed time in seconds.
        """
        self.count += 1
        return time()-self.start_time
    
    def __str__(self)->str:
        """
        Return a debug string showing elapsed time and usage count.

        Also resets the internal start time to now for next measurement.

        Returns
        -------
        str
            Debug-formatted string containing name, count, and elapsed time.
        """
        self.count += 1
        out = time() - self.start_time
        self.start_time=time()
        return "[DEBUG] name: {}, count: {}, time: {} seconds".format(self.name, self.count, out)
