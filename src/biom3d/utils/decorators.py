import warnings
import functools

def deprecated(reason="This function is deprecated."):
    def decorator(func):
        @functools.wraps(func)
        def wrapped(*args, **kwargs):
            warnings.warn(
                f"Call to deprecated function {func.__name__}(). {reason}",
                DeprecationWarning,
                stacklevel=2
            )
            return func(*args, **kwargs)
        return wrapped
    return decorator
