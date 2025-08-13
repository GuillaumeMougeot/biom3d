"""This submodule provide decorators."""

from typing import Callable
import warnings
import functools

def deprecated(reason:str="This function is deprecated.")->Callable:
    """
    Mark functions as deprecated.

    When the decorated function is called, a `DeprecationWarning` is issued with the given reason.
    This helps inform developers and users that the function is outdated and may be removed in future versions.

    Parameters
    ----------
    reason: str, default="This function is deprecated."nal
        Explanation or message indicating why the function is deprecated, or what to use instead.        

    Returns
    -------
    callable
        A decorator that wraps the given function and issues a deprecation warning when called.

    Examples
    --------
    >>> @deprecated("Use 'new_function' instead.")
    ... def old_function():
    ...     pass

    >>> old_function()
    ... # DeprecationWarning: Call to deprecated function old_function(). Use 'new_function' instead.

    Notes
    -----
    - The warning is raised using `warnings.warn` with category `DeprecationWarning`.
    - Stack level is set to 2 to show the warning at the caller's level.
    """
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
