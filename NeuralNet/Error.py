class Error(Exception):
    """Base class for exceptions in this module."""
    pass

class InitError(Error):
    """Errors in initialization

    Attributes:
        expr -- input expression in which the error occurred
        msg  -- explanation of the error
    """
    def __init__(self, expr, msg):
        self.expr = expr
        self.msg = msg

    def __str__(self):
        return repr(self.msg)