class Error(Exception):
    """Base class for exceptions in this module.

    Attributes:
        message -- explanation of the error        
    """
    def __init__(self, message):        
        self.message = message

    def __str__(self):
        return repr(self.message)    

class InitError(Error):
    """Errors in initialization

    Attributes:
        message -- explanation of the error        
    """

class EvalError(Error):
    """Errors in evaluation

    Attributes:
        message -- explanation of the error        
    """
    