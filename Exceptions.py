class CompilationError(Exception):
    """Error compiling"""
    line = None
    def __init__(self, line, message):
        self.line = line
        Exception.__init__(self, message)
    def __str__(self):
        return "Error on line %i: %s" % (self.line, self.message)