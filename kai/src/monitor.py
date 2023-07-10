class Monitor(): 

    def __init__(self, reporter, params_constructor, fix_reporter_params: dict, 
                 monitoring_skip: int): 
        """
        Initializes a monitor object. 
        Inputs: 
            self: the monitor object itself. 
            reporter: a function that carries out the reporter mechanism, 
                      typically produces a type of graphic over the model
                      training procedure. 
            params_constructor: a function that 
                      takes both fixed parameters and shifting parameters to 
                      construct a set of parameters for the reporter function
            fix_reporter_params: the fixed parameters for the reporting mechanism. 
            shift_reporter_params: the shifting (or changing) parameters for 
                      the reporting mechanism over the model training procedure. 
            monitoring_skip: the number of iterations to skip between every report. 
        """
        self._reporter = reporter
        self._params_constructor = params_constructor
        self._fix_params = fix_reporter_params
        self._monitoring_skip = monitoring_skip

    def get_monitoring_skip(self): 
        """
        Returns the monitoring skip variable. 
        """
        return self._monitoring_skip

    def eval(self, shift_params): 
        """
        Runs the evaluator function.
        """
        self._reporter(*self._params_constructor(self._fix_params, shift_params))

