class Monitor(): 

    def __init__(self, tracker: function, reporter: function,  
                 monitoring_skip: int, state_tracker: dict, 
                 reporter_args: list, tracker_args: list): 
        """
        Initializes a monitor object. 
        Inputs: 
            self: the monitor object itself. 
            tracker: a void function that tracks the model training procedure
            reporter: a void function that carries out the reporter mechanism, 
                      typically produces a type of graphic over the model
                      training procedure. 
            params_constructor: a function that 
                      takes both fixed parameters and shifting parameters to 
                      construct a set of parameters for the reporter function
            monitoring_skip: the number of iterations to skip between every report. 
            state_tracker: a dictionary tracking relevant states of the model.
            reporter_args: a list of arguments for the reporter function.
            tracker_args: a list of arguments for the tracker function.
        """
        self._tracker = tracker
        self._reporter = reporter
        self._monitoring_skip = monitoring_skip
        self._state_tracker = state_tracker
        self._reporter_args = reporter_args
        self._tracker_args = tracker_args
        self._counter = 0

    def get_monitoring_skip(self): 
        """
        Returns the monitoring skip variable. 
        """
        return self._monitoring_skip
    
    def get_states(self): 
        """
        Returns the state tracker.
        """
        return self._state_tracker

    def eval(self, model, params): 
        """
        Evaluate the monitor object. Specifically, you update the internal 
        states using the state tracker and you report the results using the
        the reporter function.

        Inputs: 
            model: the model object. 
        """
        if self._counter % self._monitoring_skip == 0:
            self._tracker(self._state_tracker, model, params, *self._tracker_args)
            self._reporter(model, params, *self._reporter_args)
        self._counter += 1

class Monitors():

    def __init__(self, monitors: list): 
        """
        Initializes a monitors object. 
        Inputs: 
            self: the monitors object itself. 
            monitors: a list of monitor objects. 
        """
        self._monitors = monitors

    def get_monitors(self): 
        """
        Returns the monitors list. 
        """
        return self._monitors
    
    def eval(self, model, params):
        """
        Runs the evaluator function for each monitor in the monitors list. 
        """
        for monitor in enumerate(self._monitors): 
            monitor.eval(model, params)
