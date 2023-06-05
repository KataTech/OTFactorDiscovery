import sympy 
import numpy as np 

class Loss(): 

    def __init__(self, model, vector = False): 
        if vector: 
            raise NotImplementedError
        self.x = sympy.symbols('x')
        self.y = sympy.symbols('y')
        # save the model
        self.orig_model = model
        self.model = model(self.x, self.y)
        # save the gradient
        self.grad_x = self.model.diff(self.x)
        self.grad_y = self.model.diff(self.y)
        # save the hessian
        self.hess_xx = self.grad_x.diff(self.x)
        self.hess_xy = self.grad_x.diff(self.y)
        self.hess_yx = self.grad_y.diff(self.x)
        self.hess_yy = self.grad_y.diff(self.y)

    def _loss_func(self):
        return self.orig_model

    def _loss(self): 
        return self.model

    def _grad(self): 
        return [self.grad_x, self.grad_y]
    
    def _hessian(self):
        return [[self.hess_xx, self.hess_xy], [self.hess_yx, self.hess_yy]]
    
    def gradient(self, x_val, y_val, as_numpy = False): 
        grad = [self.grad_x.evalf(subs = {self.x: x_val, self.y: y_val}), self.grad_y.evalf(subs = {self.x: x_val, self.y: y_val})]
        grad = [float(num) for num in grad]
        if as_numpy: 
            return np.array(grad)
        return grad
    
    def hessian(self, x_val, y_val, as_numpy = False):
        hess = [[self.hess_xx.evalf(subs = {self.x: x_val, self.y: y_val}), self.hess_xy.evalf(subs = {self.x: x_val, self.y: y_val})], 
                [self.hess_yx.evalf(subs = {self.x: x_val, self.y: y_val}), self.hess_yy.evalf(subs = {self.x: x_val, self.y: y_val})]]
        hess = [[float(num) for num in row] for row in hess]
        if as_numpy:
            return np.array(hess)
        return hess 