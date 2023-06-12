import torch
import torch.nn as nn
import numpy as np
import warnings


class MiniMaxSolver():
    """ Hongkang Yang 2019-7-6
    Implementation of the quasi implicit twisted gradient descent algorithm from [Essid_Tabak_Trigila, 2019]
    The optimization problem has two layers, i.e. min max f or max min f
    Since the Newton method is indifferent to whether we are doing minmax or maxmin except for a sign change
    at the update step, we can solve the problem by default as minmax:
    z^{n+1} = z^n - lr * B^n * grad^n
    The matrix B^n is a approximation to (J + lr H)^-1

    Use step_low_memory() when dim of input >> rounds of iteration, to avoid matrix operations
    and reduce running time and memory cost

    Args:
        model (nn.Module): model(input) is the function f in the optimization problem
        module_min (list / tuple of nn.Module): Module inside model that minimizes f
        module_max (list / tuple of nn.Module): Module that maximizes f
            P.S.: module.parameters() cannot be passed as argument
        input (Tensor, or list / tuple of Tensor): model(input) gives the value of the function f
        lr (float): the initial learning rate

    Different from the original QuasiImplicitMinimax, the step() function only update the model once, leaving
    the control of the iteration to the caller. It is convenient if we are interested in the intermediate models
    during training. The data for quasi update, namely B, is stored as a class variable.
    For the low memory version, B is stored as a list of rank-one factorizations

    Example:
        optimizer = QuasiImplicitMinimax(model, model.min_net, model.max_net, input, lr=0.1)
        for t in range(rounds):
            optimizer.step(Constrained=True)

    The update function step() has two modes. The original algorithm from [Essid_Tabak_Trigila, 2019] includes an
    anticipatory constraint for each update, and the step size or learning rate is determined by backtracking
    line search. If the constraint is satisfied by some lr, then next time step() uses lr * some increase factor.
    The increase in lr makes the update closer to the actual Newton step. Else, the line search fails and the algorithm
    either terminates or continue with Constraint=False.
    If Constraint=False, then the anticipatory constraint is ignored and lr stays constant.
    """

    def __init__(self, model, module_min, module_max, input, lr):
        # By default, the modules are stored as a list or tuple
        if isinstance(module_min, torch.nn.Module):
            module_min = (module_min,)
        if isinstance(module_max, torch.nn.Module):
            module_max = (module_max,)
        self.model = model
        self.module_min = module_min
        self.module_max = module_max
        self.input = input
        self.lr = lr
        self.d_out = np.sum([sum([para.numel() for para in module.parameters()]) for module in self.module_min])
        self.d_in = np.sum([sum([para.numel() for para in module.parameters()]) for module in self.module_max])
        # For step(), initialize the quasi Hessian matrix
        self.B = np.diagflat(np.concatenate((np.ones(self.d_out), -1 * np.ones(self.d_in))))
        # For step_low_memory(), initialize list of (weight,vector) pairs for the rank-one updates of B_n
        self.updates = []

        # Parameters for the anticipatory constraint and line search
        self.stopping_threshold = 1e-3  # stop the line search when the decay < threshold
        self.decay_rate = 0.75  # decay rate for line search
        self.increase_factor = 0.1  # increase rate for self.lr if line search is successful
        self.lr_max = lr * 5  # maximum allowed lr

        # Size of minibatch
        self.batch_size = None

    # Return a minibatch from the input sample set
    # By default, assume that the dim 0 of each tensor from self.input is the sample index
    def get_input(self):
        if not self.batch_size:
            return self.input
        else:
            if isinstance(self.input, torch.Tensor):
                N = self.input.size()[0]
                random_index_set = np.random.choice(N, self.batch_size, replace=True).tolist()
                return self.input[random_index_set]
            else:
                N = self.input[0].size()[0]
                random_index_set = np.random.choice(N, self.batch_size, replace=True).tolist()
                return [input_tensor[random_index_set] for input_tensor in self.input]

    # Compute gradient and return it as a numpy column vector, if vec
    # Else return a np.array
    def grad_vector(self, vec=True, TestMode=False, batch=None):
        # Reset gradient to be zero
        self.model.zero_grad()
        if batch is None:
            batch = self.get_input()
        L = self.model(batch)
        L.backward()
        # Print avaerage transport cost to barycenter
        if TestMode:
            print(float(L))

        p_grad = []
        for module in self.module_min:
            p_grad += torch.cat([para.grad.clone().flatten() for para in module.parameters()]).tolist()
        for module in self.module_max:
            p_grad += torch.cat([para.grad.clone().flatten() for para in module.parameters()]).tolist()
        if vec:
            p_grad = np.mat(p_grad).T
        return p_grad

    # Return all parameters as a numpy column vector
    def param_vector(self, vec=True):
        p = []
        # Tensors produced by clone() belong to the computational graph by default, so we call no_grad() to prevent it
        with torch.no_grad():
            # extract parameters from each layer
            for module in self.module_min:
                # list concatenation
                p += torch.cat([para.clone().flatten() for para in module.parameters()]).tolist()
            for module in self.module_max:
                p += torch.cat([para.clone().flatten() for para in module.parameters()]).tolist()
        if vec:
            p = np.mat(p).T
        return p

    # Update model parameters
    # It is possible to circumvent updating, if param_vector is not a clone but the acutual parameters
    def param_update(self, p):
        # keep track of the starting index
        position = 0
        if isinstance(p, np.matrix):
            # If p is a column np.matrix, convert it to np.array
            p = np.squeeze(np.asarray(p))
        with torch.no_grad():
            for module in self.module_min:
                for para in module.parameters():
                    para.copy_(
                        torch.tensor(p[position:position + para.numel()], dtype=torch.float).reshape(para.size()))
                    position += para.numel()
            for module in self.module_max:
                for para in module.parameters():
                    para.copy_(
                        torch.tensor(p[position:position + para.numel()], dtype=torch.float).reshape(para.size()))
                    position += para.numel()

    # The anticipatory constraint for twisted descent
    # Input delta is the update step. Can be either np.array or column vector
    def anticipatory_constraint(self, delta, batch=None):
        I_min = np.concatenate((np.ones(self.d_out), np.zeros(self.d_in)))
        I_max = np.concatenate((np.zeros(self.d_out), np.ones(self.d_in)))
        # Convert column vector to np.array
        if isinstance(delta, np.matrix):
            delta = np.squeeze(np.asarray(delta))
        p = self.param_vector(vec=False)

        if batch is None:
            batch = self.get_input()

        # Check the anticipatory constraints
        p_min_new = p + np.multiply(I_min, delta.T).T
        self.param_update(p_min_new)
        with torch.no_grad():
            L_min_new = self.model(batch)

        p_max_new = p + np.multiply(I_max, delta.T).T
        self.param_update(p_max_new)
        with torch.no_grad():
            L_max_new = self.model(batch)

        p_new = p + delta
        self.param_update(p_new)
        with torch.no_grad():
            L_new = self.model(batch)

        # If the constraint is not satisfied, then return to the original state
        if not L_min_new <= L_new <= L_max_new:
            self.param_update(p)

        return L_min_new <= L_new <= L_max_new

    # Normalization plans for the update:
    def normalize(self, B, grad, Normalization):
        if Normalization == 'B':
            # Normalize by the operator norm ||B||_2
            direction = -1 * B * grad / np.linalg.norm(B, ord=2)
        elif Normalization == 'F':
            # Normalize by the Frobenius norm ||B||_F, which is an upper bound for the operator norm computable
            # in quadratic time
            direction = -1 * B * grad / np.linalg.norm(B, 'fro')
        elif Normalization == 'grad':
            # Normalize by length of grad
            direction = -1 * B * grad / np.linalg.norm(grad)
        elif Normalization == 'step':
            # Normalize by step length
            direction = B * grad
            direction = -1 * direction / np.linalg.norm(direction)
        else:
            direction = -1 * B * grad
            if Normalization is not None:
                print("Please choose a valid normalization plan: 'B', 'F', 'grad', 'step', or None")
        return direction

    def step(self, Constrained=False, Normalization=None, TestMode=False):
        # Matrix for twisted update
        J = np.concatenate((np.ones(self.d_out), -1 * np.ones(self.d_in)))
        batch = self.get_input()
        p_grad = self.grad_vector(TestMode=TestMode, batch=batch)
        direction = self.normalize(self.B, p_grad, Normalization)

        # If constrained, then each update step needs to satisfy the anticipatory constraint of the minimax game
        # see [Essid_Tabak_Trigila, 2019]. Find the step length by backtracking line search.
        # If the stopping threshold is reached, proceed by the unconstrained update
        if Constrained:
            lr = self.lr
            # Assuming that the saddle point can be locally approximated by a quadratic function, e.g. f is C^2
            # then as we approach the saddle, the update of p can approach Newton's method to accelerate descent.
            # This is achieved by increasing the learning rate.
            # The update rule for lr is in section 4.1 of [Essid_Tabak_Trigila, 2019].
            while lr >= self.lr * self.stopping_threshold:
                # backtracking line search
                if self.anticipatory_constraint(delta=lr * direction, batch=batch):
                    break
                else:
                    if TestMode: print("line search decaying")
                    lr *= self.decay_rate

            # If line search fails, use the unconstrained update
            if lr < self.lr * self.stopping_threshold:
                if TestMode: print("-----line search failure-----")
                p = self.param_vector()
                p += lr * direction
                self.param_update(p)
                self.lr = lr
            # If successful, increase lr
            else:
                self.lr = min(lr * (1 + self.increase_factor), self.lr_max)

        # For the unconstrained case, simply do the update with fixed lr
        else:
            p = self.param_vector()
            p += self.lr * direction
            self.param_update(p)

        # Update quasi Hessian matrix by rank-one update
        p_grad_new = self.grad_vector(batch=batch)
        s = np.multiply(J, p_grad_new.T).T - self.B * p_grad
        if np.linalg.norm(s) > 0:
            alpha = (s.T * s) / (p_grad.T * s)
            alpha = alpha[0, 0]
            # Avoid extremely large updates
            alpha = np.sign(alpha) * min(abs(alpha), 1)  # or np.linalg.norm(self.B)
            self.B += alpha * s * s.T / (s.T * s)

    # The above computation in step() can be made more efficient: B is a sum of low-rank approximations,
    # so we only need to store the vectors s and coefficients alpha
    def step_low_memory(self, Constrained=False, Normalization=None, TestMode=False):

        # The low-rank updates to Bn are stored as vectors to reduce computational time when dim >> rounds
        # Compute matrix multiplication B * v by the sum of alpha^n sn * (sn^T * v)
        def quasi_inverse_Hessian(v):
            # Initialize the quasi inverse Hessian matrix B0 as J
            Bv = np.multiply(J, v)
            for alpha, s in self.updates:
                Bv += alpha * np.inner(s, v) * s
            return Bv

        # Estimate of the operator norm of B
        # If the low rank factorization vectors are i.i.d., then Central Limit Theorem implies that ||B_n||
        # is proportional to the sum of these vectors' norms / sqrt(n)
        # def inverse_Hessian_norm():
        #    return max(abs(self.B_norm) / (len(self.updates)+1), 1)

        # Matrix for twisted update
        J = np.concatenate((np.ones(self.d_out), -1 * np.ones(self.d_in)))
        p_grad = self.grad_vector(vec=False, TestMode=TestMode)  # Print the object function's value if TestMode

        # Normalization plans:
        if Normalization == 'grad':
            # Normalize by length of grad
            direction = -1 * quasi_inverse_Hessian(p_grad) / np.linalg.norm(p_grad)
        elif Normalization == 'step':
            # Normalize by step length
            direction = quasi_inverse_Hessian(p_grad)
            direction = -1 * direction / np.linalg.norm(direction)
        else:
            direction = -1 * quasi_inverse_Hessian(p_grad)
            if Normalization is not None:
                print('Please choose a valid normalization plan: grad, step, or None (default)')

        # If constrained, then each update step needs to satisfy the anticipatory constraint of the minimax game
        # see [Essid_Tabak_Trigila, 2019]. Find the step length by backtracking line search.
        # If the minimum threshold is reached, proceed by the unconstrained update
        if Constrained:
            lr = self.lr
            while lr >= self.lr * self.stopping_threshold:
                # backtracking line search
                if self.anticipatory_constraint(delta=lr * direction):
                    break
                else:
                    if TestMode: print("line search decaying")
                    lr *= self.decay_rate

            # If line search fails, use the unconstrained update
            if lr < self.lr * self.stopping_threshold:
                if TestMode: print("-----line search failure-----")
                self.param_update(self.param_vector(vec=False) + lr * direction)
                self.lr = lr
            else:
                self.lr = min(lr * (1 + self.increase_factor), self.lr_max)

        # For the unconstrained case, simply do the update
        else:
            p = self.param_vector(vec=False)
            p += self.lr * direction
            self.param_update(p)

        # Update quasi Hessian matrix by rank-one update
        p_grad_new = self.grad_vector(vec=False)
        s = np.multiply(J, p_grad_new) - quasi_inverse_Hessian(p_grad)
        alpha = np.linalg.norm(s) ** 2 / np.inner(p_grad, s)
        alpha = np.sign(alpha) * min(abs(alpha), 1)  # or np.linalg.norm(B)
        self.updates.append((alpha, s / np.linalg.norm(s)))

    # The Optimistic Mirror Descent algorithm from Mertikopoulos, "Optimistic mirror descent in saddle-point problems"
    def step_OMD(self, Constrained=False, TestMode=False):
        # Matrix for twisted update
        J = np.concatenate((np.ones(self.d_out), -1 * np.ones(self.d_in)))
        p = self.param_vector()
        p_grad = self.grad_vector(TestMode=TestMode)
        delta = None
        # If constrained, then each update step needs to satisfy the anticipatory constraint of the minimax game
        # see [Essid_Tabak_Trigila, 2019]. Find the step length by backtracking line search.
        # If the stopping threshold is reached, proceed by the unconstrained update
        if Constrained:
            lr = self.lr
            while lr >= self.lr * self.stopping_threshold:
                # The waiting state
                p_waiting = p - lr * np.multiply(J, p_grad.T).T
                self.param_update(p_waiting)
                p_grad_waiting = self.grad_vector()
                # The actual update
                delta = np.multiply(J, p_grad_waiting.T).T
                delta = -lr * delta  # / np.linalg.norm(delta)

                # backtracking line search
                if self.anticipatory_constraint(delta=delta):
                    break
                else:
                    if TestMode: print("line search decaying")
                    lr *= self.decay_rate

            # If line search fails, use the unconstrained update
            if lr < self.lr * self.stopping_threshold:
                if TestMode: print("-----line search failure-----")
                self.param_update(self.param_vector() + delta)
                self.lr = lr
            # If successful, increase lr
            else:
                self.lr = min(lr * (1 + self.increase_factor), self.lr_max)

        # For the unconstrained case, simply do the update with fixed lr
        else:
            # The waiting state
            p_waiting = p - self.lr * np.multiply(J, p_grad.T).T
            self.param_update(p_waiting)
            p_grad_waiting = self.grad_vector()
            # The actual update is based on the gradient at the waiting state
            delta = np.multiply(J, p_grad_waiting.T).T
            delta = -self.lr * delta  # / np.linalg.norm(delta)
            self.param_update(p + delta)
