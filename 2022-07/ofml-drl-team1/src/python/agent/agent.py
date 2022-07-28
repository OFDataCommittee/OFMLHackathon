
from typing import Callable
from abc import ABC, abstractmethod, abstractproperty
import torch as pt
from ..constants import DEFAULT_DTYPE


pt.set_default_tensor_type(DEFAULT_DTYPE)


def compute_returns(rewards: pt.Tensor, gamma: float = 0.99) -> pt.Tensor:
    n_steps = len(rewards)
    discounts = pt.logspace(0, n_steps-1, n_steps, gamma)
    returns = [(discounts[:n_steps-t] * rewards[t:]).sum()
               for t in range(n_steps)]
    return pt.tensor(returns)


def compute_gae(rewards: pt.Tensor, values: pt.Tensor, gamma: float = 0.99, lam: float = 0.97) -> pt.Tensor:
    n_steps = len(rewards)
    factor = pt.logspace(0, n_steps-1, n_steps, gamma*lam)
    delta = rewards[:-1] + gamma * values[1:] - values[:-1]
    gae = [(factor[:n_steps-t-1] * delta[t:]).sum()
           for t in range(n_steps - 1)]
    return pt.tensor(gae)


class FCPolicy(pt.nn.Module):
    def __init__(self, n_states: int, n_actions: int, action_min: pt.Tensor,
                 action_max: pt.Tensor, n_layers: int = 2, n_neurons: int = 64,
                 activation: Callable = pt.nn.functional.relu):
        super(FCPolicy, self).__init__()
        self._n_states = n_states
        self._n_actions = n_actions
        self._action_min = action_min
        self._action_max = action_max
        self._n_layers = n_layers
        self._n_neurons = n_neurons
        self._activation = activation

        # set up policy network
        self._layers = pt.nn.ModuleList()
        self._layers.append(pt.nn.Linear(self._n_states, self._n_neurons))
        if self._n_layers > 1:
            for hidden in range(self._n_layers - 1):
                self._layers.append(pt.nn.Linear(
                    self._n_neurons, self._n_neurons))
        #self._layers.append(pt.nn.Linear(self._n_neurons, 2*self._n_actions))
        self._last_layer = pt.nn.Linear(self._n_neurons, 2*self._n_actions)

    @pt.jit.ignore
    def _scale(self, actions: pt.Tensor) -> pt.Tensor:
        return (actions - self._action_min) / (self._action_max - self._action_min)

    def forward(self, x: pt.Tensor) -> pt.Tensor:
        for layer in self._layers:
            x = self._activation(layer(x))
        return 1.0 + pt.nn.functional.softplus(self._last_layer(x))

    @pt.jit.ignore
    def predict(self, states: pt.Tensor, actions: pt.Tensor) -> pt.Tensor:
        out = self.forward(states)
        c0 = out[:, :self._n_actions]
        c1 = out[:, self._n_actions:]
        beta = pt.distributions.Beta(c0, c1)
        if len(actions.shape) == 1:
            scaled_actions = self._scale(actions.unsqueeze(-1))
        else:
            scaled_actions = self._scale(actions)
        log_p = beta.log_prob(scaled_actions)
        if len(actions.shape) == 1:
            return log_p.squeeze(), beta.entropy().squeeze()
        else:
            return log_p, beta.entropy()


class FCValue(pt.nn.Module):
    def __init__(self, n_states: int, n_layers: int = 2, n_neurons: int = 64,
                 activation: Callable = pt.nn.functional.relu):
        super(FCValue, self).__init__()
        self._n_states = n_states
        self._n_layers = n_layers
        self._n_neurons = n_neurons
        self._activation = activation

        # set up value network
        self._layers = pt.nn.ModuleList()
        self._layers.append(pt.nn.Linear(self._n_states, self._n_neurons))
        if self._n_layers > 1:
            for hidden in range(self._n_layers - 1):
                self._layers.append(pt.nn.Linear(
                    self._n_neurons, self._n_neurons))
        self._layers.append(pt.nn.Linear(self._n_neurons, 1))

    def forward(self, x: pt.Tensor) -> pt.Tensor:
        for i_layer in range(len(self._layers) - 1):
            x = self._activation(self._layers[i_layer](x))
        return self._layers[-1](x).squeeze()


class Agent(ABC):
    """Common interface for all agents.
    """

    @abstractmethod
    def update(self):
        pass

    @abstractmethod
    def save(self):
        pass

    @abstractmethod
    def load(self):
        pass

    @abstractmethod
    def trace_policy(self):
        pass

    @abstractproperty
    def history(self):
        pass
