
from os import remove
from os.path import join, isfile
import torch as pt
from ..agent import compute_returns, compute_gae, FCPolicy, FCValue
from ...constants import DEFAULT_DTYPE


pt.set_default_tensor_type(DEFAULT_DTYPE)


def test_compute_rewards():
    gamma = 0.99
    rewards = pt.tensor([1.0, 0.1, 2.0])
    r0 = 0.99**0*1.0 + 0.99**1*0.1 + 0.99**2*2.0
    r1 = 0.99**0*0.1 + 0.99**1*2.0
    r2 = 0.99**0*2.0
    expected = pt.tensor([r0, r1, r2])
    assert pt.allclose(expected, compute_returns(rewards, 0.99))


def test_compute_gae():
    values = pt.tensor([0.1, 0.5, 0.2, 1.0])
    rewards = pt.tensor([1.0, 0.1, 2.0, 0.2])
    d0 = 1.0 + 0.99 * 0.5 - 0.1
    d1 = 0.1 + 0.99 * 0.2 - 0.5
    d2 = 2.0 + 0.99 * 1.0 - 0.2
    t0, t1, t2 = (0.99*0.97)**0, (0.99*0.97)**1, (0.99*0.97)**2
    A0 = t0*d0 + t1*d1 + t2*d2
    A1 = t0*d1 + t1*d2
    A2 = t0*d2
    expected = pt.tensor([A0, A1, A2])
    assert pt.allclose(expected, compute_gae(rewards, values, 0.99, 0.97))


class TestFCPolicy():
    def test_scale(self):
        policy = FCPolicy(100, 1, pt.tensor(-10), pt.tensor(10))
        scaled = policy._scale(pt.tensor([-10.0, 0.0, 10.0]))
        expected = pt.tensor([0.0, 0.5, 1.0])
        assert pt.allclose(scaled, expected)

    def test_forward(self):
        policy = FCPolicy(100, 1, pt.tensor(-10), pt.tensor(10))
        states = pt.rand((20, 100))
        out = policy(states)
        assert out.shape == (20, 2)
        assert (out > 1.0).sum().item() == 40

    def test_predict(self):
        states, actions = pt.rand((20, 100)), pt.rand(20)
        policy = FCPolicy(100, 1, pt.tensor(-10), pt.tensor(10))
        logp, entropy = policy.predict(states, actions)
        assert logp.shape == (20,)
        assert entropy.shape == (20,)
        states, actions = pt.rand((20, 100)), pt.rand((20, 2))
        policy = FCPolicy(100, 2, pt.tensor(-10), pt.tensor(10))
        logp, entropy = policy.predict(states, actions)
        assert logp.shape == (20, 2)
        assert entropy.shape == (20, 2)

    def test_tracing(self):
        policy = FCPolicy(100, 1, pt.tensor(-10), pt.tensor(10))
        test_in = pt.rand((2, 100))
        out_ref = policy(test_in)
        trace = pt.jit.script(policy)
        dest = join("/tmp/traced_test_model.pt")
        trace.save(dest)
        assert isfile(dest)
        policy_tr = pt.jit.load(dest)
        out = policy_tr(test_in)
        assert pt.allclose(out_ref, out)
        remove(dest)


def test_FCValue():
    value = FCValue(100)
    states = pt.rand((10, 100))
    out = value(states)
    assert out.shape == (10,)
