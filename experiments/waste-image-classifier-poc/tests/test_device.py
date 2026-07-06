import unittest
from types import SimpleNamespace

from waste_poc.device import resolve_device, resolve_num_workers, reset_device_print_state


class FakeDevice(str):
    pass


class FakeTensor:
    def __add__(self, _other):
        return self

    def cpu(self):
        return self

    def item(self):
        return 1.0


class FakeTorch:
    def __init__(self, cuda=False, mps_built=False, mps_available=False, mps_op=True):
        self.cuda = SimpleNamespace(is_available=lambda: cuda)
        self.backends = SimpleNamespace(mps=SimpleNamespace(is_built=lambda: mps_built, is_available=lambda: mps_available))
        self._mps_op = mps_op

    def device(self, name):
        return FakeDevice(name)

    def ones(self, *_args, **kwargs):
        if kwargs.get("device") == "mps" and not self._mps_op:
            raise RuntimeError("mps failed")
        return FakeTensor()


class DeviceTests(unittest.TestCase):
    def setUp(self):
        reset_device_print_state()

    def test_explicit_cpu_works(self):
        self.assertEqual(str(resolve_device("cpu", torch_module=FakeTorch(), print_selection=False)), "cpu")

    def test_unavailable_explicit_devices_fail_readably(self):
        with self.assertRaisesRegex(RuntimeError, "CUDA is not available"):
            resolve_device("cuda", torch_module=FakeTorch(), print_selection=False)
        with self.assertRaisesRegex(RuntimeError, "MPS is not available"):
            resolve_device("mps", torch_module=FakeTorch(), print_selection=False)

    def test_auto_prefers_cuda_then_mps_then_cpu(self):
        self.assertEqual(str(resolve_device("auto", torch_module=FakeTorch(cuda=True, mps_built=True, mps_available=True), print_selection=False)), "cuda")
        self.assertEqual(str(resolve_device("auto", torch_module=FakeTorch(mps_built=True, mps_available=True), print_selection=False)), "mps")
        self.assertEqual(str(resolve_device("auto", torch_module=FakeTorch(), print_selection=False)), "cpu")

    def test_worker_resolver(self):
        self.assertEqual(resolve_num_workers("auto", system="Darwin", cpu_count=8), 0)
        self.assertGreaterEqual(resolve_num_workers("auto", system="Linux", cpu_count=8), 0)
        self.assertEqual(resolve_num_workers(3, system="Darwin"), 3)


if __name__ == "__main__":
    unittest.main()
