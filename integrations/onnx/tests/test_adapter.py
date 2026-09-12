import unittest

import numpy as np

from flowedge_onnx import OnnxAdapter, OnnxAdapterError


class Node:
    def __init__(self, name, shape, type="tensor(float)"):
        self.name, self.shape, self.type = name, shape, type


class Session:
    def get_inputs(self):
        return [Node("condition", [1, 2])]

    def get_outputs(self):
        return [Node("action", [1, 2])]

    def run(self, outputs, feeds):
        return [feeds["condition"] * 2]


class OnnxAdapterTests(unittest.TestCase):
    def test_caller_owned_output_matches_session(self):
        output = np.empty((1, 2), dtype=np.float32)
        result = OnnxAdapter(Session()).run_into(
            np.array([[1, 3]], dtype=np.float32), output
        )
        self.assertIs(result, output)
        np.testing.assert_array_equal(output, [[2, 6]])

    def test_rejects_dynamic_contract(self):
        class Dynamic(Session):
            def get_inputs(self):
                return [Node("condition", [None, 2])]

        with self.assertRaisesRegex(OnnxAdapterError, "fixed-shape"):
            OnnxAdapter(Dynamic())


if __name__ == "__main__":
    unittest.main()
