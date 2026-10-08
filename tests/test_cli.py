import os
import unittest
from unittest.mock import patch

from experiment.cli import configure_nccl


class NcclTests(unittest.TestCase):
    def test_unsupported_gpu_disables_p2p_and_ib(self):
        with patch.dict(os.environ, {"NCCL_P2P_DISABLE": "0"}, clear=True):
            configure_nccl(False)
            self.assertEqual(os.environ["NCCL_P2P_DISABLE"], "1")
            self.assertEqual(os.environ["NCCL_IB_DISABLE"], "1")

    def test_supported_gpu_keeps_existing_settings(self):
        with patch.dict(os.environ, {}, clear=True):
            configure_nccl(True)
            self.assertNotIn("NCCL_P2P_DISABLE", os.environ)
            self.assertNotIn("NCCL_IB_DISABLE", os.environ)


if __name__ == "__main__":
    unittest.main()
