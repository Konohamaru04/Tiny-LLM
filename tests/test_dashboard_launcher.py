from __future__ import annotations

import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


class DashboardLauncherTests(unittest.TestCase):
    def test_windows_launcher_repairs_and_verifies_cuda_before_launch(self) -> None:
        launcher = (ROOT / "run_dash.bat").read_text(encoding="utf-8").lower()

        cuda_install = launcher.index("--force-reinstall")
        dependency_install = launcher.index('-r "%cd%\\requirements.txt"')
        cuda_verification = launcher.rindex("torch.cuda.is_available()")
        dashboard_launch = launcher.index("scripts\\training_dashboard.py")

        self.assertIn("https://download.pytorch.org/whl/cu130", launcher)
        self.assertIn("where nvidia-smi", launcher)
        self.assertLess(cuda_install, dependency_install)
        self.assertLess(dependency_install, cuda_verification)
        self.assertLess(cuda_verification, dashboard_launch)
        self.assertIn("refusing to start long-context training on cpu", launcher)


if __name__ == "__main__":
    unittest.main()
