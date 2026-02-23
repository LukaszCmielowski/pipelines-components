"""Pytest configuration for automl pipelines.

Mocks kfp.kubernetes so pipeline code that uses use_secret_as_env can be
imported without the kfp-kubernetes optional dependency. Must run before any
pipeline package (e.g. autogluon_tabular_training_pipeline) is loaded.
"""

import sys
from unittest.mock import MagicMock

if "kfp.kubernetes" not in sys.modules:
    _mock_kubernetes = MagicMock()
    sys.modules["kfp.kubernetes"] = _mock_kubernetes
