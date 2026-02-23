import sys
import unittest
from unittest.mock import MagicMock, patch
import os

# Create a dummy module for tests if it doesn't exist or to avoid importing the real one
# We don't import 'tests' package to avoid triggering tests/__init__.py downloads

class TestInsightFaceRefactor(unittest.TestCase):
    def setUp(self):
        # Mock dependencies in sys.modules
        self.modules_patcher = patch.dict(sys.modules, {
            "insightface": MagicMock(),
            "insightface.model_zoo": MagicMock(),
            "insightface.utils": MagicMock(),
            "insightface.utils.face_align": MagicMock(),
            "platformdirs": MagicMock(),
            "tqdm": MagicMock(),
            "numpy": MagicMock(),
            "PIL": MagicMock(),
            "dlib": MagicMock(),
        })
        self.modules_patcher.start()

        # Mock platformdirs return value
        import platformdirs
        platformdirs.user_cache_dir.return_value = "/tmp/face_engine_cache"

        # We need to reload face_engine modules to ensure they use the mocked dependencies
        # and to reset any state if they were already imported.
        # However, verifying reloads is tricky.
        # We'll assume this test is run in a fresh process or before other tests.

    def tearDown(self):
        self.modules_patcher.stop()

    def test_initialization_and_paths(self):
        # We need to mock fetch_file and os.path.exists
        with patch("face_engine.fetching.fetch_file") as mock_fetch, \
             patch("os.path.exists") as mock_exists:

            # Setup mock_exists to simulate default behavior (files exist at first check)
            mock_exists.return_value = True

            # Import classes inside test to ensure they use patched modules
            # We might need to use importlib.reload if they are already imported
            import face_engine.models.insightface_models
            from importlib import reload
            reload(face_engine.models.insightface_models)

            from face_engine.models.insightface_models import (
                RetinaFaceDetector,
                SCRFDDetector,
                ArcFaceEmbedder,
                ArcFaceResNet100Embedder,
                BUFFALO_L_URL,
                ANTELOPE_V2_URL
            )
            from face_engine import RESOURCES
            from insightface.model_zoo import model_zoo

            # Test RetinaFaceDetector
            detector = RetinaFaceDetector()
            mock_fetch.assert_called_with(BUFFALO_L_URL, os.path.join(RESOURCES, "models/buffalo_l"))
            args, _ = model_zoo.get_model.call_args
            self.assertEqual(args[0], os.path.join(RESOURCES, "models/buffalo_l/det_10g.onnx"))

            # Test SCRFDDetector
            scrfd = SCRFDDetector()
            mock_fetch.assert_called_with(ANTELOPE_V2_URL, os.path.join(RESOURCES, "models/antelopev2"))
            args, _ = model_zoo.get_model.call_args
            self.assertEqual(args[0], os.path.join(RESOURCES, "models/antelopev2/scrfd_10g_bnkps.onnx"))

            # Test ArcFaceEmbedder
            embedder = ArcFaceEmbedder()
            mock_fetch.assert_called_with(BUFFALO_L_URL, os.path.join(RESOURCES, "models/buffalo_l"))
            args, _ = model_zoo.get_model.call_args
            self.assertEqual(args[0], os.path.join(RESOURCES, "models/buffalo_l/w600k_r50.onnx"))

            # Test ArcFaceResNet100Embedder
            r100 = ArcFaceResNet100Embedder()
            mock_fetch.assert_called_with(ANTELOPE_V2_URL, os.path.join(RESOURCES, "models/antelopev2"))
            args, _ = model_zoo.get_model.call_args
            self.assertEqual(args[0], os.path.join(RESOURCES, "models/antelopev2/glintr100.onnx"))

    def test_fallback_paths(self):
        # Mock fetch_file and os.path.exists
        with patch("face_engine.fetching.fetch_file") as mock_fetch, \
             patch("os.path.exists") as mock_exists:

            # Setup mock_exists for fallback logic
            def exists_side_effect(path):
                p = str(path)
                if "buffalo_l/buffalo_l/det_10g.onnx" in p: return True
                if "antelopev2/antelopev2/scrfd_10g_bnkps.onnx" in p: return True
                if "buffalo_l/det_10g.onnx" in p: return False
                if "antelopev2/scrfd_10g_bnkps.onnx" in p: return False
                return True # Default for directory checks etc?
            mock_exists.side_effect = exists_side_effect

            # Reload module
            import face_engine.models.insightface_models
            from importlib import reload
            reload(face_engine.models.insightface_models)

            from face_engine.models.insightface_models import RetinaFaceDetector, SCRFDDetector
            from face_engine import RESOURCES
            from insightface.model_zoo import model_zoo

            # Test RetinaFaceDetector Fallback
            detector = RetinaFaceDetector()
            args, _ = model_zoo.get_model.call_args
            self.assertEqual(args[0], os.path.join(RESOURCES, "models/buffalo_l/buffalo_l/det_10g.onnx"))

            # Test SCRFDDetector Fallback
            scrfd = SCRFDDetector()
            args, _ = model_zoo.get_model.call_args
            self.assertEqual(args[0], os.path.join(RESOURCES, "models/antelopev2/antelopev2/scrfd_10g_bnkps.onnx"))

if __name__ == "__main__":
    unittest.main()
