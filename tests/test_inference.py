import unittest
import numpy as np
from face_engine import FaceEngine
from face_engine.tools import imread
from tests import TestCase

try:
    import insightface
except ImportError:
    insightface = None

try:
    import dlib
except ImportError:
    dlib = None

class TestInference(TestCase):
    @unittest.skipIf(insightface is None, "insightface not installed")
    def test_insightface_inference(self):
        engine = FaceEngine(detector="retina_face", embedder="arcface")
        self._run_inference_test(engine, "InsightFace")

    @unittest.skipIf(dlib is None, "dlib not installed")
    def test_dlib_inference(self):
        engine = FaceEngine(detector="hog", embedder="resnet")
        self._run_inference_test(engine, "Dlib")

    def _run_inference_test(self, engine, model_name):
        # Load images
        img1 = imread(self.bubbles1)
        img2 = imread(self.bubbles2)
        img3 = imread(self.drive)

        # Detect and embed
        bbs1, extra1 = engine.find_faces(img1, limit=1)
        self.assertGreater(len(bbs1), 0, f"No face detected in bubbles1 with {model_name}")
        emb1 = engine.compute_embeddings(img1, bbs1, **extra1)[0]

        bbs2, extra2 = engine.find_faces(img2, limit=1)
        self.assertGreater(len(bbs2), 0, f"No face detected in bubbles2 with {model_name}")
        emb2 = engine.compute_embeddings(img2, bbs2, **extra2)[0]

        bbs3, extra3 = engine.find_faces(img3, limit=1)
        self.assertGreater(len(bbs3), 0, f"No face detected in drive with {model_name}")
        emb3 = engine.compute_embeddings(img3, bbs3, **extra3)[0]

        # Calculate similarities
        sim_same = np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2))
        sim_diff = np.dot(emb1, emb3) / (np.linalg.norm(emb1) * np.linalg.norm(emb3))

        print(f"\n{model_name} Similarity (Same): {sim_same:.4f}")
        print(f"{model_name} Similarity (Diff): {sim_diff:.4f}")

        self.assertGreater(sim_same, 0.6, f"{model_name}: Same person similarity too low")
        self.assertLess(sim_diff, 0.4, f"{model_name}: Different person similarity too high")

if __name__ == "__main__":
    unittest.main()
