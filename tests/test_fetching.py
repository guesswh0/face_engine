import hashlib
import io
import os
import shutil
import tempfile
import unittest
from unittest import mock

from face_engine import fetching
from face_engine.fetching import fetch_file


class _FakeResponse:
    """Minimal stand-in for urlopen's response object."""

    def __init__(self, data):
        self._buffer = io.BytesIO(data)
        self.headers = {"Content-Length": str(len(data))}

    def read(self, size=-1):
        return self._buffer.read(size)

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


class TestFetchFile(unittest.TestCase):

    data = b"face-engine test payload"
    sha256 = hashlib.sha256(data).hexdigest()
    url = "https://example.com/payload.bin"

    def setUp(self):
        self.dir = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.dir, ignore_errors=True)

    def test_download_with_good_checksum(self):
        with mock.patch.object(
            fetching, "urlopen", return_value=_FakeResponse(self.data)
        ):
            fetch_file(self.url, self.dir, sha256=self.sha256)
        file = os.path.join(self.dir, "payload.bin")
        self.assertTrue(os.path.isfile(file))
        with open(file, "rb") as f:
            self.assertEqual(f.read(), self.data)

    def test_bad_checksum_raises_and_cleans_up(self):
        with mock.patch.object(
            fetching, "urlopen", return_value=_FakeResponse(self.data)
        ):
            with self.assertRaises(IOError) as context:
                fetch_file(self.url, self.dir, sha256="0" * 64)
        self.assertIn("sha256 mismatch", str(context.exception))
        # neither the temp file nor the target file is left behind
        self.assertEqual(os.listdir(self.dir), [])

    def test_known_hashes_fallback(self):
        with mock.patch.dict(fetching.KNOWN_HASHES, {"payload.bin": "0" * 64}):
            with mock.patch.object(
                fetching, "urlopen", return_value=_FakeResponse(self.data)
            ):
                with self.assertRaises(IOError):
                    fetch_file(self.url, self.dir)

    def test_existing_file_short_circuits(self):
        file = os.path.join(self.dir, "payload.bin")
        with open(file, "wb") as f:
            f.write(b"cached")
        with mock.patch.object(
            fetching, "urlopen", side_effect=AssertionError("unexpected network hit")
        ):
            fetch_file(self.url, self.dir)

    def test_rejects_non_http_scheme(self):
        with self.assertRaises(ValueError):
            fetch_file("file:///etc/passwd", self.dir)
        with self.assertRaises(ValueError):
            fetch_file("ftp://example.com/x.dat", self.dir)


if __name__ == "__main__":
    unittest.main()
