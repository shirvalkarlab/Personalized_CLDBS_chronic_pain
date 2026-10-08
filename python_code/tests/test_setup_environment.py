"""
Tests for setup_environment.py that need no network access.

Run from python_code/:
    python -m unittest discover -s tests -v
"""
import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir))
import setup_environment as setup  # noqa: E402


class SetupEnvironmentTest(unittest.TestCase):

    def test_reads_the_version_of_a_python_executable(self):
        self.assertEqual(setup.python_version(sys.executable), sys.version_info[:2])

    def test_missing_executable_has_no_version(self):
        self.assertIsNone(setup.python_version("/no/such/python3.11"))

    def test_only_python_3_10_and_3_11_are_supported(self):
        self.assertTrue(setup.is_supported((3, 10)))
        self.assertTrue(setup.is_supported((3, 11)))
        for version in [(3, 9), (3, 12), (3, 14), (2, 7), None]:
            self.assertFalse(setup.is_supported(version))

    def test_reads_every_pin_from_requirements_txt(self):
        pins = setup.read_pins(setup.REQUIREMENTS_FILE)
        self.assertEqual(pins["numpy"], "1.24.4")
        self.assertEqual(pins["scipy"], "1.10.1")
        self.assertEqual(pins["scikit-learn"], "1.2.2")
        self.assertNotIn("#", "".join(pins))

    def test_pins_ignore_comments_and_blank_lines(self):
        with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False) as handle:
            handle.write("# comment\n\nnumpy==1.24.4  # trailing\n  pandas==2.3.3\n")
        try:
            self.assertEqual(setup.read_pins(handle.name), {"numpy": "1.24.4", "pandas": "2.3.3"})
        finally:
            os.remove(handle.name)

    def test_the_running_python_is_the_first_candidate(self):
        candidates = setup.candidate_pythons()
        self.assertEqual(candidates[0], sys.executable)


if __name__ == "__main__":
    unittest.main()
