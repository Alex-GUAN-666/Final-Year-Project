"""Regression coverage for the observed --rm / rm -f Docker cleanup race."""
import subprocess
import unittest
from unittest.mock import patch

from fyp.execution import _cleanup_docker_container


def result(returncode=0, stdout="", stderr=""):
    return subprocess.CompletedProcess([], returncode, stdout, stderr)


class DockerCleanupTests(unittest.TestCase):
    def test_auto_removal_race_waits_until_daemon_confirms_absence(self):
        # The first rm conflicts with auto-removal and the container still
        # exists. A later absent listing, not the rm error text, permits success.
        replies = [
            result(1, stderr="removal of container is already in progress"),
            result(stdout="container-id\n"),
            result(1, stderr="No such container: fyp-exec-test"),
            result(),
        ]
        with patch("fyp.execution.subprocess.run", side_effect=replies) as run, \
                patch("fyp.execution.time.sleep"):
            _cleanup_docker_container("fyp-exec-test")
        listings = [call.args[0] for call in run.call_args_list if call.args[0][1] == "ps"]
        self.assertEqual(len(listings), 2)
        self.assertTrue(all("--all" in command and "name=^/fyp-exec-test$" in command
                            for command in listings))

    def test_failed_rm_is_accepted_only_after_positive_absence_check(self):
        with patch("fyp.execution.subprocess.run", side_effect=[
                result(1, stderr="removal already in progress"), result()]):
            _cleanup_docker_container("fyp-exec-test")

    def test_no_such_container_text_does_not_hide_daemon_failure(self):
        with patch("fyp.execution.subprocess.run", side_effect=[
                result(1, stderr="No such container: fyp-exec-test"),
                result(1, stderr="Cannot connect to the Docker daemon")]), \
                self.assertRaisesRegex(RuntimeError, "absence check failed"):
            _cleanup_docker_container("fyp-exec-test")

    def test_successful_rm_does_not_hide_a_surviving_container(self):
        with patch("fyp.execution.subprocess.run", side_effect=[
                result(), result(stdout="container-id\n")] * 5), \
                patch("fyp.execution.time.sleep"), \
                self.assertRaisesRegex(RuntimeError, "Container remains present"):
            _cleanup_docker_container("fyp-exec-test")

    def test_absence_query_timeout_is_fail_closed(self):
        with patch("fyp.execution.subprocess.run", side_effect=[
                result(), subprocess.TimeoutExpired(["docker", "ps"], timeout=10)]), \
                self.assertRaisesRegex(RuntimeError, "absence check failed"):
            _cleanup_docker_container("fyp-exec-test")


if __name__ == "__main__":
    unittest.main()
