import argparse
import contextlib
import io
import unittest
from unittest import mock

from train.opsd.train_self_teacher import gen_rollouts


class RolloutCliTest(unittest.TestCase):
    def test_removed_modes_are_not_part_of_the_cli(self):
        parser = gen_rollouts.build_parser()
        args = parser.parse_args([])
        self.assertFalse(hasattr(args, "stage"))
        self.assertFalse(hasattr(args, "questions_from"))

        for removed_args in (
            ["--stage", "score"],
            ["--questions-from", "dataset"],
            ["--score-batch-size", "2"],
        ):
            with self.subTest(args=removed_args):
                with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                    parser.parse_args(removed_args)

    def test_main_always_generates_and_scores(self):
        args = argparse.Namespace(
            model="student",
            dataset="deepmath",
            output_root="rollouts",
            max_samples=2,
            n=3,
            force=False,
        )
        parser = mock.Mock()
        parser.parse_args.return_value = args

        with (
            mock.patch.object(gen_rollouts, "build_parser", return_value=parser),
            mock.patch.object(gen_rollouts, "rollout_path", return_value="rollouts/cache"),
            mock.patch.object(gen_rollouts.os.path, "isdir", return_value=False),
            mock.patch.object(gen_rollouts, "generate") as generate,
            mock.patch.object(gen_rollouts, "score_in_clean_process") as score,
        ):
            gen_rollouts.main()

        generate.assert_called_once_with(args, "rollouts/cache")
        score.assert_called_once_with(args, "rollouts/cache")

    def test_cache_without_hint_provenance_is_regenerated(self):
        args = argparse.Namespace(
            model="student",
            dataset="deepmath",
            output_root="rollouts",
            max_samples=2,
            n=3,
            force=False,
        )
        parser = mock.Mock()
        parser.parse_args.return_value = args
        cached = mock.MagicMock()
        cached.column_names = ["gen_model"]
        cached.unique.return_value = ["student"]
        cached.__len__.return_value = 6

        with (
            mock.patch.object(gen_rollouts, "build_parser", return_value=parser),
            mock.patch.object(gen_rollouts, "rollout_path", return_value="rollouts/cache"),
            mock.patch.object(gen_rollouts.os.path, "isdir", return_value=True),
            mock.patch.object(gen_rollouts, "load_from_disk", return_value=cached),
            mock.patch.object(gen_rollouts, "generate") as generate,
            mock.patch.object(gen_rollouts, "score_in_clean_process") as score,
        ):
            gen_rollouts.main()

        generate.assert_called_once_with(args, "rollouts/cache")
        score.assert_called_once_with(args, "rollouts/cache")



class RolloutScoringProcessTest(unittest.TestCase):
    def test_scoring_uses_spawn_and_propagates_the_worker_target(self):
        args = argparse.Namespace(model="student")
        process = mock.Mock(exitcode=0)
        context = mock.Mock()
        context.Process.return_value = process

        with mock.patch.object(
            gen_rollouts.multiprocessing, "get_context", return_value=context
        ) as get_context:
            gen_rollouts.score_in_clean_process(args, "rollouts/cache")

        get_context.assert_called_once_with("spawn")
        context.Process.assert_called_once_with(
            target=gen_rollouts.score, args=(args, "rollouts/cache")
        )
        process.start.assert_called_once_with()
        process.join.assert_called_once_with()
        process.close.assert_called_once_with()

    def test_scoring_worker_failure_is_not_silenced(self):
        process = mock.Mock(exitcode=7)
        context = mock.Mock()
        context.Process.return_value = process
        with mock.patch.object(gen_rollouts.multiprocessing, "get_context", return_value=context):
            with self.assertRaisesRegex(RuntimeError, "exit code 7"):
                gen_rollouts.score_in_clean_process(argparse.Namespace(), "rollouts/cache")

        process.close.assert_called_once_with()


if __name__ == "__main__":
    unittest.main()

