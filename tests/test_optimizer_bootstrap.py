import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from agent.config import ExperimentConfig, ProblemConfig
from agent.core.optimizer import Optimizer
from agent.eval.modal_evaluator import ModalEvaluator


class OptimizerBootstrapTests(unittest.TestCase):
    @patch("agent.core.optimizer.OpenAIPatchClient")
    def test_seed_candidate_bootstraps_when_seed_file_missing(self, mock_client_cls):
        fake_client = Mock()
        fake_client.generate_initial_candidate.return_value = (
            "import torch\n\n"
            "def solution(*args, **kwargs):\n"
            "    return args[0]\n"
        )
        mock_client_cls.return_value = fake_client

        with tempfile.TemporaryDirectory() as tmp:
            repo = Path(tmp)
            (repo / "reference").mkdir(parents=True)
            (repo / "solutions_triton").mkdir(parents=True)
            (repo / "solutions_agent").mkdir(parents=True)

            (repo / "reference" / "5.py").write_text(
                "def solution(x):\n    return x\n",
                encoding="utf-8",
            )
            (repo / "solutions_triton" / "1_triton.py").write_text(
                "def solution(x):\n    return x\n",
                encoding="utf-8",
            )

            cfg = ExperimentConfig(
                name="test_bootstrap",
                seed_from_backend="triton",
                candidate_dir="solutions_agent",
                problems=[ProblemConfig(problem_id=5)],
            )
            optimizer = Optimizer(repo_root=repo, cfg=cfg)

            result = optimizer.bootstrap.ensure_candidate(5)

            candidate_file = repo / "solutions_agent" / "5_agent.py"
            self.assertTrue(candidate_file.exists())
            self.assertIn("def solution", candidate_file.read_text(encoding="utf-8"))
            self.assertEqual(fake_client.generate_initial_candidate.call_count, 1)
            self.assertEqual(result.event, "bootstrap_generated")

    @patch("agent.core.optimizer.OpenAIPatchClient")
    def test_seed_candidate_uses_seed_file_when_available(self, mock_client_cls):
        fake_client = Mock()
        mock_client_cls.return_value = fake_client

        with tempfile.TemporaryDirectory() as tmp:
            repo = Path(tmp)
            (repo / "reference").mkdir(parents=True)
            (repo / "solutions_triton").mkdir(parents=True)
            (repo / "solutions_agent").mkdir(parents=True)

            seed_code = "def solution(x):\n    return x + 1\n"
            (repo / "solutions_triton" / "5_triton.py").write_text(seed_code, encoding="utf-8")

            cfg = ExperimentConfig(
                name="test_seed_copy",
                seed_from_backend="triton",
                candidate_dir="solutions_agent",
                problems=[ProblemConfig(problem_id=5)],
            )
            optimizer = Optimizer(repo_root=repo, cfg=cfg)

            result = optimizer.bootstrap.ensure_candidate(5)

            candidate_file = repo / "solutions_agent" / "5_agent.py"
            self.assertEqual(seed_code, candidate_file.read_text(encoding="utf-8"))
            self.assertEqual(fake_client.generate_initial_candidate.call_count, 0)
            self.assertEqual(result.event, "bootstrap_seed_copy")

    @patch("agent.core.optimizer.OpenAIPatchClient")
    def test_optimizer_routes_correctness_fix_mode_and_logs_cycle_event(self, mock_client_cls):
        fake_client = Mock()
        mock_client_cls.return_value = fake_client

        with tempfile.TemporaryDirectory() as tmp:
            repo = Path(tmp)
            (repo / "reference").mkdir(parents=True)
            (repo / "solutions_triton").mkdir(parents=True)
            (repo / "solutions_agent").mkdir(parents=True)
            (repo / "runs").mkdir(parents=True)

            (repo / "solutions_triton" / "5_triton.py").write_text(
                "def solution(x):\n    return x\n",
                encoding="utf-8",
            )

            cfg = ExperimentConfig(
                name="test_mode_routing",
                seed_from_backend="triton",
                candidate_dir="solutions_agent",
                max_iterations=1,
                problems=[ProblemConfig(problem_id=5)],
            )
            optimizer = Optimizer(repo_root=repo, cfg=cfg)

            reference = Mock()
            reference.summary_rank0 = {"aggregate": {"reference_mean_ms": 2.0, "candidate_mean_ms": 2.0}}
            reference.eval_feedback = {"correctness": {"all_ok": True}}
            candidate = Mock()
            candidate.summary_rank0 = {"aggregate": {"candidate_mean_ms": 1.0}}
            candidate.eval_feedback = {"correctness": {"all_ok": False, "failed_ranks": [0]}}

            optimizer.evaluator = Mock()
            optimizer.evaluator.precompute_references = Mock()
            optimizer.evaluator.evaluate_pair = Mock(return_value=(reference, candidate))
            optimizer.proposer = Mock()
            optimizer.proposer.propose = Mock(return_value={"candidate_code": "def solution(x):\n    return x\n"})
            optimizer._apply_code = Mock()

            optimizer.run()

            proposal_ctx = optimizer.proposer.propose.call_args.args[0]
            self.assertEqual(proposal_ctx.proposal_mode, "correctness_fix")

            log_path = optimizer.run_dir / "iterations.jsonl"
            log_content = log_path.read_text(encoding="utf-8")
            self.assertIn('"event": "bootstrap_seed_copy"', log_content)
            self.assertIn('"event": "correctness_fix_cycle"', log_content)
            self.assertIn('"event": "proposal_applied"', log_content)


class ModalEvaluatorCacheTests(unittest.TestCase):
    def test_precompute_reference_then_candidate_uses_cached_reference_flag(self):
        with tempfile.TemporaryDirectory() as tmp:
            repo = Path(tmp)
            ref_dir = repo / "logs" / "problem_5" / "reference"
            cand_dir = repo / "logs" / "problem_5" / "agent"
            ref_dir.mkdir(parents=True, exist_ok=True)
            cand_dir.mkdir(parents=True, exist_ok=True)
            (ref_dir / "summary_rank0.json").write_text(
                '{"status":"ok","aggregate":{"candidate_mean_ms":2.0},"ranks":[]}',
                encoding="utf-8",
            )
            (cand_dir / "summary_rank0.json").write_text(
                '{"status":"ok","aggregate":{"candidate_mean_ms":1.0},"ranks":[]}',
                encoding="utf-8",
            )

            cfg = ExperimentConfig(name="eval_cache", problems=[ProblemConfig(problem_id=5)])
            evaluator = ModalEvaluator(repo, cfg.eval)
            problem = ProblemConfig(problem_id=5)

            with patch("agent.eval.modal_evaluator.subprocess.run") as mock_run:
                evaluator.precompute_references([problem])
                evaluator.evaluate_pair(problem)
                evaluator.evaluate_pair(problem)

                self.assertEqual(mock_run.call_count, 3)
                first_cmd = mock_run.call_args_list[0].args[0]
                second_cmd = mock_run.call_args_list[1].args[0]
                third_cmd = mock_run.call_args_list[2].args[0]
                self.assertNotIn("true", first_cmd)
                self.assertNotIn("true", second_cmd)
                self.assertNotIn("true", third_cmd)
                self.assertNotIn("--use-cached-reference", first_cmd)
                self.assertIn("--use-cached-reference", second_cmd)
                self.assertIn("--use-cached-reference", third_cmd)


if __name__ == "__main__":
    unittest.main()
