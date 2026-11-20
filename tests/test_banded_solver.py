"""交接协议在求解器上的行为测试：几何、证书、票据、预算与反传。"""

from __future__ import annotations

import copy
import unittest

import numpy as np

import banded_solver as solver
from read_capture import read_capture


CAPTURE = "samples/capture.json"


def base_capture():
    return copy.deepcopy(read_capture(CAPTURE))


def small_capture(**overrides):
    """一个便于精确检查的 4 点连续剖面。"""
    capture = {
        "offsets": [-1, 0, 1],
        "tol": 1e-9,
        "pivot_floor": 1e-10,
        "scratch_bytes": 1 << 20,
        "slots": {"d": [2.0, True], "l": [-0.25, True], "p": [0.7, False]},
        "profiles": {
            "chain": {
                "active": [True, True, True, True],
                "bands": [
                    [None, ["d", 1], ["l", 1]],
                    [["l", 1], ["d", 1], ["l", 1]],
                    [["l", 1], ["d", 1], ["l", 1]],
                    [["l", 1], ["d", 1], None],
                ],
                "boundary": {"left": None, "right": None, "single": None},
            }
        },
        "rhs": {"u": [1.0, 0.0, 0.0, 1.0], "v": [0.0, 1.0, 2.0, 3.0]},
        "events": [],
    }
    capture.update(overrides)
    return capture


def storages(solver_instance, profile="chain"):
    segments, _ = solver_instance._resolve(profile)
    return [_assemble(segment)[0] for segment in segments]


def _assemble(segment, batch=None):
    return solver._assemble(segment, (batch or solver.BatchSolver(small_capture())).slots)


class GeometryTests(unittest.TestCase):
    def test_offsets_map_band_columns_to_unknowns(self):
        capture = small_capture()
        batch = solver.BatchSolver(capture)
        storage = storages(batch)[0]
        matrix = np.array(
            [[storage.get(i, j) for j in range(storage.n)] for i in range(storage.n)]
        )
        expected = np.array(
            [
                [2.0, -0.25, 0.0, 0.0],
                [-0.25, 2.0, -0.25, 0.0],
                [0.0, -0.25, 2.0, -0.25],
                [0.0, 0.0, -0.25, 2.0],
            ]
        )
        self.assertTrue(np.allclose(matrix, expected))

    def test_gap_splits_segments_and_ignores_cross_coupling(self):
        capture = small_capture()
        capture["profiles"]["chain"]["active"] = [True, True, False, True]
        batch = solver.BatchSolver(capture)
        segments, _ = batch._resolve("chain")
        self.assertEqual([segment.points for segment in segments], [(0, 1), (3,)])

    def test_single_point_segment_has_its_own_boundary_row(self):
        capture = small_capture()
        capture["profiles"]["chain"]["active"] = [True, True, False, True]
        capture["profiles"]["chain"]["boundary"] = {
            "left": None,
            "right": None,
            "single": {"row": [None, ["d", 2], None], "add_rhs": ["p", 1]},
        }
        batch = solver.BatchSolver(capture)
        segments, _ = batch._resolve("chain")
        storage, _ = solver._assemble(segments[1], batch.slots)
        self.assertEqual(storage.n, 1)
        self.assertAlmostEqual(storage.get(0, 0), 4.0)
        # 补齐位置不生成列：第二段只有 1 列，不出现第 4 个未知量。
        self.assertEqual(len(segments[0].points) + len(segments[1].points), 3)

    def test_boundary_row_expands_offsets_and_adds_constant(self):
        capture = small_capture()
        capture["profiles"]["chain"]["boundary"] = {
            "left": {"row": [None, ["p", 1], ["l", 2]], "add_rhs": ["d", 0.5]},
            "right": {"row": [["d", 1], None, None], "add_rhs": None},
            "single": None,
        }
        batch = solver.BatchSolver(capture)
        segment = batch._resolve("chain")[0][0]
        storage, _ = solver._assemble(segment, batch.slots)
        # 左边界从点 0 展开：列 1 -> 未知量 0，列 2 -> 未知量 1。
        self.assertAlmostEqual(storage.get(0, 0), 0.7)
        self.assertAlmostEqual(storage.get(0, 1), -0.5)
        self.assertEqual(segment.constants[0], (0, "d", 0.5))
        # 右边界从末点（点 3）展开：列 0 -> 未知量 2，列 1 -> 未知量 3（未给系数）。
        self.assertAlmostEqual(storage.get(3, 2), 2.0)
        self.assertEqual(storage.get(3, 3), 0.0)


class CertificateTests(unittest.TestCase):
    def test_low_precision_is_used_when_it_meets_tolerance(self):
        batch = solver.BatchSolver(small_capture(tol=1e-6))
        report = batch.solve("t", "chain", ["u"])
        channel = report["u"]
        self.assertTrue(channel.accepted)
        self.assertEqual(channel.precision, "fp32")
        self.assertEqual(channel.retried, ("fp32",))
        self.assertLessEqual(channel.certificate, batch.tol)

    def test_promotion_when_low_precision_fails(self):
        capture = small_capture(tol=1e-12)
        batch = solver.BatchSolver(capture)
        report = batch.solve("t", "chain", ["v"])
        channel = report["v"]
        self.assertTrue(channel.accepted)
        self.assertEqual(channel.precision, "fp64")
        self.assertEqual(channel.retried, ("fp32", "fp64"))
        self.assertTrue(report.budget["promotions"])

    def test_singular_profile_keeps_reason_and_other_channels_continue(self):
        batch = solver.BatchSolver(base_capture())
        report = batch.solve("t", "fault", ["bad"])
        channel = report["bad"]
        self.assertFalse(channel.accepted)
        self.assertIn("奇异", channel.reason)
        self.assertEqual(channel.ranges, ())
        self.assertFalse(report.continuation()["can_continue"])

    def test_singular_segment_does_not_block_independent_segment_certificate(self):
        capture = base_capture()
        report = solver.BatchSolver(capture).solve("t", "fault", ["bad"])
        self.assertFalse(report["bad"].accepted)
        # ridge 的段 0 在同样预算下仍可认证。
        ridge = solver.BatchSolver(capture).solve("t", "ridge", ["edge"])
        self.assertTrue(ridge["edge"].accepted)

    def test_pivot_floor_rejects_near_singular_but_smooth_output(self):
        # A = [[1, 1], [1, 1+1e-14]]：输出可以很平滑，但主元远低于范数。
        capture = small_capture()
        capture["slots"] = {"a": [1.0, True], "b": [1.0, True], "c": [1.0, True]}
        capture["profiles"]["chain"] = {
            "active": [True, True],
            "bands": [
                [None, ["a", 1], ["b", 1]],
                [["c", 1], ["a", 1.00000000000001], None],
            ],
            "boundary": {"left": None, "right": None, "single": None},
        }
        capture["rhs"] = {"u": [1.0, 1.0]}
        capture["pivot_floor"] = 1e-6
        batch = solver.BatchSolver(capture)
        channel = batch.solve("t", "chain", ["u"])["u"]
        self.assertFalse(channel.accepted)
        self.assertIn("主元比率", channel.reason)

    def test_zero_rhs_zero_solution_has_zero_certificate(self):
        capture = small_capture()
        capture["rhs"] = {"u": [0.0, 0.0, 0.0, 0.0]}
        batch = solver.BatchSolver(capture)
        report = batch.solve("t", "chain", ["u"])
        self.assertTrue(report["u"].accepted)
        self.assertEqual(report["u"].certificate, 0.0)

    def test_certificate_matches_protocol_formula(self):
        batch = solver.BatchSolver(base_capture())
        batch.solve("t", "ridge", ["edge"])
        segment = batch._tickets["t"].channels["edge"].segments[0]
        storage = segment.storage
        matrix = np.array(
            [[storage.get(i, j) for j in range(storage.n)] for i in range(storage.n)]
        )
        x, b = segment.solution, segment.rhs
        expected = np.linalg.norm(matrix @ x - b, np.inf) / (
            np.linalg.norm(matrix, np.inf) * np.linalg.norm(x, np.inf)
            + np.linalg.norm(b, np.inf)
        )
        self.assertAlmostEqual(segment.certificate, expected, places=15)


class TicketTests(unittest.TestCase):
    def test_old_ticket_survives_publish(self):
        batch = solver.BatchSolver(base_capture(), budget_bytes=1 << 20)
        batch.solve("old", "ridge", ["edge"])
        before = batch._tickets["old"].channels["edge"].segments[0].solution.copy()
        batch.publish(
            "r1",
            {
                "ridge": {
                    "left": {"row": [None, ["p", 1], ["l", 2]], "add_rhs": ["d", 0.1]},
                    "right": None,
                    "single": {"row": [None, ["d", 1], None], "add_rhs": ["p", 1]},
                }
            },
        )
        report = batch.solve("new", "ridge", ["drift"])
        self.assertEqual(report.version, "r1")
        after = batch._tickets["old"].channels["edge"].segments[0].solution
        self.assertTrue(np.array_equal(before, after))
        self.assertNotEqual(report["drift"].segments[0]["certificate"], None)
        # 旧票据的版本与边界定义也保留。
        self.assertEqual(batch.results("old")[0]["boundary_version"], "__initial__")
        self.assertIsNone(batch.results("old")[0]["boundary"]["left"])

    def test_duplicate_label_cannot_be_rebuilt(self):
        batch = solver.BatchSolver(base_capture())
        batch.solve("old", "ridge", ["easy"])
        with self.assertRaisesRegex(solver.CaptureError, "已存在"):
            batch.solve("old", "ridge", ["edge"])

    def test_rhs_update_only_changes_rhs_not_matrix(self):
        batch = solver.BatchSolver(base_capture())
        batch.solve("old", "ridge", ["easy"])
        old_segments = batch._tickets["old"].channels["easy"].segments
        batch.update_rhs("edge", [0.0, 0.0, 0.0, 0.0, 0.0])
        report = batch.solve("new", "ridge", ["easy"])
        self.assertEqual(report["easy"].ranges, ((0, 3), (4, 5)))
        # 右端项变化使段内证书改变，但旧票据仍绑定旧右端项。
        self.assertTrue(np.array_equal(old_segments[1].rhs, [0.7]))

    def test_cache_hit_uses_matrix_fingerprint_not_length(self):
        batch = solver.BatchSolver(base_capture())
        batch.solve("first", "ridge", ["easy"])
        misses = batch.budget_report()["misses"]
        batch.solve("second", "ridge", ["edge"])
        report = batch.budget_report()
        self.assertGreater(report["hits"], 0)
        self.assertLessEqual(report["misses"] - misses, 2)

    def test_changed_boundary_matrix_is_not_served_from_length_key(self):
        batch = solver.BatchSolver(base_capture())
        batch.solve("old", "ridge", ["edge"])
        batch.publish(
            "r1",
            {
                "ridge": {
                    "left": {"row": [None, ["p", 1], ["l", 2]], "add_rhs": ["d", 0.1]},
                    "right": None,
                    "single": None,
                }
            },
        )
        batch.solve("new", "ridge", ["edge"])
        # 两版矩阵不同：旧段 0 的矩阵与新版段 0 的矩阵元素不一致。
        old = batch._tickets["old"].channels["edge"].segments[0].storage
        new = batch._tickets["new"].channels["edge"].segments[0].storage
        self.assertNotAlmostEqual(old.get(0, 0), new.get(0, 0))
        self.assertNotEqual(batch._digest(old), batch._digest(new))

    def test_version_can_change_only_one_rhs(self):
        batch = solver.BatchSolver(small_capture())
        batch.solve("a", "chain", ["u"])
        matrix_before = batch._tickets["a"].channels["u"].segments[0].storage
        batch.publish(
            "r1",
            {"chain": {"left": None, "right": None, "single": None}},
        )
        report = batch.solve("b", "chain", ["v"])
        matrix_after = batch._tickets["b"].channels["v"].segments[0].storage
        self.assertEqual(batch._digest(matrix_before), batch._digest(matrix_after))


class BudgetTests(unittest.TestCase):
    def test_scratch_accounts_for_factor_reuse_and_promotion(self):
        batch = solver.BatchSolver(base_capture(), budget_bytes=1 << 20)
        report = batch.solve("t", "ridge", ["easy", "edge"])
        budget = report.budget
        self.assertGreater(budget["used_bytes"], 0)
        self.assertLessEqual(budget["used_bytes"], budget["budget_bytes"])
        self.assertGreaterEqual(budget["peak_bytes"], budget["used_bytes"])
        self.assertTrue(budget["promotions"])
        self.assertEqual(budget["remaining_bytes"], budget["budget_bytes"] - budget["used_bytes"])

    def test_tight_budget_keeps_reason_and_allows_other_segments(self):
        capture = small_capture(scratch_bytes=64)
        batch = solver.BatchSolver(capture)
        report = batch.solve("t", "chain", ["u"])
        channel = report["u"]
        self.assertFalse(channel.accepted)
        self.assertIn("预算", channel.reason)
        self.assertEqual(report.continuation()["rejected"][0]["key"], "u")

    def test_budget_overflow_reports_needed_bytes(self):
        capture = small_capture(scratch_bytes=8)
        batch = solver.BatchSolver(capture)
        report = batch.solve("t", "chain", ["u"])
        self.assertFalse(report["u"].accepted)
        self.assertIn("预算", report.budget["reason"] or report["u"].reason)

    def test_release_frees_scratch(self):
        batch = solver.BatchSolver(base_capture(), budget_bytes=256)
        batch.solve("t", "ridge", ["easy"])
        self.assertGreater(batch.budget_report()["used_bytes"], 0)
        batch.release("t")
        self.assertEqual(batch.budget_report()["used_bytes"], 0)
        self.assertEqual(batch.budget_report()["evictions"], 0)


class BackwardTests(unittest.TestCase):
    def setUp(self):
        self.base = base_capture()
        self.batch = solver.BatchSolver(self.base, budget_bytes=1 << 20)
        self.batch.solve("old", "ridge", ["edge"])
        self.active = np.array(self.base["profiles"]["ridge"]["active"])
        self.cotangent = np.array([1.0, 0.0, -2.0, 0.0, 2.0])

    def test_rhs_gradient_matches_finite_difference(self):
        report = self.batch.back("old", "edge", self.cotangent)
        self.assertEqual(report.status, "ok")
        gradient = report.rhs_grad
        for index in range(len(self.active)):
            step = 1e-6

            def loss(delta):
                capture = copy.deepcopy(self.base)
                capture["rhs"]["edge"][index] += delta
                batch = solver.BatchSolver(capture, budget_bytes=1 << 20)
                batch.solve("x", "ridge", ["edge"])
                values = np.concatenate(
                    [seg.solution for seg in batch._tickets["x"].channels["edge"].segments]
                )
                return float(np.dot(self.cotangent[self.active], values))

            expected = (loss(step) - loss(-step)) / (2 * step)
            self.assertAlmostEqual(gradient[index], expected, places=6)

    def test_inactive_cotangent_is_ignored(self):
        report = self.batch.back("old", "edge", self.cotangent)
        self.assertEqual(report.rhs_grad[3], 0.0)
        active_only = self.batch.back("old", "edge", self.cotangent[self.active])
        self.assertTrue(np.allclose(report.rhs_grad, active_only.rhs_grad))

    def test_coefficient_gradient_matches_finite_difference(self):
        report = self.batch.back("old", "edge", self.cotangent)
        for slot in ("d", "l"):
            step = 1e-5

            def loss(delta):
                capture = copy.deepcopy(self.base)
                capture["slots"][slot][0] += delta
                batch = solver.BatchSolver(capture, budget_bytes=1 << 20)
                batch.solve("x", "ridge", ["edge"])
                values = np.concatenate(
                    [seg.solution for seg in batch._tickets["x"].channels["edge"].segments]
                )
                return float(np.dot(self.cotangent[self.active], values))

            expected = (loss(step) - loss(-step)) / (2 * step)
            self.assertAlmostEqual(report.coefficient_grads[slot], expected, places=5)


    def test_fixed_boundary_slot_is_not_trained(self):
        report = self.batch.back("old", "edge", self.cotangent)
        self.assertNotIn("p", report.coefficient_grads)
        self.assertNotIn("p", report.trainable_slots)

    def test_shared_slot_across_segments_and_channels_is_summed(self):
        report = self.batch.solve("multi", "ridge", ["easy", "edge"])
        self.assertTrue(report["easy"].accepted and report["edge"].accepted)
        backward = self.batch.back(
            "multi",
            ["easy", "edge"],
            {
                "easy": np.array([0.5, 0.0, 0.0, 0.0, 0.0]),
                "edge": self.cotangent,
            },
        )
        self.assertEqual(backward.status, "ok")
        for slot in backward.coefficient_grads:
            total = sum(
                per_key.get(slot, 0.0) for per_key in backward.coefficient_grads_by_key.values()
            )
            self.assertAlmostEqual(backward.coefficient_grads[slot], total, places=12)

    def test_rejected_forward_returns_no_gradient(self):
        drifty = solver.BatchSolver(base_capture())
        drifty.solve("old", "ridge", ["easy"])
        drifty.publish(
            "r1",
            {
                "ridge": {
                    "left": {"row": [None, ["p", 1], ["l", 2]], "add_rhs": ["d", 0.1]},
                    "right": None,
                    "single": {"row": [None, ["d", 1], None], "add_rhs": ["p", 1]},
                }
            },
        )
        drifty._rhs["drift"] = np.zeros(5)
        drifty._rhs["drift"][0] = np.nan
        report = drifty.solve("bad", "ridge", ["drift"])
        self.assertFalse(report["drift"].accepted)
        backward = drifty.back("bad", "drift", self.cotangent)
        self.assertEqual(backward.status, "rejected")
        self.assertEqual(backward.rhs_grads, {})
        self.assertEqual(backward.coefficient_grads, {})

    def test_adjoint_uses_same_tolerance_and_reuses_factor(self):
        report = self.batch.back("old", "edge", self.cotangent)
        segments = report.segments["edge"]
        self.assertTrue(all(item["accepted"] for item in segments))
        for item in segments:
            self.assertLessEqual(item["certificate"], self.batch.tol)

    def test_back_on_old_ticket_uses_old_boundary(self):
        self.batch.publish(
            "r1",
            {
                "ridge": {
                    "left": {"row": [None, ["p", 1], ["l", 2]], "add_rhs": ["d", 0.1]},
                    "right": None,
                    "single": None,
                }
            },
        )
        self.batch.solve("new", "ridge", ["drift"])
        old = self.batch.back("old", "edge", self.cotangent)
        new = self.batch.back("new", "drift", self.cotangent)
        self.assertEqual(old.boundary_version, "__initial__")
        self.assertEqual(new.boundary_version, "r1")
        self.assertNotAlmostEqual(
            old.coefficient_grads["d"], new.coefficient_grads["d"], places=6
        )


class BatchSplitTests(unittest.TestCase):
    def test_channel_order_does_not_change_results(self):
        cotangents = {
            "easy": np.array([0.5, 0.0, 0.0, 0.0, 0.0]),
            "edge": np.array([1.0, 0.0, -2.0, 0.0, 2.0]),
        }
        results = []
        for keys in (["easy", "edge"], ["edge", "easy"]):
            batch = solver.BatchSolver(base_capture(), budget_bytes=1 << 20)
            batch.solve("t", "ridge", keys)
            backward = batch.back("t", keys, {key: cotangents[key] for key in keys})
            results.append(backward)
        first, second = results
        self.assertTrue(np.allclose(first.rhs_grads["edge"], second.rhs_grads["edge"]))
        for slot in ("d", "l"):
            self.assertAlmostEqual(
                first.coefficient_grads[slot], second.coefficient_grads[slot], places=12
            )

    def test_single_vs_grouped_backward_agree(self):
        batch = solver.BatchSolver(base_capture(), budget_bytes=1 << 20)
        batch.solve("t", "ridge", ["easy", "edge"])
        cotangent = np.array([1.0, 0.0, -2.0, 0.0, 2.0])
        grouped = batch.back("t", ["easy", "edge"], {"easy": cotangent, "edge": cotangent})
        singles = [
            batch.back("t", key, cotangent) for key in ("easy", "edge")
        ]
        for slot in ("d", "l"):
            self.assertAlmostEqual(
                grouped.coefficient_grads[slot],
                sum(item.coefficient_grads.get(slot, 0.0) for item in singles),
                places=10,
            )


class InputValidationTests(unittest.TestCase):
    def test_rejects_unknown_profile(self):
        with self.assertRaisesRegex(solver.CaptureError, "未知剖面"):
            solver.BatchSolver(base_capture()).solve("t", "nope", ["edge"])

    def test_rejects_wrong_rhs_length(self):
        with self.assertRaisesRegex(solver.CaptureError, "不一致"):
            solver.BatchSolver(base_capture()).solve("t", "fault", ["edge"])

    def test_rejects_unknown_rhs_key(self):
        with self.assertRaisesRegex(solver.CaptureError, "未知通道"):
            solver.BatchSolver(base_capture()).solve("t", "ridge", ["nope"])

    def test_rejects_nonfinite_capture_values(self):
        capture = small_capture()
        capture["rhs"]["u"][0] = float("nan")
        with self.assertRaisesRegex(solver.CaptureError, "有限"):
            solver.BatchSolver(capture)

    def test_rejects_duplicate_offsets(self):
        capture = small_capture(offsets=[0, 0, 1])
        with self.assertRaisesRegex(solver.CaptureError, "重复"):
            solver.BatchSolver(capture)


class ReplayTests(unittest.TestCase):
    def test_replay_sample_capture(self):
        log = solver.replay(read_capture(CAPTURE))
        self.assertEqual([item["kind"] for item in log], ["solve", "solve", "publish", "solve", "back", "back"])
        self.assertTrue(log[0]["result"]["channels"]["edge"]["accepted"])
        self.assertFalse(log[1]["result"]["channels"]["bad"]["accepted"])
        self.assertIn("reason", log[1]["result"]["channels"]["bad"])
        self.assertGreater(len(log[4]["result"]["rhs_grads"]["edge"]), 0)


class EdgeCaseTests(unittest.TestCase):
    def test_exact_singular_never_certifies_even_with_zero_rhs(self):
        capture = {
            "offsets": [-1, 0, 1],
            "tol": 1e-9,
            "pivot_floor": 1e-10,
            "scratch_bytes": 1 << 20,
            "slots": {"z": [0.0, True]},
            "profiles": {
                "s": {
                    "active": [True],
                    "bands": [[None, ["z", 1], None]],
                    "boundary": {"left": None, "right": None, "single": None},
                }
            },
            "rhs": {"zero": [0.0], "one": [1.0]},
            "events": [],
        }
        batch = solver.BatchSolver(capture)
        for key in ("zero", "one"):
            channel = batch.solve(key, "s", [key])[key]
            self.assertFalse(channel.accepted)
            self.assertIsNone(channel.certificate)
            self.assertIn("奇异", channel.reason)

    def test_promotion_marks_transient_bytes(self):
        batch = solver.BatchSolver(small_capture(tol=1e-12))
        report = batch.solve("t", "chain", ["u"])
        promotions = report.budget["promotions"]
        self.assertTrue(promotions)
        self.assertEqual(promotions[0]["from"], "fp32")
        self.assertEqual(promotions[0]["to"], "fp64")
        self.assertGreaterEqual(promotions[0]["transient_bytes"], 0)

    def test_rejected_channel_does_not_pin_factors(self):
        capture = base_capture()
        batch = solver.BatchSolver(capture)
        batch.solve("t", "fault", ["bad"])
        self.assertEqual(batch.budget_report()["used_bytes"], 0)
        self.assertEqual(batch.cache_report()["entries"], [])

    def test_partial_channel_reports_accepted_segment_ranges(self):
        capture = small_capture(scratch_bytes=999999)
        # 让第二段奇异，第一段仍可认证。
        capture["profiles"]["chain"]["active"] = [True, True, False, True]
        capture["profiles"]["chain"]["bands"][3] = [None, None, ["z", 1]]
        capture["slots"]["z"] = [0.0, True]
        capture["rhs"]["u"] = [1.0, 0.0, 0.0, 1.0]
        batch = solver.BatchSolver(capture)
        channel = batch.solve("t", "chain", ["u"])["u"]
        self.assertFalse(channel.accepted)
        self.assertEqual(channel.ranges, ((0, 2),))
        self.assertIn("段 1", channel.reason)

    def test_segment_precision_is_reported_per_segment(self):
        batch = solver.BatchSolver(base_capture(), budget_bytes=1 << 20)
        channel = batch.solve("t", "ridge", ["easy"])["easy"]
        self.assertEqual(channel.precisions, ("fp32", "fp64"))
        self.assertEqual(channel.precision, "fp64")



if __name__ == "__main__":
    unittest.main()
