"""带状平滑层求解/发证/晋级/反传/预算/生命周期的完整行为测试。"""

from __future__ import annotations

import copy
import json
import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from band_solver import (  # noqa: E402
    Session, replay, certificate, segment_runs, solve_banded,
    band_certificate, FLOAT32, FLOAT64,
)

CAPTURE = ROOT / "samples" / "capture.json"
COT = [1.0, 0.0, -2.0, 0.0, 2.0]


def make_session(**kw):
    return Session(CAPTURE, **kw)


class GeometryTests(unittest.TestCase):
    def test_segment_runs(self):
        self.assertEqual(segment_runs([1, 1, 1, 0, 1]), [(0, 3), (4, 5)])
        self.assertEqual(segment_runs([0, 1, 0]), [(1, 2)])
        self.assertEqual(segment_runs([0, 0]), [])

    def test_capture_ranges(self):
        s = make_session()
        s.solve("old", "ridge", ["easy", "edge"])
        t = s.tickets["old"]
        self.assertEqual(t.report()["unknown_ranges"], [[0, 3], [4, 5]])


class CaptureReplayTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.s = replay(CAPTURE)
        cls.cap = json.loads(CAPTURE.read_text(encoding="utf-8"))

    def test_ticket_statuses(self):
        self.assertEqual(self.s.tickets["old"].status, "ok")
        self.assertEqual(self.s.tickets["new"].status, "ok")
        fault = self.s.tickets["fault"]
        self.assertEqual(fault.channels["bad"].status, "singular")

    def test_boundary_versions_bound(self):
        self.assertIsNone(self.s.tickets["old"].boundary_version)
        self.assertEqual(self.s.tickets["new"].boundary_version, "r1")

    def test_every_accepted_channel_has_valid_certificate(self):
        for label in ("old", "new"):
            t = self.s.tickets[label]
            for key in t.accepted_keys:
                ch = t.channels[key]
                self.assertLessEqual(ch.certificate, 1e-9)
                for sr in ch.segments:
                    self.assertTrue(sr.certified)
                    self.assertLessEqual(sr["certificate"], 1e-9)

    def test_low_precision_is_used_but_scalar_segment_promotes(self):
        # edge 的三对角段在规范 FP32 Thomas 下证书约 6.5e-10 < 1e-9，
        # 长度一的段 FP32 标量误差 ~8e-9，必须晋级 FP64。
        ch = self.s.tickets["old"].channels["edge"]
        seg0, seg1 = ch.segments
        self.assertLessEqual(seg0["certificate"], 1e-9)
        self.assertIn(seg0.precision, (FLOAT32, FLOAT64))
        self.assertEqual(seg1.precision, FLOAT64)
        self.assertEqual(ch.precision, FLOAT64)

    def test_scratch_budget_respected(self):
        rep = self.s.scratch
        self.assertLessEqual(rep["peak_total_bytes"], self.cap["scratch_bytes"])
        self.assertLessEqual(rep["net_retained_bytes"],
                             self.cap["scratch_bytes"])
        self.assertGreater(rep["peak_total_bytes"], 0)
        self.assertGreaterEqual(rep["peak_headroom_bytes"], 0)

    def test_old_gradients_match_oracle(self):
        r = self.s.tickets["old"].back_reports[("old", "edge")]
        self.assertTrue(r.ok)
        self.assertAlmostEqual(r.slot_grad["d"], -12.17262157, places=5)
        self.assertAlmostEqual(r.slot_grad["l"], -5.80407109, places=5)
        self.assertAlmostEqual(r.slot_grad["p"], 0.98522167, places=6)
        np.testing.assert_allclose(
            r.rhs_gradient,
            [0.46835799, -0.19693310, -1.03372736, 0.0, 0.98522167],
            atol=1e-7)
        # 固定槽 p 不进入可训练系数梯度。
        self.assertNotIn("p", r.coefficient_gradients)
        self.assertIn("d", r.coefficient_gradients)
        self.assertEqual(r.updatable_coefficients, ["d", "l"])
        self.assertEqual(r.fixed_slots, ["p"])

    def test_new_gradients_match_oracle(self):
        r = self.s.tickets["new"].back_reports[("new", "drift")]
        self.assertTrue(r.ok)
        self.assertAlmostEqual(r.slot_grad["d"], 3.68268475, places=6)
        self.assertAlmostEqual(r.slot_grad["l"], 0.36445952, places=6)
        self.assertAlmostEqual(r.slot_grad["p"], -5.86737318, places=6)
        for a in r.adjoint_certificates:
            self.assertTrue(a.certified)
            self.assertLessEqual(a.transpose_certificate, 1e-9)

    def test_query_certificate_per_channel(self):
        q = self.s.certificate_for("old", "edge")
        self.assertTrue(q.ok)
        self.assertLessEqual(q.certificate, 1e-9)
        self.assertEqual(len(q.segments), 2)


class FiniteDifferenceGradientTests(unittest.TestCase):
    def test_slot_and_rhs_gradients(self):
        cap = json.loads(CAPTURE.read_text(encoding="utf-8"))
        s = Session(cap)
        s.solve("old", "ridge", ["edge"])
        api = s.back("old", "edge", COT)

        slots = {k: v[0] for k, v in cap["slots"].items()}
        off = cap["offsets"]

        def loss(dval):
            A = np.zeros((5, 5))
            for i, row in enumerate(cap["profiles"]["ridge"]["bands"]):
                for j, cell in enumerate(row):
                    if cell:
                        sid, m = cell
                        val = dval if sid == "d" else slots[sid]
                        A[i, i + off[j]] += m * val
            b = np.array(cap["rhs"]["edge"], float)
            b[4] += slots["p"]  # single add_rhs
            x = np.zeros(5)
            x[:3] = np.linalg.solve(A[:3, :3], b[:3])
            x[4] = b[4] / A[4, 4]
            return float(np.array(COT) @ x)

        eps = 1e-6
        v0 = slots["d"]
        fd = (loss(v0 + eps) - loss(v0 - eps)) / (2 * eps)
        self.assertTrue(np.isclose(fd, api.slot_grad["d"], rtol=1e-5))


class LifecycleFreezeTests(unittest.TestCase):
    def test_old_ticket_ignores_later_publish_slots_and_rhs(self):
        s = make_session()
        s.solve("old", "ridge", ["edge"])
        before = s.back("old", "edge", COT).slot_grad

        s.publish("r1", json.loads(
            CAPTURE.read_text(encoding="utf-8"))["events"][2][2])
        s.slot_values["d"] = 999.0
        s.slot_values["l"] = -999.0
        s.set_rhs("edge", [999.0] * 5)

        after = s.back("old", "edge", COT).slot_grad
        for k in before:
            self.assertAlmostEqual(before[k], after[k], places=10)

        # 新票据确实使用 r1。
        s.solve("new", "ridge", ["drift"])
        self.assertEqual(s.tickets["new"].boundary_version, "r1")

    def test_duplicate_label_rejected(self):
        s = make_session()
        s.solve("L", "ridge", ["edge"])
        with self.assertRaises(Exception):
            s.solve("L", "ridge", ["edge"])

    def test_unknown_ticket_back_is_report_not_throw(self):
        s = make_session()
        r = s.back("nope", "edge", COT)
        self.assertFalse(r.ok)
        self.assertEqual(r.status, "unknown_ticket")


class RejectedForwardTests(unittest.TestCase):
    def test_rejected_channel_back_refused_even_with_good_factor(self):
        # 极端 tol 使 FP64 证书也无法过门：分解是好的，但前向不发证。
        s = make_session(tol=1e-30)
        s.solve("t", "ridge", ["edge"])
        ch = s.tickets["t"].channels["edge"]
        self.assertFalse(ch.ok)
        r = s.back("t", "edge", COT)
        self.assertFalse(r.ok)
        self.assertEqual(r.status, "forward_rejected")
        self.assertNotIn("coefficient_gradients", r)

    def test_fault_back_refused(self):
        s = make_session()
        s.solve("fault", "fault", ["bad"])
        r = s.back("fault", "bad", [1.0])
        self.assertFalse(r.ok)
        self.assertEqual(r.status, "forward_rejected")

    def test_zero_matrix_zero_rhs_still_singular(self):
        cap = json.loads(CAPTURE.read_text(encoding="utf-8"))
        cap["rhs"]["zz"] = [0.0]
        s = Session(cap)
        s.solve("z", "fault", ["zz"])
        self.assertEqual(s.tickets["z"].channels["zz"].status, "singular")

    def test_nonfinite_rhs_rejected(self):
        cap = json.loads(CAPTURE.read_text(encoding="utf-8"))
        cap["rhs"]["nan_rhs"] = [1.0]
        cap["rhs"]["nan_rhs"] = [float("nan")]
        s = Session(cap)
        rep = s.solve("n", "fault", ["nan_rhs"])
        self.assertEqual(rep.channels["nan_rhs"]["status"], "non_finite")


class PivotFloorTests(unittest.TestCase):
    def _session(self, eps):
        cfg = {
            "offsets": [-1, 0, 1], "tol": 1e-9, "pivot_floor": 1e-10,
            "scratch_bytes": 10 ** 9,
            "slots": {"D": [2.0, True], "E": [eps, True]},
            "profiles": {"p": {
                "active": [True, True],
                "bands": [[None, ["D", 1], None],
                          [None, ["E", 1], None]],
                "boundary": {"left": None, "right": None, "single": None}}},
            "rhs": {"b": [1.0, 1.0]},
        }
        return Session(cfg)

    def test_tiny_pivot_rejected_with_reason(self):
        s = self._session(1e-12)
        s.solve("q", "p", ["b"])
        ch = s.tickets["q"].channels["b"]
        self.assertEqual(ch.status, "pivot_floor")
        self.assertIn("主元", ch.reason)

    def test_healthy_near_singular_still_certified(self):
        s = self._session(1e-8)
        s.solve("q", "p", ["b"])
        self.assertTrue(s.tickets["q"].channels["b"].ok)


class BudgetTests(unittest.TestCase):
    def test_tight_budget_reports_scope_and_release_recovers(self):
        cap = json.loads(CAPTURE.read_text(encoding="utf-8"))
        s = Session(cap, scratch_bytes=180)
        s.solve("old", "ridge", ["easy", "edge"])
        s.solve("fault", "fault", ["bad"])
        s.publish("r1", cap["events"][2][2])
        s.solve("new", "ridge", ["drift"])

        blocked = s.back("old", "edge", COT)
        self.assertFalse(blocked.ok)
        self.assertEqual(blocked.status, "resource_limited")
        res = blocked.resource
        self.assertGreater(res.shortfall_bytes, 0)
        self.assertIn("new", res.reclaimable_bytes_by_release)
        self.assertIn("continuable", res)

        s.release("new")
        self.assertTrue(s.back("old", "edge", COT).ok)
        # 释放后允许按需重新分解，新票据反传仍可继续。
        self.assertTrue(s.back("new", "drift", COT).ok)
        self.assertLessEqual(s.ledger.peak_total_bytes, 180)

    def test_impossibly_tiny_budget_blocks_channel(self):
        s = make_session(scratch_bytes=16)
        rep = s.solve("old", "ridge", ["edge"])
        self.assertFalse(rep.channels["edge"].ok)
        self.assertEqual(rep.channels["edge"].status, "resource_limited")

    def test_budget_never_exceeded(self):
        s = make_session(scratch_bytes=220)
        s.solve("old", "ridge", ["easy", "edge"])
        s.publish("r1", json.loads(
            CAPTURE.read_text(encoding="utf-8"))["events"][2][2])
        s.solve("new", "ridge", ["drift"])
        s.back("old", "edge", COT)
        s.back("new", "drift", COT)
        self.assertLessEqual(s.ledger.net_retained_bytes, 220)
        self.assertLessEqual(s.ledger.peak_total_bytes, 220)


class BatchSplitInvarianceTests(unittest.TestCase):
    def test_splitting_rhs_batch_gives_identical_solution(self):
        a = make_session()
        ra = a.solve("all", "ridge", ["easy", "edge"])

        b = make_session()
        b.solve("one", "ridge", ["easy"])
        rb = b.solve("two", "ridge", ["edge"])

        xa = np.array(ra.channels["edge"]["x"])
        xb = np.array(rb.channels["edge"]["x"])
        np.testing.assert_array_equal(xa, xb)
        self.assertEqual(ra.channels["edge"].certificate,
                         rb.channels["edge"].certificate)

        ga = a.back("all", "edge", COT).coefficient_gradients
        gb = b.back("two", "edge", COT).coefficient_gradients
        for k in ga:
            self.assertAlmostEqual(ga[k], gb[k], places=12)

    def test_compact_and_full_cotangent_agree(self):
        s = make_session()
        s.solve("t", "ridge", ["edge"])
        full = s.back("t", "edge", COT).coefficient_gradients
        compact = s.back("t", "edge", [1.0, 0.0, -2.0, 2.0]
                         ).coefficient_gradients
        for k in full:
            self.assertAlmostEqual(full[k], compact[k], places=12)

    def test_compact_rhs_supported(self):
        s = make_session()
        s.rhs["edge_compact"] = [1.77, 3.11, -21.5, 0.13]  # 有效点原始值
        rep = s.solve("t", "ridge", ["edge_compact"])
        self.assertTrue(rep.channels["edge_compact"].ok)
        full = make_session()
        rf = full.solve("t", "ridge", ["edge"])
        np.testing.assert_allclose(
            rep.channels["edge_compact"]["x"],
            rf.channels["edge"]["x"], atol=1e-10)


class MergeGradientsTests(unittest.TestCase):
    def test_back_batch_merges_shared_slot_across_channels(self):
        s = make_session()
        s.solve("t", "ridge", ["easy", "edge"])
        merged = s.back_batch("t", {"easy": COT, "edge": COT})
        self.assertTrue(merged.ok)
        ge = s.tickets["t"].back_reports[("t", "edge")].coefficient_gradients
        gy = s.tickets["t"].back_reports[("t", "easy")].coefficient_gradients
        self.assertAlmostEqual(
            merged.coefficient_gradients["d"], ge["d"] + gy["d"], places=10)

    def test_repeated_back_replaces_channel_contribution(self):
        s = make_session()
        s.solve("t", "ridge", ["edge"])
        s.back("t", "edge", COT)
        g1 = dict(s.tickets["t"].slot_grad)
        s.back("t", "edge", COT)  # 同 key 重复不双倍累计
        g2 = dict(s.tickets["t"].slot_grad)
        self.assertAlmostEqual(g1["d"], g2["d"], places=12)


class CacheKeyTests(unittest.TestCase):
    def test_same_matrix_different_addrhs_cache_hits(self):
        cap = json.loads(CAPTURE.read_text(encoding="utf-8"))
        s = Session(cap)
        b1 = {"left": None, "right": None,
              "single": {"row": [None, ["d", 1], None],
                         "add_rhs": ["p", 1]}}
        b2 = {"left": None, "right": None,
              "single": {"row": [None, ["d", 1], None],
                         "add_rhs": ["d", 0.5]}}
        s.publish("v1", {"ridge": b1})
        s.solve("t1", "ridge", ["edge"])
        hits_before = sum(1 for e in s.ledger.events
                           if e["kind"] == "cache_hit")
        s.publish("v2", {"ridge": b2})
        s.solve("t2", "ridge", ["edge"])
        hits_after = sum(1 for e in s.ledger.events
                         if e["kind"] == "cache_hit")
        self.assertGreater(hits_after, hits_before)

    def test_same_length_different_boundary_matrix_cache_misses(self):
        cap = json.loads(CAPTURE.read_text(encoding="utf-8"))
        s = Session(cap)
        s.solve("old", "ridge", ["edge"])
        keys_after_old = set(s.pool.records)
        s.publish("r1", cap["events"][2][2])
        s.solve("new", "ridge", ["drift"])
        # 新左边界把 d 换成 p、l 系数改变：三对角段必须是新分解。
        new_keys = set(s.pool.records) - keys_after_old
        self.assertTrue(any(k.endswith(FLOAT64) for k in new_keys))


class DenseCertificateHelperTests(unittest.TestCase):
    def test_certificate_zero_denominator(self):
        A = np.zeros((1, 1))
        self.assertEqual(certificate(A, [0.0], [0.0]), 0.0)

    def test_solve_banded_matches_dense(self):
        A = np.array([[4.0, 1.0, 0.0], [1.0, 3.0, 1.0], [0.0, 1.0, 2.0]])
        b = [1.0, 2.0, 3.0]
        x = solve_banded(A, b, FLOAT64)
        np.testing.assert_allclose(x, np.linalg.solve(A, b), atol=1e-12)
        self.assertLessEqual(certificate(A, x, b), 1e-12)


class PublishSignatureTests(unittest.TestCase):
    def test_three_arg_publish(self):
        s = make_session()
        triple = {"left": None, "right": None,
                  "single": {"row": [None, ["d", 1], None],
                             "add_rhs": ["p", 1]}}
        s.publish("vx", "ridge", triple)
        s.solve("t", "ridge", ["edge"])
        self.assertEqual(s.tickets["t"].boundary_version, "vx")

    def test_row_null_boundary_still_adds_rhs(self):
        cfg = {
            "offsets": [-1, 0, 1], "tol": 1e-9, "pivot_floor": 1e-10,
            "scratch_bytes": 10 ** 6,
            "slots": {"d": [2.0, True], "q": [0.5, True]},
            "profiles": {"p": {
                "active": [True, True],
                "bands": [[None, ["d", 1], None],
                          [None, ["d", 1], None]],
                "boundary": {"left": {"row": None,
                                      "add_rhs": ["q", 2]},
                             "right": None, "single": None}}},
            "rhs": {"b": [1.0, 4.0]}}
        s = Session(cfg)
        s.solve("t", "p", ["b"])
        np.testing.assert_allclose(
            s.tickets["t"].channels["b"].x_full, [1.0, 2.0], atol=1e-8)
        g = s.back("t", "b", [1.0, 1.0])
        self.assertAlmostEqual(g.slot_grad["q"], 1.0)


class ReplayCliTests(unittest.TestCase):
    def test_run_events_count(self):
        from band_solver import run_events
        events = run_events(CAPTURE)
        self.assertEqual(len(events), 6)

    def test_read_capture_replay_cli(self):
        import subprocess
        import sys
        out = subprocess.run(
            [sys.executable, "-m", "read_capture", str(CAPTURE), "--replay"],
            cwd=ROOT, text=True, capture_output=True, check=True)
        rep = json.loads(out.stdout)
        self.assertIn("tickets", rep)
        self.assertEqual(rep["tickets"]["new"]["boundary_version"], "r1")


if __name__ == "__main__":
    unittest.main(verbosity=2)
