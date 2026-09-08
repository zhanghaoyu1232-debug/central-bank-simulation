"""Check source provenance and the specific default-resolution audit cases.

Uses the original cascade methods with a controlled contract-book fixture
and the original shared estate resolver; it does not run Monte Carlo paths.
Only NumPy is required. --artifacts-dir additionally checks raw result files.
"""
from __future__ import annotations

import argparse
import ast
import csv
import hashlib
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "reproduction" / "simulation_source"
sys.path.insert(0, str(SOURCE))
from bank_econ_shared import assets_ex_equity
from interbank_resolution import settle_absorbing_defaults_batch


class Book:
    def __init__(self, contracts):
        self.contracts = list(contracts)

    def remove_contract(self, c):
        self.contracts = [x for x in self.contracts if x is not c]

    def add_contract(self, c):
        self.contracts.append(c)


def load_cascade_methods(filename):
    tree = ast.parse((SOURCE / filename).read_text(encoding="utf-8"))
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef)
               and n.name == "BankNetworkSimulator")
    names = {"_cascade_negative_equity_defaults", "_classify_equity_defaults"}
    nodes = [n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name in names]
    namespace = {}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), filename, "exec"), namespace)
    return namespace


def cascade_fixture(filename):
    methods = load_cascade_methods(filename)

    class Fixture:
        num_banks = 4
        bank_types = ["central", "commercial", "commercial", "commercial"]
        _cascade_negative_equity_defaults = methods["_cascade_negative_equity_defaults"]
        _classify_equity_defaults = methods["_classify_equity_defaults"]

        def __init__(self):
            self.banks = []
            for i, cash, external_debt in [(0, 1000, 0), (1, 10, 80), (2, 0, 200), (3, 500, 100)]:
                self.banks.append({"name": "CentralBank" if i == 0 else f"Bank{i}",
                    "type": self.bank_types[i], "is_active": True,
                    "liquid_assets": float(cash), "current_liabilities": float(external_debt),
                    "interbank_assets": 0.0, "interbank_liabilities": 0.0,
                    "investment": {"projects": {"amount": 0.0}}})
            self.book = Book([SimpleNamespace(contract_id="AB120", lender_idx=1,
                borrower_idx=2, principal=120.0, remaining_principal=120.0,
                arrears_due=0.0, maturity_step=5, schedule_type="single_payment")])
            self.refreshes = 0
            self.writeoff = 0.0

        def _bank_equity(self, b):
            return assets_ex_equity(b) - b["current_liabilities"] - b["interbank_liabilities"]

        def _refresh_regulatory_metrics(self, step):
            self.refreshes += 1
            for i, b in enumerate(self.banks):
                if not b.get("is_active", True):
                    continue
                b["interbank_assets"] = sum(c.remaining_principal for c in self.book.contracts if c.lender_idx == i)
                b["interbank_liabilities"] = sum(c.remaining_principal for c in self.book.contracts if c.borrower_idx == i)
                b["core_capital"] = max(0.0, self._bank_equity(b))

        def _resolve_bank_defaults_batch(self, newly, step, reason):
            summaries = settle_absorbing_defaults_batch(
                banks=self.banks, book=self.book, exposure_matrix=np.zeros((4, 4)),
                failed_indices=newly, step=step, reason=reason, lgd=1.0,
                make_contract=lambda **kw: SimpleNamespace(**kw),
                effective_notional=lambda c: c.remaining_principal + c.arrears_due,
            )
            self.writeoff += sum(s["creditor_writeoff"] for s in summaries)

    sim = Fixture()
    # No initial payment-default list is supplied to the actual method.
    resolved = sim._cascade_negative_equity_defaults(0, reason="audit_fixture")
    assert resolved == [1, 2], resolved
    assert sim.banks[3]["is_active"]
    assert sim.refreshes >= 3
    assert abs(sim.writeoff - 120.0) < 1e-9
    return {"resolved": resolved, "refreshes": sim.refreshes, "creditor_writeoff": sim.writeoff}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifacts-dir", type=Path)
    args = parser.parse_args()
    manifest = json.loads((ROOT / "reproduction/manifest.json").read_text())
    for filename, expected in manifest["source_files"].items():
        assert hashlib.sha256((SOURCE / filename).read_bytes()).hexdigest() == expected, filename
    from output_paths import simulation_code_fingerprint
    assert simulation_code_fingerprint() == manifest["core_source_fingerprint"]
    results = {"source_fingerprint": simulation_code_fingerprint(), "cascade": {}}
    for mechanism in ["centralized", "decentralized"]:
        filename = f"bank_simulation_model_{mechanism}_central_policy.py"
        results["cascade"][mechanism] = cascade_fixture(filename)
    with (ROOT / "backmatter/centralized_central_policy_initial_bank_data.csv").open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    for row in rows:
        c, la, cl, iba, ibl, pa = [float(row[key]) for key in
            ["core_capital", "liquid_assets", "current_liabilities", "interbank_assets", "interbank_liabilities", "project_amount"]]
        equity = la + iba + pa - cl - ibl
        assert abs(equity - c) < 1e-9
        assert abs(equity / (cl + ibl + 1e-9) - float(row["solvency_ratio"])) < 1e-9
        initial_outflow = 0.2 if row["type"] == "central" else 0.4
        assert abs(min(3, la / ((cl + ibl) * initial_outflow)) - float(row["liquidity_coverage_ratio"])) < 1e-9
    results["initial_rows_verified"] = len(rows)
    if args.artifacts_dir:
        for artifact in manifest["artifacts"]:
            raw = (args.artifacts_dir / artifact["path"]).read_bytes()
            assert hashlib.sha256(raw).hexdigest() == artifact["sha256"], artifact["path"]
        results["raw_artifacts_verified"] = len(manifest["artifacts"])
    (ROOT / "reproduction/verification_results.json").write_text(json.dumps(results, indent=2))
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
