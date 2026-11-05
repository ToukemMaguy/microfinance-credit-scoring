"""
Synthetic data generator for the micro-loan credit scoring project.

Produces two files that mimic core-banking extracts:
  - loans.csv       one row per loan (customer, branch, booking and maturity dates, amount, status)
  - repayments.csv  the repayment ledger: one row per transaction (principal, interest, penalty)

All customers, branches and amounts are FICTITIOUS. The behaviour is designed to reproduce
patterns seen in real micro-lending: repeat borrowers, short single-instalment loans,
behaviour that persists from one loan to the next, branch effects and a temporary external shock.

Usage:  python data/generate_data.py
"""
from __future__ import annotations

import datetime as dt
import math
import random
from pathlib import Path

import pandas as pd

SEED = 42
START = dt.date(2023, 1, 2)          # first possible booking date
CUTOFF = dt.date(2025, 12, 31)       # extraction date: nothing after this day is known
SHOCK = (dt.date(2025, 6, 16), dt.date(2025, 7, 31))   # temporary market disruption (maturities in this window)
N_CUSTOMERS = 3600

# branch code, size weight, risk multiplier
BRANCHES = [("BR01", 1.3, 1.6), ("BR02", 1.2, 1.1), ("BR03", 0.8, 1.3), ("BR04", 1.0, 0.6), ("BR05", 0.9, 1.1),
            ("BR06", 0.9, 0.5), ("BR07", 0.9, 0.45), ("BR08", 0.8, 0.95), ("BR09", 0.7, 0.85), ("BR10", 0.7, 0.9),
            ("BR11", 0.5, 1.0), ("BR12", 0.3, 1.5)]
AMOUNTS = [50_000, 100_000, 150_000, 200_000, 250_000, 300_000, 400_000, 500_000, 600_000, 700_000]
TERMS = [29, 30, 31, 31, 31, 31, 31, 32, 33, 35, 38]


def next_business_day(d: dt.date) -> dt.date:
    while d.weekday() >= 5:
        d += dt.timedelta(days=1)
    return d


def simulate(seed: int = SEED) -> tuple[pd.DataFrame, pd.DataFrame]:
    rng = random.Random(seed)
    loans, ledger, loan_no = [], [], 0
    for c in range(N_CUSTOMERS):
        branch, _, branch_risk = rng.choices(BRANCHES, weights=[b[1] for b in BRANCHES])[0]
        risk = math.exp(rng.gauss(0, 1.1)) * branch_risk           # latent, persistent customer risk
        level = rng.choices(AMOUNTS, weights=[4, 8, 8, 10, 12, 12, 12, 14, 10, 10])[0]
        offset = 0 if rng.random() < 0.3 else int(rng.random() * 1060)
        booking = next_business_day(START + dt.timedelta(days=offset))
        last_delay = 0
        while booking <= CUTOFF:
            loan_no += 1
            loan_id = f"ML{loan_no:06d}"
            term = rng.choice(TERMS)
            maturity = next_business_day(booking + dt.timedelta(days=term))
            amount = float(max(50_000, min(700_000, level + rng.choice([0, 0, 0, 50_000, -50_000]))))

            # probability of paying 7+ days late: customer risk, past behaviour, external shock
            p_late = min(0.6, 0.025 * risk * (2.5 if last_delay >= 7 else 1))
            if SHOCK[0] <= maturity <= SHOCK[1]:
                p_late = min(0.85, p_late * 4)
            u = rng.random()
            if u < p_late:
                delay = 7 + int(rng.expovariate(1 / 12)) + (rng.randint(20, 100) if rng.random() < 0.15 else 0)
            elif u < p_late + 0.28:
                delay = rng.randint(1, 6)
            else:
                delay = -rng.choice([0, 0, 0, 0, 1, 2, 3])
            last_delay = delay
            settlement = maturity + dt.timedelta(days=delay)

            # repayment ledger: principal in one or several payments, interest, penalty if late
            if settlement <= CUTOFF:
                n_parts = 1 if delay <= 2 else rng.choice([1, 2, 2, 3])
                paid_days = sorted({(settlement - dt.timedelta(days=rng.randint(1, max(1, delay)))) for _ in range(n_parts - 1)})
                paid_days = [d for d in paid_days if d > booking] + [settlement]
                shares = [rng.uniform(0.2, 0.5) for _ in paid_days[:-1]]
                remaining = amount
                for d, s in zip(paid_days[:-1], shares):
                    part = round(amount * s, -3)
                    ledger.append((loan_id, d, part, "PRINCIPAL"))
                    remaining -= part
                ledger.append((loan_id, paid_days[-1], remaining, "PRINCIPAL"))
                ledger.append((loan_id, paid_days[0], round(amount * 0.05, -2), "INTEREST"))
                if delay >= 3:
                    ledger.append((loan_id, settlement, round(amount * 0.002 * delay, -2), "PENALTY"))
                status = "LIQUIDATED"
            else:
                if maturity < CUTOFF and rng.random() < 0.4:      # partial payment on an unsettled loan
                    ledger.append((loan_id, min(CUTOFF, maturity + dt.timedelta(days=rng.randint(0, 10))),
                                   round(amount * rng.uniform(0.1, 0.5), -3), "PRINCIPAL"))
                status = "ACTIVE"
            loans.append((loan_id, f"C{c + 1:05d}", branch, booking, maturity, amount, status))

            if settlement > CUTOFF or rng.random() < 0.10:           # customer stops borrowing
                break
            gap = rng.choices([0, rng.randint(1, 3), rng.randint(4, 8), rng.randint(9, 40), rng.randint(60, 150)],
                              weights=[55, 20, 10, 10 * min(3, risk), 5 * min(3, risk)])[0]
            booking = next_business_day(settlement + dt.timedelta(days=gap))

    loans_df = pd.DataFrame(loans, columns=["loan_id", "customer_id", "branch", "booking_date", "maturity_date", "amount", "status"])
    ledger_df = pd.DataFrame(ledger, columns=["loan_id", "txn_date", "amount", "tag"]).sort_values(["txn_date", "loan_id"])
    return loans_df, ledger_df.reset_index(drop=True)


if __name__ == "__main__":
    out = Path(__file__).resolve().parent
    loans_df, ledger_df = simulate()
    loans_df.to_csv(out / "loans.csv", index=False)
    ledger_df.to_csv(out / "repayments.csv", index=False)
    print(f"loans.csv: {len(loans_df):,} loans | repayments.csv: {len(ledger_df):,} transactions")
