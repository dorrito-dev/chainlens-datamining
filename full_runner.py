#!/usr/bin/env python3
"""ChainLens interactive runner.

Usage:
    uv run python full_runner.py

Menu: fetch via Etherscan, rebuild from cache, sample run, browse datasets,
start/stop the dashboard, run checks.
"""

from __future__ import annotations

import json
import os
import shutil
import socket
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent
OUTPUT = ROOT / "data" / "output"
DB = OUTPUT / "chainlens.duckdb"
ENV_FILE = ROOT / ".env"
DASH_LOG = Path("/tmp/opencode/chainlens_dashboard.log")
DEFAULT_PORT = 8501

UV = shutil.which("uv")


# ── helpers ───────────────────────────────────────────────────────────────────


def base(module: str) -> list[str]:
    """Command prefix: prefer uv, fall back to the current interpreter."""
    if UV:
        return [UV, "run", module]
    return [sys.executable, "-m", module]


def run(args: list[str]) -> int:
    """Run in the foreground with inherited stdio."""
    print(f"\n$ {' '.join(args)}\n", flush=True)
    try:
        return subprocess.call(args, cwd=ROOT)
    except KeyboardInterrupt:
        print("\n[interrupted]")
        return 130
    except FileNotFoundError as exc:
        print(f"[error] {exc}")
        return 127


def api_key() -> str | None:
    if (key := os.environ.get("ETHERSCAN_API_KEY", "").strip()):
        return key
    if ENV_FILE.exists():
        for line in ENV_FILE.read_text().splitlines():
            if line.strip().startswith("ETHERSCAN_API_KEY="):
                key = line.split("=", 1)[1].strip().strip("'\"")
                if key and "your_etherscan" not in key:
                    return key
    return None


def port_open(port: int) -> bool:
    with socket.socket() as sock:
        sock.settimeout(0.3)
        return sock.connect_ex(("127.0.0.1", port)) == 0


def free_port(start: int = DEFAULT_PORT) -> int:
    for port in range(start, start + 20):
        if not port_open(port):
            return port
    return start


def connect():
    if not DB.exists():
        print(f"[missing] {DB}\nRun a pipeline first (menu 1-3).")
        return None
    try:
        import duckdb
    except ImportError:
        print("[error] duckdb is not installed. Run: uv sync")
        return None
    return duckdb.connect(str(DB), read_only=True)


def tables(con) -> list[str]:
    rows = con.execute("select table_name from duckdb_tables() order by 1").fetchall()
    return [row[0] for row in rows]


def pause() -> None:
    try:
        input("\n[enter] to continue ")
    except EOFError:
        pass


# ── pipeline ──────────────────────────────────────────────────────────────────


def pipeline(mode: str, fetch: bool) -> None:
    if mode == "real" and fetch and not api_key():
        print("[error] ETHERSCAN_API_KEY missing. Copy .env.example to .env and add a key.")
        pause()
        return
    args = [*base("python"), "run.py", "--mode", mode]
    if mode == "real" and not fetch:
        args.append("--skip-fetch")
    code = run(args)
    print(f"\n[exit {code}] artifacts in {OUTPUT}")
    evaluation = OUTPUT / "evaluation.md"
    if evaluation.exists():
        print(f"\n{evaluation.read_text()}")
    pause()


# ── dashboard ─────────────────────────────────────────────────────────────────


def dashboard_start() -> None:
    if port_open(DEFAULT_PORT):
        print(f"[running] something is already on port {DEFAULT_PORT}: http://localhost:{DEFAULT_PORT}")
        pause()
        return
    port = free_port()
    DASH_LOG.parent.mkdir(parents=True, exist_ok=True)
    cmd = [*base("streamlit"), "run", "dashboard.py",
           "--server.headless", "true", "--server.port", str(port),
           "--browser.gatherUsageStats", "false"]
    print(f"\n$ {' '.join(cmd)}\nlog -> {DASH_LOG}", flush=True)
    log = DASH_LOG.open("a")
    subprocess.Popen(cmd, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT,
                     stdin=subprocess.DEVNULL, start_new_session=True)
    url = f"http://localhost:{port}"
    for _ in range(15):
        time.sleep(1)
        try:
            with urllib.request.urlopen(f"{url}/_stcore/health", timeout=2) as resp:
                if resp.status == 200:
                    print(f"[ready] {url}")
                    break
        except Exception:
            continue
    else:
        print(f"[warning] no health response yet; check {DASH_LOG}")
    print("Stop it with menu option 6.")
    pause()


def dashboard_stop() -> None:
    result = subprocess.run(["pkill", "-f", "streamlit run dashboard.py"],
                            capture_output=True, text=True)
    time.sleep(1)
    with socket.socket() as sock:
        sock.settimeout(0.3)
        still_up = sock.connect_ex(("127.0.0.1", DEFAULT_PORT)) == 0
    if still_up:
        print(f"[warning] port {DEFAULT_PORT} still busy")
    elif result.returncode == 0:
        print("[stopped] dashboard terminated")
    else:
        print("[info] no dashboard was running")
    pause()


# ── datasets ──────────────────────────────────────────────────────────────────


def show_tables(con) -> None:
    rows = []
    for name in tables(con):
        count = con.execute(f'select count(*) from "{name}"').fetchone()[0]
        rows.append((name, count))
    width = max(len(n) for n, _ in rows)
    print()
    for name, count in rows:
        print(f"  {name:<{width}}  {count:>10,}")


def preview(con) -> None:
    names = tables(con)
    print()
    for i, name in enumerate(names, 1):
        print(f"  {i}. {name}")
    raw = input("Table number (0 = back): ").strip()
    if not raw.isdigit() or not 1 <= int(raw) <= len(names):
        return
    name = names[int(raw) - 1]
    con.sql(f'select * from "{name}" limit 20').show()


def top_scores(con) -> None:
    con.sql(
        "select address, score, tier, top_driver, round(raw_score, 2) as raw "
        "from risk_scores order by score desc, address limit 20"
    ).show()


def wallet_lookup(con) -> None:
    address = input("Address: ").strip().lower()
    if not address:
        return
    row = con.execute(
        "select score, tier, top_driver, raw_score from risk_scores where address = ?",
        [address],
    ).fetchone()
    if not row:
        print(f"[not found] {address}")
        return
    score, tier, driver, raw = row
    print(f"\n{address}\n  score={score}  tier={tier}  driver={driver}  raw={raw:.2f}")

    trace = con.execute(
        "select trace_json from score_trace where address = ?", [address]
    ).fetchone()
    if trace:
        rules = json.loads(trace[0]).get("rules", {})
        print("\n  rule  hit    weight  strength  contribution")
        for rid, rule in rules.items():
            print(
                f"  {rid:<5} {str(rule.get('hit')):<6} "
                f"{rule.get('weight', 0):>6}  {rule.get('strength', 0):>8.3f}  "
                f"{rule.get('contribution', 0):>12.3f}"
            )

    flags = con.execute(
        "select signal_name, evidence from wallet_flags where address = ? order by signal_name",
        [address],
    ).fetchall()
    if flags:
        print("\n  signals")
        for name, evidence in flags:
            text = (evidence or "")[:120]
            print(f"    {name}: {text}{'…' if len(evidence or '') > 120 else ''}")


def show_inputs(con) -> None:
    print("\nlabels")
    con.sql("select source, label, count(*) n from labels group by 1,2 order by 3 desc").show()
    print("\nknown addresses (excluded from R4)")
    con.sql("select address, category from known_addresses order by category").show()
    sanctions = ROOT / "config" / "sanctions.json"
    if sanctions.exists():
        data = json.loads(sanctions.read_text())
        count = len(data if isinstance(data, list) else data.get("addresses", []))
        print(f"\nsanctions seed: {count} addresses ({sanctions.relative_to(ROOT)})")


def cache_stats() -> None:
    folder = ROOT / "data" / "cache" / "addresses"
    metas = list(folder.glob("*.meta.json"))
    truncated = 0
    for meta in metas:
        try:
            if json.loads(meta.read_text()).get("truncated"):
                truncated += 1
        except Exception:
            pass
    size = sum(f.stat().st_size for f in folder.glob("*")) / 1e6 if folder.exists() else 0
    print(f"\ncache: {len(metas):,} addresses, {size:,.0f} MB, {truncated:,} truncated")


def output_files() -> None:
    if not OUTPUT.exists():
        print(f"[missing] {OUTPUT}")
        return
    print()
    for path in sorted(OUTPUT.iterdir()):
        if path.is_file():
            print(f"  {path.stat().st_size / 1e6:9.1f} MB  {path.name}")


def datasets_menu() -> None:
    while True:
        print("\nDatasets\n"
              "  1. List tables and row counts\n"
              "  2. Preview a table\n"
              "  3. Top scored wallets\n"
              "  4. Look up a wallet (score, rules, signals)\n"
              "  5. Inputs (labels, sanctions, known addresses)\n"
              "  6. Etherscan cache stats\n"
              "  7. Output files\n"
              "  0. Back")
        choice = ask()
        if choice == "0":
            return
        con = connect()
        if con is None:
            pause()
            continue
        try:
            if choice == "1":
                show_tables(con)
            elif choice == "2":
                preview(con)
            elif choice == "3":
                top_scores(con)
            elif choice == "4":
                wallet_lookup(con)
            elif choice == "5":
                show_inputs(con)
            elif choice == "6":
                cache_stats()
            elif choice == "7":
                output_files()
            else:
                continue
        finally:
            con.close()
        pause()


# ── checks ────────────────────────────────────────────────────────────────────


def checks() -> None:
    lint = run([*base("ruff"), "check", "patterns.py", "scoring.py", "run.py",
                "dashboard.py", "full_runner.py", "tests/"])
    tests = run([*base("pytest"), "tests/", "-q"])
    print(f"\nruff exit {lint}, pytest exit {tests}")
    pause()


# ── main ──────────────────────────────────────────────────────────────────────


def ask() -> str:
    try:
        return input("\nSelect: ").strip()
    except (EOFError, KeyboardInterrupt):
        print()
        raise SystemExit(0) from None


def main() -> int:
    print("ChainLens runner")
    while True:
        print("\n  1. Full real run (fetch from Etherscan)\n"
              "  2. Rebuild from cache (no API calls)\n"
              "  3. Sample run (fixtures, no API key)\n"
              "  4. Datasets\n"
              "  5. Start dashboard (cached results)\n"
              "  6. Stop dashboard\n"
              "  7. Quality checks (ruff + pytest)\n"
              "  0. Exit")
        choice = ask()
        if choice == "0":
            print("bye")
            return 0
        if choice == "1":
            pipeline("real", fetch=True)
        elif choice == "2":
            pipeline("real", fetch=False)
        elif choice == "3":
            pipeline("sample", fetch=False)
        elif choice == "4":
            datasets_menu()
        elif choice == "5":
            dashboard_start()
        elif choice == "6":
            dashboard_stop()
        elif choice == "7":
            checks()


if __name__ == "__main__":
    raise SystemExit(main())
