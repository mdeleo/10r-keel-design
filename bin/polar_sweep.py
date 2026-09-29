#!/usr/bin/env python3
"""
polar_sweep.py - batch 2D polars for the flap-airfoil set (from airfoil_flap.py).

Runs XFoil (same viscous engine flow5 uses) over every combination of
  airfoil file x Reynolds number x Ncrit x transition mode
and collects everything into one CSV plus a max-L/D summary.

Transition modes
  free   : natural transition (trip at x/c = 1.0 on both surfaces)
  forced : trip at --xtrip (default 0.10) on both surfaces, like a tape trip
  turb   : trip at --xturb (default 0.02), fully turbulent worst case

Example
  python polar_sweep.py airfoils_out --re 75000 100000 150000 200000 300000 \
      --ncrit 4.5 --modes free forced turb --amax 10 --amin -6

Use --dry-run to only write the XFoil command files (no XFoil needed).
Requires: numpy is not needed; XFoil on PATH (or --xfoil /path/to/xfoil).
"""
import argparse
import csv
import glob
import os
import re
import subprocess
import sys


def modes_xtrip(args):
    return {"free": 1.0, "forced": args.xtrip, "turb": args.xturb}


def xfoil_script(dat, polar, re_, ncrit, xtrip, amax, amin, da, itr):
    lines = [
        "PLOP", "G", "",             # graphics off
        f"LOAD {dat}",
        "PANE",
        "OPER",
        f"VISC {re_:g}",
        "MACH 0",
        "VPAR", f"N {ncrit:g}", "XTR", f"{xtrip:g}", f"{xtrip:g}", "",
        f"ITER {itr}",
        "PACC", polar, "",
        f"ASEQ 0 {amax:g} {da:g}",
        "INIT",
        f"ASEQ 0 {amin:g} {-da:g}",
        "PACC", "",
        "", "QUIT",
    ]
    return "\n".join(lines) + "\n"


def parse_polar(path):
    rows, started = [], False
    if not os.path.exists(path):
        return rows
    with open(path) as f:
        for line in f:
            if line.strip().startswith("-----"):
                started = True
                continue
            if started:
                p = line.split()
                if len(p) >= 7:
                    try:
                        rows.append([float(v) for v in p[:7]])
                    except ValueError:
                        pass
    return rows


def flap_angle(fn):
    m = re.search(r"flap([+-]?[\d.]+)", os.path.basename(fn))
    return float(m.group(1)) if m else float("nan")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("airfoil_dir", help="folder of .dat files (output of airfoil_flap.py)")
    ap.add_argument("--re", type=float, nargs="+",
                    default=[75000, 100000, 150000, 200000, 300000])
    ap.add_argument("--ncrit", type=float, nargs="+", default=[4.5])
    ap.add_argument("--modes", nargs="+", default=["free", "forced", "turb"],
                    choices=["free", "forced", "turb"])
    ap.add_argument("--xtrip", type=float, default=0.10)
    ap.add_argument("--xturb", type=float, default=0.02)
    ap.add_argument("--amax", type=float, default=10.0)
    ap.add_argument("--amin", type=float, default=-6.0)
    ap.add_argument("--da", type=float, default=0.5)
    ap.add_argument("--iter", type=int, default=200)
    ap.add_argument("--timeout", type=int, default=90, help="seconds per run")
    ap.add_argument("--xfoil", default="xfoil")
    ap.add_argument("--outdir", default="polars_out")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()

    files = sorted(glob.glob(os.path.join(a.airfoil_dir, "*.dat")))
    if not files:
        sys.exit(f"no .dat files in {a.airfoil_dir}")
    os.makedirs(a.outdir, exist_ok=True)
    trips = modes_xtrip(a)
    out_rows, total = [], len(files) * len(a.re) * len(a.ncrit) * len(a.modes)
    n = 0

    for dat in files:
        fa = flap_angle(dat)
        stem = os.path.splitext(os.path.basename(dat))[0]
        for re_ in a.re:
            for nc in a.ncrit:
                for mode in a.modes:
                    n += 1
                    tag = f"{stem}_Re{re_/1000:g}k_N{nc:g}_{mode}"
                    polar = os.path.abspath(os.path.join(a.outdir, tag + ".pol"))
                    if os.path.exists(polar):
                        os.remove(polar)
                    script = xfoil_script(os.path.abspath(dat), polar, re_, nc,
                                          trips[mode], a.amax, a.amin, a.da, a.iter)
                    with open(os.path.join(a.outdir, tag + ".in"), "w") as f:
                        f.write(script)
                    if a.dry_run:
                        continue
                    try:
                        subprocess.run([a.xfoil], input=script, text=True,
                                       capture_output=True, timeout=a.timeout)
                    except subprocess.TimeoutExpired:
                        print(f"[{n}/{total}] {tag}: timeout (partial results kept)")
                    except FileNotFoundError:
                        sys.exit(f"XFoil not found: {a.xfoil} (use --xfoil or --dry-run)")
                    rows = parse_polar(polar)
                    print(f"[{n}/{total}] {tag}: {len(rows)} points")
                    for r in rows:
                        out_rows.append([fa, re_, nc, mode] + r)

    if a.dry_run:
        print(f"wrote {total} XFoil command files to {a.outdir}/")
        return

    hdr = ["flap_deg", "Re", "Ncrit", "mode", "alpha", "CL", "CD", "CDp", "CM",
           "Xtr_top", "Xtr_bot"]
    csv_path = os.path.join(a.outdir, "all_polars.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(hdr)
        w.writerows(out_rows)

    # summary: best CL/CD per case
    best = {}
    for r in out_rows:
        fa, re_, nc, mode, alpha, cl, cd = r[:7]
        if cd > 0:
            k = (mode, nc, re_, fa)
            if k not in best or cl / cd > best[k][0]:
                best[k] = (cl / cd, alpha, cl)
    sum_path = os.path.join(a.outdir, "max_LD_summary.csv")
    with open(sum_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["mode", "Ncrit", "Re", "flap_deg", "max_CL/CD", "alpha_at_max", "CL_at_max"])
        for k in sorted(best):
            w.writerow(list(k) + [round(v, 4) for v in best[k]])
    print(f"\n{csv_path}\n{sum_path}")


if __name__ == "__main__":
    main()

