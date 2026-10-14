# =================================================================================
# Mold export of a V4 run — main wing only, in millimetres
# -------------------------------------------------------------------------------
# Re-reads `x_best_v4.npy` from a V4 run folder, rebuilds the wing with a dense
# spanwise section count and writes, into `<run_dir>/mold/`:
#   - wing.stl                     full wing surface (mm), for CAM / mold design
#   - sections/sec_XXXmm.dat       real-size section at each span station (mm),
#                                  already twisted and placed (x = chord axis,
#                                  z = thickness axis) — for templates / CNC
#   - sections/sec_XXXmm_norm.dat  same section, chord-normalised (Selig format)
#   - stations.csv                 y, LE x, TE x, chord, twist, z offset, t/c
#
# Usage:
#   python src/helpers/export_mold_V4.py outputs/<case>_<date>_<nn>_v4_<airfoil>
#   [--step-mm 10] [--n-span 80] [--n-chord 150]
# The case and reference airfoil are read from the run folder name.
# =================================================================================

import os
import re
import sys
import argparse
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent   # src/
sys.path.append(str(ROOT))
sys.path.append(str(ROOT / "helpers"))


def main() -> None:
    ap = argparse.ArgumentParser(description="Mold export (wing) of a V4 run.")
    ap.add_argument("run_dir", type=Path)
    ap.add_argument("--step-mm", type=float, default=10.0, help="spanwise station step (mm)")
    ap.add_argument("--n-span", type=int, default=80, help="spanwise sections for the STL")
    ap.add_argument("--n-chord", type=int, default=150, help="points per side per section")
    args = ap.parse_args()

    run = args.run_dir.resolve()
    mt = re.match(r"^(.*)_\d{8}_\d+_v4_(.+)$", run.name)
    if not mt or not (run / "x_best_v4.npy").exists():
        sys.exit(f"✗ {run} is not a V4 run folder (x_best_v4.npy + <case>_<date>_<nn>_v4_<airfoil>)")
    os.environ["FOIL_CASE"], os.environ["FOIL_AIRFOIL"] = mt.group(1), mt.group(2)

    import optAlgoV4 as V
    from export_STL import write_stl_binary
    import aerosandbox as asb

    xv = np.load(run / "x_best_v4.npy")
    x = {n: float(v) for n, v in zip(V.VAR_NAMES, xv[:-1])}

    V.N_WING = args.n_span
    airplane, geo = V.build_airplane(x)
    wing = airplane.wings[0]

    # Dense, real-coordinate airfoils so the STL is smooth chordwise too.
    for xs in wing.xsecs:
        xs.airfoil = xs.airfoil.to_airfoil().repanel(n_points_per_side=args.n_chord)

    out = run / "mold"
    (out / "sections").mkdir(parents=True, exist_ok=True)

    verts, faces = wing.mesh_body(method="tri", chordwise_resolution=args.n_chord)
    write_stl_binary(out / "wing.stl", verts * 1000.0, faces)

    # ── Sections at fixed span stations (interpolated from the dense build) ──
    semi = x["wing_span"] / 2.0
    ys = np.arange(0.0, semi * 1000.0 + 1e-6, args.step_mm) / 1000.0
    if ys[-1] < semi - 1e-4:
        ys = np.append(ys, semi)
    ys[-1] = min(ys[-1], semi * 0.995)            # the r=1 section has a degenerate chord

    yk = np.array([xs.xyz_le[1] for xs in wing.xsecs], dtype=float)
    rows = []
    for y in ys:
        i = int(np.clip(np.searchsorted(yk, y) - 1, 0, len(yk) - 2))
        t = (y - yk[i]) / (yk[i + 1] - yk[i])
        a, b = wing.xsecs[i], wing.xsecs[i + 1]
        chord = (1 - t) * a.chord + t * b.chord
        twist = (1 - t) * a.twist + t * b.twist
        le = (1 - t) * np.array(a.xyz_le, float) + t * np.array(b.xyz_le, float)
        ca = np.array(a.airfoil.coordinates); cb = np.array(b.airfoil.coordinates)
        co = (1 - t) * ca + t * cb                 # same repanel → same point count
        tc = float(asb.Airfoil(coordinates=co).max_thickness())

        name = f"sec_{y*1000:05.1f}mm".replace(".", "p")
        with open(out / "sections" / f"{name}_norm.dat", "w") as f:
            f.write(f"{name} t/c={tc*100:.2f}%\n")
            for px, pz in co:
                f.write(f" {px:.6f} {pz:.6f}\n")
        # Real size: scale, rotate by twist about the LE (nose-up positive), place.
        th = np.radians(twist)
        X = co[:, 0] * chord; Z = co[:, 1] * chord
        Xr = X * np.cos(th) + Z * np.sin(th)
        Zr = -X * np.sin(th) + Z * np.cos(th)
        with open(out / "sections" / f"{name}.dat", "w") as f:
            f.write(f"{name} y={y*1000:.1f}mm chord={chord*1000:.2f}mm twist={twist:+.2f}deg (mm, LE-placed)\n")
            for px, pz in zip(Xr + le[0], Zr + le[2]):
                f.write(f" {px*1000:.3f} {pz*1000:.3f}\n")
        rows.append((y * 1000, le[0] * 1000, (le[0] + chord) * 1000, chord * 1000,
                     twist, le[2] * 1000, tc * 100))

    with open(out / "stations.csv", "w") as f:
        f.write("y_mm,x_LE_mm,x_TE_mm,chord_mm,twist_deg,z_mm,t_over_c_pct\n")
        for r in rows:
            f.write(",".join(f"{v:.3f}" for v in r) + "\n")

    print(f"✓ wing.stl ({len(faces)} tris, mm), {len(rows)} sections, stations.csv → {out}")


if __name__ == "__main__":
    main()
