# =================================================================================
# Export STL d'un run optimisé — wing, stab, fuselage, mast
# -------------------------------------------------------------------------------
# Relit `x_best.npy` d'un dossier de run, ré-instancie la géométrie avec un nombre
# de sections élevé + profils repanelés haute résolution (continuité visuelle/CFD),
# et écrit 4 fichiers STL binaires dans `<run_dir>/stl/`.
#
# Usage:
#   python3 src/export_STL.py outputs/file_name
# =================================================================================

import sys
import struct
import argparse
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.append(str(ROOT))


# ========= STL RESOLUTION PARAMETERS ===========
n_wing_default=150
n_stab_default=100
n_fuse_default=60
n_mast_default=30
chordwise_default=200
tangential_default=64




def write_stl_binary(path: Path, verts: np.ndarray, faces: np.ndarray) -> None:
    """STL binaire — verts (Nv,3) float, faces (Nf,3) indices entiers (triangles)."""
    tris = verts[faces].astype(np.float32)             # (Nf, 3, 3)
    e1   = tris[:, 1] - tris[:, 0]
    e2   = tris[:, 2] - tris[:, 0]
    n    = np.cross(e1, e2)
    norm = np.linalg.norm(n, axis=1, keepdims=True)
    n    = (n / np.where(norm > 1e-12, norm, 1.0)).astype(np.float32)

    with open(path, "wb") as f:
        f.write(b"\0" * 80)                            # header (80 bytes vide)
        f.write(struct.pack("<I", len(faces)))         # nb de triangles
        # Bloc par triangle : 50 bytes — 12 normale, 36 vertices, 2 attrib.
        block = np.zeros(len(faces), dtype=[
            ("normal",  "<3f4"), ("v0", "<3f4"),
            ("v1", "<3f4"),      ("v2", "<3f4"),
            ("attr", "<u2"),
        ])
        block["normal"] = n
        block["v0"]     = tris[:, 0]
        block["v1"]     = tris[:, 1]
        block["v2"]     = tris[:, 2]
        f.write(block.tobytes())


def main() -> None:
    parser = argparse.ArgumentParser(description="Export STL d'un run optimisé.")
    parser.add_argument("run_dir", type=Path, help="dossier de run (contient x_best.npy)")
    parser.add_argument("--n-wing",     type=int, default=n_wing_default, help="sections aile (def: 150)")
    parser.add_argument("--n-stab",     type=int, default=n_stab_default, help="sections stab (def: 100)")
    parser.add_argument("--n-fuse",     type=int, default=n_fuse_default,  help="sections fuselage (def: 60)")
    parser.add_argument("--n-mast",     type=int, default=n_mast_default,  help="sections mât (def: 30)")
    parser.add_argument("--chordwise",  type=int, default=chordwise_default, help="résolution chordwise (def: 200)")
    parser.add_argument("--tangential", type=int, default=tangential_default,  help="résolution autour du fuselage (def: 64)")
    args = parser.parse_args()

    if not (args.run_dir / "x_best.npy").exists():
        print(f"✗ {args.run_dir}/x_best.npy introuvable.")
        sys.exit(1)

    # Import V2 — déclenche la calibration de_da (cache → instantané si déjà fait)
    print("Import V2…")
    import optFixedProfileV2 as V2

    # Override des résolutions pour le maillage final
    V2.N_WING = args.n_wing
    V2.N_STAB = args.n_stab
    V2.N_FUSE = args.n_fuse
    V2.N_MAST = args.n_mast

    # Repanel des profils — pour que la coque soit lisse longitudinalement aussi
    V2.WING_AIRFOIL = V2.WING_AIRFOIL.repanel(n_points_per_side=args.chordwise)
    V2.STAB_AIRFOIL = V2.STAB_AIRFOIL.repanel(n_points_per_side=args.chordwise)

    print(f"Lecture x_best : {args.run_dir.name}")
    x_best = np.load(args.run_dir / "x_best.npy")
    p = V2.decode(x_best)

    print(f"Build airplane @ N_wing={args.n_wing}, N_stab={args.n_stab}, "
          f"N_fuse={args.n_fuse}, N_mast={args.n_mast}, chordwise={args.chordwise}…")
    _, wing, stab, _, mast, fuselage = V2.build_airplane(p)

    out_dir = args.run_dir / "stl"
    out_dir.mkdir(exist_ok=True)

    print(f"\nMaillage & export → {out_dir.relative_to(args.run_dir.parent)}/")
    parts_wings = [("wing", wing), ("stab", stab), ("mast", mast)]
    for name, surface in parts_wings:
        verts, faces = surface.mesh_body(method="tri",
                                         chordwise_resolution=args.chordwise)
        path = out_dir / f"{name}.stl"
        write_stl_binary(path, verts, faces)
        print(f"  ✓ {name:8s}  {len(verts):6d} verts  {len(faces):6d} tris  "
              f"→ {path.stat().st_size / 1024:.0f} KB")

    verts, faces = fuselage.mesh_body(method="tri",
                                      tangential_resolution=args.tangential)
    path = out_dir / "fuselage.stl"
    write_stl_binary(path, verts, faces)
    print(f"  ✓ fuselage {len(verts):6d} verts  {len(faces):6d} tris  "
          f"→ {path.stat().st_size / 1024:.0f} KB")

    print(f"\n✓ 4 STL générés dans {out_dir}")


if __name__ == "__main__":
    main()
