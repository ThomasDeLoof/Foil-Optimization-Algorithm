# Fiche Technique — WINDSURF | NACA1415

*Générée le 2026-05-16 14:15:20*

---

## 0. Paramètres du run

### Recherche paramétrique
| Paramètre | Valeur |
|:---|:---|
| Cambrures testées | [0, 1, 2] |
| Épaisseurs root testées | [15] |
| Angles reflex testés | [0.0, 1.0, 2.0, 3.0] |
| Profil root retenu | NACA1415 |
| Angle reflex retenu | 3.0° |

### Bornes d'optimisation — Aile
| Variable | Borne min | Borne max | Init |
|:---|:---|:---|:---|
| Envergure (m) | 0.9 | 1.3 | 1.0 |
| Corde root (m) | 0.12 | 0.28 | 0.18 |
| Corde tip (m) | 0.02 | 0.12 | 0.03 |
| Vrillage (°) | -4.0 | -1.3 | -1.6 |

### Bornes d'optimisation — Stabilisateur
| Variable | Borne min | Borne max | Init |
|:---|:---|:---|:---|
| Envergure (m) | 0.32 | 0.56 | 0.42 |
| Corde root (m) | 0.06 | 0.2 | 0.1 |
| Corde tip (m) | 0.015 | 0.07 | 0.02 |
| Calage (°) | -5.0 | 0.5 | -3.0 |

### Bornes d'optimisation — Fuselage & CG
| Variable | Borne min | Borne max | Init |
|:---|:---|:---|:---|
| Longueur fuselage (m) | 0.65 | 0.9 | 0.7 |
| Ratio CG | 0.2 | 0.4 | 0.27 |

### Contraintes scénario — WINDSURF
| Contrainte | Min | Max |
|:---|:---|:---|
| Surface aile (m²) | 0.07 | 0.13 |
| Surface stab (m²) | 0.019 | 0.028 |
| Marge statique | 0.1 | 0.4 |
| Force stab (N) | -30.0 | -5.0 |
| Volume de queue | 0.45 | 0.9 |

---

## Conditions de vol

| Paramètre | Valeur |
|:---|:---|
| Cas | windsurf |
| Poids total | 892.7 N (91 kg) |
| Vitesse décollage | 7.5 m/s |
| Vitesse croisière | 13.0 m/s |

---

## 1. Géométrie Aile

| Paramètre | Valeur |
|:---|:---|
| Profil | NACA1415 |
| Surface | 1125.3 cm² |
| Envergure | 90.0 cm |
| Allongement | 7.57 |
| Corde emplanture | 171.4 mm |
| Corde saumon | 10.0 mm |
| Vrillage | -1.30° |
| Charge alaire | 7933.11 N/m² |
| Re emplanture | 2.23e+06 |
| Re saumon | 4.46e+05 |

---

## 2. Géométrie Stabilisateur

| Paramètre | Valeur |
|:---|:---|
| Surface | 190.0 cm² |
| Envergure | 32.0 cm |
| Allongement | 5.42 |
| Corde emplanture | 89.6 mm |
| Calage | -2.13° |
| Longueur fuselage | 65.0 cm |

---

## 3. Performances

| Paramètre | Valeur |
|:---|:---|
| Finesse (L/D) | 8.55 |
| Traînée | 153.29 N |
| Incidence croisière | 1.86° |
| CL croisière | 0.094 |
| CD croisière | 0.0110 |
| CL stabilisateur | -0.019 |

---

## 4. Stabilité & Équilibre

| Paramètre | Valeur |
|:---|:---|
| Marge statique | 97.10 % |
| Position CG | 36.8% (4.5 cm du BA) |
| Force stabilisateur | -30.00 N |
| Moment résiduel | -8.0279 N·m |
| Volume de queue | 0.9000 |

---

## ⚠️ Avertissements

- ⚠️ Marge statique élevée (97.1%) - risque de manœuvrabilité réduite.
- ⚠️ Moment résiduel important (-8.03 N·m) - vérifier l'équilibre.
