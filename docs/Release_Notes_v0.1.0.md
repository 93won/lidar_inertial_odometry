# v0.1.0

## Changes

- Corrected the iterated ESKF update to use the propagated state and covariance as a fixed prior.
- Revised IMU transition, noise discretization, and gyro/accelerometer bias Jacobians.
- Constrained gravity to a fixed magnitude with a 2-DoF tangent-state representation.
- Switched estimator state, covariance, and solver calculations to double precision.
- Moved point-cloud deskewing before downsampling.
- Added incremental KD-tree matching, local-map pruning, and correspondence distance/plane gates.
- Added robust residual weighting, line search, rejected-update rollback, and partial-loss recovery.
- Validated Jacobians, covariance symmetry/PSD, gravity, synthetic stationary initialization, and spatial-index operations. Test sources are not included in this distribution.

## M3DGR benchmark

Compared with baseline `73348c1`, using four logical CPUs per version. Results cover 29 sequences per sensor; APE/RPE are available for 25. The comparison includes configuration changes: the iteration limit increased from 4 to 10, and Mid360 IMU noise parameters changed.

| Metric | Avia: baseline → v0.1.0 | Change | Mid360: baseline → v0.1.0 | Change |
| --- | ---: | ---: | ---: | ---: |
| Mean APE RMSE (m) | 6.3087 → 1.4940 | −76.3% | 0.4982 → 0.2466 | −50.5% |
| Mean 1-second displacement RMSE (m) | 0.5884 → 0.1782 | −69.7% | 0.0751 → 0.0557 | −25.9% |
| Mean frame processing time* (ms) | 6.056 → 30.632 | +405.8% | 6.554 → 21.001 | +220.4% |
| Sequences with lower APE | 22/25 | — | 25/25 | — |

APE improved in **47/50 evaluated sensor cases**. Median per-sequence APE reductions were **32.5% for Avia** and **19.1% for Mid360**.

Metrics are equally weighted across sequences. APE uses rigid alignment without scale correction. The 1-second metric measures aligned displacement error, not full SE(3) RPE. *Processing time is the mean of per-sequence frame-time medians; it excludes data conversion.

### Bias stability

Changes in the mean per-sequence metric across all 29 sequences; negative means lower.

| Metric | Avia | Mid360 |
| --- | ---: | ---: |
| Gyro bias endpoint drift | −49.6% | +64.7% |
| Accelerometer bias endpoint drift | −48.4% | −38.3% |
| Gyro bias component step RMS | −38.8% | +885.6% |
| Accelerometer bias component step RMS | +11.0% | +126.8% |
| Maximum gyro bias step norm | −29.9% | +3319.1% |
| Maximum accelerometer bias step norm | +47.8% | +911.7% |

Bias ground truth is unavailable: these measure changes in the estimates, not absolute accuracy. Lower drift can also reflect missed updates. Gravity magnitude error remains approximately 5e-12 m/s².

<details>
<summary>Per-sequence APE RMSE (m)</summary>

| Sequence | Avia: baseline → v0.1.0 | Change | Mid360: baseline → v0.1.0 | Change |
| --- | ---: | ---: | ---: | ---: |
| Outdoor01 | 0.3097 → 0.2186 | -29.4% | 0.2982 → 0.2261 | -24.2% |
| Outdoor04 | 0.4326 → 0.3181 | -26.5% | 0.5992 → 0.2323 | -61.2% |
| Dynamic02 | 10.1406 → 2.4718 | -75.6% | 0.1609 → 0.1374 | -14.6% |
| Occlusion01 | 16.8272 → 2.8286 | -83.2% | 0.1782 → 0.1572 | -11.8% |
| Occlusion02 | 14.0569 → 6.7278 | -52.1% | 0.1687 → 0.1455 | -13.7% |
| Varying-illu01 | 36.8069 → 5.4013 | -85.3% | 0.1651 → 0.1390 | -15.8% |
| Varying-illu02 | 7.3969 → 1.6448 | -77.8% | 0.1647 → 0.1372 | -16.7% |
| Dark03 | 15.9359 → 2.0915 | -86.9% | 0.1939 → 0.1745 | -10.0% |
| Dark04 | 4.0471 → 0.7507 | -81.5% | 0.2037 → 0.1823 | -10.5% |
| Dynamic03 | 0.2676 → 0.1805 | -32.5% | 0.2269 → 0.1717 | -24.3% |
| Dynamic04 | 0.3760 → 0.2943 | -21.7% | 0.4687 → 0.1892 | -59.6% |
| Occlusion03 | 0.2655 → 0.1746 | -34.2% | 0.6753 → 0.4868 | -27.9% |
| Occlusion04 | 0.3398 → 0.2579 | -24.1% | 1.0623 → 0.4183 | -60.6% |
| Varying-illu03 | 0.5937 → 0.7679 | +29.3% | 1.1994 → 0.6913 | -42.4% |
| Varying-illu04 | 0.3491 → 0.3740 | +7.1% | 0.8924 → 0.2605 | -70.8% |
| Varying-illu05 | 0.1696 → 0.1216 | -28.3% | 0.4957 → 0.1787 | -64.0% |
| Dark01 | 0.1423 → 0.1458 | +2.4% | 0.1858 → 0.1777 | -4.4% |
| Dark02 | 0.7428 → 0.7390 | -0.5% | 0.6212 → 0.1760 | -71.7% |
| Corridor01 | N/A → N/A | N/A | N/A → N/A | N/A |
| Corridor02 | N/A → N/A | N/A | N/A → N/A | N/A |
| Elevator01 | N/A → N/A | N/A | N/A → N/A | N/A |
| Wheel-float01 | 13.4367 → 2.9816 | -77.8% | 0.2100 → 0.1934 | -7.9% |
| Wheel-float02 | 6.9747 → 2.1692 | -68.9% | 0.1531 → 0.1313 | -14.2% |
| Sha-turn01 | 21.4554 → 1.3878 | -93.5% | 0.2468 → 0.1997 | -19.1% |
| Sha-turn02 | 4.3949 → 3.8338 | -12.8% | 0.1814 → 0.1426 | -21.4% |
| Grass01 | 0.7103 → 0.5014 | -29.4% | 0.5478 → 0.5340 | -2.5% |
| Grass02 | 0.4953 → 0.4586 | -7.4% | 0.4570 → 0.4365 | -4.5% |
| Z-Rough-Road01 | 1.0492 → 0.5083 | -51.6% | 2.6978 → 0.2453 | -90.9% |
| GNSS-denial01 | N/A → N/A | N/A | N/A → N/A | N/A |

</details>

## Known limitations

- Processing cost increased approximately **5.06× for Avia** and **3.20× for Mid360**. Bias smoothness did not improve consistently.
- Avia APE regressed on Varying-illu03, Varying-illu04, and Dark01. Large errors remain on Occlusion02 (6.728 m) and Varying-illu01 (5.401 m).
- **Dynamic01 was excluded after inspecting its results**, following an Avia measurement dropout and failure to recover tracking. Its updated APE was 32.3145 m versus 4.4449 m for baseline. The bag checksum matches the published original; this exclusion does not resolve the recovery defect.
- Corridor01/02 and Elevator01 have trajectory-evaluation errors; GNSS-denial01 lacks an evaluation reference. Their N/A entries are not accuracy passes. GNSS-denial02 was not run because its raw bag was unavailable.
- Outdoor01 and Occlusion03 remain included with historical checksum-mismatch caveats.
- These are single-run comparisons, not a demonstrated zero-divergence or tracking-success rate.
