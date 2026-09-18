# mcsirius

Small source-to-Cup-1 beam optimizer for the FLAVIA AMS beamline.

## Scope

mcsirius controls only the components required to maximize beam current at Cup 1:

- sputter voltage
- extraction voltage
- analyzing magnet current
- Einzel lens voltage

The user-facing input will ultimately be the ion mass. The optimizer then prepares
the source, calculates the expected magnet setting, finds the beam on Cup 1 and
optimizes the source and focusing settings.

## Initial machine constraints

- Sputter voltage: 4 to 9 kV
- Extraction voltage: 14 to 25 kV
- Einzel lens voltage: 14 to 25 kV
- Einzel lens must remain within 2 kV of extraction voltage
- Initial operating point: 4 kV sputter, 14 kV extraction, 14 kV Einzel lens

## Design

The optimizer will use a deterministic optimization chain rather than a
multi-dimensional grid scan.

Hardware access, measurement and optimization logic will remain separate so the
optimizer can be tested against simulated hardware before controlling FLAVIA.
