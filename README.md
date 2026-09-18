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
## Magnet seed model

The initial analyzing-magnet setpoint is calculated from ion mass and total
source energy before any measured beam search is performed.

The model uses the same geometry and current/field calibration as FLAVIA:

- bending radius: 0.5 m
- total ion energy: sputter voltage + extraction voltage
- current calibration: I = (B_kG + 0.0348763) / 0.10886

This value is only the predicted center for the later local magnet search.
The optimizer will not treat magnet current as an independent global parameter.
## First optimization stage

The first executable optimization chain is intentionally simple:

1. calculate the expected magnet current from ion mass and source energy
2. run a bounded coarse-to-fine magnet scan around that prediction
3. restore the best measured magnet setting
4. scan the Einzel lens while keeping it within 2 kV of extraction
5. restore the best magnet and Einzel settings
6. use the magnitude of the Cup 1 current as the optimization score

The scan distances and step sizes remain explicit inputs until suitable
machine values have been verified experimentally.