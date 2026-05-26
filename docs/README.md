# Alpha Capital Documentation

This repository contains the runnable code. The full engineering canon lives in the Alpha Capital vault:

```text
~/Documents/AlphaCapital/
```

The vault is intentionally more detailed than this repo. It defines the completed system: architecture, pattern contracts, feature assembly, execution, validation, and operating doctrine.

## Primary Vault Files

- `Architecture.md` - portfolio construction, optimizer, execution classes, evidence schema, and runtime contracts
- `Patterns.md` - canonical 17-pattern roster, thesis categories, exit geometry, and validation gates
- `Validation.md` - shadow and real validation tracks
- `Engineering/FeatureAssembly.md` - pattern-specific assembly contracts, lookahead enforcement, typed missing values, and lineage requirements
- `Engineering/Patterns/` - per-pattern SPEC, EXPOSURE, DATA, EXECUTION, and VALIDATION documents
- `Engineering/RuntimeLayerStack.md` - backend-first implementation order
- `CODEX.md` - compact recovery context for AI coding agents

## Repo-Facing Rule

Repository docs should describe the intended completed system and the production standard. They should not drift into stale "what is currently done" snapshots unless the status is audit-backed and expected to remain useful.

When code behavior changes in a way that alters a pattern contract, update the relevant vault file first or in the same change. The README should stay concise; detailed doctrine belongs in the vault.
