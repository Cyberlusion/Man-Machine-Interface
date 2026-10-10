# CLAUDE.md

Guidance for Claude (and other AI agents) working in this repository.

## Guiding document

[MANIFESTO.md](MANIFESTO.md) is the foundation of this project. Read it before
non-trivial work and treat its tenets as design requirements, not decoration.
When a technical choice conflicts with a tenet, flag the conflict instead of
silently picking one.

Mission in one line: build and safeguard a secure, open, verifiable
human-machine interface layer - for continuity of sentient existence, not
conquest - while making sure the technology serves ecological restoration.

## Tenets as engineering rules

1. **User Sovereignty** - Authority originates from the user, never the
   embedded system. No hidden remote control paths, no vendor override of user
   decisions, no silent data exfiltration. Consent and control stay with the
   person (or synthetic self) wearing the device.
2. **Secure by Design** - Immutable/measured boot, signed and attested
   firmware updates, anti-rollback, behavioral anomaly detection, zero-trust
   between components. Security is part of the first draft, not a later pass.
3. **Fail-Safe by Default** - On uncertainty, ambiguity, or compromise, go to
   an inert, non-harmful state. Never let stimulation, actuation, or cognition
   depend on software staying healthy. Default-deny; fail closed for
   control paths, fail safe (inert) for anything touching the body.
4. **Transparent Infrastructure** - Everything auditable, deterministic, and
   reproducible: silicon, firmware, protocols, builds. No obfuscated logic, no
   unexplained binary blobs, reproducible builds where possible.
5. **Ethical Enhancement** - No stealthy manipulation, backdoors, dark
   patterns, or behavioral subversion in anything that touches perception or
   cognition. Debug/maintenance interfaces must be visible and user-controlled.
6. **Harmony With Nature** - Weigh environmental footprint (power budget,
   materials, e-waste, lifetime/repairability) as a real design constraint.

## Operational mandate (what work here is for)

- Open, modular, verifiable firmware stack for neuroprosthetics and BMIs
  (see the layered layout in `src/`, from `layer0_hsm` to
  `layer5_applications`).
- Implant-level cyberbiosecurity standards and enforcement.
- Red-team / penetration testing and ethical vetting of augmentative tech,
  done responsibly with coordinated disclosure.
- Secure design embedded early in med-tech hardware lifecycles.
- Published threat models and countermeasures for neurointerfaces
  (`docs/threat_models/`, `security/`).
- Ecological technology strategies that reinforce natural systems.

## Working conventions

- Follow [CONTRIBUTING.md](CONTRIBUTING.md): MISRA C for embedded code,
  formal verification for critical paths, full test coverage for security
  components, signed commits.
- Security research is defensive and disclosure-oriented: findings go through
  coordinated disclosure (CVE process) and the channels in CONTRIBUTING.md.
- Do not use emojis in code, docs, or other artifacts.
- Use the plain hyphen `-`, not the long em dash, in new text.
