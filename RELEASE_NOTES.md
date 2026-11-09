# Version 1.0.0

This release accompanies the completed revision of *Relational Behavioral Records
for Requirement-Conditioned Language Model Policy Selection*.

- Includes the controlled primary comparisons, additional controlled baselines,
  repeated selection, and the LunarLander and Highway auxiliary evidence.
- Includes offline scoring code, frozen inputs and retained model responses,
  including failed attempts. No additional model calls were made for this release.
- The public source is an export with its own history. Local development notes,
  execution recovery tools, manuscript drafts, reviewer correspondence, and
  credentials are not included.
- Relevant algorithm and statistical comments are retained. Internal workflow
  comments were removed without changing the original program syntax tree.
- An offline Highway entry point exposes the same scoring functions without
  importing the historical launcher or relying on the author's filesystem.
- Local user paths and account/transport fields were sanitized in a copy of the
  evidence. The source archive remains unchanged. Public and original hashes are
  listed in the evidence manifest. Frozen judge request bytes are preserved.
- No trained neural weights are included. Offline result reproduction and a new
  training/model run are distinct procedures.

`PUBLICATION_CHECKS.json` records the final local validation of this release.
