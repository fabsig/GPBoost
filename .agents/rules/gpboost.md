---
trigger: always_on
description: GPBoost repository rules for coding agents
---

The rules for this repository are kept in a single file at its root, so that every agent reads the
same ones. Follow them: @../../AGENTS.md

They cover, among other things, that thread parameters such as `num_parallel_threads` and
`num_threads` must not be set to 1 as a generic precaution.
