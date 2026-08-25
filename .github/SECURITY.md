# Security Policy

AutoVQE is a research harness and does not run a hosted service. Security
reports are still welcome, especially for dependency, supply-chain, or unsafe
file-handling issues.

## Supported Versions

The `main` branch is the supported development line.

## Reporting a Vulnerability

Please do not open a public issue for a security vulnerability.

Use GitHub's private vulnerability reporting for this repository if available,
or contact the maintainer through the GitHub profile linked from the project.

Include:

- affected files or commands,
- a minimal reproduction,
- expected impact,
- dependency versions if relevant.

## Scope

In scope:

- unsafe parsing or file handling in problem JSON workflows,
- dependency or CI supply-chain issues,
- commands that could unexpectedly modify files outside the repo.

Out of scope:

- inaccurate results caused by malformed or incorrect Hamiltonian coefficients,
- quantum algorithm performance limitations,
- intentionally long-running optimization workloads.
