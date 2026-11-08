# Contributing to ForceCouplingMethod.jl

Contributions are welcome. The [README](README.md#contributing) sets out the basics. [ColPrac](https://github.com/SciML/ColPrac) could be a good reference too. We ask you to say if you used an LLM to
write code, and we ask that pull requests be opened by people.

Where to look before you start:

- **The working rules** are in [CLAUDE.md](CLAUDE.md). It is written for coding agents, but
  the rules apply to people too. The paper (Su & Keaveny 2024) is the specification. Every
  new behaviour starts with its docstring and a failing test. The hot path allocates
  nothing. `Float32` and `Float64` both work. Code follows Blue style with 92-character
  lines.
- **Tests:** [test/README.md](test/README.md) explains the suite and how to run it. CI has
  no GPU, so if you change GPU code, run the GPU tests yourself on a machine with an NVIDIA
  GPU, as that guide shows.
- **Performance changes** need a measurement before and after, made with the suite in
  [benchmark/](benchmark/README.md).
- **Open work** is listed in the
  [issues](https://github.com/Biofluid-Dynamics-Group/ForceCouplingMethod.jl/issues),
  some of them marked *good first issue*.

Maintainers make a release by registering a commit that bumps `version` in
`Project.toml`: comment `@JuliaRegistrator register` on it, and TagBot tags and releases it
once the General registry has merged the new version. The note in
`.github/workflows/TagBot.yml` explains why that commit should not change any workflow
file.
