# API reference

The package exports three names: a configuration constructor (the cold path, run once),
the mutating operator application (the hot path, run per solver iteration), and a
matrix-free operator wrapper for iterative solvers. The
[method overview](method/overview.md) explains how they fit together; internal step
functions are documented in the [architecture](devdocs/architecture.md) page of the
developer documentation.

```@docs
FFCMConfig
mobility!
FFCMMobility
```

A fourth name is public but not exported, because it is a tool you call once, by its full
name, to choose the settings the constructor takes.
[Choosing parameters](choosing-parameters.md) explains what it does.

```@docs
ForceCouplingMethod.calibrate
```
