---
title: "11.6. Linking Directives"
section: "11.6"
source: "https://docs.nvidia.com/cuda/parallel-thread-execution/#linking-directives"
---

## [11.6. Linking Directives](https://docs.nvidia.com/cuda/parallel-thread-execution#linking-directives)[](https://docs.nvidia.com/cuda/parallel-thread-execution/#linking-directives "Permalink to this headline")

- `.extern`
- `.visible`
- `.weak`

### [11.6.1. Linking Directives: .extern](https://docs.nvidia.com/cuda/parallel-thread-execution#linking-directives-extern)[](https://docs.nvidia.com/cuda/parallel-thread-execution/#linking-directives-extern "Permalink to this headline")

**`.extern`**

External symbol declaration.

**Syntax**

```text
.extern identifier
```

**Description**

Declares identifier to be defined external to the current module. The module defining such
identifier must define it as `.weak` or `.visible` only once in a single object file. Extern
declaration of symbol may appear multiple times and references to that get resolved against the
single definition of that symbol.

**PTX ISA Notes**

Introduced in PTX ISA version 1.0.

**Target ISA Notes**

Supported on all target architectures.

**Examples**

```text
.extern .global .b32 foo;  // foo is defined in another module
```

### [11.6.2. Linking Directives: .visible](https://docs.nvidia.com/cuda/parallel-thread-execution#linking-directives-visible)[](https://docs.nvidia.com/cuda/parallel-thread-execution/#linking-directives-visible "Permalink to this headline")

**`.visible`**

Visible (externally) symbol declaration.

**Syntax**

```text
.visible identifier
```

**Description**

Declares identifier to be globally visible. Unlike C, where identifiers are globally visible unless
declared static, PTX identifiers are visible only within the current module unless declared
`.visible` outside the current.

**PTX ISA Notes**

Introduced in PTX ISA version 1.0.

**Target ISA Notes**

Supported on all target architectures.

**Examples**

```text
.visible .global .b32 foo;  // foo will be externally visible
```

### [11.6.3. Linking Directives: .weak](https://docs.nvidia.com/cuda/parallel-thread-execution#linking-directives-weak)[](https://docs.nvidia.com/cuda/parallel-thread-execution/#linking-directives-weak "Permalink to this headline")

**`.weak`**

Visible (externally) symbol declaration.

**Syntax**

```text
.weak identifier
```

**Description**

Declares identifier to be globally visible but _weak_. Weak symbols are similar to globally visible
symbols, except during linking, weak symbols are only chosen after globally visible symbols during
symbol resolution. Unlike globally visible symbols, multiple object files may declare the same weak
symbol, and references to a symbol get resolved against a weak symbol only if no global symbols have
the same name.

**PTX ISA Notes**

Introduced in PTX ISA version 3.1.

**Target ISA Notes**

Supported on all target architectures.

**Examples**

```text
.weak .func (.reg .b32 val) foo;  // foo will be externally visible
```

### [11.6.4. Linking Directives: .common](https://docs.nvidia.com/cuda/parallel-thread-execution#linking-directives-common)[](https://docs.nvidia.com/cuda/parallel-thread-execution/#linking-directives-common "Permalink to this headline")

**`.common`**

Visible (externally) symbol declaration.

**Syntax**

```text
.common identifier
```

**Description**

Declares identifier to be globally visible but “common”.

Common symbols are similar to globally visible symbols. However multiple object files may declare
the same common symbol and they may have different types and sizes and references to a symbol get
resolved against a common symbol with the largest size.

Only one object file can initialize a common symbol and that must have the largest size among all
other definitions of that common symbol from different object files.

`.common` linking directive can be used only on variables with `.global` storage. It cannot be
used on function symbols or on symbols with opaque type.

**PTX ISA Notes**

Introduced in PTX ISA version 5.0.

**Target ISA Notes**

`.common` directive requires `sm_20` or higher.

**Examples**

```text
.common .global .u32 gbl;
```
