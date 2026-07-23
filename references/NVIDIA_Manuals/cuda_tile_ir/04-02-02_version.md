---
title: "4.2.2. Version"
section: "4.2.2"
source: "https://docs.nvidia.com/cuda/tile-ir/latest/sections/bytecode.html#version"
---

### [4.2.2. Version](https://docs.nvidia.com/cuda/tile-ir/latest/sections#version)[](https://docs.nvidia.com/cuda/tile-ir/latest/sections/#version "Permalink to this headline")

Following the magic number is the version of the bytecode format. The version allows both
forwards and backwards compatibility of the format.

The version is encoded as three little-endian fixed-width fields immediately following the magic number:

```text
version {
  major: uint8_t,    // Major version number
  minor: uint8_t,    // Minor version number
  tag: uint16_t      // Version tag (little-endian)
}
```

Each file encodes the version which allows parsers to adapt to the specific version.
We do not break old versions of the format, if a newer file uses features an older parser
can’t interpret, it skips unknown optional features but will fail gracefully in cases where
new features are unsupported. A new producer can also target older versions of the bytecode
directly ensuring compatibility with older drivers.

#### [Version Compatibility Guarantees](https://docs.nvidia.com/cuda/tile-ir/latest/sections#version-compatibility-guarantees)[](https://docs.nvidia.com/cuda/tile-ir/latest/sections/#version-compatibility-guarantees "Permalink to this headline")

1. Backward Compatibility (a new deserializer can read old bytecode):
  - The deserializer must support all previous versions of the bytecode format.
  - The deserializer must handle all previously existing opcodes, types, and sections.
  - The deserializer must maintain original program semantics.
2. Forward Compatibility (an old deserializer can read new bytecode):
  - The deserializer must fail only if it encounters unknown required features.
  - The deserializer must only read bytecode up to its version.
  - The deserializer must error with a clear message if there is a version mismatch.
3. Version Targeting (a serializer can target specific older versions):
  - The serializer may target one or more specific older versions.
  - The serializer must validate all features are supported by the requested target version.
  - The serializer must fail if attempting to serialize unsupported features.
4. Optional vs Required Features (required changes are clearly documented):
  - Required changes must be introduced in a new bytecode version and gracefully failure must be designed.
  - Optional changes must be clearly documented and must preserve the above guarantees.

**Examples**:

- For new ops → Add new opcodes
- For changes to existing ops → Typically keep the old format for backward compatibility
and add new opcode rather than modifying existing ones
