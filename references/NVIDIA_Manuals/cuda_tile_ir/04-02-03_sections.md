---
title: "4.2.3. Sections"
section: "4.2.3"
source: "https://docs.nvidia.com/cuda/tile-ir/latest/sections/bytecode.html#sections"
---

### [4.2.3. Sections](https://docs.nvidia.com/cuda/tile-ir/latest/sections#sections)[](https://docs.nvidia.com/cuda/tile-ir/latest/sections/#sections "Permalink to this headline")

The remainder of the bytecode stream is composed of sections.  Sections are used to group data within the
bytecode and allow operations on the stream to be per-section enabling out-of-order processing of data and/or lazy-loading.
Each section contains a section ID, whose high bit indicates if the section has alignment requirements, a
length and an optional alignment. A section ID is a 7-bit integer with the high bit indicating if the section
has alignment requirements. When an alignment is present, a variable number of padding bytes (each byte = `0xCB`)
may appear before the section data. The alignment of a section must be a power of 2. The alignment is represented
as a `VarInt`. The padding ensures the section data starts at the specified alignment boundary.

```text
section {
  idAndIsAligned: byte   // low 7 bits = section ID, high bit = alignment bit
  length: varint,
  alignment: varint?,    // present only if high bit was set
  padding: byte[],       // bytes of 0xCB as needed
  data: byte[]
}
```

#### [Deserialization](https://docs.nvidia.com/cuda/tile-ir/latest/sections#deserialization)[](https://docs.nvidia.com/cuda/tile-ir/latest/sections/#deserialization "Permalink to this headline")

The following are the implications of the section format on the deserialization process.

1. The deserializer must read the section ID and length (`idAndIsAligned`) as a single byte.
2. If the section ID has the high bit set, the deserializer must read the alignment (`alignment`) and apply the alignment by skipping
`0xCB` bytes as needed.
3. The deserializer must read the length (`length`) bytes of the payload. The length is a `VarInt`.
4. The deserializer must move on to the next section until `EOF` (end of file).

#### [String Section](https://docs.nvidia.com/cuda/tile-ir/latest/sections#string-section)[](https://docs.nvidia.com/cuda/tile-ir/latest/sections/#string-section "Permalink to this headline")

The string section holds all textual names used by the module to avoid repeating them inline.
The section is encoded with the total number of strings, followed by the start
index of each of the individual strings. The remaining encoding contains a single blob containing all the strings
concatenated together. This design allows loading a specific string without reading the whole section. Strings in
the bytecode are stored as raw byte sequences (in UTF-8 encoding) with an associated size. Finding the i-th
string involves jumping to `stringStartIndex[i]` and reading until the next offset or end of the blob.

```text
strings {
  numStrings: varint,
  stringStartIndex: uint32[],
  stringData: byte[]
}
```

#### [Function Table Section](https://docs.nvidia.com/cuda/tile-ir/latest/sections#function-table-section)[](https://docs.nvidia.com/cuda/tile-ir/latest/sections/#function-table-section "Permalink to this headline")

The function table section enumerates the module’s functions and embeds their code inline. First,
`numFunctions` indicates how many functions follow; for each function `i`,
`nameIndex[i]` is an index into the string section naming the function and
`signatureIndex[i]` is an index into the Type section specifying its parameter, return
types, global flags, etc (so multiple functions with identical signatures can share the
same type entry). The field `functionLocIndex[i]` is a `VarInt` referencing an entry
in the Debug Section that describes the function’s definition location (e.g., source
file and line). If `functionLocIndex[i]` is zero, there is no associated debug
metadata for the function’s definition scope. `lengthOfFunction[i]` states how many
bytes of code belong to function `i`, and `functionBody[i]` contains exactly those
bytes of instruction encodings. This layout avoids a separate code section and makes
parsing each function straightforward: once you read the metadata for function `i`,
you can either parse its instructions directly or skip them by advancing
`lengthOfFunction[i]` bytes. The instruction encoding itself allows each operation to
include a slot for its source location, referencing an entry in the debug section.

The instruction encodings (opcodes, operands, etc.) are described later under Operation
Opcodes and Encodings Section.

The function table encoding has been updated to include function flags and optional optimization hints:

```text
functionTable {
  numFunctions : varint
  // for each function i in [0..numFunctions-1]:
  nameIndex[i] : varint                    // References the function's name in the StringSec
  signatureIndex[i] : varint               // References the extended function signature in the TypeSec
  entryFlag[i] : byte                      // Function flags (visibility, kind, optimization hints)
  functionLocIndex[i] : varint             // 0 means no debug location
  optimizationHints[i]? : self-contained  // Present only if HasOptimizationHints flag is set
  lengthOfFunction[i] : varint
  functionBody[i] : byte[lengthOfFunction[i]]
}
```

The `entryFlag` byte encodes the following information:

- **Bit 0 (0x01)**: Visibility Flag (0 = Public, 1 = Private)
- **Bit 1 (0x02)**: Function Kind Flag (0 = Device Function, 1 = Kernel Entry Point)
- **Bit 2 (0x04)**: Has Optimization Hints Flag (0 = No, 1 = Yes)
- **Bits 3-7**: Reserved for future extensions

#### [Constant Data Section](https://docs.nvidia.com/cuda/tile-ir/latest/sections#constant-data-section)[](https://docs.nvidia.com/cuda/tile-ir/latest/sections/#constant-data-section "Permalink to this headline")

The constant data section holds large constants (e.g., dense tensors, large arrays) separately from code.

The current implementation does not impose explicit size limits on individual constants.
Constants are stored using `uint64_t` offsets, allowing for very large constant data.
However, practical limits may be imposed by available memory and any downstream
compiler or runtime limitations (such as CUBIN generation constraints).

As we have done with the string section we move the constants into their own section to avoid bloating
the function table section and allow for the lazy loading of specific constants when needed.
Individual operations may reference constants by an index into this section.
The section is encoded with the total number of constants, followed by the start index of each
of the individual constants. The remaining encoding contains a single blob containing all the
constants concatenated together.

```text
constant {
  numConstants: varint,
  constantStartIndex: uint64_t[],
  constantData: byte[]
}
```

#### [Type Section](https://docs.nvidia.com/cuda/tile-ir/latest/sections#type-section)[](https://docs.nvidia.com/cuda/tile-ir/latest/sections/#type-section "Permalink to this headline")

The type section stores all type definitions (scalar types, function signatures, parametric
types, etc.) used in the module. The section begins with `numTypes` specifying how many types
follow, then an array of offsets (`typeStartIndex`) into the encoded blob (`typeData`). To load the
i-th type definition, you use the offset contained in `typeStartIndex[i]` to index into
the `typeData` blob and parse the definition from there. The `typeData` blob is a single encoded binary
blob containing the type definitions concatenated together.

```text
type {
  numTypes: varint
  typeStartIndex: uint32_t[] // array of offsets, length = numTypes
  typeData: byte[]           // concatenated bytes for all type definitions
}
```

Each individual type definition will consist of a `typeTag` followed by a predefined payload
structure specific to that `typeTag`. This deterministic approach allows parsers to
understand the `payload` layout based solely on the `typeTag`.

```text
typeTag : byte
payload : byte[lengthOfPayload]
```

- `typeTag` indicates the “kind” of type. This can be a simple scalar, a function
signature, or a more complex structure like a tensor.
- `payload` is interpreted based on `typeTag`. For instance, a function signature might
store the number of parameters, references to their types, etc, while a tensor type
might store dimensions and an element type.

Example `typeTag` Values:

| `typeTag`  | Meaning         | Expected Payload |  |  |
| --- | --- | --- |
| 0x01             \| i1 (1-bit int)  \| None |  |  |
| 0x02 | i32 (32-bit int) | None |
| 0x03 | i64 (64-bit int) | None |
| 0x04 | tensor | `elementTypeIndex + rank + dimensions[]` |
| 0x10 | Function | Extended function signature |
| … | Future types | Depends on the type |

#### [Detailed Type Encodings](https://docs.nvidia.com/cuda/tile-ir/latest/sections#detailed-type-encodings)[](https://docs.nvidia.com/cuda/tile-ir/latest/sections/#detailed-type-encodings "Permalink to this headline")

The following sections describe the specific encoding format for each type tag:

**Integer Types** (`typeTag` = 0x00-0x04)

Integer types require no additional payload data beyond the type tag:

```text
I1:   typeTag = 0x00  // 1-bit boolean
I8:   typeTag = 0x01  // 8-bit integer
I16:  typeTag = 0x02  // 16-bit integer
I32:  typeTag = 0x03  // 32-bit integer
I64:  typeTag = 0x04  // 64-bit integer
```

**Float Types** (`typeTag` = 0x05-0x0B)

Float types require no additional payload data beyond the type tag:

```text
F16:      typeTag = 0x05  // 16-bit IEEE float
BF16:     typeTag = 0x06  // 16-bit bfloat
F32:      typeTag = 0x07  // 32-bit IEEE float
TF32:     typeTag = 0x08  // TensorFloat-32
F64:      typeTag = 0x09  // 64-bit IEEE float
F8E4M3FN: typeTag = 0x0A  // 8-bit float (E4M3FN format)
F8E5M2:   typeTag = 0x0B  // 8-bit float (E5M2 format)
```

**Pointer Type** (`typeTag` = 0x0C)

```text
typeTag : byte = 0x0C      // Pointer type
pointeeTypeIndex : varint  // Index of the pointee type
```

**Tile Type** (`typeTag` = 0x0D)

```text
typeTag : byte = 0x0D      // Tile type
elementTypeIndex : varint  // Index of the element type
shape : int64_t[]          // Shape dimensions (var-length array)
```

**TensorView Type** (`typeTag` = 0x0E)

```text
typeTag : byte = 0x0E      // TensorView type
elementTypeIndex : varint  // Index of the element type
shape : int64_t[]          // Shape dimensions (var-length array)
strides : int64_t[]        // Stride values (var-length array)
indexTypeTag : byte        // Index type (I32=0x03 or I64=0x04)
```

**PartitionView Type** (`typeTag` = 0x0F)

```text
typeTag : byte = 0x0F        // PartitionView type
tileShape : int32_t[]        // Tile shape (var-length array)
tensorViewTypeIndex : varint // Index of the TensorView type
dimMap : int32_t[]           // Dimension mapping (var-length array)
masked : byte                // Masked flag (0x00=false, 0x01=true)
```

**Function Type** (`typeTag` = 0x10)

```text
typeTag : byte = 0x10 // Function Type
numParams : varint    // Number of input parameters
paramTypeIndices : varint[]  // Array of type indices for each parameter
numResults : varint   // Number of results
resultTypeIndices : varint[] // Array of type indices for each return value
```

The function type encoding stores only the essential type information (input and result types).
Argument attributes and other function metadata are stored separately in the function table section.

**Token Type** (`typeTag` = 0x11)

```text
typeTag : byte = 0x11  // Token type (no additional payload)
```

#### [Attribute Encoding](https://docs.nvidia.com/cuda/tile-ir/latest/sections#attribute-encoding)[](https://docs.nvidia.com/cuda/tile-ir/latest/sections/#attribute-encoding "Permalink to this headline")

Attributes in **Tile IR** bytecode can be encoded in two ways:

1. **Inline encoding** - Simple attributes are encoded directly in the instruction stream
2. **Self-contained encoding** - Complex attributes include a type tag followed by their data

Self-contained attributes use the following format:

```text
attributeTag : byte
attributeData : byte[]  // Format depends on attributeTag
```

The following attribute tags are supported:

**Integer Attribute** (`attributeTag` = 0x01)

```text
attributeTag : byte = 0x01  // Integer attribute
typeIndex : varint          // Type index for the integer type
value : varint              // Integer value (zero-extended)
```

**Float Attribute** (`attributeTag` = 0x02)

```text
attributeTag : byte = 0x02  // Float attribute
typeIndex : varint          // Type index for the float type
value : byte[]              // APFloat representation (variable length)
```

**Bool Attribute** (`attributeTag` = 0x03)

```text
attributeTag : byte = 0x03  // Bool attribute
value : byte                // 0x00=false, 0x01=true
```

**Type Attribute** (`attributeTag` = 0x04)

```text
attributeTag : byte = 0x04  // Type attribute
typeIndex : varint          // Index of the referenced type
```

**String Attribute** (`attributeTag` = 0x05)

```text
attributeTag : byte = 0x05  // String attribute
stringIndex : varint        // Index into the string section
```

**Array Attribute** (`attributeTag` = 0x06)

```text
attributeTag : byte = 0x06    // Array attribute
numElements : varint          // Number of elements
elements : self-contained[]   // Array of self-contained attributes
```

**DenseElements Attribute** (`attributeTag` = 0x07)

```text
attributeTag : byte = 0x07  // DenseElements attribute
typeIndex : varint          // Type index for the shaped type
constantIndex : varint      // Index into constant section (for numeric data)
// OR for string data:
numStrings : varint         // Number of string elements
stringIndices : varint[]    // Indices into string section
```

**DivBy Attribute** (`attributeTag` = 0x08)

```text
attributeTag : byte = 0x08    // DivBy attribute
divisor : varint              // Divisor value
flags : byte                  // Bit 0: unsignedInt, Bit 1: hasEvery, Bit 2: hasAlong
every : signed_varint?        // Present if Bit 1 set
along : signed_varint?        // Present if Bit 2 set
```

**SameElements Attribute** (`attributeTag` = 0x09)

```text
attributeTag : byte = 0x09  // SameElements attribute
values : int64_t[]          // Array of int64 values (var-length)
```

**Dictionary Attribute** (`attributeTag` = 0x0A)

```text
attributeTag : byte = 0x0A      // Dictionary attribute
numEntries : varint             // Number of key-value pairs
entries : dictEntry[]           // Array of dictionary entries

dictEntry {
  keyStringIndex : varint       // Index of key string
  value : self-contained        // Self-contained attribute value
}
```

**OptimizationHints Attribute** (`attributeTag` = 0x0B)

```text
attributeTag : byte = 0x0B  // OptimizationHints attribute
dictionary : dictionary    // Dictionary attribute (without tag)
```

**NonNegative Attribute** (`attributeTag` = 0x0C)

```text
attributeTag : byte = 0x0C  // NonNegative attribute
// No additional payload - presence indicates non-negative constraint
```

#### [Global Section](https://docs.nvidia.com/cuda/tile-ir/latest/sections#global-section)[](https://docs.nvidia.com/cuda/tile-ir/latest/sections/#global-section "Permalink to this headline")

The global section stores module-level global variables. This section is optional and is only present if the module contains `cuda_tile.global` operations.

```text
global {
  numGlobals: varint
  // for each global i in [0..numGlobals-1]:
  symbolNameIndex[i] : varint   // References the global's symbol name in the StringSec
  valueTypeIndex[i] : varint    // Type index for the global's value type
  constantValueIndex[i] : varint // Index into constant section for the global's initial value
}
```

Each global variable is encoded with:

- **symbolNameIndex**: Index into the string section for the global’s symbol name
- **valueTypeIndex**: Index into the type section for the global’s type (typically a shaped type like tensor)
- **constantValueIndex**: Index into the constant section containing the global’s initial value

#### [Debug Section](https://docs.nvidia.com/cuda/tile-ir/latest/sections#debug-section)[](https://docs.nvidia.com/cuda/tile-ir/latest/sections/#debug-section "Permalink to this headline")

The debug section stores the serialized debug information (for more details about debug information see [Debug Info](https://docs.nvidia.com/cuda/tile-ir/latest/sections/debug_info.html#section-debug-info)).
This section is optional as certain tools may ignore it and serializers may leave it empty for release builds.

```text
debug {
  diOpsNum: varint          // Total number of operations with debug info
  padding: bytes            // Align to 4 bytes
  diIndexOffsets: uint32_t[] // Per op offset into the debug info indices
  diIndicesNum: varint      // Total number of debug info indices
  padding: bytes            // Align to 4 bytes
  diIndices: uint64_t[]     // Array of debug indices to debug info attributes
  diAttrNum: varint         // Total number of debug info attributes
  padding: bytes            // Align to 4 bytes
  diOffsets: uint32_t[]     // Per debug info attribute offset into the debug info data
  diData: bytes             // Data for each debug info attribute
}
```

The debug section uses a multi-level indirection scheme:

1. **Operations**: `diOpsNum` operations have debug info, with `diIndexOffsets` pointing into the indices array
2. **Indices**: `diIndicesNum` total indices in `diIndices`, referencing debug info attributes
3. **Attributes**: `diAttrNum` debug info attributes stored in `diData` with offsets in `diOffsets`

Each debug info attribute begins with a `debugEntryType` indicating what kind of debug info it is:

- `0x00` = “Unknown”
- `0x01` = “DICompileUnit”
- `0x02` = “DIFile”
- `0x03` = “DILexicalBlock”
- `0x04` = “DILoc”
- `0x05` = “DISubprogram”
- `0x06` = “CallSite”

`debugEntryPayload` describes line, file, variable name, function index, instruction
offset, etc. Each `debugEntryType` has a fixed, known structure. If
`functionLocIndex[i]` or an instruction’s `locationIndex` is non-zero, it references
`debugEntryOffset[...]` in this section, whose payload can store file/line info or
other metadata.

#### [Debug Entry Payload Formats](https://docs.nvidia.com/cuda/tile-ir/latest/sections#debug-entry-payload-formats)[](https://docs.nvidia.com/cuda/tile-ir/latest/sections/#debug-entry-payload-formats "Permalink to this headline")

Each debug entry in `diData` begins with a `debugEntryType` byte followed by type-specific data:

```text
debugEntry {
  debugEntryType : byte     // Identifies the debug info type
  entryData : byte[]        // Type-specific payload (format below)
}
```

The following debug entry types are supported:

**Unknown Debug Info** (`debugEntryType` = 0x00)

```text
debugEntryType : varint = 0x00  // Unknown debug info
// No additional payload
```

**DICompileUnit** (`debugEntryType` = 0x01)

```text
debugEntryType : varint = 0x01  // DICompileUnit
language : varint               // Source language identifier
fileIndex : varint              // Index of associated DIFile
producer : varint               // String index for compiler producer
optimized : byte                // 0x00=false, 0x01=true
emissionKind : varint           // Emission kind enumeration
```

**DIFile** (`debugEntryType` = 0x02)

```text
debugEntryType : varint = 0x02  // DIFile
filename : varint               // String index for filename
directory : varint              // String index for directory
```

**DILexicalBlock** (`debugEntryType` = 0x03)

```text
debugEntryType : varint = 0x03  // DILexicalBlock
line : varint                   // Line number
column : varint                 // Column number
scopeIndex : varint             // Index of parent scope (DIFile or DISubprogram)
```

**DILoc** (`debugEntryType` = 0x04)

```text
debugEntryType : varint = 0x04  // DILoc (source location)
line : varint                   // Line number
column : varint                 // Column number
scopeIndex : varint             // Index of scope (DISubprogram, DILexicalBlock, etc.)
inlinedAtIndex : varint         // Index of inlined location (0 if not inlined)
```

**DISubprogram** (`debugEntryType` = 0x05)

```text
debugEntryType : varint = 0x05  // DISubprogram (function debug info)
name : varint                   // String index for function name
linkageName : varint            // String index for linkage name
fileIndex : varint              // Index of associated DIFile
line : varint                   // Line number where function is defined
typeIndex : varint              // Index of function type
scopeLineIndex : varint         // Line number where scope begins
flags : varint                  // Function flags (visibility, etc.)
unitIndex : varint              // Index of associated DICompileUnit
```

**CallSite** (`debugEntryType` = 0x06)

```text
debugEntryType : varint = 0x06  // CallSite location
calleeIndex : varint            // Index of called location
callerIndex : varint            // Index of calling location
```
