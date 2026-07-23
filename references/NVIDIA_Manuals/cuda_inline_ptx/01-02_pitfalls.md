---
title: "1.2. Pitfalls"
section: "1.2"
source: "https://docs.nvidia.com/cuda/inline-ptx-assembly/#pitfalls"
---

## [1.2. Pitfalls](https://docs.nvidia.com/cuda/inline-ptx-assembly#pitfalls)[](https://docs.nvidia.com/cuda/inline-ptx-assembly/#pitfalls "Permalink to this headline")

Although `asm()` statements are very flexible and powerful, you may encounter some pitfalls—these are listed in this section.

### [1.2.1. Namespace Conflicts](https://docs.nvidia.com/cuda/inline-ptx-assembly#namespace-conflicts)[](https://docs.nvidia.com/cuda/inline-ptx-assembly/#namespace-conflicts "Permalink to this headline")

If the cube function (described before) is called and inlined multiple times in the code, it generates an error about duplicate definitions of the temp register t1. To avoid this error you need to:

- not inline the cube function, or,
- nest the t1 use inside `{}` so that it has a separate scope for each invocation, e.g.:

```c++
__device__ int cube (int x)
{
  int y;
  asm("{\n\t"                        // use braces for local scope
      " reg .u32 t1;\n\t"            // temp reg t1,
      " mul.lo.u32 t1, %1, %1;\n\t"  // t1 = x * x
      " mul.lo.u32 %0, t1, %1;\n\t"  // y = t1 * x
      "}"
      : "=r"(y) : "r" (x));
  return y;
}
```

Note that you can similarly use braces for local labels inside the `asm()` statement.

### [1.2.2. Memory Space Conflicts](https://docs.nvidia.com/cuda/inline-ptx-assembly#memory-space-conflicts)[](https://docs.nvidia.com/cuda/inline-ptx-assembly/#memory-space-conflicts "Permalink to this headline")

Since `asm()` statements have no way of knowing what memory space a register is in, the user must make sure that the appropriate PTX instruction is used. For `sm_20` and greater, any pointer argument to an `asm()` statement is passed as a generic address.

### [1.2.3. Incorrect Optimization](https://docs.nvidia.com/cuda/inline-ptx-assembly#incorrect-optimization)[](https://docs.nvidia.com/cuda/inline-ptx-assembly/#incorrect-optimization "Permalink to this headline")

The compiler assumes that an `asm()` statement has no side effects except to change the output operands. To ensure that the asm is not deleted or moved during generation of PTX, you should use the volatile keyword, e.g.:

```c++
asm volatile ("mov.u32 %0, %%clock;" : "=r"(x));
```

Normally any memory that is written to will be specified as an out operand, but if there is a hidden read or write on user memory (for example, indirect access of a memory location via an operand), or if you want to stop any memory optimizations around the `asm()` statement performed during generation of PTX, you can add a “memory” clobbers specification after a 3rd colon. For example:

```c++
asm volatile ("mov.u32 %0, %%clock;" : "=r"(x) :: "memory");
asm ("st.u32 [%0], %1;" :: "l"(p), "r"(x) : "memory");
```

### [1.2.4. Incorrect PTX](https://docs.nvidia.com/cuda/inline-ptx-assembly#incorrect-ptx)[](https://docs.nvidia.com/cuda/inline-ptx-assembly/#incorrect-ptx "Permalink to this headline")

The compiler front end does not parse the `asm()` statement template string and does not know what it means or even whether it is valid PTX input. So if there are any errors in the string it will not show up until `ptxas`. For example, if you pass a value with an “r” constraint but use it in an `add.f64` you will get a parse error from ptxas. Similarly, operand modifiers are not supported. For example, in

```c++
asm("mov.u32 %0, %n1;" : "=r"(n) : "r"(1));
```

the ‘n’ modifier in “%n1” is not supported and will be passed to `ptxas`, where it can cause undefined behavior. Refer to the document nvcc.pdf for further compiler related details.
