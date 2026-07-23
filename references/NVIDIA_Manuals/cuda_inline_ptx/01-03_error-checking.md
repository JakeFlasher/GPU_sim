---
title: "1.3. Error Checking"
section: "1.3"
source: "https://docs.nvidia.com/cuda/inline-ptx-assembly/#error-checking"
---

## [1.3. Error Checking](https://docs.nvidia.com/cuda/inline-ptx-assembly#error-checking)[](https://docs.nvidia.com/cuda/inline-ptx-assembly/#error-checking "Permalink to this headline")

The following are some of the error checks that the compiler will do on inlinePTXasm:

- Multiple constraint letters for a single asm operand are not allowed, e.g.:

```c++
asm("add.s32 %0, %1, %2;" : "=r"(i) : "rf"(j), "r"(k));
```

error: an asm operand may specify only one constraint letter in a __device__/__global__ function
- Only scalar variables are allowed as asm operands. Specifically aggregates like ‘struct’ type variables are not allowed, e.g.

```c++
int4 i4;
asm("add.s32 %0, %1, %2;" : "=r"(i4) : "r"(j), "r"(k));
```

error: an asm operand must have scalar type
- The type and size implied by a PTX asm constraint must match that of the associated operand. Example where size does not match:

> For ‘char’ type variable “ci”,
>
>
>
> ```c++
> asm("add.s32 %0,%1,%2;":"=r"(ci):"r"(j),"r"(k));
> ```
>
>
> error: asm operand type size(1) does not match type/size implied by constraint ‘r’
>
>
> In order to use ‘char’ type variables “ci”, “cj”, and “ck” in the above asm statement, code segment similar to the following may be used,
>
>
>
> ```c++
> int temp = ci;
> asm("add.s32 %0,%1,%2;":"=r"(temp):"r"((int)cj),"r"((int)ck));
> ci = temp;
> ```
>
>
> Another example where type does not match:
>
>
> For ‘float’ type variable “fi”,
>
>
>
> ```c++
> asm("add.s32 %0,%1,%2;":"=r"(fi):"r"(j),"r"(k));
> ```
>
>
> error: asm operand type size(4) does not match type/size implied by constraint ‘r’
