---
title: "5.3. Library Availability"
section: "5.3"
source: "https://docs.nvidia.com/cuda/cuda-binary-utilities/#library-availability"
---

## [5.3. Library Availability](https://docs.nvidia.com/cuda/cuda-binary-utilities#library-availability)[](https://docs.nvidia.com/cuda/cuda-binary-utilities/#library-availability "Permalink to this headline")

`cu++filt` is also available as a static library (libcufilt) that can be linked against an existing project. The following interface describes it’s usage:

```text
char* __cu_demangle(const char *id, char *output_buffer, size_t *length, int *status)
```

This interface can be found in the file “nv_decode.h” located in the SDK.

**Input Parameters**

_id_ Input mangled string.

_output_buffer_ Pointer to where the demangled buffer will be stored. This memory must be allocated with malloc. If output-buffer is NULL, memory will be malloc’d to store the demangled name and returned through the function return value. If the output-buffer is too small, it is expanded using realloc.

_length_ It is necessary to provide the size of the output buffer if the user is providing pre-allocated memory. This is needed by the demangler in case the size needs to be reallocated. If the length is non-null, the length of the demangled buffer is placed in length.

_status_ *status is set to one of the following values:

> - 0 - The demangling operation succeeded
> - -1 - A memory allocation failure occurred
> - -2 - Not a valid mangled id
> - -3 - An input validation failure has occurred (one or more arguments are invalid)

**Return Value**

A pointer to the start of the NUL-terminated demangled name, or NULL if the demangling fails. The caller is responsible for deallocating this memory using free.

**Note**: This function is thread-safe.

**Example Usage**

```text
#include <stdio.h>
#include <stdlib.h>
#include "nv_decode.h"

int main()
{
  int     status;
  const char *real_mangled_name="_ZN8clstmp01I5cls01E13clstmp01_mf01Ev";
  const char *fake_mangled_name="B@d_iDentiFier";

  char* realname = __cu_demangle(fake_mangled_name, 0, 0, &status);
  printf("fake_mangled_name:\t result => %s\t status => %d\n", realname, status);
  free(realname);

  size_t size = sizeof(char)*1000;
  realname = (char*)malloc(size);
  __cu_demangle(real_mangled_name, realname, &size, &status);
  printf("real_mangled_name:\t result => %s\t status => %d\n", realname, status);
  free(realname);

  return 0;
}
```

This prints:

```text
fake_mangled_name:   result => (null)     status => -2
real_mangled_name:   result => clstmp01<cls01>::clstmp01_mf01()   status => 0
```
