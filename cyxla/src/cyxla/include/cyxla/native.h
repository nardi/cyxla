/* The C interface of native functions that programs call through cyxla.
 *
 * A native function is registered under a name with `register_native_function` (in Python or
 * in Cython). A program calls it with `cyxla.native_call(name, ...)`. The function receives
 * the buffers of the arguments and of the results, the attributes of the call, and the
 * pointer that it was registered with. It runs on the threads of the runtime, without the
 * GIL, so it has to be thread safe. Code in any language that can export a C function can
 * implement it.
 */

#ifndef CYXLA_NATIVE_H_
#define CYXLA_NATIVE_H_

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* The version of the structures below. Registration checks it. */
#define CYXLA_NATIVE_ABI_VERSION 1

/* The codes of element types, which are the ones of the XLA FFI API. */
enum {
  CYXLA_NATIVE_PRED = 1,
  CYXLA_NATIVE_S8 = 2,
  CYXLA_NATIVE_S16 = 3,
  CYXLA_NATIVE_S32 = 4,
  CYXLA_NATIVE_S64 = 5,
  CYXLA_NATIVE_U8 = 6,
  CYXLA_NATIVE_U16 = 7,
  CYXLA_NATIVE_U32 = 8,
  CYXLA_NATIVE_U64 = 9,
  CYXLA_NATIVE_F16 = 10,
  CYXLA_NATIVE_F32 = 11,
  CYXLA_NATIVE_F64 = 12,
  CYXLA_NATIVE_C64 = 15,
  CYXLA_NATIVE_BF16 = 16,
  CYXLA_NATIVE_C128 = 18
};

/* The kinds of attributes. */
enum { CYXLA_NATIVE_ATTRIBUTE_SCALAR = 0, CYXLA_NATIVE_ATTRIBUTE_STRING = 1 };

/* An argument or a result: dense, row major, with `rank` dimensions. The data of an argument
 * must not be written. The data of a result is allocated by the runtime and has the size that
 * the program declared. */
typedef struct CyxlaNativeArray {
  int32_t dtype;
  int64_t rank;
  const int64_t* dimensions;
  void* data;
} CyxlaNativeArray;

/* An attribute of the call besides `name`. A scalar has a `dtype` and `value` points to it. A
 * string has `value` pointing to `length` bytes without a terminator. */
typedef struct CyxlaNativeAttribute {
  const char* name;
  int32_t kind;
  int32_t dtype;
  const void* value;
  int64_t length;
} CyxlaNativeAttribute;

/* A native function returns NULL when it succeeds, and otherwise a message that stays valid
 * until the function is called again on the same thread. */
typedef const char* (*CyxlaNativeFunction)(
    void* user_data, int64_t argument_count, const CyxlaNativeArray* arguments,
    int64_t result_count, CyxlaNativeArray* results, int64_t attribute_count,
    const CyxlaNativeAttribute* attributes);

#ifdef __cplusplus
}
#endif

#endif /* CYXLA_NATIVE_H_ */
