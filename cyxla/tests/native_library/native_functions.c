/* Native functions for the tests of cyxla. */

#include <math.h>
#include <stdint.h>
#include <string.h>
#include <time.h>

#include "cyxla/native.h"

#ifdef _WIN32
#include <windows.h>
#define EXPORT __declspec(dllexport)
#else
#define EXPORT __attribute__((visibility("default")))
#endif

/* Adds one to a counter that several threads may update. */
static void increment(int64_t* counter) {
#ifdef _WIN32
  InterlockedIncrement64((volatile LONG64*)counter);
#else
  __atomic_fetch_add(counter, 1, __ATOMIC_SEQ_CST);
#endif
}

/* Sleeps for a number of seconds, which may have a fraction. */
static void sleep_seconds(double value) {
#ifdef _WIN32
  Sleep((DWORD)(value * 1000.0));
#else
  struct timespec duration;
  duration.tv_sec = (time_t)value;
  duration.tv_nsec = (long)((value - (double)duration.tv_sec) * 1e9);
  nanosleep(&duration, NULL);
#endif
}

static int64_t element_count(const CyxlaNativeArray* array) {
  int64_t count = 1;
  for (int64_t axis = 0; axis < array->rank; ++axis) count *= array->dimensions[axis];
  return count;
}

static const CyxlaNativeAttribute* find_attribute(
    const CyxlaNativeAttribute* attributes, int64_t count, const char* name) {
  for (int64_t position = 0; position < count; ++position)
    if (strcmp(attributes[position].name, name) == 0) return &attributes[position];
  return NULL;
}

/* results[0] = factor * arguments[0] + offset, where the offset is the state of the function. */
EXPORT const char* scale(void* user_data, int64_t argument_count,
                         const CyxlaNativeArray* arguments, int64_t result_count,
                         CyxlaNativeArray* results, int64_t attribute_count,
                         const CyxlaNativeAttribute* attributes) {
  const CyxlaNativeAttribute* factor = find_attribute(attributes, attribute_count, "factor");
  if (argument_count != 1 || result_count != 1) return "scale takes one argument and one result";
  if (factor == NULL || factor->dtype != CYXLA_NATIVE_F64) return "scale needs the float64 attribute factor";
  double offset = user_data == NULL ? 0.0 : *(const double*)user_data;
  const double* source = (const double*)arguments[0].data;
  double* target = (double*)results[0].data;
  for (int64_t index = 0, count = element_count(&arguments[0]); index < count; ++index)
    target[index] = *(const double*)factor->value * source[index] + offset;
  return NULL;
}

/* results[0] = arguments[0] + arguments[1] and results[1] = arguments[0] - arguments[1]. */
EXPORT const char* sum_and_difference(void* user_data, int64_t argument_count,
                                      const CyxlaNativeArray* arguments, int64_t result_count,
                                      CyxlaNativeArray* results, int64_t attribute_count,
                                      const CyxlaNativeAttribute* attributes) {
  if (argument_count != 2 || result_count != 2) return "sum_and_difference takes two arguments and two results";
  const double* first = (const double*)arguments[0].data;
  const double* second = (const double*)arguments[1].data;
  for (int64_t index = 0, count = element_count(&arguments[0]); index < count; ++index) {
    ((double*)results[0].data)[index] = first[index] + second[index];
    ((double*)results[1].data)[index] = first[index] - second[index];
  }
  return NULL;
}

/* results[0][0] is the length of the string attribute label, as a float64. */
EXPORT const char* label_length(void* user_data, int64_t argument_count,
                                const CyxlaNativeArray* arguments, int64_t result_count,
                                CyxlaNativeArray* results, int64_t attribute_count,
                                const CyxlaNativeAttribute* attributes) {
  const CyxlaNativeAttribute* label = find_attribute(attributes, attribute_count, "label");
  if (label == NULL || label->kind != CYXLA_NATIVE_ATTRIBUTE_STRING) return "label_length needs the string attribute label";
  ((double*)results[0].data)[0] = (double)label->length;
  return NULL;
}

EXPORT const char* always_fails(void* user_data, int64_t argument_count,
                                const CyxlaNativeArray* arguments, int64_t result_count,
                                CyxlaNativeArray* results, int64_t attribute_count,
                                const CyxlaNativeAttribute* attributes) {
  return "the native function failed on purpose";
}

/* results[0] = arguments[0] squared, and the cotangent version for the derivative. */
EXPORT const char* square(void* user_data, int64_t argument_count,
                          const CyxlaNativeArray* arguments, int64_t result_count,
                          CyxlaNativeArray* results, int64_t attribute_count,
                          const CyxlaNativeAttribute* attributes) {
  const double* source = (const double*)arguments[0].data;
  for (int64_t index = 0, count = element_count(&arguments[0]); index < count; ++index)
    ((double*)results[0].data)[index] = source[index] * source[index];
  return NULL;
}

EXPORT const char* square_backward(void* user_data, int64_t argument_count,
                                   const CyxlaNativeArray* arguments, int64_t result_count,
                                   CyxlaNativeArray* results, int64_t attribute_count,
                                   const CyxlaNativeAttribute* attributes) {
  const double* point = (const double*)arguments[0].data;
  const double* cotangent = (const double*)arguments[1].data;
  for (int64_t index = 0, count = element_count(&arguments[0]); index < count; ++index)
    ((double*)results[0].data)[index] = 2.0 * point[index] * cotangent[index];
  return NULL;
}

/* Counts its calls in the state, copies the argument, and sleeps for the attribute seconds. */
EXPORT const char* sleep_and_count(void* user_data, int64_t argument_count,
                                   const CyxlaNativeArray* arguments, int64_t result_count,
                                   CyxlaNativeArray* results, int64_t attribute_count,
                                   const CyxlaNativeAttribute* attributes) {
  const CyxlaNativeAttribute* seconds = find_attribute(attributes, attribute_count, "seconds");
  if (user_data != NULL) increment((int64_t*)user_data);
  if (seconds != NULL) sleep_seconds(*(const double*)seconds->value);
  memcpy(results[0].data, arguments[0].data, (size_t)element_count(&arguments[0]) * sizeof(double));
  return NULL;
}

/* Copies its argument and records the address of the argument data and the dtype. */
EXPORT const void* last_argument_data = NULL;
EXPORT int32_t last_argument_dtype = 0;
EXPORT const char* record_argument(void* user_data, int64_t argument_count,
                                   const CyxlaNativeArray* arguments, int64_t result_count,
                                   CyxlaNativeArray* results, int64_t attribute_count,
                                   const CyxlaNativeAttribute* attributes) {
  last_argument_data = arguments[0].data;
  last_argument_dtype = arguments[0].dtype;
  if (result_count > 0 && element_count(&results[0]) > 0)
    memcpy(results[0].data, arguments[0].data, (size_t)element_count(&arguments[0]) * sizeof(double));
  return NULL;
}
