/* A plugin that reports a PJRT C API version older than the runtime needs. */
#include "export.h"
#include "xla/pjrt/c/pjrt_c_api.h"

static PJRT_Api api = {
    .struct_size = PJRT_Api_STRUCT_SIZE,
    .pjrt_api_version =
        {
            .struct_size = PJRT_Api_Version_STRUCT_SIZE,
            .major_version = 0,
            .minor_version = 1,
        },
};

TEST_EXPORT const PJRT_Api* GetPjrtApi(void) { return &api; }
