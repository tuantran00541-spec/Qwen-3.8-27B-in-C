#ifndef QWEN38_GLIBC_EXPF_REDIRECT_H
#define QWEN38_GLIBC_EXPF_REDIRECT_H

#include <math.h>

/* Windows-only build redirect used by the exact gate.  The compatibility
 * implementation reproduces the pinned Linux/glibc expf value semantics.
 * Proven kernel source remains unchanged; this header is force-included only
 * for Win32 DLLs whose exact arithmetic already calls expf(). */
#ifdef __cplusplus
extern "C" {
#endif
float qwen38_glibc_expf_compat(float x);
#ifdef __cplusplus
}
#endif

#define expf qwen38_glibc_expf_compat

#endif
