#ifndef UTILS_CONFIG_H
#define UTILS_CONFIG_H

// paths
#define LIBTORCH_LIB_PATH ""
// Build options
/* #undef __ENABLE_DEBUG__ */
#define __USE_FORTRAN__
/* #undef __GTEST__BUILD_ */
#define __STRICT_VALIDATION__

#define STRICT_CHECK_ENABLED defined(__STRICT_VALIDATION__)
#define DEBUG_SPAN_ENABLED (defined(__ENABLE_DEBUG__))
#endif // UTILS_CONFIG_H
