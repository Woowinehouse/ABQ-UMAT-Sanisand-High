/*****************************************************************************
 * @file macro.h
 *
 * @brief Macros for assertion checking with automatic error location
 *
 * @details This header file defines macros that simplify assertion checking
 *          by automatically passing file and line information to assertion
 *          procedures. The macros handle compiler-specific differences and
 *          are used in ABAQUS UMAT subroutines for runtime error checking.
 *****************************************************************************/

#define ERROR_LOCATION __FILE__, __LINE__ /*< Macro for file and line location */

#ifdef __GFORTRAN__
/* Assert that a logical expression is true (gfortran version) */
#define CHECK_TRUE(EXPR, FORMAT) call ASSERT_TRUE(EXPR, FORMAT, ERROR_LOCATION)
/* Assert that two integer values are equal (gfortran version) */
#define CHECK_EQUAL(VALUE, EXPECTED, FORMAT) call ASSERT_EQUAL(VALUE, EXPECTED, FORMAT, ERROR_LOCATION)
/* Assert that two floating-point values are equal within tolerance (gfortran version) */
#define CHECK_FLOAT_EQUAL(VALUE, EXPECTED, FORMAT) call ASSERT_FLOAT_EQUAL(VALUE, EXPECTED, FORMAT, ERROR_LOCATION)
#else
/* Assert that a logical expression is true */
#define CHECK_TRUE(EXPR, MSG) call ASSERT_TRUE(EXPR, MSG, ERROR_LOCATION)
/* Assert that two integer values are equal */
#define CHECK_EQUAL(VALUE, EXPECTED, FORMAT) call ASSERT_EQUAL(VALUE, EXPECTED, FORMAT, ERROR_LOCATION)
/* Assert that two floating-point values are equal within tolerance */
#define CHECK_FLOAT_EQUAL(VALUE, EXPECTED, FORMAT) call ASSERT_FLOAT_EQUAL(VALUE, EXPECTED, FORMAT, ERROR_LOCATION)
#endif
