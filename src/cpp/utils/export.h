#ifndef UTILS_EXPORT_H
#define UTILS_EXPORT_H

// Export macro for the myumat library
#if defined(_WIN32) || defined(__CYGWIN__)
#ifdef MYUMAT_EXPORTS
#define MYUMAT_API __declspec(dllexport)
#else
#define MYUMAT_API __declspec(dllimport)
#endif
#else
#ifdef MYUMAT_EXPORTS
#define MYUMAT_API __attribute__((visibility("default")))
#else
#define MYUMAT_API
#endif
#endif

#endif // UTILS_EXPORT_H
