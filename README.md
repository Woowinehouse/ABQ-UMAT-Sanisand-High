# ABQ-UMAT-Sanisand-High

A high-performance SANISAND sand constitutive model User Material Subroutine (UMAT) for Abaqus finite element software, employing hybrid C++/Fortran programming for efficient numerical simulation of sand mechanical behavior.

## Overview

This project implements the SANISAND (Simple ANIsotropic SAND) constitutive model as a User Material Subroutine (UMAT) for Abaqus. The model captures the anisotropic behavior of sands under various loading conditions, including monotonic and cyclic loading. The implementation leverages modern C++20 features for performance-critical computations while maintaining Fortran compatibility with Abaqus.

## Key Features

- **SANISAND Constitutive Model**: Implementation of the SANISAND model for accurate simulation of sand behavior
- **Hybrid C++/Fortran Architecture**: Combines C++ performance with Fortran's Abaqus compatibility
- **High-Performance Computing**: Utilizes LibTorch for tensor operations and MKL for linear algebra
- **Cross-Platform Support**: Compatible with Windows and Linux systems
- **Comprehensive Error Handling**: Detailed error reporting and validation mechanisms
- **Testing Framework**: GoogleTest integration for unit testing
- **Python Automation**: Scripts for automated model generation and parameter studies

## Project Structure

```
ABQ-UMAT-Sanisand-High/
├── cmake/                    # CMake configuration files
├── src/                      # Source code
│   ├── cpp/                  # C++ implementation
│   │   ├── core/            # Core model implementation
│   │   ├── ops/             # Mathematical operations
│   │   ├── umat/            # UMAT interface components
│   │   └── utils/           # Utility functions
│   ├── fortran/             # Fortran implementation
│   └── interface/           # Interface files (umat.cpp, umat.f90)
├── tests/                    # Unit tests
├── scripts/                  # Python automation scripts
├── input/                    # Example input files
├── model/                    # Abaqus model files
├── third_party/              # Third-party dependencies
└── build-vs2026/            # Build directory (generated)
```

## Requirements

### For Development and Building
- **CMake** 3.25 or newer
- **C++ Compiler** with C++20 support (MSVC, GCC, Clang, or Intel oneAPI)
- **Fortran Compiler** Fortran 2018 compliant (ifx, ifort, or gfortran)
- **LibTorch** >= 2.2.0 (PyTorch C++ API)
- **Intel MKL** (optional, for optimized linear algebra)
- **GoogleTest** (for testing, automatically fetched by CMake)

### For Abaqus Integration
- **Abaqus** 2022 or newer
- **Python** 3.10+ (for automation scripts)

## Building the Project

### Basic Build

1. **Create a build directory** (out-of-source build recommended):
   ```powershell
   mkdir build
   cd build
   ```

2. **Configure with CMake**:
   ```powershell
   cmake -G "Visual Studio 17 2022" -A x64 ..
   ```
   For Linux:
   ```bash
   cmake -DCMAKE_BUILD_TYPE=Release ..
   ```

3. **Build the project**:
   ```powershell
   cmake --build . --config Release
   ```

### Build Options

You can customize the build with the following CMake options:

| Option | Default | Description |
|--------|---------|-------------|
| `CMAKE_BUILD_TYPE` | `Release` | Build type (Debug, Release, RelWithDebInfo, MinSizeRel) |
| `BUILD_CXX` | `OFF` | Build the UMAT in C++ language |
| `BUILD_FORTRAN` | `OFF` | Build the UMAT in Fortran language |
| `BUILD_SHARED_LIBS` | `OFF` | Build shared libraries instead of static |
| `BUILD_TESTING` | `OFF` | Enable unit tests (requires GoogleTest) |
| `USE_MIMALLOC` | `OFF` | Use mimalloc memory allocator |
| `USE_ABAQUS` | `OFF` | Enable Abaqus-specific features |
| `Merge_Fortran` | `OFF` | Merge Fortran project files |

Example with custom options:
```powershell
cmake -G "Visual Studio 17 2022" -A x64 -DBUILD_CXX=ON -DBUILD_TESTING=ON ..
```

## Usage

### 1. Building the UMAT Library

Build the UMAT library with your preferred configuration:
```powershell
cmake -G "Visual Studio 17 2022" -A x64 -DBUILD_CXX=ON ..
cmake --build . --config Release
```

### 2. Integrating with Abaqus

1. Copy the generated UMAT library to your Abaqus working directory
2. Configure the `abaqus_v6.env` file to point to the library
3. Use the UMAT in your Abaqus input files:

```inp
*MATERIAL, NAME=SAND
*USER MATERIAL, CONSTANTS=20
  [material properties...]
*DEPVAR
  14
```

### 3. Running Example Simulations

Use the provided Python scripts to generate and run example models:

```python
cd scripts
python compression_model.py
```

This will generate Abaqus input files for different loading conditions in the `input/` directory.

### 4. Running Tests

If tests are enabled during build:
```powershell
ctest -C Release --output-on-failure
```

## Development

### Code Style

- C++ code follows the Google C++ Style Guide
- Fortran code uses modern Fortran 2018 features
- Automatic formatting with `.clang-format` for C++ and `fprettify.cfg` for Fortran

### Adding New Features

1. Add C++ implementation in `src/cpp/`
2. Add Fortran interface if needed in `src/fortran/`
3. Update the UMAT interface in `src/interface/`
4. Add tests in `tests/`
5. Update CMakeLists.txt as needed

### Debugging

For debugging, build in Debug mode:
```powershell
cmake -G "Visual Studio 17 2022" -A x64 -DCMAKE_BUILD_TYPE=Debug -DBUILD_TESTING=ON ..
```

The UMAT includes extensive debug output that can be enabled via the `abaqus_debug` function.

## Error Handling

The UMAT includes comprehensive error handling for:
- NaN and infinity detection
- Input validation
- Convergence issues
- Memory errors

Errors are reported to Abaqus's message file with detailed diagnostic information.

## Performance Optimization

- Uses LibTorch for efficient tensor operations
- Optional MKL integration for linear algebra
- Memory pooling with mimalloc (optional)
- Thread-safe design for parallel execution in Abaqus

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests for new functionality
5. Ensure all tests pass
6. Submit a pull request

Please ensure your code follows the project's coding standards and includes appropriate documentation.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- Based on the SANISAND constitutive model by Dafalias and Manzari
- Uses LibTorch for tensor operations
- CMake configuration inspired by modern C++ project templates

## Support

For issues and questions:
1. Check the existing issues on GitHub
2. Review the example files in `input/` and `scripts/`
3. Contact the maintainers through GitHub issues

## Citation

If you use this code in your research, please cite:

```bibtex
@software{ABQ_UMAT_Sanisand_High,
  author = {Woowinehouse},
  title = {ABQ-UMAT-Sanisand-High: High-performance SANISAND UMAT for Abaqus},
  year = {2026},
  url = {https://github.com/Woowinehouse/ABQ-UMAT-Sanisand-High}
}
```

---

*This project is maintained by Woowinehouse. For more information, visit the [GitHub repository](https://github.com/Woowinehouse/ABQ-UMAT-Sanisand-High).*
