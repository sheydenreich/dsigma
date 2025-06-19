cmake_minimum_required(VERSION 3.21)
project(dsigma LANGUAGES CXX CUDA)

# Find the remaining dependencies.
# scikit-build-core will automatically find NumPy and Cython.
find_package(CUDAToolkit REQUIRED)
find_package(Python REQUIRED COMPONENTS Interpreter Development)

# --- Add OpenMP support ---
find_package(OpenMP)
if(OpenMP_CXX_FOUND)
    message(STATUS "Found OpenMP: ${OpenMP_CXX_VERSION}")
    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
    
    # Fix for OpenMP library path issues
    string(REPLACE ";" " " OpenMP_CXX_LIBRARIES_FIXED "${OpenMP_CXX_LIBRARIES}")
    set(OpenMP_CXX_LIBRARIES "${OpenMP_CXX_LIBRARIES_FIXED}")

    if(WIN32)
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /openmp:llvm")
        set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Xcompiler=/openmp:llvm")
    else()
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
        # For NVCC compiler, need to pass OpenMP flags differently
        set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Xcompiler=${OpenMP_CXX_FLAGS}")
        set(CMAKE_CUDA_HOST_LINK_LAUNCHER "${CMAKE_CXX_COMPILER}")
        set(CMAKE_CUDA_STANDARD_LIBRARIES "${CMAKE_CUDA_STANDARD_LIBRARIES} ${OpenMP_CXX_LIBRARIES}")
    endif()
else()
    message(WARNING "OpenMP not found. Multi-GPU processing will be disabled (sequential execution).")
endif()

# Find NumPy - modern approach that works with current NumPy versions
execute_process(
    COMMAND ${Python_EXECUTABLE} -c "import numpy; print(numpy.get_include())"
    OUTPUT_VARIABLE NumPy_INCLUDE_DIRS
    OUTPUT_STRIP_TRAILING_WHITESPACE
    RESULT_VARIABLE NumPy_RESULT
)
if(NOT NumPy_RESULT EQUAL 0)
    message(FATAL_ERROR "Could not find NumPy. Please install NumPy.")
endif()
message(STATUS "Found NumPy include directory: ${NumPy_INCLUDE_DIRS}")

# Find Cython - modern approach
execute_process(
    COMMAND ${Python_EXECUTABLE} -c "import Cython; print(Cython.__version__)"
    OUTPUT_VARIABLE Cython_VERSION
    OUTPUT_STRIP_TRAILING_WHITESPACE
    RESULT_VARIABLE Cython_RESULT
)
if(NOT Cython_RESULT EQUAL 0)
    message(FATAL_ERROR "Could not find Cython. Please install Cython.")
endif()
message(STATUS "Found Cython version: ${Cython_VERSION}")

# Find cython executable
find_program(CYTHON_EXECUTABLE cython REQUIRED)
if(NOT CYTHON_EXECUTABLE)
    find_program(CYTHON_EXECUTABLE cython3 REQUIRED)
endif()
message(STATUS "Found Cython executable: ${CYTHON_EXECUTABLE}")

# Function to compile Cython files
function(cython_sources OUTPUT_VAR)
    set(CYTHON_SRC_FILES ${ARGN})
    set(OUTPUT_FILES)
    
    foreach(CYTHON_FILE ${CYTHON_SRC_FILES})
        get_filename_component(CYTHON_FILE_WE ${CYTHON_FILE} NAME_WE)
        get_filename_component(CYTHON_FILE_DIR ${CYTHON_FILE} DIRECTORY)
        set(C_FILE "${CMAKE_CURRENT_BINARY_DIR}/${CYTHON_FILE_WE}.cpp")
        
        add_custom_command(
            OUTPUT ${C_FILE}
            COMMAND ${CYTHON_EXECUTABLE} --cplus -3 -I${CMAKE_CURRENT_SOURCE_DIR}/dsigma -o ${C_FILE} ${CMAKE_CURRENT_SOURCE_DIR}/${CYTHON_FILE}
            DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/${CYTHON_FILE} ${CMAKE_CURRENT_SOURCE_DIR}/dsigma/_precompute_cuda.pxd
            COMMENT "Compiling Cython file ${CYTHON_FILE}"
        )
        
        list(APPEND OUTPUT_FILES ${C_FILE})
    endforeach()
    
    set(${OUTPUT_VAR} ${OUTPUT_FILES} PARENT_SCOPE)
endfunction()

# Function to create Cython module (replacement for cython_add_module)
function(cython_add_module MODULE_NAME CYTHON_FILE)
    get_filename_component(CYTHON_FILE_WE ${CYTHON_FILE} NAME_WE)
    set(C_FILE "${CMAKE_CURRENT_BINARY_DIR}/${CYTHON_FILE_WE}.cpp")
    
    add_custom_command(
        OUTPUT ${C_FILE}
        COMMAND ${CYTHON_EXECUTABLE} --cplus -3 -I${CMAKE_CURRENT_SOURCE_DIR}/dsigma -o ${C_FILE} ${CMAKE_CURRENT_SOURCE_DIR}/${CYTHON_FILE}
        DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/${CYTHON_FILE}
        COMMENT "Compiling Cython file ${CYTHON_FILE}"
    )
    
    add_library(${MODULE_NAME} SHARED ${C_FILE})
    set_target_properties(${MODULE_NAME} PROPERTIES PREFIX "")
    if(WIN32)
        set_target_properties(${MODULE_NAME} PROPERTIES SUFFIX ".pyd")
    endif()
    
    target_link_libraries(${MODULE_NAME} PRIVATE ${Python_LIBRARIES})
    target_include_directories(${MODULE_NAME} PRIVATE ${Python_INCLUDE_DIRS})
endfunction()

# Find HEALPIX library with multiple search paths
find_path(HEALPIX_INCLUDE_DIR 
    NAMES healpix_cxx.h healpix_base.h
    HINTS 
        ${CONDA_PREFIX}/include
        $ENV{CONDA_PREFIX}/include
        /usr/local/include
        /usr/include
        /opt/conda/include
        ${CMAKE_PREFIX_PATH}/include
    PATH_SUFFIXES healpix_cxx healpix
)

find_library(HEALPIX_LIBRARY 
    NAMES healpix_cxx libhealpix_cxx
    HINTS 
        ${CONDA_PREFIX}/lib
        $ENV{CONDA_PREFIX}/lib
        /usr/local/lib
        /usr/lib
        /opt/conda/lib
        ${CMAKE_PREFIX_PATH}/lib
    PATH_SUFFIXES healpix_cxx healpix
)

if(NOT HEALPIX_INCLUDE_DIR OR NOT HEALPIX_LIBRARY)
    message(WARNING "Could not find healpix_cxx library. Trying alternative approaches...")
    
    # Try pkg-config as fallback
    find_package(PkgConfig QUIET)
    if(PkgConfig_FOUND)
        pkg_check_modules(HEALPIX QUIET healpix_cxx)
        if(HEALPIX_FOUND)
            set(HEALPIX_INCLUDE_DIR ${HEALPIX_INCLUDE_DIRS})
            set(HEALPIX_LIBRARY ${HEALPIX_LIBRARIES})
            message(STATUS "Found HEALPix via pkg-config")
        endif()
    endif()
    
    if(NOT HEALPIX_INCLUDE_DIR OR NOT HEALPIX_LIBRARY)
        message(WARNING "Could not find healpix_cxx. Building without HEALPix support.")
        set(HEALPIX_FOUND 0)
    endif()
endif()

if(HEALPIX_INCLUDE_DIR AND HEALPIX_LIBRARY)
    message(STATUS "Found HEALPix include directory: ${HEALPIX_INCLUDE_DIR}")
    message(STATUS "Found HEALPix library: ${HEALPIX_LIBRARY}")
    add_definitions(-DHEALPIX_FOUND=1)
    set(HEALPIX_FOUND 1)
else()
    add_definitions(-DHEALPIX_FOUND=0)
    set(HEALPIX_FOUND 0)
endif()

# --- Build the CUDA Extension (_precompute_cuda) ---
cython_sources(CUDA_CYTHON_SRC dsigma/_precompute_cuda.pyx)
add_library(_precompute_cuda SHARED
    ${CUDA_CYTHON_SRC}
    dsigma/precompute_engine_cuda.cu
    dsigma/precompute_interface.cu
    dsigma/cuda_host_utils.cpp
    dsigma/healpix_gpu.cu
)
set_property(TARGET _precompute_cuda PROPERTY CUDA_SEPARABLE_COMPILATION ON)

# Set CUDA architectures for compatibility with older GPUs
set_property(TARGET _precompute_cuda PROPERTY CUDA_ARCHITECTURES "70;75;80;86")
set_target_properties(_precompute_cuda PROPERTIES PREFIX "")
if(WIN32)
    set_target_properties(_precompute_cuda PROPERTIES SUFFIX ".pyd")
endif()
target_link_libraries(_precompute_cuda PRIVATE
    CUDA::cudart
    ${Python_LIBRARIES}
)
if(OpenMP_CXX_FOUND)
    target_link_libraries(_precompute_cuda PRIVATE OpenMP::OpenMP_CXX)
    target_compile_definitions(_precompute_cuda PRIVATE OPENMP_FOUND)
endif()
if(HEALPIX_FOUND)
    target_link_libraries(_precompute_cuda PRIVATE ${HEALPIX_LIBRARY})
endif()

target_include_directories(_precompute_cuda PRIVATE
    ${Python_INCLUDE_DIRS}
    ${CUDAToolkit_INCLUDE_DIRS}
    ${NumPy_INCLUDE_DIRS}
    ${CMAKE_CURRENT_SOURCE_DIR}/dsigma
    ${CMAKE_CURRENT_SOURCE_DIR}/cudaKDTree
)
if(HEALPIX_FOUND)
    target_include_directories(_precompute_cuda PRIVATE ${HEALPIX_INCLUDE_DIR})
endif()

# --- Build the regular Cython Extension (precompute_engine) ---
cython_add_module(precompute_engine
    dsigma/precompute_engine.pyx
)
target_include_directories(precompute_engine PRIVATE
    ${Python_INCLUDE_DIRS}
    ${NumPy_INCLUDE_DIRS}
)

# --- Install both compiled libraries into the package ---
install(TARGETS _precompute_cuda precompute_engine
    DESTINATION dsigma)