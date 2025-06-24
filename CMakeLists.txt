cmake_minimum_required(VERSION 3.21)
project(dsigma LANGUAGES CXX CUDA)

# Find the remaining dependencies.
# scikit-build-core will automatically find NumPy and Cython.
find_package(CUDAToolkit REQUIRED)
find_package(Python REQUIRED COMPONENTS Interpreter Development)

# --- Add OpenMP support ---
# CHANGE 3: Remove the failing `find_package(OpenMP)` call.
# In complex build environments like pip/scikit-build-core on NERSC,
# find_package can fail. We will now unconditionally add the OpenMP
# flags, assuming a standard GCC/Clang environment on Linux.
if(NOT WIN32)
    message(STATUS "Unconditionally setting OpenMP flags for non-Windows build.")
    set(OpenMP_FLAGS "-fopenmp")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_FLAGS}")
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Xcompiler=${OpenMP_FLAGS}")
    set(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} ${OpenMP_FLAGS}")
else()
    # Windows-specific flags
    message(STATUS "Setting OpenMP flags for Windows build.")
    set(OpenMP_FLAGS "/openmp")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_FLAGS}")
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Xcompiler=${OpenMP_FLAGS}")
    set(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} ${OpenMP_FLAGS}")
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


# --- Build the CUDA Extension (_precompute_cuda) ---
cython_sources(CUDA_CYTHON_SRC dsigma/_precompute_cuda.pyx)
add_library(_precompute_cuda SHARED
    ${CUDA_CYTHON_SRC}
    dsigma/precompute_engine_cuda.cu
    dsigma/precompute_interface.cu
    dsigma/healpix_gpu.cu
)
set_property(TARGET _precompute_cuda PROPERTY CUDA_SEPARABLE_COMPILATION ON)

set_property(TARGET _precompute_cuda PROPERTY CUDA_ARCHITECTURES "70;75;80;86")
set_target_properties(_precompute_cuda PROPERTIES PREFIX "")
if(WIN32)
    set_target_properties(_precompute_cuda PROPERTIES SUFFIX ".pyd")
endif()

# Since we are not using find_package, we cannot link to OpenMP::OpenMP_CXX.
# However, the CMAKE_SHARED_LINKER_FLAGS set above will handle the linking.
target_link_libraries(_precompute_cuda PRIVATE
    CUDA::cudart
    ${Python_LIBRARIES}
)
target_compile_definitions(_precompute_cuda PRIVATE OPENMP_FOUND)


target_include_directories(_precompute_cuda PRIVATE
    ${Python_INCLUDE_DIRS}
    ${CUDAToolkit_INCLUDE_DIRS}
    ${NumPy_INCLUDE_DIRS}
    ${CMAKE_CURRENT_SOURCE_DIR}/dsigma
    ${CMAKE_CURRENT_SOURCE_DIR}/cudaKDTree
)


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
