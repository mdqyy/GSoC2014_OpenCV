# Provides an option that the user can optionally select.
# Can accept condition to control when option is available for user.
# Usage:
#   option(<option_variable> "help string describing the option" 
#             <initial value or boolean expression> [IF <condition>])
macro(cond_option variable description value)
    set(__value ${value})
    set(__condition "")
    set(__varname "__value")
    foreach(arg ${ARGN})
        if(arg STREQUAL "IF" OR arg STREQUAL "if")
            set(__varname "__condition")
        else()
            list(APPEND ${__varname} ${arg})
        endif()
    endforeach()
    unset(__varname)
    if("${__condition}" STREQUAL "")
        set(__condition 2 GREATER 1)
    endif()
  
    if(${__condition})
        if("${__value}" MATCHES ";")
            if(${__value})
                option(${variable} "${description}" ON)
            else()
                option(${variable} "${description}" OFF)
            endif()
        elseif(DEFINED ${__value})
            if(${__value})
                option(${variable} "${description}" ON)
            else()
                option(${variable} "${description}" OFF)
            endif()
        else()
            option(${variable} "${description}" ${__value})
        endif()
    else()
        unset(${variable} CACHE)
    endif()
    unset(__condition)
    unset(__value)
endmacro()

macro(append_or_remove __dst __op)
    string(REPLACE " " ";" __dst_lst "${${__dst}}")
    string(REPLACE " " ";" __op_lst "${${__op}}")

    set(__condition "")
    set(__varname "__value")
    foreach(arg ${ARGN})
        if(arg STREQUAL "IF" OR arg STREQUAL "if")
            set(__varname "__condition")
        else()
            list(APPEND ${__varname} ${arg})
        endif()
    endforeach()
    unset(__varname)
    if("${__condition}" STREQUAL "")
        set(__condition 2 GREATER 1)
    endif()
      
    if (${__condition})
        LIST(APPEND __dst_lst ${__op_lst})
        LIST(REMOVE_DUPLICATES __dst_lst)
    else()
        foreach(item ${__op_lst})
            LIST(REMOVE_ITEM __dst_lst "${item}")
        endforeach()
    endif()
    
    string(REPLACE ";" " " ${__dst} "${__dst_lst}")
    string(REPLACE ";" " " ${__op} "${__op_lst}")    
endmacro()

#-------------------------------------------------------------
#-------------------------------------------------------------
set(CMAKE_MODULE_PATH ${CMAKE_CURRENT_LIST_DIR})

# openmp option
    find_package(OpenMP REQUIRED)
    option(WITH_OPENMP "Build with OpenMP" OFF)
    
    append_or_remove(CMAKE_CXX_FLAGS OpenMP_CXX_FLAGS IF (WITH_OPENMP))
    append_or_remove(CMAKE_C_FLAGS OpenMP_C_FLAGS IF (WITH_OPENMP))
    append_or_remove(CMAKE_EXE_LINKER_FLAGS OpenMP_EXE_LINKER_FLAGS IF (WITH_OPENMP))
    
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}" CACHE STRING 
        "Flags used by the compiler during all build types." FORCE)
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS}" CACHE STRING 
        "Flags used by the compiler during all build types." FORCE)
    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS}" 
        CACHE STRING "Flags used by the linker." FORCE)

# option for static compiling
    cond_option(STATIC_RUNTIME "Build with /MT and /MTd" ON IF (MSVC))
    
    set(FLAGS_LIST CMAKE_C_FLAGS_DEBUG CMAKE_C_FLAGS_RELEASE 
                   CMAKE_C_FLAGS_MINSIZEREL CMAKE_C_FLAGS_RELWITHDEBINFO 
                   CMAKE_CXX_FLAGS_DEBUG CMAKE_CXX_FLAGS_RELEASE                    
                   CMAKE_CXX_FLAGS_MINSIZEREL CMAKE_CXX_FLAGS_RELWITHDEBINFO 
                   CACHE INTERNAL "list of all compiler-flag variables")
    
    if (STATIC_RUNTIME)
        foreach(flag_var ${FLAGS_LIST})
                string(REPLACE "/MD" "/MT" ${flag_var} "${${flag_var}}")
                set(${flag_var} ${${flag_var}} CACHE STRING "" FORCE)
        endforeach(flag_var)
    else()
        foreach(flag_var ${FLAGS_LIST})
                string(REPLACE "/MT" "/MD" ${flag_var} "${${flag_var}}")
                set(${flag_var} ${${flag_var}} CACHE STRING "" FORCE)
        endforeach(flag_var)    
    endif()

# find OpenCV, absolutely required
    cond_option(STATIC_OPENCV "Build with statically compiled OpenCV libs" ON IF (WIN32))
    if (STATIC_OPENCV)
        set(BUILD_SHARED_LIBS OFF)
    endif()
    
    find_package(OpenCV REQUIRED QUIET)

# find CUDA option
    option(WITH_CUDA "Build with CUDA" OFF)
    if (WITH_CUDA)
        find_package(CUDA QUIET)
    endif()
    
# find Qt5 option
    option(WITH_QT "Build with QT" OFF)
    if (WITH_QT)
        cmake_policy(SET CMP0020 NEW)
        add_definitions(-DWITH_QT)    
    
        set(CMAKE_INCLUDE_CURRENT_DIR ON)    
        set(CMAKE_AUTOMOC ON)
        
        find_package(Qt5Core)
        find_package(Qt5Gui)
    endif()