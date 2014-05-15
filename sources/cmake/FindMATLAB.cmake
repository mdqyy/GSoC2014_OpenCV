# Provides an find_file with no caching
macro(find_file_no_cache)
    find_file(${ARGN})
    
    set(__ARGN ${ARGN})

    list(GET __ARGN 0 __varname)
    set(__value "${${__varname}}")
    
    unset(${__varname} CACHE)
    set(${__varname} "${__value}")
endmacro()

# Provides an find_library with no caching
macro(find_library_no_cache)
    find_library(${ARGN})
    
    set(__ARGN ${ARGN})

    list(GET __ARGN 0 __varname)
    set(__value "${${__varname}}")
    
    unset(${__varname} CACHE)
    set(${__varname} "${__value}")
endmacro()

# Provides an find_path with no caching
macro(find_path_no_cache)
    find_path(${ARGN})
    
    set(__ARGN ${ARGN})

    list(GET __ARGN 0 __varname)
    set(__value "${${__varname}}")
    
    unset(${__varname} CACHE)
    set(${__varname} "${__value}")
endmacro()

# Provides an find_program with no caching
macro(find_program_no_cache)
    find_program(${ARGN})
    
    set(__ARGN ${ARGN})

    list(GET __ARGN 0 __varname)
    set(__value "${${__varname}}")
    
    unset(${__varname} CACHE)
    set(${__varname} "${__value}")
endmacro()

#-------------------------------------------------------------
#-------------------------------------------------------------


# ----- set_library_presuffix ----- #
# This function modifies the library prefixes and suffixes used by
# find_library when finding Matlab libraries. It does not affect scopes
# outside of this file.

function(set_libarch_prefix_suffix)

    if (UNIX AND NOT APPLE)
        set(CMAKE_FIND_LIBRARY_PREFIXES "lib" PARENT_SCOPE)
        set(CMAKE_FIND_LIBRARY_SUFFIXES ".so" ".a" PARENT_SCOPE)
    elseif (APPLE)
        set(CMAKE_FIND_LIBRARY_PREFIXES "lib" PARENT_SCOPE)
        set(CMAKE_FIND_LIBRARY_SUFFIXES ".dylib" ".a" PARENT_SCOPE)
    elseif (WIN32)
        set(CMAKE_FIND_LIBRARY_PREFIXES "lib" PARENT_SCOPE)
        set(CMAKE_FIND_LIBRARY_SUFFIXES ".lib" ".dll" PARENT_SCOPE)
    endif()

endfunction()


# ----- locate_matlab_root -----
#
# Attempt to find the path to the Matlab installation. If successful, sets
# the absolute path in the variable MATLAB_DIR

function(locate_matlab_root)
    if (DEFINED MATLAB_DIR)
        return()
    endif()
    
    # search the path environment variable
    find_program_no_cache(MATLAB_DIR_ matlab PATHS ENV PATH)
    if (MATLAB_DIR_)
        # get the root directory from the full path
        # /path/to/matlab/rootdir/bin/matlab.exe
        get_filename_component(MATLAB_DIR_ ${MATLAB_DIR_} PATH)
        get_filename_component(MATLAB_DIR_ ${MATLAB_DIR_} PATH)
        set(MATLAB_DIR ${MATLAB_DIR_} CACHE STRING "Root matlab directory" FORCE)
        
        return()
    endif()
    
    # --- UNIX/APPLE ---
    if (UNIX)
        # possible root locations, in order of likelihood
        set(SEARCH_DIRS_ /Applications /usr/local /opt/local /usr /opt)
        set(MATLABS Matlab/R20 MATLAB/R20 matlab/R20 Matlab MATLAB matlab)
  
      foreach (MATLABS_S ${MATLABS})
        foreach (DIR_ ${SEARCH_DIRS_})
            file(GLOB MATLAB_DIR_ ${DIR_}/${MATLABS_S}*)
            if (MATLAB_DIR_)
                list(SORT MATLAB_DIR_)
                list(REVERSE MATLAB_DIR_)
                list(GET MATLAB_DIR_ 0 MATLAB_DIR_)
                set(MATLAB_DIR ${MATLAB_DIR_} CACHE STRING "Root matlab directory" FORCE)
              
                return()
            endif()
        endforeach()
      endforeach()
  
    # --- WINDOWS ---
    elseif (WIN32)
  
        # search the registry determine the available Matlab versions
        set(REG_EXTENSION_ "SOFTWARE\\Mathworks\\MATLAB")
        set(REG_ROOTS_ "HKEY_LOCAL_MACHINE" "HKEY_CURRENT_USER")
        foreach(REG_ROOT_ ${REG_ROOTS_})
            execute_process(COMMAND reg query "${REG_ROOT_}\\${REG_EXTENSION_}" OUTPUT_VARIABLE QUERY_RESPONSE_)
            if (QUERY_RESPONSE_)
                string(REGEX MATCHALL "[0-9]\\.[0-9]" VERSION_STRINGS_ ${QUERY_RESPONSE_})
                list(APPEND VERSIONS_ ${VERSION_STRINGS_})
            endif()
        endforeach()
  
        # select the highest version
        list(APPEND VERSIONS_ "0.0")
        list(SORT VERSIONS_)
        list(REVERSE VERSIONS_)
        list(GET VERSIONS_ 0 VERSION_)
  
        # request the MATLABROOT from the registry
        foreach(REG_ROOT_ ${REG_ROOTS_})
            get_filename_component(QUERY_RESPONSE_ [${REG_ROOT_}\\${REG_EXTENSION_}\\${VERSION_};MATLABROOT] ABSOLUTE)
            if (NOT ${QUERY_RESPONSE_} MATCHES "registry$")
                set(MATLAB_DIR ${QUERY_RESPONSE_} CACHE STRING "Matlab root directory" FORCE)
                
                return()
            endif()
        endforeach()
    endif()

endfunction()


# ----- locate_matlab -----
#
# Attempt to find the Matlab components
# (include directory and libraries) under the root. 
# If everything is found,
# sets the variable MATLAB_FOUND to TRUE
function(locate_matlab)

    # attempt to find the Matlab root folder
    locate_matlab_root()
  
    # get the mex shell script
    find_program(MATLAB_MEX_SCRIPT NAMES mex mex.bat 
                                   PATHS ${MATLAB_DIR}/bin 
                                   DOC "Mex script" NO_DEFAULT_PATH)
    
    # get the Matlab executable
    find_program(MATLAB_BIN NAMES matlab matlab.exe
                            PATHS ${MATLAB_DIR}/bin 
                            DOC "Matlab executable" NO_DEFAULT_PATH)

    # get the mex extension
    if (NOT DEFINED MATLAB_MEXEXT)    
        find_file_no_cache(MATLAB_MEXEXT_SCRIPT NAMES mexext mexext.bat 
                                               PATHS ${MATLAB_DIR}/bin 
                                               DOC "Script to get expected mex extension" 
                                               NO_DEFAULT_PATH)
                                 
        execute_process(COMMAND ${MATLAB_MEXEXT_SCRIPT}
                        OUTPUT_VARIABLE MATLAB_MEXEXT_
                        OUTPUT_STRIP_TRAILING_WHITESPACE)

        set(MATLAB_MEXEXT ".${MATLAB_MEXEXT_}" CACHE STRING "Expected extension of mex plugins" FORCE)
    endif()
    
    # map the mexext to an architecture extension
    if (NOT DEFINED MATLAB_ARCH)
        set(ARCHITECTURES_ "maci64" "maci" "glnxa64" "glnx64" "sol64" "sola64" "win32" "win64" )
        foreach(ARCHITECTURE_ ${ARCHITECTURES_})
            if(EXISTS ${MATLAB_DIR}/bin/${ARCHITECTURE_})
                set(MATLAB_ARCH ${ARCHITECTURE_} CACHE STRING "Matlab arch" FORCE)
                break()
            endif()
        endforeach()
    endif()
    
    # get the include path
    find_path(MATLAB_INCLUDE_DIRS mex.h PATHS ${MATLAB_DIR}/*
                                              ${MATLAB_DIR}/*/*
                                              ${MATLAB_DIR}/*/*/*
                                        DOC "Directory with matlab-specific headers")
    include_directories(${MATLAB_INCLUDE_DIRS})

    # get the path to the libraries
    if (NOT DEFINED MATLAB_LIB_DIR)
        if (WIN32)
            set(MATLAB_LIB_DIR ${MATLAB_DIR}/extern/lib/${MATLAB_ARCH}/microsoft
                CACHE STRING "Path where matlab libraries are located" FORCE)
                
            # get the libraries
            set_libarch_prefix_suffix()
            find_library_no_cache(MATLAB_LIB_MX  mx  PATHS ${MATLAB_LIB_DIR} NO_DEFAULT_PATH)
            find_library_no_cache(MATLAB_LIB_MEX mex PATHS ${MATLAB_LIB_DIR} NO_DEFAULT_PATH)
            find_library_no_cache(MATLAB_LIB_MAT mat PATHS ${MATLAB_LIB_DIR} NO_DEFAULT_PATH)
            find_library_no_cache(MATLAB_LIB_ENG eng PATHS ${MATLAB_LIB_DIR} NO_DEFAULT_PATH)

            set(MATLAB_LIBS ${MATLAB_LIB_MX} ${MATLAB_LIB_MEX} 
                            ${MATLAB_LIB_MAT} ${MATLAB_LIB_ENG}
                CACHE STRING "List of matlab libraries" FORCE)
        
            mark_as_advanced(MATLAB_LIB_DIR MATLAB_LIBS)
        elseif(UNIX)
            set(MATLAB_LIB_DIR ${MATLAB_DIR}/bin
                CACHE STRING "Path where matlab libraries are located" FORCE)            
        endif()
    endif()

    if (WIN32)
        add_definitions(-DDLL_EXPORT_SYM=__declspec\(dllexport\))
    endif()
    
    mark_as_advanced(MATLAB_MEX_SCRIPT
                     MATLAB_BIN
                     MATLAB_MEXEXT
                     MATLAB_ARCH
                     MATLAB_INCLUDE_DIRS)    
                     
endfunction()



#----------------------------------------------------------------------------
#----------------------------------------------------------------------------
if (NOT MATLAB_FOUND)
    locate_matlab()

    set(MATLAB_LIST MATLAB_MEX_SCRIPT   
                    MATLAB_INCLUDE_DIRS
                    MATLAB_BIN                                           
                    MATLAB_MEXEXT
                    MATLAB_ARCH)
    if (WIN32)
        set(MATLAB_LIST ${MATLAB_LIST} MATLAB_LIB_DIR MATLAB_LIBS)
    endif()

    find_package_handle_standard_args(Matlab DEFAULT_MSG ${MATLAB_LIST})  
endif()