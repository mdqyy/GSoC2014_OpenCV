cmake_minimum_required(VERSION 2.8.3) 
project(algorithms)	
    
add_library(structuredEdgeDetection STATIC structuredEdgeDetection/structuredEdgeDetection.cpp)		

#-------------------------------------------------------
#-------------------------------------------------------

set(ALG_LIBS 
			 structuredEdgeDetection
	CACHE INTERNAL "List of libs from algorithm subproject" FORCE)
			 
set(ALG_INCLUDE_DIRS 
					 ${CMAKE_CURRENT_LIST_DIR}/../algorithms/structuredEdgeDetection
	CACHE INTERNAL "List of include directories from algorithm subproject" FORCE)              