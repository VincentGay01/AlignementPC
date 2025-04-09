# CMakeList.txt : projet CMake pour alignementPly, incluez la source et définissez
# la logique spécifique au projet ici.
#
cmake_minimum_required (VERSION 3.8)

# Activez Rechargement à chaud pour les compilateurs MSVC si cela est pris en charge.
if (POLICY CMP0141)
  cmake_policy(SET CMP0141 NEW)
  set(CMAKE_MSVC_DEBUG_INFORMATION_FORMAT "$<IF:$<AND:$<C_COMPILER_ID:MSVC>,$<CXX_COMPILER_ID:MSVC>>,$<$<CONFIG:Debug,RelWithDebInfo>:EditAndContinue>,$<$<CONFIG:Debug,RelWithDebInfo>:ProgramDatabase>>")
endif()
include(C:/vcpkg/scripts/buildsystems/vcpkg.cmake)
project ("alignementPly")
set(Boost_DEBUG ON)
set(Boost_USE_STATIC_LIBS ON)
set(Boost_USE_STATIC ON)
find_package(Boost REQUIRED COMPONENTS signals2)

find_package(OpenMP REQUIRED)

find_package(PCL 1.14 REQUIRED)
include_directories(${PCL_INCLUDE_DIRS})
link_directories(${PCL_LIBRARY_DIRS})
add_definitions(${PCL_DEFINITIONS})
find_package(CGAL REQUIRED)
# Ajoutez une source à l'exécutable de ce projet.
add_executable (alignementPly "alignementPly.cpp" )
target_link_libraries (alignementPly ${PCL_LIBRARIES} CGAL OpenMP::OpenMP_CXX Boost::signals2)
if (CMAKE_VERSION VERSION_GREATER 3.12)
  set_property(TARGET alignementPly PROPERTY CXX_STANDARD 20)
endif()

# TODO: Ajoutez des tests et installez des cibles si nécessaire.
