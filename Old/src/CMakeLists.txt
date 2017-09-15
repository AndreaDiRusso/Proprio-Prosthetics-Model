PROJECT(Tutorial)
cmake_minimum_required(VERSION 2.8)

set(MUJOCO_DIR "E:/mjpro")
set(ipcdir "E:/Google Drive/Borton Lab/Inter Process Communication/")
set(zmqdir "E:/Google Drive/libzmq/bin/x64/Release/v140/dynamic/")
set(protodir "E:/protobuf/install")
# list of source files
set(project_files
	simulate.cpp
	${ipcdir}mujoco2py.pb.cc
	)
	# add this folder to the include
include_directories("./")
include_directories("E:/Google Drive/MATLAB/Eigen-328")
include_directories("E:/Google Drive/libzmq/include")
include_directories("E:/protobuf/install/include")
include_directories("E:/Google Drive/Borton Lab/Inter Process Communication")
include_directories(${MUJOCO_DIR}/include)
# define the executable
if(WIN32)
 add_executable(tutorial WIN32 ${project_files})
else()
 add_executable(tutorial ${project_files})
endif()

# in Windows + Visual Studio we need this to make it a console application
if(WIN32)
 if(MSVC)
  set_target_properties(tutorial PROPERTIES LINK_FLAGS "/SUBSYSTEM:CONSOLE")
 endif(MSVC)
endif(WIN32)

# install application
set(BIN_DIR ${CMAKE_CURRENT_SOURCE_DIR}/../bin)

target_link_libraries(tutorial ${MUJOCO_DIR}/bin/mujoco130.lib)
target_link_libraries(tutorial ${MUJOCO_DIR}/bin/glfw3.lib)
target_link_libraries(tutorial ${zmqdir}/libzmq.lib)
target_link_libraries(tutorial debug ${protodir}/lib/libprotobufd.lib optimized ${protodir}/lib/libprotobuf.lib)
install(TARGETS tutorial DESTINATION ${BIN_DIR})
