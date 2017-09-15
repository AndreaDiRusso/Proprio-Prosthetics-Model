PROJECT(rendering_view)
cmake_minimum_required(VERSION 2.8)
include("E:/Google Drive/Code Libraries/Cpp/ucm/cmake/ucm.cmake")

ucm_set_runtime(STATIC)

set(MUJOCO_DIR "E:/mjpro140")
set(ZMQ_DIR "E:/Google Drive/Code Libraries/Cpp/libzmq/bin/x64/Release/v140/dynamic")
set(PROTOBUF_DIR "E:/protobuf/install")
set(IMGSAVE_DIR "E:/Google Drive/Code Libraries/ImageSaverCode")

# list of source files
set(project_files
	simulate.cpp
	mujoco2py_functions.cpp
	mujoco2py_functions.hpp
	mujoco2py.pb.cc
	${IMGSAVE_DIR}/Image.h
	${IMGSAVE_DIR}/Image.cpp
	)
# add this folder to the include
include_directories("./")
include_directories("E:/Google Drive/Code Libraries/Cpp/Eigen-328")
include_directories("E:/Google Drive/Code Libraries/Cpp/libzmq/include")
include_directories(${PROTOBUF_DIR}/include)
include_directories(${MUJOCO_DIR}/include)
include_directories(${IMGSAVE_DIR})
# define the executable
if(WIN32)
 add_executable(renderer WIN32 ${project_files})
else()
 add_executable(renderer ${project_files})
endif()

# in Windows + Visual Studio we need this to make it a console application
if(WIN32)
 if(MSVC)
  set_target_properties(renderer PROPERTIES LINK_FLAGS "/SUBSYSTEM:CONSOLE")
 endif(MSVC)
endif(WIN32)

# install application
set(BIN_DIR ${CMAKE_CURRENT_SOURCE_DIR}/../bin)

target_link_libraries(renderer ${MUJOCO_DIR}/bin/mujoco140.lib)
target_link_libraries(renderer ${MUJOCO_DIR}/bin/glfw3.lib)
target_link_libraries(renderer ${ZMQ_DIR}/libzmq.lib)
target_link_libraries(renderer debug ${PROTOBUF_DIR}/lib/libprotobufd.lib optimized ${PROTOBUF_DIR}/lib/libprotobuf.lib)
install(TARGETS renderer DESTINATION ${BIN_DIR})
