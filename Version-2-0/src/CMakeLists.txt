PROJECT(version_2_0)
cmake_minimum_required(VERSION 2.8)
include("E:/Google Drive/Code Libraries/Cpp/ucm/cmake/ucm.cmake")
# TODO: remove absolute paths
ucm_set_runtime(STATIC)
set(MUJOCO_DIR "E:/mjpro150")
set(ZMQ_DIR "E:/Google Drive/Code Libraries/Cpp/zeromq-4.2.2/bin/x64/Release/v140/dynamic")
set(PROTOBUF_DIR "E:/protobuf/install")

# list of source files
set(project_files
	basic.cpp
	basic_config.hpp
	mujoco2py_functions.cpp
	mujoco2py_functions.hpp
	functions_config.hpp
	mujoco2py.pb.cc
	)
# add this folder to the include
include_directories("./")
include_directories("E:/Google Drive/Code Libraries/Cpp/eigen-eigen-5a0156e40feb")
include_directories("E:/Google Drive/Code Libraries/Cpp/zeromq-4.2.2/include")
include_directories(${PROTOBUF_DIR}/include)
include_directories(${MUJOCO_DIR}/include)
# define the executable
if(WIN32)
 add_executable(proprioModel WIN32 ${project_files})
 set_property(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR} PROPERTY VS_STARTUP_PROJECT proprioModel)
else()
 add_executable(proprioModel ${project_files})
endif()

# in Windows + Visual Studio we need this to make it a console application
if(WIN32)
 if(MSVC)
  set_target_properties(proprioModel PROPERTIES LINK_FLAGS "/SUBSYSTEM:CONSOLE")
 endif(MSVC)
endif(WIN32)

# install application
set(BIN_DIR ${CMAKE_CURRENT_SOURCE_DIR}/../bin)

target_link_libraries(proprioModel ${MUJOCO_DIR}/bin/mujoco150.lib)
target_link_libraries(proprioModel ${MUJOCO_DIR}/bin/glfw3.lib)
target_link_libraries(proprioModel ${ZMQ_DIR}/libzmq.lib)
target_link_libraries(proprioModel debug ${PROTOBUF_DIR}/lib/libprotobufd.lib optimized ${PROTOBUF_DIR}/lib/libprotobuf.lib)
install(TARGETS proprioModel DESTINATION ${BIN_DIR})
