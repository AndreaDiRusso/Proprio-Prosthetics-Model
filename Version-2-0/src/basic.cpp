//---------------------------------//
//  This file is part of MuJoCo    //
//  Written by Emo Todorov         //
//  Copyright (C) 2017 Roboti LLC  //
//---------------------------------//
#include "mujoco.h"
#include "glfw3.h"
#include "zmq.hpp"
#include "stdlib.h"
#include "string.h"
#include "stdio.h"

//Proprio specific includes and define:
#include "mujoco2py_functions.hpp"
#include "basic_config.hpp"

//end Proprio specific

// MuJoCo data structures
mjModel* m = NULL;                  // MuJoCo model
mjData* d = NULL;                   // MuJoCo data
mjvCamera cam;                      // abstract camera
mjvOption opt;                      // visualization options
mjvScene scn;                       // abstract scene
mjrContext con;                     // custom GPU context

// mouse interaction
bool button_left = false;
bool button_middle = false;
bool button_right =  false;
double lastx = 0;
double lasty = 0;

// video output:
// create output rgb file
FILE* fp;
errno_t errOut = fopen_s(&fp, "D:\\tempdata\\Biomechanical Model\\figures\\flywheel_design_out.raw", "wb");

//-------------------------------- IPC variables ------------------------------------

zmq::context_t context(1);
zmq::socket_t incoming_publisher = zmq::socket_t(context, ZMQ_REQ);
zmq::socket_t neuron_publisher = zmq::socket_t(context, ZMQ_REQ);

//-------------------------------- Pose Fitting variables ------------------------------------

std::vector<mjtNum> actuator_length0;
std::vector<mjtNum> tendon_length0;
std::vector<mjtNum> actuator_velocity0;
Eigen::VectorXd solution_pose;
Eigen::VectorXd target_pose = Eigen::VectorXd::Zero(NUM_SITES);

// keyboard callback
void keyboard(GLFWwindow* window, int key, int scancode, int act, int mods)
{
    // backspace: reset simulation
    if( act==GLFW_PRESS && key==GLFW_KEY_BACKSPACE )
    {
        mj_resetData(m, d);
        mj_forward(m, d);
    }
}


// mouse button callback
void mouse_button(GLFWwindow* window, int button, int act, int mods)
{
    // update button state
    button_left =   (glfwGetMouseButton(window, GLFW_MOUSE_BUTTON_LEFT)==GLFW_PRESS);
    button_middle = (glfwGetMouseButton(window, GLFW_MOUSE_BUTTON_MIDDLE)==GLFW_PRESS);
    button_right =  (glfwGetMouseButton(window, GLFW_MOUSE_BUTTON_RIGHT)==GLFW_PRESS);

    // update mouse position
    glfwGetCursorPos(window, &lastx, &lasty);
}


// mouse move callback
void mouse_move(GLFWwindow* window, double xpos, double ypos)
{
    // no buttons down: nothing to do
    if( !button_left && !button_middle && !button_right )
        return;

    // compute mouse displacement, save
    double dx = xpos - lastx;
    double dy = ypos - lasty;
    lastx = xpos;
    lasty = ypos;

    // get current window size
    int width, height;
    glfwGetWindowSize(window, &width, &height);

    // get shift key state
    bool mod_shift = (glfwGetKey(window, GLFW_KEY_LEFT_SHIFT)==GLFW_PRESS ||
                      glfwGetKey(window, GLFW_KEY_RIGHT_SHIFT)==GLFW_PRESS);

    // determine action based on mouse button
    mjtMouse action;
    if( button_right )
        action = mod_shift ? mjMOUSE_MOVE_H : mjMOUSE_MOVE_V;
    else if( button_left )
        action = mod_shift ? mjMOUSE_ROTATE_H : mjMOUSE_ROTATE_V;
    else
        action = mjMOUSE_ZOOM;

    // move camera
    mjv_moveCamera(m, action, dx/height, dy/height, &scn, &cam);
}


// scroll callback
void scroll(GLFWwindow* window, double xoffset, double yoffset)
{
    // emulate vertical mouse motion = 5% of window height
    mjv_moveCamera(m, mjMOUSE_ZOOM, 0, -0.05*yoffset, &scn, &cam);
}


// main function
int main(int argc, const char** argv)
{
    // check command-line arguments
    if( argc!=2 )
    {
        printf(" USAGE:  basic modelfile\n");
        return 0;
    }

    // activate software
    mj_activate("E:\\Google Drive\\Borton Lab\\mjkey.txt");

    // load and compile model
    char error[1000] = "Could not load binary model";
    if( strlen(argv[1])>4 && !strcmp(argv[1]+strlen(argv[1])-4, ".mjb") )
        m = mj_loadModel(argv[1], 0);
    else
        m = mj_loadXML(argv[1], 0, error, 1000);
    if( !m )
        mju_error_s("Load model error: %s", error);

    // make data
    d = mj_makeData(m);

	// save initial lengths of tendons
	for (int a = 0; a < m->nu; a++) {
		actuator_length0.push_back(d->actuator_length[a]);
		tendon_length0.push_back(d->ten_length[a]);
	}

    // init GLFW
    if( !glfwInit() )
        mju_error("Could not initialize GLFW");

    // create window, make OpenGL context current, request v-sync
    GLFWwindow* window = glfwCreateWindow(1200, 900, "Demo", NULL, NULL);
    glfwMakeContextCurrent(window);
    glfwSwapInterval(1);

    // initialize visualization data structures
    mjv_defaultCamera(&cam);
    mjv_defaultOption(&opt);
    mjr_defaultContext(&con);
    mjv_makeScene(&scn, 1000);                   // space for 1000 objects
    mjr_makeContext(m, &con, mjFONTSCALE_100);   // model-specific context


	//change intial camera angle:
	cam.lookat[0] = m->stat.center[0];
	cam.lookat[1] = m->stat.center[1];
	cam.lookat[2] = m->stat.center[2];
	cam.azimuth = 210;
	cam.elevation = 0;
	cam.distance = 1.5 * m->stat.extent;

	// set to free camera
	cam.type = mjCAMERA_FREE;

    // install GLFW mouse and keyboard callbacks
    glfwSetKeyCallback(window, keyboard);
    glfwSetCursorPosCallback(window, mouse_move);
    glfwSetMouseButtonCallback(window, mouse_button);
    glfwSetScrollCallback(window, scroll);

	// Connect ZMQ publishers
	std::cout << "Connecting to python server 1 ..." << std::endl;
	incoming_publisher.connect("tcp://localhost:5556");

	int request_timeout = 1500;
	incoming_publisher.setsockopt(ZMQ_RCVTIMEO, &request_timeout, sizeof(request_timeout));

	std::cout << "Connecting to python server 2 ..." << std::endl;
	neuron_publisher.connect("tcp://localhost:5555");
	neuron_publisher.setsockopt(ZMQ_RCVTIMEO, &request_timeout, sizeof(request_timeout));

	// Start connections to python slaves
	int a = std::system("start python \"C:\\Users\\Radu\\Documents\\GitHub\\Proprio-Prosthetics-Model\\Version-2-0\\src\\ServeSiteCoords.py\" &");
	int b = std::system("start python \"C:\\Users\\Radu\\Documents\\GitHub\\Proprio-Prosthetics-Model\\Version-2-0\\src\\NeuronServer.py\" &");
	mju_copy(d->qpos, m->key_qpos, m->nq * 1);
	// TODO: remove all absolute paths


    // run main loop, target real-time simulation and 60 fps rendering
    while( !glfwWindowShouldClose(window) )
    {
        // advance interactive simulation for 1/60 sec
        //  Assuming MuJoCo can simulate faster than real-time, which it usually can,
        //  this loop will finish on time for the next frame to be rendered at 60 fps.
        //  Otherwise add a cpu timer and exit this loop when it is time to render.
        mjtNum simstart = d->time;
        
		while (d->time - simstart < 1.0 / 60.0) {
			try {
				fitPoseToSites(d, m, &scn, &incoming_publisher, tendons, joints, &solution_pose, &target_pose);
			}
			catch (zmq::error_t e) {
				std::cout << "ERROR";
			}
			mj_step(m, d);
		}
        // get framebuffer viewport
        mjrRect viewport = {0, 0, 0, 0};
        glfwGetFramebufferSize(window, &viewport.width, &viewport.height);

        // update scene and render
        mjv_updateScene(m, d, &opt, NULL, &cam, mjCAT_ALL, &scn);
		// add new targets
		for (int i = 0; i < NUM_SITES / 3; i++) {
			double sz = 30e-3;
			if (i < 8) {
				renderTargetSite(&scn, target_pose(3 * i), target_pose(3 * i + 1), target_pose(3 * i + 2), target_color[3 * i], target_color[3 * i + 1], target_color[3 * i + 2], target_size[i]);
			}
		}
        mjr_render(viewport, &scn, &con);

        // swap OpenGL buffers (blocking call due to v-sync)
        glfwSwapBuffers(window);

        // process pending GUI events, call GLFW callbacks
        glfwPollEvents();

		try {
			//updateNeuron(d, m, &neuron_publisher, tendons, tendon_length0);
		}
		catch (zmq::error_t e) {
			break;
		}
    }

    // close GLFW, free visualization storage
    glfwTerminate();
    mjv_freeScene(&scn);
    mjr_freeContext(&con);

    // free MuJoCo model and data, deactivate
    mj_deleteData(d);
    mj_deleteModel(m);
    mj_deactivate();
	fclose(fp);

    return 1;
}
