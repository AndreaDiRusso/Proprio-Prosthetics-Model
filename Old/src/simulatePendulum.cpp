//-----------------------------------//
//  This file is part of MuJoCo.     //
//  Copyright 2009-2015 Roboti LLC.  //
//-----------------------------------//

#include "mujoco.h"
#include "glfw3.h"
#include "zmq.hpp"
#include "stdlib.h"
#include "string.h"
#include <mutex>
#include <iostream>
#include <Eigen\Dense>
#include "mujoco2py.pb.h"
//#include <unsupported/Eigen/NonLinearOptimization>
#include <unsupported/Eigen/LevenbergMarquardt>
//#include "unsupported/Eigen/src/LevenbergMarquardt/LevenbergMarquardt.h"

//-------------------------------- global variables -------------------------------------

// synchronization
std::mutex gui_mutex;

//communication
zmq::context_t context(1);
zmq::socket_t publisher = zmq::socket_t(context, ZMQ_REQ);;

// model
mjModel* m = 0;
mjData* d = 0;

std::vector<mjtNum> actuator_length0;
std::vector<mjtNum> actuator_velocity0;

char lastfile[100] = "";

// user state
bool update_cmd = true;
bool paused = false;
bool showoption = false;
bool showinfo = true;
bool showdepth = false;
int showhelp = 1;                   // 0: none; 1: brief; 2: full
int speedtype = 1;                  // 0: slow; 1: normal; 2: max

// abstract visualization
mjvObjects objects;
mjvCamera cam;
mjvOption vopt;
char status[1000] = "";

// OpenGL rendering
mjrContext con;
mjrOption ropt;
double scale = 1;
bool stereoavailable = false;
float depth_buffer[3840*2160];       // big enough for 4K screen
unsigned char depth_rgb[960*540*3];  // 1/4th of screen

// selection and perturbation
bool button_left = false;
bool button_middle = false;
bool button_right =  false;
int lastx = 0;
int lasty = 0;
int selbody = 0;
int perturb = 0;
mjtNum selpos[3] = {0, 0, 0};
mjtNum refpos[3] = {0, 0, 0};
mjtNum refquat[4] = {1, 0, 0, 0};
int needselect = 0;                 // 0: none, 1: select, 2: center 

// help strings
const char help_title[] = 
"Help\n"
"Option\n"
"Info\n"
"Depth map\n"
"Stereo\n"
"Speed\n"
"Pause\n"
"Reset\n"
"Forward\n"
"Back\n"
"Forward 100\n"
"Back 100\n"
"Autoscale\n"
"Reload\n"
"Geoms\n"
"Sites\n"
"Select\n"
"Center\n"
"Zoom\n"
"Camera\n"
"Perturb\n"
"Switch Cam";

const char help_content[] = 
"F1\n"
"F2\n"
"F3\n"
"F4\n"
"F5\n"
"Enter\n"
"Space\n"
"BackSpace\n"
"Right arrow\n"
"Left arrow\n"
"Page Down\n"
"Page Up\n"
"Ctrl A\n"
"Ctrl L\n"
"0 - 4\n"
"Shift 0 - 4\n"
"L double-click\n"
"R double-click\n"
"Scroll or M drag\n"
"[Shift] L/R drag\n"
"Ctrl [Shift] drag\n"
"[ ]";

char opt_title[1000] = "";
char opt_content[1000];

//-------------------------------- IK Solver Testing -----------------------------------------


struct test_functor : Eigen::DenseFunctor<double>
{
	mjModel* model;
	mjData* data;
	mjtNum target_site_xpos[3];

	test_functor(mjModel* m, mjData* d, mjtNum* tgt) : DenseFunctor<double>(1, 9) {
		model = m;
		//std::cout << "dim(qpos) = " << model->nq << std::endl;
		data = d;
		//std::cout << "nsite = " << model->nsite << std::endl;
		target_site_xpos[0] = tgt[0];
		target_site_xpos[1] = tgt[1];
		target_site_xpos[2] = tgt[2];
	}

	int operator()(const Eigen::VectorXd &x, Eigen::VectorXd &fvec) const
	{
		double tmp1, tmp2, tmp3;
		double y[15] = { 1.4e-1, 1.8e-1, 2.2e-1, 2.5e-1, 2.9e-1, 3.2e-1, 3.5e-1,
			3.9e-1, 3.7e-1, 5.8e-1, 7.3e-1, 9.6e-1, 1.34, 2.1, 4.39 };

		for (int i = 0; i < values(); i++)
		{
			tmp1 = i + 1;
			tmp2 = 16 - i - 1;
			tmp3 = (i >= 8) ? tmp2 : tmp1;
			fvec[i] = y[i] - (x[0] + tmp1 / (x[1] * tmp2 + x[2] * tmp3));
		}
		return 0;
	}

	int df(const Eigen::VectorXd &x, Eigen::MatrixXd &fjac) const
	{
		double tmp1, tmp2, tmp3, tmp4;
		for (int i = 0; i < values(); i++)
		{
			tmp1 = i + 1;
			tmp2 = 16 - i - 1;
			tmp3 = (i >= 8) ? tmp2 : tmp1;
			tmp4 = (x[1] * tmp2 + x[2] * tmp3); tmp4 = tmp4*tmp4;
			fjac(i, 0) = -1;
			fjac(i, 1) = tmp1*tmp2 / tmp4;
			fjac(i, 2) = tmp1*tmp3 / tmp4;
		}
		return 0;
	}
};

struct site_functor : Eigen::DenseFunctor<double>
{
	mjModel* _model;
	mjData* _data;
	double _target_site_xpos[100];
	int _n_sites;
	std::vector<std::string> _site_name;

	site_functor(mjModel* m, mjData* d, double* tgt, std::vector<std::string> siteNames) : DenseFunctor<double>(1, 9) {
		_model = new mjModel(*m);
		//std::cout << "inputs = " << inputs() << std::endl;
		_data = new mjData(*d);
		//std::cout << "values = " << values() << std::endl;
		_site_name = siteNames;
		_n_sites = siteNames.size();

		for (int i = 0; i < 3*_n_sites; i++) {
			_target_site_xpos[i] = tgt[i];
		}

	}

	void reset_model() {
	}

	int operator()(const Eigen::VectorXd &x, Eigen::VectorXd &fvec) const
	{
		double current_site_xpos[100];
		int count = 0;
		
		for (int i = 0; i < inputs(); i++) {
			_data->qpos[i] = x[i];
			_data->qvel[i] = 0;
			_data->qacc[i] = 0;
		}

		mj_forward(_model,_data);

		for (int i = 0; i < _n_sites; i++) {
			int siteID = mj_name2id(_model, mjOBJ_SITE, _site_name[i].c_str());
			current_site_xpos[count] = _data->site_xpos[3 * siteID + 0];
			count++;
			current_site_xpos[count] = _data->site_xpos[3 * siteID + 1];
			count++;
			current_site_xpos[count] = _data->site_xpos[3 * siteID + 2];
			count++;
		}

		//Reset Model

		mj_resetData(m, d);

		mj_forward(_model, _data);

		//std::cout << "current_site_xpos = [" << std::endl;
		for (int i = 0; i < values(); i++)
		{
			fvec[i] = _target_site_xpos[i] - current_site_xpos[i];
			// Radu Debugging:
			//std::cout << current_site_xpos[i] << std::endl;
		}
		//std::cout << "]" << std::endl;
		//std::cout << "Fvec = " << std::endl << fvec << std::endl;
		return 0;
	}

	int df(const Eigen::VectorXd &x, Eigen::MatrixXd &fjac) const
	{
		double jacp[100], jacr[100];
		for (int i = 0; i < _n_sites; i++) {

			int siteID = mj_name2id(m, mjOBJ_SITE, _site_name[i].c_str());

			for (int i = 0; i < inputs(); i++) {
				_data->qpos[i] = x[i];
				_data->qvel[i] = 0;
				_data->qacc[i] = 0;
			}

			mj_forward(_model, _data);

			mj_jacSite(_model, _data, jacp, jacr, siteID);

			mj_resetData(m, d);

			mj_forward(_model, _data);

			for (int j = 0; j < values()/_n_sites; j++)
			{
				for (int k = 0; k < inputs(); k++) {

					fjac(3*i+j, k) = -jacp[inputs()*j+k];

				}
			}
		}
		return 0;
	}
};

Eigen::VectorXd testLmder1(double factor)
{
	int n = 1, info;

	Eigen::VectorXd x;

	/* the following starting values provide a rough fit. */
	x.setConstant(n, 0);
	x[0] = 0.785398;
	/*
	for (int j = 0; j < n; j++) {
		x[j] = m->qpos0[j];
	}
	*/
	// do the computation
	double targ[9] = { 0, 0, 0.06,
		0, 0.6 -0.06,
		0.06, 1, 0
	};

	std::vector<std::string> sites;
	sites.push_back("top");
	sites.push_back("mid");
	sites.push_back("bot");
	site_functor functor(m, d, targ, sites);
	Eigen::LevenbergMarquardt<site_functor> lm(functor);

	lm.setFactor(factor);
	//lm.setEpsilon(std::numeric_limits<double>::epsilon());

	//std::cout << "lm.info says: " << lm.info() << std::endl;
	//std::cout << "Starting with a guess of: " << std::endl << x << std::endl;

	info = lm.minimize(x);
	
	// check return value
	//std::cout << "lm.minimize info was " << info << std::endl;
	//std::cout << "Levenberg Computed" << std::endl << x << std::endl;

	return x;
	//std::cout << "Levenberg Check 1" << (lm.nfev() == 6) << std::endl;
	//std::cout << "Levenberg Check 1" << (lm.njev() == 5) << std::endl;
	/*
	// check norm
	//std::cout << "Levenberg Check 1" << (lm.fvec().blueNorm() == 0.09063596) << std::endl;

	// check x
	Eigen::VectorXd x_ref(n);
	x_ref << 0.08241058, 1.133037, 2.343695;
	std::cout << "Levenberg Computed" << std::endl << x << std::endl << "Reference value: " << std::endl << x_ref << std:: endl;
	*/
}

//-------------------------------- Radu Defined Functions ------------------------------------
double tendonMomentArm(mjData* d, mjModel* m, std::string tendon_name, std::string joint_name) {

	int jointID = mj_name2id(m, mjOBJ_JOINT, joint_name.c_str());
	int tendonID = mj_name2id(m, mjOBJ_TENDON, tendon_name.c_str());
	int nv = m->nv;
	double ma = d->ten_moment[nv*tendonID + jointID];
	return ma;
}

double k_ogi(double sai) {
	return 0.32 + 0.71  *exp(-1.112 * (sai - 1)) + sin(3.722 * (sai - 0.656));
}

double h_ogi(double eta) {
	return 1 + tanh(3 * eta);
}

double extended_ctrl(double ctrl, std::string muscle_name) {
	int actID = mj_name2id(m, mjOBJ_ACTUATOR, muscle_name.c_str());
	
	double ext_ctrl = ctrl;

	double L = d->actuator_length[actID];
	double L_bar = actuator_length0[actID];

	double v = d->actuator_velocity[actID];
	//double v_bar = actuator_velocity0[actID];

	double k = k_ogi(L / L_bar);
	double h = h_ogi(v / 3);

	//double gain = m->actuator_gainprm[actID];
	//double PE = 1.6*exp(15 * (L - L_bar) - 1)/gain;

	return k*h*ext_ctrl;
}

double momentArmCalc(Eigen::Vector3d v, Eigen::Vector3d a, Eigen::Vector3d r) {
	Eigen::Vector3d cp = a.cross(r);
	return v.dot(cp);
}

double momentArm(std::string joint, std::string mus1, std::string mus2) {

	int jointID = mj_name2id(m, mjOBJ_JOINT, joint.c_str());

	int mus1ID = mj_name2id(m, mjOBJ_SITE, mus1.c_str());
	int mus2ID = mj_name2id(m, mjOBJ_SITE, mus2.c_str());

	Eigen::Vector3d mus_origin_vec, mus_insertion_vec, joint_vec, joint_dir;

	double ma = -1;

	if (jointID != -1 && mus1ID != -1 && mus2ID != -1) {

		mus_origin_vec = Eigen::Vector3d(d->geom_xpos[3 * mus1ID], d->geom_xpos[3 * mus1ID + 1], d->geom_xpos[3 * mus1ID + 2]);

		mus_insertion_vec = Eigen::Vector3d(d->geom_xpos[3 * mus2ID], d->geom_xpos[3 * mus2ID + 1], d->geom_xpos[3 * mus2ID + 2]);

		joint_vec = Eigen::Vector3d(d->xanchor[3 * jointID], d->xanchor[3 * jointID + 1], d->xanchor[3 * jointID + 2]);
		joint_dir = Eigen::Vector3d(d->xaxis[3 * jointID], d->xaxis[3 * jointID + 1], d->xaxis[3 * jointID + 2]);

		Eigen::Vector3d v = (mus_insertion_vec - mus_origin_vec);
		v /= v.norm();

		ma = momentArmCalc(v, joint_dir, joint_vec -mus_insertion_vec);
	}

	return ma;
}

void cycleJoint(mjData* d, mjModel* m, std::string tendon_name, std::string joint_name, zmq::socket_t *publisher) {
	
	double ma_model = tendonMomentArm(d, m, tendon_name, joint_name);

	// protocol buffer stuff
	mujoco2py::mujoco_msg to_msg;
	mujoco2py::mujoco_msg from_msg;

	to_msg.set_tnd_ma(ma_model);
	to_msg.set_tnd_name(tendon_name);
	to_msg.set_jnt_name(joint_name);
	std::string msg_str;
	to_msg.SerializeToString(&msg_str);
	
	//  Send message to all subscribers
	zmq::message_t message(msg_str.length());
	memcpy(message.data(), msg_str.c_str(), msg_str.length());
	(*publisher).send(message);
	std::cout << "moment arm was: " << ma_model << std::endl;

	//  Get the reply.
	zmq::message_t reply;
	(*publisher).recv(&reply);

	from_msg.ParseFromArray(reply.data(), reply.size());
	double reply_dbl = from_msg.jnt_qpos();

	int jointID = mj_name2id(m, mjOBJ_JOINT, joint_name.c_str());
	d->qpos[jointID] = reply_dbl;

	std::cout << "The reply was: " << reply_dbl << std::endl;
}

void cycleMuscle(mjData* d, mjModel* m, std::string muscle_name, std::string tendon_name, zmq::socket_t *publisher) {
	// protocol buffer stuff
	mujoco2py::mujoco_msg to_msg;
	mujoco2py::mujoco_msg from_msg;
	int actID = mj_name2id(m, mjOBJ_ACTUATOR, muscle_name.c_str());
	int tendID = mj_name2id(m, mjOBJ_TENDON, tendon_name.c_str());

	to_msg.set_mus_name(muscle_name);
	to_msg.set_tnd_name(tendon_name);
	to_msg.set_mus_force(d->actuator_force[actID]);
	to_msg.set_tnd_len(d->ten_length[tendID]);

	std::string msg_str;
	to_msg.SerializeToString(&msg_str);

	//  Send message to all subscribers
	zmq::message_t message(msg_str.length());
	memcpy(message.data(), msg_str.c_str(), msg_str.length());
	(*publisher).send(message);

	//  Get the reply.
	zmq::message_t reply;
	(*publisher).recv(&reply);

	from_msg.ParseFromArray(reply.data(), reply.size());
	double reply_dbl = from_msg.mus_ctrl();

	double ext_ctrl = extended_ctrl(reply_dbl, muscle_name);
	d->ctrl[actID] = ext_ctrl;

	std::cout << "The reply was: " << reply_dbl << std::endl;
}
//-------------------------------- utility functions ------------------------------------

// center and scale view
void autoscale(GLFWwindow* window)
{
    // autoscale
	cam.azimuth = 45;
	cam.elevation = 10;
	cam.fovy = 25;
    cam.lookat[0] = m->stat.center[0];
    cam.lookat[1] = m->stat.center[1];
    cam.lookat[2] = m->stat.center[2];
    cam.distance = 1.15 * m->stat.extent;
    cam.camid = -1;
    cam.trackbodyid = -1;
    if( window )
    {
        int width, height;
        glfwGetFramebufferSize(window, &width, &height);
        mjv_updateCameraPose(&cam, (mjtNum)width/(mjtNum)height);
    }
}

// load mjb or xml model
void loadmodel(GLFWwindow* window, const char* filename)
{
    // load and compile
    char error[1000] = "could not load binary model";
    mjModel* mnew = 0;
    if( strlen(filename)>4 && !strcmp(filename+strlen(filename)-4, ".mjb") )
        mnew = mj_loadModel(filename, 0, 0);
    else
        mnew = mj_loadXML(filename, error);
    if( !mnew )
    {
        printf("%s\n", error);
        return;
    }

    // delete old model, assign new
    mj_deleteData(d);
    mj_deleteModel(m);
    m = mnew;
    d = mj_makeData(m);

	//d->qpos = m->qpos0;

	//actuator_velocity0 = new mjtNum(*(d->actuator_velocity));

    mj_forward(m, d);
	for (int a = 0; a < m->nu; a++) {
		actuator_length0.push_back(d->actuator_length[a]);
	}

    // save filename for reload
    strcpy(lastfile, filename);

    // re-create custom context
    mjr_makeContext(m, &con, 150);

    // clear perturbation state
    perturb = 0;
    selbody = 0;
    needselect = 0;

    // set title
    if( window && m->names )
        glfwSetWindowTitle(window, m->names);

    // center and scale view
    autoscale(window);
}


//--------------------------------- callbacks -------------------------------------------

// keyboard
void keyboard(GLFWwindow* window, int key, int scancode, int act, int mods)
{
    int n;

    // require model
    if( !m )
        return;

    // do not act on release
    if( act==GLFW_RELEASE )
        return;

    gui_mutex.lock();

    switch( key )
    {
    case GLFW_KEY_F1:                   // help
        showhelp++;
        if( showhelp>2 )
            showhelp = 0;
        break;

    case GLFW_KEY_F2:                   // option
        showoption = !showoption;
        break;

    case GLFW_KEY_F3:                   // info
        showinfo = !showinfo;
        break;

    case GLFW_KEY_F4:                   // depthmap
        showdepth = !showdepth;
        break;

    case GLFW_KEY_F5:                   // stereo
        if( stereoavailable )
            ropt.stereo = !ropt.stereo;
        break;

    case GLFW_KEY_ENTER:                // speed
        speedtype += 1;
        if( speedtype>2 )
            speedtype = 0;
        break;

    case GLFW_KEY_SPACE:                // pause
        paused = !paused;
		update_cmd = !update_cmd;
        break;

    case GLFW_KEY_BACKSPACE:            // reset
        mj_resetData(m, d);
        mj_forward(m, d);
        break;

    case GLFW_KEY_RIGHT:                // step forward
        if( paused )
            mj_step(m, d);
        break;

    case GLFW_KEY_LEFT:                 // step back
        if( paused )
        {
            m->opt.timestep = -m->opt.timestep;
            mj_step(m, d);
            m->opt.timestep = -m->opt.timestep;
        }
        break;

    case GLFW_KEY_PAGE_DOWN:            // step forward 100
        if( paused )
            for( n=0; n<100; n++ )
                mj_step(m,d);
        break;

    case GLFW_KEY_PAGE_UP:              // step back 100
        if( paused )
        {
            m->opt.timestep = -m->opt.timestep;
            for( n=0; n<100; n++ )
                mj_step(m,d);
            m->opt.timestep = -m->opt.timestep;
        }
        break;

    case GLFW_KEY_LEFT_BRACKET:         // previous camera
        if( cam.camid>-1 )
            cam.camid--;
        break;

    case GLFW_KEY_RIGHT_BRACKET:        // next camera
        if( cam.camid<m->ncam-1 )
            cam.camid++;
        break;

    default:
        // control keys
        if( mods & GLFW_MOD_CONTROL )
        {
            if( key==GLFW_KEY_A )
                autoscale(window);
            else if( key==GLFW_KEY_L && lastfile[0] )
                loadmodel(window, lastfile);

            break;
        }

        // toggle visualization flag
        for( int i=0; i<mjNVISFLAG; i++ )
            if( key==mjVISSTRING[i][2][0] )
                vopt.flags[i] = !vopt.flags[i];

        // toggle rendering flag
        for( int i=0; i<mjNRNDFLAG; i++ )
            if( key==mjRNDSTRING[i][2][0] )
                ropt.flags[i] = !ropt.flags[i];

        // toggle geom/site group
        for( int i=0; i<mjNGROUP; i++ )
            if( key==i+'0')
            {
                if( mods & GLFW_MOD_SHIFT )
                    vopt.sitegroup[i] = !vopt.sitegroup[i];
                else
                    vopt.geomgroup[i] = !vopt.geomgroup[i];
            }
    }

    gui_mutex.unlock();
}


// mouse button
void mouse_button(GLFWwindow* window, int button, int act, int mods)
{
    // past data for double-click detection
    static int lastbutton = 0;
    static double lastclicktm = 0;

    // update button state
    button_left =   (glfwGetMouseButton(window, GLFW_MOUSE_BUTTON_LEFT)==GLFW_PRESS);
    button_middle = (glfwGetMouseButton(window, GLFW_MOUSE_BUTTON_MIDDLE)==GLFW_PRESS);
    button_right =  (glfwGetMouseButton(window, GLFW_MOUSE_BUTTON_RIGHT)==GLFW_PRESS);

    // update mouse position
    double x, y;
    glfwGetCursorPos(window, &x, &y);
    lastx = (int)(scale*x);
    lasty = (int)(scale*y);

    // require model
    if( !m )
        return;

    gui_mutex.lock();

    // set perturbation
    int newperturb = 0;
    if( (mods & GLFW_MOD_CONTROL) && selbody>0 )
    {
        // right: translate;  left: rotate
        if( button_right )
            newperturb = mjPERT_TRANSLATE;
        else if( button_left )
            newperturb = mjPERT_ROTATE;

        // perturbation onset: reset reference
        if( newperturb && !perturb )
        {
            int id = paused ? m->body_rootid[selbody] : selbody;
            mju_copy3(refpos, d->xpos+3*id);
            mju_copy(refquat, d->xquat+4*id, 4);
        }
    }
    perturb = newperturb;

    // detect double-click (250 msec)
    if( act==GLFW_PRESS && glfwGetTime()-lastclicktm<0.25 && button==lastbutton )
    {
        if( button==GLFW_MOUSE_BUTTON_LEFT )
            needselect = 1;
        else
            needselect = 2;

        // stop perturbation on select
        perturb = 0;
    }

    // save info
    if( act==GLFW_PRESS )
    {
        lastbutton = button;
        lastclicktm = glfwGetTime();
    }

    gui_mutex.unlock();
}


// mouse move
void mouse_move(GLFWwindow* window, double xpos, double ypos)
{
    // no buttons down: nothing to do
    if( !button_left && !button_middle && !button_right )
        return;

    // compute mouse displacement, save
    float dx = (int)(scale*xpos) - (float)lastx;
    float dy = (int)(scale*ypos) - (float)lasty;
    lastx = (int)(scale*xpos);
    lasty = (int)(scale*ypos);

    // require model
    if( !m )
        return;

    // get current window size
    int width, height;
    glfwGetFramebufferSize(window, &width, &height);

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

    gui_mutex.lock();

    // perturbation
    if( perturb )
    {
        if( selbody>0 )
            mjv_moveObject(action, dx, dy, &cam.pose, 
                           (float)width, (float)height, refpos, refquat);
    }

    // camera control
    else
        mjv_moveCamera(action, dx, dy, &cam, (float)width, (float)height);

    gui_mutex.unlock();
}


// scroll
void scroll(GLFWwindow* window, double xoffset, double yoffset)
{
    // require model
    if( !m )
        return;

    // get current window size
    int width, height;
    glfwGetFramebufferSize(window, &width, &height);

    // scroll
    gui_mutex.lock();
    mjv_moveCamera(mjMOUSE_ZOOM, 0, (float)(-20*yoffset), &cam, (float)width, (float)height);
    gui_mutex.unlock();
}


// drop
void drop(GLFWwindow* window, int count, const char** paths)
{
    // make sure list is non-empty
    if( count>0 )
    {
        gui_mutex.lock();
        loadmodel(window, paths[0]);
        gui_mutex.unlock();
    }
}


//-------------------------------- simulation and rendering -----------------------------

// make option string
void makeoptionstring(const char* name, char key, char* buf)
{
    int i=0, cnt=0;

    // copy non-& characters
    while( name[i] && i<50 )
    {
        if( name[i]!='&' )
            buf[cnt++] = name[i];

        i++;
    }

    // finish
    buf[cnt] = ' ';
    buf[cnt+1] = '(';
    buf[cnt+2] = key;
    buf[cnt+3] = ')';
    buf[cnt+4] = 0;
}


// advance simulation
void advance(void)
{
    // perturbations
    if( selbody>0 )
    {
        // fixed object: edit
        if( m->body_jntnum[selbody]==0 && m->body_parentid[selbody]==0 )
            mjv_mouseEdit(m, d, selbody, perturb, refpos, refquat);
    
        // movable object: set mouse perturbation
        else
            mjv_mousePerturb(m, d, selbody, perturb, refpos, refquat, 
                             d->xfrc_applied+6*selbody);
    }

    // advance simulation
    mj_step(m, d);

    // clear perturbation
    if( selbody>0 )
        mju_zero(d->xfrc_applied+6*selbody, 6);
}


// render
void render(GLFWwindow* window)
{
    // past data for FPS calculation
    static double lastrendertm = 0;

    // get current window rectangle
    mjrRect rect = {0, 0, 0, 0};
    glfwGetFramebufferSize(window, &rect.width, &rect.height);

    double duration = 0;
    gui_mutex.lock();

    // no model: empty screen
    if( !m )
    {
		mjr_rectangle(1, rect, 0, 0, rect.width, rect.height, 
			0.2, 0.3, 0.4, 1);
        mjr_overlay(rect, mjGRID_TOPLEFT, 0, "Drag-and-drop model file here", 0, &con);
        gui_mutex.unlock();
        return;
    }

    // start timers
    double starttm = glfwGetTime();
    mjtNum startsimtm = d->time;

    // paused
    if( paused )
    {
        // edit
        mjv_mouseEdit(m, d, selbody, perturb, refpos, refquat);

        // recompute to refresh rendering
        mj_forward(m, d);

        // 15 msec delay
        while( glfwGetTime()-starttm<0.015 );
    }

    // running
    else
    {
        // simulate for 15 msec of CPU time
        int n = 0;
        while( glfwGetTime()-starttm<0.015 )
        {
            // step at specified speed
            if( (speedtype==0 && n==0) || (speedtype==1 && d->time-startsimtm<0.016) || speedtype==2 )
            {
                advance();
                n++;
            }

            // simulation already done: compute duration
            else if( duration==0 && n )
                duration = 1000*(glfwGetTime() - starttm)/n;

        }

        // compute duration if not already computed
        if( duration==0 && n )
            duration = 1000*(glfwGetTime() - starttm)/n;
    }

    // update simulation statistics
    if( !paused )
        sprintf(status, "%.1f\n%d (%d)\n%.2f\n%.0f          \n%.2f\n%.2f\n%d",
                d->time, d->nefc, d->ncon, 
                duration, 1.0/(glfwGetTime()-lastrendertm),
                d->energy[0]+d->energy[1],
                mju_log10(mju_max(mjMINVAL,
                                  mju_abs(d->solverstat[0]-d->solverstat[1]) /
                                  mju_max(mjMINVAL,mju_abs(d->solverstat[0])+mju_abs(d->solverstat[1])))),
                cam.camid );
    lastrendertm = glfwGetTime();

    // create geoms and lights
    mjv_makeGeoms(m, d, &objects, &vopt, mjCAT_ALL, selbody, 
                  (perturb & mjPERT_TRANSLATE) ? refpos : 0, 
                  (perturb & mjPERT_ROTATE) ? refquat : 0, selpos); 
    mjv_makeLights(m, d, &objects);

    // update camera
    mjv_setCamera(m, d, &cam);
    mjv_updateCameraPose(&cam, (mjtNum)rect.width/(mjtNum)rect.height);

    // selection
    if( needselect )
    {
        // find selected geom
        mjtNum pos[3];
        int selgeom = mjr_select(rect, &objects, lastx, rect.height - lasty, 
                                 pos, 0, &ropt, &cam.pose, &con);

        // set lookat point
        if( needselect==2 )
        {
            if( selgeom >= 0 )
                mju_copy3(cam.lookat, pos);
        }

        // set body selection
        else
        {
            if( selgeom>=0 && objects.geoms[selgeom].objtype==mjOBJ_GEOM )
            {
                // record selection
                selbody = m->geom_bodyid[objects.geoms[selgeom].objid];

                // clear if invalid
                if( selbody<0 || selbody>=m->nbody )
                    selbody = 0;

                // otherwise compute selpos
                else
                {
                    mjtNum tmp[3];
                    mju_sub3(tmp, pos, d->xpos+3*selbody);
                    mju_mulMatTVec(selpos, d->xmat+9*selbody, tmp, 3, 3);
                }
            }
            else
                selbody = 0;
        }

        needselect = 0;
    }

    // render rgb
    mjr_render(0, rect, &objects, &ropt, &cam.pose, &con);

    // show depth map
    if( showdepth )
    {
        // get the depth buffer
        mjr_getBackbuffer(0, depth_buffer, rect, &con);

        // convert to RGB, subsample by 4
        for( int r=0; r<rect.height; r+=4 )
            for( int c=0; c<rect.width; c+=4 )
            {
                // get subsampled address
                int adr = (r/4)*(rect.width/4) + c/4;

                // assign rgb
                depth_rgb[3*adr] = depth_rgb[3*adr+1] = depth_rgb[3*adr+2] = 
                    (unsigned char)((1.0f-depth_buffer[r*rect.width+c])*255.0f);
            }

        // show in bottom-right corner
        mjr_showBuffer(depth_rgb, rect.width/4, rect.height/4, (3*rect.width)/4, 0, &con);
    }

    // show overlays
    if( showhelp==1 )
        mjr_overlay(rect, mjGRID_TOPLEFT, 0, "Help  ", "F1  ", &con);
    else if( showhelp==2 )
        mjr_overlay(rect, mjGRID_TOPLEFT, 0, help_title, help_content, &con);

    if( showinfo )
    {
        if( paused )
            mjr_overlay(rect, mjGRID_BOTTOMLEFT, 0, "PAUSED", 0, &con);
        else
            mjr_overlay(rect, mjGRID_BOTTOMLEFT, 0, 
                "Time\nSize\nCPU\nFPS\nEngy\nStat\nCam", status, &con);
    }

    if( showoption )
    {
        int i;
        char buf[100];

        // fill titles on first pass
        if( !opt_title[0] )
        {
            for( i=0; i<mjNRNDFLAG; i++)
            {
                makeoptionstring(mjRNDSTRING[i][0], mjRNDSTRING[i][2][0], buf);
                strcat(opt_title, buf);
                strcat(opt_title, "\n");
            }
            for( i=0; i<mjNVISFLAG; i++)
            {
                makeoptionstring(mjVISSTRING[i][0], mjVISSTRING[i][2][0], buf);
                strcat(opt_title, buf);
                if( i<mjNVISFLAG-1 )
                    strcat(opt_title, "\n");
            }
        }

        // fill content
        opt_content[0] = 0;
        for( i=0; i<mjNRNDFLAG; i++)
        {
            strcat(opt_content, ropt.flags[i] ? " + " : "   ");
            strcat(opt_content, "\n");
        }
        for( i=0; i<mjNVISFLAG; i++)
        {
            strcat(opt_content, vopt.flags[i] ? " + " : "   ");
            if( i<mjNVISFLAG-1 )
                strcat(opt_content, "\n");
        }

        // show
        mjr_overlay(rect, mjGRID_TOPRIGHT, 0, opt_title, opt_content, &con);
    }

    gui_mutex.unlock();
}


//-------------------------------- main function ----------------------------------------

int main(int argc, const char** argv)
{
	// activate MuJoCo license
	mj_activate("E:\\mjpro\\mjkey.txt");

	// activate python server
	int a = std::system("start python \"E:\\Google Drive\\Borton Lab\\Inter Process Communication\\CycleMuscleServer.py\" &");
	
	// init GLFW, set multisampling
    if (!glfwInit())
        return 1;
    glfwWindowHint(GLFW_SAMPLES, 4);

    // try stereo if refresh rate is at least 100Hz
    GLFWwindow* window = 0;
    if( glfwGetVideoMode(glfwGetPrimaryMonitor())->refreshRate>=100 )
    {
        glfwWindowHint(GLFW_STEREO, 1);
        window = glfwCreateWindow(1200, 900, "Simulate", NULL, NULL);
        if( window )
            stereoavailable = true;
    }

    // no stereo: try mono
    if( !window )
    {
        glfwWindowHint(GLFW_STEREO, 0);
        window = glfwCreateWindow(1200, 900, "Simulate", NULL, NULL);
    }
    if( !window )
    {
        glfwTerminate();
        return 1;
    }
    glfwMakeContextCurrent(window);

    // determine retina scaling
    int width, width1, height;
    glfwGetFramebufferSize(window, &width, &height);
    glfwGetWindowSize(window, &width1, &height);
    scale = (double)width/(double)width1;

    // init MuJoCo rendering
    mjv_makeObjects(&objects, 1000);
    mjv_defaultCamera(&cam);
    mjv_defaultOption(&vopt);
    mjr_defaultOption(&ropt);
    mjr_defaultContext(&con);
    mjr_makeContext(m, &con, 150);

    // load model if filename given as argument
    if( argc==2 )
        loadmodel(window, argv[1]);

    // set GLFW callbacks
    glfwSetKeyCallback(window, keyboard);
    glfwSetCursorPosCallback(window, mouse_move);
    glfwSetMouseButtonCallback(window, mouse_button);
    glfwSetScrollCallback(window, scroll);
    glfwSetDropCallback(window, drop);

    // print version
    printf("MuJoCo Pro version %.2lf\n\n", mj_version());

	std::cout << "Connecting to python server ..." << std::endl;
	publisher.connect("tcp://localhost:5556");

	// get a target configuration

	//Reset Model
	
	mj_resetData(m, d);
	d->qpos[0] = 90*2*M_PI/360;
	mj_forward(m, d);
	/*
	std::string target = "top";
	int siteID = mj_name2id(m, mjOBJ_SITE, target.c_str());
	std::cout << "top: " << d->site_xpos[3 * siteID + 0] << "  " << d->site_xpos[3 * siteID + 1] << "  " << d->site_xpos[3 * siteID + 2] << std::endl;

	target = "mid";
	siteID = mj_name2id(m, mjOBJ_SITE, target.c_str());
	std::cout << "mid: " << d->site_xpos[3 * siteID + 0] << "  " << d->site_xpos[3 * siteID + 1] << "  " << d->site_xpos[3 * siteID + 2] << std::endl;

	target = "bot";
	siteID = mj_name2id(m, mjOBJ_SITE, target.c_str());
	std::cout << "bot: " << d->site_xpos[3 * siteID + 0] << "  " << d->site_xpos[3 * siteID + 1] << "  " << d->site_xpos[3 * siteID + 2] << std::endl;
	*/
	//Levenberg Test
	
	Eigen::VectorXd x = testLmder1(200);
		
	// main loop
    while( !glfwWindowShouldClose(window) )
    {
        // simulate and render
        render(window);
		
		if (update_cmd) {
			d->qpos[0] = x[0];
			//cycleJoint(d, m, "r_GAS", "right_ankle_x", &publisher);
			//cycleMuscle(d, m, "r_GAS", "r_GAS", &publisher);
		}
		
        // finalize
        glfwSwapBuffers(window);
        glfwPollEvents();
    }

    // delete everything we allocated
    mj_deleteData(d);
    mj_deleteModel(m);
    mjr_freeContext(&con);
    mjv_freeObjects(&objects);

    // terminate
    glfwTerminate();
	mj_deactivate();
    return 0;
}
