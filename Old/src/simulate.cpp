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
zmq::socket_t publisher = zmq::socket_t(context, ZMQ_REQ);
zmq::socket_t neuron_publisher = zmq::socket_t(context, ZMQ_REQ);

// model
mjModel* m = 0;
mjData* d = 0;

std::vector<mjtNum> actuator_length0;
std::vector<mjtNum> tendon_length0;
std::vector<mjtNum> actuator_velocity0;
Eigen::VectorXd solution_pose;

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

	test_functor(mjModel* m, mjData* d, mjtNum* tgt) : DenseFunctor<double>(6, 3) {
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
	mjModel* _model = new mjModel();
	mjData* _data = new mjData();
	double _target_site_xpos[100];
	int _n_sites;

	std::vector<std::string> _site_name;

	site_functor(mjModel* m, mjData* d, double* tgt, std::vector<std::string> siteNames) : DenseFunctor<double>(6, 24) {
		//
		//_model = new mjModel();
		_model = NULL;
		_model = mj_copyModel(_model, m);
		//std::cout << "inputs = " << inputs() << std::endl;

		_data = NULL;
		_data = mj_makeData(_model);
		//std::cout << "values = " << values() << std::endl;
		_site_name = siteNames;
		_n_sites = siteNames.size();

		//std::cout << "tgt = [" << std::endl;
		for (int i = 0; i < 3*_n_sites; i++) {
			_target_site_xpos[i] = tgt[i];
			// Radu Debugging:
			//std::cout << tgt[i] << std::endl;
		}
		//std::cout << "]" << std::endl;

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
			int siteID = mj_name2id(m, mjOBJ_SITE, _site_name[i].c_str());
			current_site_xpos[count] = _data->site_xpos[3 * siteID + 0];
			count++;
			current_site_xpos[count] = _data->site_xpos[3 * siteID + 1];
			count++;
			current_site_xpos[count] = _data->site_xpos[3 * siteID + 2];
			count++;
		}

		//Reset Model
		//mj_resetData(_model, _data);
		//mj_forward(_model, _data);

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

			//Reset Model
			//mj_resetData(_model, _data);
			//mj_forward(_model, _data);

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

Eigen::VectorXd testLmder1(double *targ, Eigen::VectorXd x, std::vector<std::string> sites)
{
	int n = 6, info;

	// do the computation

	site_functor functor(m, d, targ, sites);
	Eigen::LevenbergMarquardt<site_functor> lm(functor);

	double factor = 200;
	lm.setFactor(factor);
	//lm.setEpsilon(std::numeric_limits<double>::epsilon());

	//std::cout << "lm.info says: " << lm.info() << std::endl;
	//std::cout << "Starting with a guess of: " << std::endl << x << std::endl;

	info = lm.minimize(x);

	// check return value
	//std::cout << "lm.minimize info was " << info << std::endl;
	//std::cout << "Levenberg Computed" << std::endl << x << std::endl;

	return x;
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

	mujoco2py::mujoco_msg_mj_tend *tendon_msg = to_msg.add_tend();
	(*tendon_msg).set_ma(ma_model);
	(*tendon_msg).set_name(tendon_name);
	mujoco2py::mujoco_msg_mj_joint *joint_msg = to_msg.add_joint();
	(*joint_msg).set_name(joint_name);

	std::string msg_str;
	to_msg.SerializeToString(&msg_str);

	//  Send message to all subscribers
	zmq::message_t message(msg_str.length());
	memcpy(message.data(), msg_str.c_str(), msg_str.length());
	(*publisher).send(message);
	//std::cout << "moment arm was: " << ma_model << std::endl;

	//  Get the reply.
	zmq::message_t reply;
	(*publisher).recv(&reply);

	from_msg.ParseFromArray(reply.data(), reply.size());

	for (int i = 0; i < from_msg.joint_size(); i++) {
		std::string current_joint = from_msg.joint(i).name();
		int jointID = mj_name2id(m, mjOBJ_JOINT, current_joint.c_str());
		d->qpos[jointID] = from_msg.joint(i).qpos();
		//std::cout << "The reply moved joint " << from_msg.joint(i).name() << " to position " << from_msg.joint(i).qpos() << std::endl;
	}
}

void SweepMomentArms(mjData* d, mjModel* m, std::vector<std::string> *pair_names, int num_positions, double* start_pos, double* end_pos, zmq::socket_t *publisher) {
	// protocol buffer stuff
	mujoco2py::mujoco_msg to_msg;
	mujoco2py::mujoco_msg from_msg;
	mujoco2py::general_msg from_gen_msg;
	mujoco2py::general_msg init_msg;
	mujoco2py::general_msg end_msg;

	mujoco2py::mujoco_msg_mj_tend *tendon_msg;
	mujoco2py::mujoco_msg_mj_joint *joint_msg;

	std::vector<std::string>::iterator jnt_it = pair_names[1].begin();

	init_msg.set_instruction("begin");
	init_msg.add_value(num_positions);
	init_msg.add_value(pair_names[1].size());
	std::string init_msg_str;
	init_msg.SerializeToString(&init_msg_str);
	//  Send message to all subscribers
	zmq::message_t init_msg_zmq(init_msg_str.length());
	memcpy(init_msg_zmq.data(), init_msg_str.c_str(), init_msg_str.length());
	(*publisher).send(init_msg_zmq);
	//  Get the reply.
	zmq::message_t reply;
	(*publisher).recv(&reply);

	int i = 0;
	for (std::vector<std::string>::iterator tnd_it = pair_names[0].begin(); tnd_it < pair_names[0].end(); tnd_it++, jnt_it++, i++)
	{
		mujoco2py::general_msg next_joint_msg;

		next_joint_msg.set_instruction("next");
		next_joint_msg.add_value(start_pos[i]);
		next_joint_msg.add_value(end_pos[i]);

		std::string next_joint_msg_str;
		next_joint_msg.SerializeToString(&next_joint_msg_str);

		//  Send message to all subscriber

		zmq::message_t next_joint_msg_zmq(next_joint_msg_str.length());
		memcpy(next_joint_msg_zmq.data(), next_joint_msg_str.c_str(), next_joint_msg_str.length());
		(*publisher).send(next_joint_msg_zmq);
		//  Get the reply.
		(*publisher).recv(&reply);

		for (int a = 0; a < num_positions; a++){
			double ma_model = tendonMomentArm(d, m, (*tnd_it), (*jnt_it));

			tendon_msg = to_msg.add_tend();
			(*tendon_msg).set_ma(ma_model);
			(*tendon_msg).set_name(*tnd_it);
			joint_msg = to_msg.add_joint();
			(*joint_msg).set_name(*jnt_it);

			std::string msg_str;
			to_msg.SerializeToString(&msg_str);

			//  Send message to all subscribers
			zmq::message_t message(msg_str.length());
			memcpy(message.data(), msg_str.c_str(), msg_str.length());
			(*publisher).send(message);
			//std::cout << "moment arm was: " << ma_model << std::endl;

			//  Get the reply.
			zmq::message_t reply;
			(*publisher).recv(&reply);

			from_msg.ParseFromArray(reply.data(), reply.size());

			mj_resetData(m, d);
			for (int b = 0; b < from_msg.joint_size(); b++) {
				std::string current_joint = from_msg.joint(b).name();
				int jointID = mj_name2id(m, mjOBJ_JOINT, current_joint.c_str());
				d->qpos[jointID] = from_msg.joint(b).qpos();
				//std::cout << "The reply moved joint " << from_msg.joint(i).name() << " to position " << from_msg.joint(i).qpos() << std::endl;
			}

			mj_inverse(m, d);
		}
	}

	end_msg.set_instruction("end");
	std::string end_msg_str;
	end_msg.SerializeToString(&end_msg_str);
	//  Send message to all subscribers
	zmq::message_t end_msg_zmq(end_msg_str.length());
	memcpy(end_msg_zmq.data(), end_msg_str.c_str(), end_msg_str.length());
	(*publisher).send(end_msg_zmq);
}

void cycleMuscle(mjData* d, mjModel* m, std::string muscle_name, std::string tendon_name, zmq::socket_t *publisher) {
	// protocol buffer stuff
	mujoco2py::mujoco_msg to_msg;
	mujoco2py::mujoco_msg from_msg;

	int actID = mj_name2id(m, mjOBJ_ACTUATOR, muscle_name.c_str());
	int tendID = mj_name2id(m, mjOBJ_TENDON, tendon_name.c_str());

	mujoco2py::mujoco_msg_mj_act *act_msg = to_msg.add_act();
	mujoco2py::mujoco_msg_mj_tend *tend_msg = to_msg.add_tend();

	(*act_msg).set_name(muscle_name);
	(*act_msg).set_force(d->actuator_force[actID]);

	(*tend_msg).set_name(tendon_name);
	(*tend_msg).set_len(d->ten_length[tendID]);

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
	double reply_dbl = from_msg.act(0).ctrl();

	double ext_ctrl = extended_ctrl(reply_dbl, muscle_name);
	d->ctrl[actID] = ext_ctrl;

	//std::cout << "The reply was: " << reply_dbl << std::endl;
}

void cycleSitePos(mjData* d, mjModel*  m, zmq::socket_t *publisher, std::vector<std::string> tendon_names, std::vector<std::string> joint_names) {
	// protocol buffer stuff
	mujoco2py::mujoco_msg to_msg;
	mujoco2py::mujoco_msg from_msg;

	mujoco2py::mujoco_msg_mj_act *act_msg = 0;
	mujoco2py::mujoco_msg_mj_joint *joint_msg = 0;
	mujoco2py::mujoco_msg_mj_site *site_msg = 0;

	Eigen::VectorXd actuator_forces(m->nu);
	Eigen::VectorXd joint_forces(m->nv);

	int n = 6;
	Eigen::VectorXd x;

	std::vector<std::string> sites;
	sites.push_back("right_hip");
	sites.push_back("right_knee");
	sites.push_back("right_ankle");
	sites.push_back("right_toe");
	sites.push_back("left_hip");
	sites.push_back("left_knee");
	sites.push_back("left_ankle");
	sites.push_back("left_toe");

	/* the following starting values provide a rough fit. */
	x.setConstant(n, 0);
	for (int j = 0; j < n; j++) {
		x[j] = d->qpos[j];
		//std::cout << "x_i[" << j << "] = " << x[j] << " ";
	}
	//std::cout << std::endl;
	double targ[24];
	//left hip:
	targ[12] = 0.03143;
	targ[13] = 0;
	targ[14] = 0;
	//left knee
	targ[15] = 0.03143;
	targ[16] = 0;
	targ[17] = -0.163;
	//left ankle
	targ[18] = 0.03143;
	targ[19] = 0;
	targ[20] = -0.345;
	//left toe
	targ[21] = 0.03143;
	targ[22] = -0.074;
	targ[23] = -0.345;

	int i = 0;
	for (std::vector<std::string>::iterator it = joint_names.begin(); it < joint_names.end(); it++, i++) {
		int jointID = mj_name2id(m, mjOBJ_JOINT, (*it).c_str());

		joint_msg = to_msg.add_joint();
		(*joint_msg).set_name(*it);
		joint_forces(i) = d->qfrc_inverse[jointID];

		(*joint_msg).set_force(joint_forces(i));
	}

	for (std::vector<std::string>::iterator it = sites.begin(); it < sites.end(); it++) {
		int siteID = mj_name2id(m, mjOBJ_SITE, (*it).c_str());

		site_msg = to_msg.add_site();
		(*site_msg).set_name(*it);
		double y = d->site_xpos[3 * siteID + 1];
		double z = d->site_xpos[3 * siteID + 2];

		(*site_msg).set_y(y);
		(*site_msg).set_z(z);
	}


	Eigen::MatrixXd moment_arm_matrix(m->nu, m->nv);

	for (int i = 0; i < m->nu; i++) {
		for (int j = 0; j < m->nv; j++) {
			moment_arm_matrix(i, j) = d->actuator_moment[m->nv*i + j];
		}
	}

	Eigen::MatrixXd inv_ma_matrix(m->nv, m->nu);
	inv_ma_matrix = moment_arm_matrix.transpose();
	Eigen::JacobiSVD<Eigen::MatrixXd>pinv(inv_ma_matrix);

	actuator_forces = inv_ma_matrix.transpose()*joint_forces;
	//std::cout << "Moment arm matrix: " << std::endl << moment_arm_matrix << std::endl;
	//std::cout << "Actuator Force Vector: " << std::endl << actuator_forces << std::endl;

	i = 0;
	for (std::vector<std::string>::iterator it = tendon_names.begin(); it < tendon_names.end(); it++, i++) {
		int actID = mj_name2id(m, mjOBJ_ACTUATOR, (*it).c_str());

		act_msg = to_msg.add_act();
		(*act_msg).set_name(*it);
		(*act_msg).set_force(actuator_forces(i));
	}

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

	for (int i = 0; i < from_msg.site_size(); i++) {
		targ[3 * i    ] = from_msg.site(i).x();
		targ[3 * i + 1] = from_msg.site(i).y();
		targ[3 * i + 2] = from_msg.site(i).z();
		/*std::cout << "The reply expects site: " << from_msg.site(i).site_name()
			<< " to be at position <" << from_msg.site(i).x() << ", "
			<< from_msg.site(i).y() << ", " << from_msg.site(i).z() << ">" << std::endl;*/
	}

	solution_pose = testLmder1(targ,x,sites);
	//std::cout << "solution pose: " << solution_pose.transpose() << std::endl;
	mj_resetData(m, d);
	for (int a = 0; a < m->nq; a++) {
		d->qpos[a] = solution_pose[a];
	}

}

void updateNeuron(mjData* d, mjModel*  m, zmq::socket_t *publisher, std::vector<std::string> tendon_names) {
	// protocol buffer stuff
	mujoco2py::mujoco_msg to_msg;
	mujoco2py::mujoco_msg from_msg;

	mujoco2py::mujoco_msg_mj_tend *tend_msg = 0;

	for (std::vector<std::string>::iterator it = tendon_names.begin(); it < tendon_names.end(); it++) {
		int siteID = mj_name2id(m, mjOBJ_TENDON, (*it).c_str());

		tend_msg = to_msg.add_tend();
		(*tend_msg).set_name(*it);
		(*tend_msg).set_len(d->ten_length[siteID]);
		(*tend_msg).set_len_dot(d->ten_velocity[siteID]);
		//std::cout << "tendon velocity was: " << d->ten_velocity[siteID] << std::endl;
		(*tend_msg).set_len0(tendon_length0[siteID]);
	}

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
void loadmodel(GLFWwindow* window, const char* filename, const char* xmlstring)
{
	// load and compile
	char error[1000] = "could not load binary model";
	mjModel* mnew = 0;
	if (xmlstring)
		mnew = mj_loadXML(0, xmlstring, error, 1000);
	else if (strlen(filename)>4 && !strcmp(filename + strlen(filename) - 4, ".mjb"))
		mnew = mj_loadModel(filename, 0, 0);
	else
		mnew = mj_loadXML(filename, 0, error, 1000);
	if (!mnew)
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
		tendon_length0.push_back(d->ten_length[a]);
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
                loadmodel(window, lastfile,0);

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
        loadmodel(window, paths[0],0);
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
	mj_inverse(m, d);
    //mj_step(m, d);

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
		mjr_rectangle(rect, 0, 0, rect.width, rect.height, 0.2, 0.3, 0.4, 1);
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
	mj_activate("E:\\mjpro\\key_5511.txt");

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
        loadmodel(window, argv[1],0);

    // set GLFW callbacks
    glfwSetKeyCallback(window, keyboard);
    glfwSetCursorPosCallback(window, mouse_move);
    glfwSetMouseButtonCallback(window, mouse_button);
    glfwSetScrollCallback(window, scroll);
    glfwSetDropCallback(window, drop);

    // print version
    printf("MuJoCo Pro version %.2lf\n\n", mj_version());

	std::cout << "Connecting to python server 1 ..." << std::endl;
	publisher.connect("tcp://localhost:5556");
	std::cout << "Connecting to python server 2 ..." << std::endl;
  neuron_publisher.connect("tcp://localhost:5555");

	// activate python server
	//int a = std::system("start python \"E:\\Google Drive\\Borton Lab\\Inter Process Communication\\CycleGeneralizedCoordsServer.py\" &");
	//int a = std::system("start python \"E:\\Google Drive\\Borton Lab\\Inter Process Communication\\CycleJointServer.py\" &");
	//int a = std::system("start python \"E:\\Google Drive\\Borton Lab\\Inter Process Communication\\CycleMuscleServer.py\" &");
	int a = std::system("start python \"E:\\Google Drive\\Borton Lab\\Inter Process Communication\\CycleSiteCoordsServer.py\" &");
	int b = std::system("start python \"E:\\Google Drive\\Borton Lab\\Inter Process Communication\\NeuronServer.py\" &");

	//Levenberg Test
	/*
	double targ[18] = { -0.04343, -0.0479416, 0.932763,
	0.04343, -0.0672373, 1.04794,
	-0.02423, 0.074101, 0.743362,
	0.02423, -0.2566, 0.92589,
	-0.03433, -0.0993, 0.849,
	0.03433, -0.150931, 1.09938
	};
		solution_pose = testLmder1(200, targ);
		paused = true;
		mj_resetData(m, d);
		for (int a = 0; a < m->nq; a++) {
			d->qpos[a] = solution_pose[a];
		}
		mj_forward(m, d);
	*/

	// get a target configuration
	/*
	std::string target = "right_hip";
	int siteID = mj_name2id(m, mjOBJ_SITE, target.c_str());
	std::cout << "right_hip: " << d->site_xpos[3 * siteID + 0] << "  " << d->site_xpos[3 * siteID + 1] << "  " << d->site_xpos[3 * siteID + 2] << std::endl;

	target = "right_knee";
	siteID = mj_name2id(m, mjOBJ_SITE, target.c_str());
	std::cout << "right_knee: " << d->site_xpos[3 * siteID + 0] << "  " << d->site_xpos[3 * siteID + 1] << "  " << d->site_xpos[3 * siteID + 2] << std::endl;

	target = "right_ankle";
	siteID = mj_name2id(m, mjOBJ_SITE, target.c_str());
	std::cout << "right_ankle: " << d->site_xpos[3 * siteID + 0] << "  " << d->site_xpos[3 * siteID + 1] << "  " << d->site_xpos[3 * siteID + 2] << std::endl;

	target = "right_toe";
	siteID = mj_name2id(m, mjOBJ_SITE, target.c_str());
	std::cout << "right_toe: " << d->site_xpos[3 * siteID + 0] << "  " << d->site_xpos[3 * siteID + 1] << "  " << d->site_xpos[3 * siteID + 2] << std::endl;

	target = "left_hip";
	siteID = mj_name2id(m, mjOBJ_SITE, target.c_str());
	std::cout << "left_hip: " << d->site_xpos[3 * siteID + 0] << "  " << d->site_xpos[3 * siteID + 1] << "  " << d->site_xpos[3 * siteID + 2] << std::endl;

	target = "left_knee";
	siteID = mj_name2id(m, mjOBJ_SITE, target.c_str());
	std::cout << "left_knee: " << d->site_xpos[3 * siteID + 0] << "  " << d->site_xpos[3 * siteID + 1] << "  " << d->site_xpos[3 * siteID + 2] << std::endl;

	target = "left_ankle";
	siteID = mj_name2id(m, mjOBJ_SITE, target.c_str());
	std::cout << "left_ankle: " << d->site_xpos[3 * siteID + 0] << "  " << d->site_xpos[3 * siteID + 1] << "  " << d->site_xpos[3 * siteID + 2] << std::endl;

	target = "left_toe";
	siteID = mj_name2id(m, mjOBJ_SITE, target.c_str());
	std::cout << "left_toe: " << d->site_xpos[3 * siteID + 0] << "  " << d->site_xpos[3 * siteID + 1] << "  " << d->site_xpos[3 * siteID + 2] << std::endl;
	*/
	// main loop
	std::vector<std::string> tendons;

	tendons.push_back("r_IL");
	tendons.push_back("r_GMED");
	tendons.push_back("r_VAS");
	tendons.push_back("r_TA");
	tendons.push_back("r_SOL");
	tendons.push_back("r_RF");
	tendons.push_back("r_BF");
	tendons.push_back("r_GAS");
	tendons.push_back("l_IL");
	tendons.push_back("l_GMED");
	tendons.push_back("l_VAS");
	tendons.push_back("l_TA");
	tendons.push_back("l_SOL");
	tendons.push_back("l_RF");
	tendons.push_back("l_BF");
	tendons.push_back("l_GAS");

	std::vector<std::string> joints;

	joints.push_back("right_hip_x");
	joints.push_back("right_knee");
	joints.push_back("right_ankle_x");
	joints.push_back("left_hip_x");
	joints.push_back("left_knee");
	joints.push_back("left_ankle_x");

	std::vector<std::string> name_pairs[2];
	double start_poses[11];
	double end_poses[11];

	name_pairs[0].push_back("r_IL");
	name_pairs[1].push_back("right_hip_x");

	start_poses[0] = -40;
	end_poses[0] = 90;

	name_pairs[0].push_back("r_GMED");
	name_pairs[1].push_back("right_hip_x");

	start_poses[1] = -40;
	end_poses[1] = 90;

	name_pairs[0].push_back("r_VAS");
	name_pairs[1].push_back("right_knee");

	start_poses[2] = -60;
	end_poses[2] = 90;

	name_pairs[0].push_back("r_TA");
	name_pairs[1].push_back("right_ankle_x");

	start_poses[3] = -60;
	end_poses[3] = 30;

	name_pairs[0].push_back("r_SOL");
	name_pairs[1].push_back("right_ankle_x");

	start_poses[4] = -60;
	end_poses[4] = 30;

	name_pairs[0].push_back("r_RF");
	name_pairs[1].push_back("right_hip_x");

	start_poses[5] = -40;
	end_poses[5] = 90;

	name_pairs[0].push_back("r_RF");
	name_pairs[1].push_back("right_knee");

	start_poses[6] = -60;
	end_poses[6] = 90;

	name_pairs[0].push_back("r_BF");
	name_pairs[1].push_back("right_hip_x");

	start_poses[7] = -40;
	end_poses[7] = 90;

	name_pairs[0].push_back("r_BF");
	name_pairs[1].push_back("right_knee");

	start_poses[8] = -60;
	end_poses[8] = 90;

	name_pairs[0].push_back("r_GAS");
	name_pairs[1].push_back("right_knee");

	start_poses[9] = -60;
	end_poses[9] = 90;

	name_pairs[0].push_back("r_GAS");
	name_pairs[1].push_back("right_ankle");

	start_poses[10] = -60;
	end_poses[10] = 30;

    while( !glfwWindowShouldClose(window) )
    {
		if (update_cmd) {
			//SweepMomentArms(d, m, name_pairs, 100, start_poses, end_poses, &publisher);
			//Sleep(100);
			//cycleMuscle(d, m, "r_TA", "r_TA", &publisher);

			cycleSitePos(d, m, &publisher, tendons, joints);
		}
        // simulate and render
        render(window);
		//mj_inverse(m, d);
		if (update_cmd) {
			updateNeuron(d, m, &neuron_publisher, tendons);
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
