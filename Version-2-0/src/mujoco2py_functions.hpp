#include "mujoco.h"
#include "glfw3.h"
#include "zmq.hpp"
#include "stdlib.h"
#include "string.h"
#include "stdio.h"

#include <iostream>
#include <limits>
#include <random>

#include <Eigen\Dense>
#include "mujoco2py.pb.h"
#include <unsupported/Eigen/LevenbergMarquardt>

#define PI 3.141592
#define ERROR_MULTIPLIER 1
#define ALPHA 0.005
#define NUM_JOINTS 12
#define NUM_SITES 36

struct site_functor;

void v_defaultGeom(mjvGeom* geom);
double angleOfSegments(int idx, int idy, int idz, double* targ);
Eigen::VectorXd apply_lm(mjModel* m, mjData* d, double *targ, Eigen::VectorXd x, std::vector<std::string> sites,int *info_holder, double tol);

void renderTargetSite(mjvScene* scn, double x, double y, double z, double re, double gr, double bl, double sz);

/*fitPoseToSites
Connects to a publisher which serves the position of a list of pre-recorded sites and fits the pose of the model to the site positions.
*/
void fitPoseToSites(mjData* d, mjModel*  m, mjvScene* scn, zmq::socket_t *publisher, std::vector<std::string> tendon_names, std::vector<std::string> joint_names, Eigen::VectorXd* solution_pose, Eigen::VectorXd* target_pose);

void updateNeuron(mjData* d, mjModel*  m, zmq::socket_t *publisher, std::vector<std::string> tendon_names, std::vector<mjtNum> tendon_length0);

double getTendonMomentArm(mjData* d, mjModel* m, std::string tendon_name, std::string joint_name);

void poseJoints(mjData* d, mjModel* m, zmq::socket_t *publisher);