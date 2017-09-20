
std::vector<std::vector<std::string>> name_pairs = {
	{ "r_IL","right_hip_x" },
	{ "r_GMED","right_hip_x" },
	{ "r_VAS","right_knee" },
	{ "r_TA","right_ankle_x" },
	{ "r_SOL","right_ankle_x" },
	{ "r_RF","right_hip_x" },
	{ "r_RF","right_knee" },
	{ "r_BF","right_hip_x" },
	{ "r_BF","right_knee" },
	{ "r_GAS","right_knee" },
	{ "r_GAS","right_ankle" }
};

double start_poses[11] = {
	-40,
	-40,
	-60,
	-60,
	-60,
	-40,
	-60,
	-40,
	-60,
	-60,
	-60
};

double end_poses[11] = {
	90,
	90,
	90,
	30,
	30,
	90,
	90,
	90,
	90,
	90,
	30
};

std::vector<std::string> sites = {
	"right_iliac_crest",
	"right_hip",
	"right_knee",
	"right_ankle",
	"right_knuckle",
	"right_toe",
	"left_iliac_crest",
	"left_hip",
	"left_knee",
	"left_ankle",
	"left_knuckle",
	"left_toe" };

double weights[NUM_SITES] = {
	1, // right iliac crest
	1,
	1,
	1, // right hip
	1,
	1,
	1, // right knee
	1,
	1,
	2, // right ankle
	2,
	2,
	1, // right knuckle
	1,
	1,
	1, // right toe
	1,
	1,
	1, // left iliac crest
	1,
	1,
	1, // left hip
	1,
	1,
	0, // left knee
	0,
	0,
	0, // left ankle
	0,
	0,
	0, // left knuckle
	0,
	0,
	0, // left toe
	0,
	0
};
