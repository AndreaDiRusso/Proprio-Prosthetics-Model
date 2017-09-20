#define WINW 3000
#define WINH 1500

// Target visualization
double target_color[NUM_SITES] = {
	1, // Right iliac Crest, WHITE
	1,
	1,
	1, // Right Hip, YELLOW
	1,
	0,
	1, // Right Knee, RED
	0,
	0,
	0.5, // Right Ankle, GRAY
	0.5,
	0.5,
	0, // Right Knuckle, BLUE
	0,
	1,
	0, // Right Toe, TEAL
	1,
	1,
	1, // Left Iliac Crest, WHITE
	1,
	1,
	1, // Left Hip, YELLOW
	1,
	0,
	1, // Left Knee
	1,
	1,
	1, // Left Ankle
	1,
	1,
	1, // Left Knuckle
	1,
	1,
	1, // Left Toe
	1,
	1
};

double target_size[NUM_SITES / 3] = {
	15e-3,
	15e-3,
	15e-3,
	15e-3,
	15e-3,
	15e-3,
	15e-3,
	15e-3,
	1e-3,
	1e-3,
	1e-3,
	1e-3
};
std::vector<std::string> tendons = {
	"r_IL",
	"r_GMED",
	"r_VAS",
	"r_TA",
	"r_SOL",
	"r_RF",
	"r_BF",
	"r_GAS",
	"l_IL",
	"l_GMED",
	"l_VAS",
	"l_TA",
	"l_SOL",
	"l_RF",
	"l_BF",
	"l_GAS"
};

std::vector<std::string> joints = {
	"world_x",
	"world_y",
	"world_z",
	"world_xt",
	"world_yt",
	"world_zt",
	"right_hip_x",
	"right_hip_z",
	"right_knee",
	"right_ankle_x",
	//"left_hip_x",
	//"left_hip_z",
	"left_knee",
	"left_ankle_x"
};