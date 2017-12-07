import pstats, argparse, pickle

parser = argparse.ArgumentParser()
parser.add_argument('--folder', default = 'W:/ENG_Neuromotion_Shared/group/Proprioprosthetics/Data/201709261100-Proprio/')
args = parser.parse_args()
folder = args.folder

folder = 'W:/ENG_Neuromotion_Shared/group/Proprioprosthetics/Data/Starbuck Example'

timingStats = {}
timingStats.update({'nelder' : pstats.Stats(folder + '/nelder_timing.txt')})
timingStats.update({'leastsq' : pstats.Stats(folder + '/leastsq_timing.txt')})
timingStats.update({'lbfgsb' : pstats.Stats(folder + '/lbfgsb_timing.txt')})

print(timingStats['nelder'].total_tt)
print(timingStats['leastsq'].total_tt)
print(timingStats['lbfgsb'].total_tt)

with open(folder + '/T_1_interp_fit_statistics.pickle', 'rb') as f:
    accuracyStats = pickle.load(f)

accuracyStats
