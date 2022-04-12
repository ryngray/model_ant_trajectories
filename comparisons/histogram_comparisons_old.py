import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import skmob
from scipy import stats
from skmob.measures.individual import distance_straight_line, maximum_distance, number_of_visits, waiting_times, real_entropy
import math
from tqdm import tqdm
from numpy.linalg import norm
import traja
import nolds

import warnings
warnings.filterwarnings("ignore")

def ccw(A,B,C):
    return ((C[1]-A[1]) * (B[0]-A[0]) > (B[1]-A[1]) * (C[0]-A[0]))

# Return true if line segments AB and CD intersect
def intersect(A,B,C,D):
    return (ccw(A,C,D) != ccw(B,C,D) and ccw(A,B,C) != ccw(A,B,D))

def get_crossings(a,get_crossings_vals = False):
    a = np.asarray(a)
    count = 0
    ni = 0
    crossings =[]
    for i in range(len(a)-1):
        for j in range(i+2, len(a)-1):
            A = [a[i][0], a[i][1]]
            B = [a[i+1][0], a[i+1][1]]
            C = [a[j][0], a[j][1]]
            D = [a[j+1][0], a[j+1][1]]
            if(intersect(A,B,C,D)):
                count += 1
                if(get_crossings_vals):
                    crossings.append([A,B,C,D])
            else:
                ni += 1
    if(get_crossings_vals):
        return [count, crossings]
    return count
            
def triangle_area(arr, step_size = 1):
    if(step_size == 0):
        print("Cannot have step size be 0")
    tot_area = 0
    for i in range(len(arr)):
        [x1,y1,t] = arr[i]
        [x2,y2,t] = arr[i+step_size]
        try:
            [x3,y3,t] = arr[i+2*step_size]
        except:
            return tot_area
        Area = 1/2 *(x1*abs(y2 - y3) + x2*abs(y3 - y1) + x3*abs(y1 - y2))
        tot_area += Area
    return tot_area

def get_change_dir(a):
    a = np.asarray(a)
    asign = np.sign(a)
    signchange = ((np.roll(asign, 1) - asign) != 0).astype(int)
    signchange[0]=0
    return sum(signchange)

def max_from_start(a):
    dists = []
    return max(abs(a-a[0]))


def get_speed(a,t=True):
    f_np = np.asarray(a)
    xs = np.subtract(f_np[1,0:-1],f_np[1,1:])
    ys = np.subtract(f_np[2,0:-1],f_np[2,1:])
    if(t):
        speed = np.divide(np.sqrt(np.add(np.power(xs, 2), np.power(ys, 2))),np.subtract(f_np[1:,2],f_np[:-1,2]))
    else:
        speed = np.sqrt(np.add(np.power(xs, 2), np.power(ys, 2)))
    speed = np.insert(speed, 0,0)
    return speed

def get_acceleration(a, t = True):
    f_np = np.asarray(a)
    xs = np.subtract(f_np[1,0:-1],f_np[1,1:])
    ys = np.subtract(f_np[2,0:-1],f_np[2,1:])
    if(t):
        speed = np.divide(np.sqrt(np.add(np.power(xs, 2), np.power(ys, 2))),np.subtract(f_np[1:,2],f_np[:-1,2]))
        acc = np.divide(np.subtract(speed[1:], speed[:-1]),np.subtract(f_np[1:,3],f_np[:-1,3]))
    else:
        speed = np.sqrt(np.add(np.power(xs, 2), np.power(ys, 2)))
        acc = np.subtract(speed[1:], speed[:-1])
    acc = np.insert(acc, 0,0)
    return acc

def calcAngle(lineA,lineB):
    lineA = np.asarray(lineA)
    lineB = np.asarray(lineB)
    line1Y1 = lineA[1]
    line1X1 = lineA[0]
    line1Y2 = lineA[3]
    line1X2 = lineA[2]

    line2Y1 = lineB[1]
    line2X1 = lineB[0]
    line2Y2 = lineB[3]
    line2X2 = lineB[2]
#     print(lineA)

    #calculate angle between pairs of lines
    angle1 = math.atan2(line1Y1-line1Y2,line1X1-line1X2)
    angle2 = math.atan2(line2Y1-line2Y2,line2X1-line2X2)
    angleDegrees = (angle1-angle2) * 360 / (2*math.pi)
    return angleDegrees

def get_angles(a):
    angles = []
    a = np.asarray(a)
#     print(a)
    for i in range(len(a)-2):
        v1 = [a[i][0]]
        ang = calcAngle([a[i][0],a[i][1],a[i+1][0],a[i+1][1]],[a[i+1][0],a[i+1][1],a[i+2][0],a[i+2][1]])
        angles.append(ang)
    return angles

def get_angle(traj):
    f_np = np.asarray(traj)
    angle = np.arctan2(np.subtract(f_np[1,:-1],f_np[1,1:]),np.subtract(f_np[0,:-1],f_np[0,1:]))
    angle = np.insert(angle, 0,0)
    deg = [math.degrees(x) for x in angle]
    return deg



def speed_calcs(dat_given, dat_avg):
	dat_speed = []

	num_generate = min([len(data_given), len(dat_avg)])
	gan_speed = []
	gan_mode = []
	gan_max = []

	print(np.shape(data_given))
	for i in tqdm(range(num_generate)):
	    rs_tmp = (get_speed(data_given[data_given[:,2] == i],0))
	    gan_speed.append(np.mean(rs_tmp))
	    gan_max.append(np.max(rs_tmp))
	    gan_mode.append(stats.mode(rs_tmp)[0])
	gan_mode = np.reshape(gan_mode, (num_generate))

	print(np.shape(gan_mode),gan_mode, max(gan_mode),min(gan_mode))


	plt.figure(0)
	plt.hist(dat_avg['mode_speed'],bins=50,density = True,alpha = 0.5,label="Real Data")
	plt.hist(gan_mode,density = True,alpha = 0.5,label="Generated Data")
	plt.title("Histogram of mode of speed")
	plt.xlabel("m/s")
	plt.legend()
	md = np.mean(dat_avg['mode_speed'])
	vd = np.std(dat_avg['mode_speed'])
	mg = np.mean(gan_mode)
	vg = np.std(gan_mode)
	str_box = "Data Mean: " + str(round(md,2)) + "\nData STD: " + str(round(vd,2)) + "\nGenerated Mean: " +str(round(mg,2)) + "\nGenerated STD: " + str(round(vg,2))
	plt.annotate(str_box, xy=(0.05, 0.8), xycoords='axes fraction')
	plt.savefig("mode_speed.png")

	print("KS Mode Speed", stats.ks_2samp(dat_avg['mode_speed'], gan_mode))
        
	plt.figure(1)
	plt.hist(dat_avg['max_speed'],bins=50,label="Real Data",alpha=0.5)
	plt.hist(gan_max,bins=50,label="Generated Data",alpha=0.5)
	plt.title("Histogram of max of speed")
	plt.xlabel("m/s")
	plt.savefig("max_speed.png")
	plt.legend()
	md = np.mean(dat_avg['max_speed'])
	vd = np.std(dat_avg['max_speed'])
	mg = np.mean(gan_mode)
	vg = np.std(gan_mode)
	str_box = "Data Mean: " + str(md) + "\nData STD: " + str(vd) + "\nGenerated Mean: " +str(mg) + "\nGenerated STD: " + str(vg)
	# plt.text(0.05, 0.95, str_box, verticalalignment = "top")
	plt.annotate(str_box, xy=(0.05, 0.8), xycoords='axes fraction')
	plt.savefig("max_speed")

	print("KS Max Speed", stats.ks_2samp(dat_avg['max_speed'],gan_max))

	return [stats.ks_2samp(dat_avg['mode_speed'], gan_mode), stats.ks_2samp(dat_avg['max_speed'],gan_max)]


def acceleration_calcs(dat_given, dat_avg):
	gan_acc = []
	gan_acc_max = []
	gan_acc_mode = []
	num_generate = min([len(data_given), len(dat_avg)])
	for i in tqdm(range(num_generate)):
	    rs_tmp = get_acceleration(data_given[data_given[:,2] == i],0)
	    gan_acc.append(np.mean(rs_tmp))
	    gan_acc_max.append(np.max(rs_tmp))
	    gan_acc_mode.append(stats.mode(rs_tmp)[0])
	gan_acc_mode = np.reshape(gan_acc_mode, (num_generate))
	# rw_speed  = [item for sublist in s for item in sublist]

	plt.figure(2)
	plt.hist(dat_avg['mode_acceleration'],bins = 50, label="Real Data",alpha=0.5)
	plt.hist(gan_acc_mode, bins= 50, label="Generated Data",alpha=0.5)
	plt.title("Comparison of Mode Acceleration")
	plt.legend()
	md = np.mean(dat_avg['mode_acceleration'])
	vd = np.std(dat_avg['mode_acceleration'])
	mg = np.mean(gan_acc_mode)
	vg = np.std(gan_acc_mode)
	str_box = "Data Mean: " + str(round(md,2)) + "\nData STD: " + str(round(vd,2)) + "\nGenerated Mean: " +str(round(mg,2)) + "\nGenerated STD: " + str(round(vg,2))
	plt.annotate(str_box, xy=(0.05, 0.8), xycoords='axes fraction')
	plt.savefig("acceleration_gan.png")

	print("KS Mode Acceleration:", stats.ks_2samp(dat_avg["mode_acceleration"], gan_acc_mode))

	plt.figure(3)
	plt.hist(dat_avg['max_acceleration'],bins = 50, label="Real Data",alpha=0.5)
	plt.hist(gan_acc_max, bins= 50, label="Generated Data",alpha=0.5)
	plt.title("Comparison of Max Acceleration")
	plt.legend()
	md = np.mean(dat_avg['max_acceleration'])
	vd = np.std(dat_avg['max_acceleration'])
	mg = np.mean(gan_acc_max)
	vg = np.std(gan_acc_max)
	str_box = "Data Mean: " + str(round(md,2)) + "\nData STD: " + str(round(vd,2)) + "\nGenerated Mean: " +str(round(mg,2)) + "\nGenerated STD: " + str(round(vg,2))
	plt.annotate(str_box, xy=(0.05, 0.8), xycoords='axes fraction')	
	plt.savefig("acceleration_max_gan.png")

	print("KS Max Acceleration:", stats.ks_2samp(dat_avg['max_acceleration'], gan_acc_max))

	return [stats.ks_2samp(dat_avg["mode_acceleration"], gan_acc_mode), stats.ks_2samp(dat_avg['max_acceleration'], gan_acc_max)]

def angle_calcs(dat_given, dat_avg):
	gan_ang = []
	gan_ang_max = []
	gan_ang_mode = []
	num_generate = min([len(data_given), len(dat_avg)])
	for i in range(num_generate):
	    rs_tmp = get_angle(data_given[data_given[:,2] == i])
	    gan_ang.append(math.radians(np.mean(rs_tmp)))
	    gan_ang_max.append(math.radians(np.max(rs_tmp)))
	    gan_ang_mode.append(math.radians(stats.mode(rs_tmp)[0]))
	    
	gan_ang_mode = np.reshape(gan_ang_mode, (num_generate))
	# rw_speed  = [item for sublist in s for item in sublist]

	plt.figure(4)
	plt.hist(dat_avg['max_angle'],bins = 50, label="Real Data",alpha=0.5)
	plt.hist(gan_ang_max, bins= 50, label="GAN Generated Data",alpha=0.5)
	plt.title("Comparison of Max Angle")
	plt.legend()
	md = np.mean(dat_avg['max_angle'])
	vd = np.std(dat_avg['max_angle'])
	mg = np.mean(gan_ang_max)
	vg = np.std(gan_ang_max)
	str_box = "Data Mean: " + str(round(md,2)) + "\nData STD: " + str(round(vd,2)) + "\nGenerated Mean: " +str(round(mg,2)) + "\nGenerated STD: " + str(round(vg,2))
	plt.annotate(str_box, xy=(0.05, 0.8), xycoords='axes fraction')
	plt.savefig("angle_gan.png")
	print("KS Angle Mode:", stats.ks_2samp(dat_avg['mode_angle'], gan_ang_max))


	plt.figure(5)
	plt.hist(dat_avg['mode_angle'],bins = 50, label="Real Data",alpha=0.5)
	plt.hist(gan_ang_mode, bins= 50, label="GAN Generated Data",alpha=0.5)
	plt.title("Comparison of Mode Angle (in degrees)")
	plt.legend()
	md = np.mean(dat_avg['mode_angle'])
	vd = np.std(dat_avg['mode_angle'])
	mg = np.mean(gan_ang_mode)
	vg = np.std(gan_ang_mode)
	str_box = "Data Mean: " + str(round(md,2)) + "\nData STD: " + str(round(vd,2)) + "\nGenerated Mean: " +str(round(mg,2)) + "\nGenerated STD: " + str(round(vg,2))
	plt.annotate(str_box, xy=(0.05, 0.8), xycoords='axes fraction')
	plt.savefig("angle_gan.png")

	print("KS Angle Max:", stats.ks_2samp(dat_avg['max_angle'], gan_ang_max))

	return [stats.ks_2samp(dat_avg["mode_angle"], gan_ang_mode), stats.ks_2samp(dat_avg['max_angle'], gan_ang_max)]


def get_skmob(traj_sim, num_generate, padding=200):
	num_generate =int(len(traj_sim)/padding)
	traj_sim = traj_sim[:,0:2]
	gan_ind = [i for i in range(num_generate) for j in range(padding)]
	gan_ind = np.reshape(gan_ind, (len(gan_ind),1))
	gan_t = [i/100 for i in range(num_generate*padding)]
	gan_reshape = np.reshape(traj_sim, (padding*num_generate, 2))
	print(np.shape(gan_reshape),np.shape(gan_t))
	gan_t = np.reshape(gan_t, (len(gan_t),1))
	all_data = np.append(gan_reshape, gan_t, 1)
	all_data = np.append(all_data, gan_ind,1)
	# all_data[:,2] = all_data[:,2]*100

	gan_pd = pd.DataFrame(all_data, columns=["x",'y','t','user'])
	# rw_pd = pd.Dataframe(all_data, latitude=1, longitude=2, datetime=3, user_id=4)

	tdf = skmob.TrajDataFrame(gan_pd, latitude='x',longitude='y',datetime='t',user_id='user')


	return tdf

def get_ganpd(traj_sim, num_generate, padding=200):
	num_generate =int(len(traj_sim)/padding)
	traj_sim = traj_sim[:,0:2]
	gan_ind = [i for i in range(num_generate) for j in range(padding)]
	gan_ind = np.reshape(gan_ind, (len(gan_ind),1))
	gan_t = [i/100 for i in range(num_generate*padding)]
	gan_reshape = np.reshape(traj_sim, (padding*num_generate, 2))
	print(np.shape(gan_reshape),np.shape(gan_t))
	gan_t = np.reshape(gan_t, (len(gan_t),1))
	all_data = np.append(gan_reshape, gan_t, 1)
	all_data = np.append(all_data, gan_ind,1)
	# all_data[:,2] = all_data[:,2]*100

	gan_pd = pd.DataFrame(all_data, columns=["x",'y','t','user'])

	return gan_pd	

def straight_line_distance_calcs(tdf, dat_avg):
	try:
		dsl_df = distance_straight_line(tdf)
	except:
		return 0

	plt.figure(6)
	plt.hist(dat_avg['dist_traveled'],bins=50, label="Real Data",alpha=0.5)
	plt.hist(dsl_df['distance_straight_line'],bins=50, label="GAN Generated Data",alpha=0.5)
	plt.title("Comparison of Straight Line Distance")
	plt.legend()
	md = np.mean(dat_avg['dist_traveled'])
	vd = np.std(dat_avg['dist_traveled'])
	mg = np.mean(dsl_df['distance_straight_line'],)
	vg = np.std(dsl_df['distance_straight_line'],)
	str_box = "Data Mean: " + str(round(md,2)) + "\nData STD: " + str(round(vd,2)) + "\nGenerated Mean: " +str(round(mg,2)) + "\nGenerated STD: " + str(round(vg,2))
	plt.annotate(str_box, xy=(0.05, 0.8), xycoords='axes fraction')
	plt.savefig("line_dist_gan.png")

	print("KS Straight Line Distance:", stats.ks_2samp(dat_avg['dist_to_line'], dsl_df['distance_straight_line']))

	return [stats.ks_2samp(dat_avg['dist_to_line'], dsl_df['distance_straight_line'])]


def max_dist_calcs(tdf, dat_avg):
	try:
		md_df = maximum_distance(tdf)
	except:
		return 0
	plt.figure(7)
	plt.hist(dat_avg['max_dist'],bins=50, label="Real Data",alpha=0.5)
	plt.hist(md_df['maximum_distance'],bins=50, label="GAN Generated Data",alpha=0.5)
	plt.title("Comparison of Maximum distance between any two points")
	plt.legend()
	md = np.mean(dat_avg['max_dist'])
	vd = np.std(dat_avg['max_dist'])
	mg = np.mean(md_df['maximum_distance'])
	vg = np.std(md_df['maximum_distance'])
	str_box = "Data Mean: " + str(round(md,2)) + "\nData STD: " + str(round(vd,2)) + "\nGenerated Mean: " +str(round(mg,2)) + "\nGenerated STD: " + str(round(vg,2))
	plt.annotate(str_box, xy=(0.05, 0.8), xycoords='axes fraction')
	plt.savefig("max_dist_gan.png")

	print("KS Maximum distance between two points", stats.ks_2samp(dat_avg['max_dist'], md_df['maximum_distance']))

	return [stats.ks_2samp(dat_avg['max_dist'], md_df['maximum_distance'])]


def distance_straight_line_calcs(tdf, dat_avg):
	d_sl = []
	num_generate =int(len(traj_sim)/padding)
	step_size = 50
	for i in tqdm(range(num_generate)):
	    d_tmp = 0
	    for j in range(1,199,step_size):
	        p1 = gan_pd[gan_pd['user']==i].iloc[0][['x','y']]
	        p2 = gan_pd[gan_pd['user']==i].iloc[199][['x','y']]
	        p3 = gan_pd[gan_pd['user']==i].iloc[j][['x','y']]
	        d_tmp += norm(np.cross(p2-p1, p1-p3))/norm(p2-p1)
	    d_sl.append(d_tmp)

	plt.hist(dat_avg['dist_to_line'],bins=50, label="Real Data",alpha=0.5)
	plt.figure(8)
	plt.hist(d_sl, bins=10, label="GAN Generated Data",alpha=0.5)
	plt.title("Comparison of Distance to a straight line")
	# plt.xlim((0,1))
	plt.ylim((0,400))
	plt.legend()
	md = np.mean(dat_avg['dist_to_line'])
	vd = np.std(dat_avg['dist_to_line'])
	mg = np.mean(d_sl)
	vg = np.std(d_sl)
	str_box = "Data Mean: " + str(round(md,2)) + "\nData STD: " + str(round(vd,2)) + "\nGenerated Mean: " +str(round(mg,2)) + "\nGenerated STD: " + str(round(vg,2))
	plt.annotate(str_box, xy=(0.05, 0.8), xycoords='axes fraction')
	plt.savefig("max_dist_gan.png")

def direction_changes_calcs(gan_pd, dat_avg):
	dc = []
	num_generate = min([len(gan_pd), len(dat_avg)])

	for i in tqdm(range(num_generate)):
	    ang = get_angle(np.asarray(gan_pd[gan_pd['user']==i]))
	    dc.append(get_change_dir(ang))
	plt.figure(9)
	plt.hist(dat_avg['direction_changes'],bins=50, label="Real Data",alpha=0.5)
	plt.hist(dc,bins=50, label="GAN Generated Data",alpha=0.5)
	plt.title("Comparison of Direction Changes")
	plt.legend()
	md = np.mean(dat_avg['direction_changes'])
	vd = np.std(dat_avg['direction_changes'])
	mg = np.mean(dc)
	vg = np.std(dc)
	str_box = "Data Mean: " + str(round(md,2)) + "\nData STD: " + str(round(vd,2)) + "\nGenerated Mean: " +str(round(mg,2)) + "\nGenerated STD: " + str(round(vg,2))
	plt.annotate(str_box, xy=(0.05, 0.8), xycoords='axes fraction')
	plt.savefig("dir_changes_gan.png")

	print("KS Direction Changes:", stats.ks_2samp(dat_avg['direction_changes'], dc))

	return [stats.ks_2samp(dat_avg['direction_changes'], dc)]


def triangle_area_calcs(gan_pd, dat_avg):
	newa = gan_pd[['x','y','user']]
	newa = np.asarray(newa)
	max_dists = []
	areas = []
	num_generate = min([len(gan_pd), len(dat_avg)])
	for plotid in tqdm(range(num_generate)):
	    c = newa[newa[:,2]==plotid]
	    area = triangle_area(c,step_size=30)
	    areas.append(abs(area))
	plt.figure(10)
	plt.hist(dat_avg['triangle_area'],bins=50, label="Real Data")
	plt.hist(areas, bins=50, label="GAN Generated Data")
	plt.title("Comparison of triangle area")
	plt.legend()
	md = np.mean(dat_avg['triangle_area'])
	vd = np.std(dat_avg['triangle_area'])
	mg = np.mean(areas)
	vg = np.std(areas)
	str_box = "Data Mean: " + str(round(md,2)) + "\nData STD: " + str(round(vd,2)) + "\nGenerated Mean: " +str(round(mg,2)) + "\nGenerated STD: " + str(round(vg,2))
	plt.annotate(str_box, xy=(0.05, 0.8), xycoords='axes fraction')
	# plt.xlim((0,10000000))
	plt.savefig("max_from_start_gan.png")


	print("KS Triangle Area:", stats.ks_2samp(dat_avg['triangle_area'], areas))

	return [stats.ks_2samp(dat_avg['triangle_area'], areas)]

def max_from_start_calcs(gan_pd, dat_avg):
	newa = gan_pd[['x','y','user']]
	newa = np.asarray(newa)
	max_dists = []
	num_generate = min([len(gan_pd), len(dat_avg)])
	for plotid in tqdm(range(num_generate)):
	    c = newa[newa[:,2]==plotid]
	    xs = np.subtract(c[0,0],c[:,0])
	    ys = np.subtract(c[0,1],c[:,1])
	    dists = np.sqrt(np.add(np.power(xs, 2), np.power(ys, 2)))
	    m = max(dists)
	    max_dists.append(m)
	plt.figure(11)
	plt.hist(dat_avg['max_dist_start'],bins=50, label="Real Data",alpha=0.5)
	plt.hist(max_dists, bins=50, label="GAN Generated Data",alpha=0.5)
	plt.title("Comparison of max distance from start")
	plt.legend()
	md = np.mean(dat_avg['max_dist_start'])
	vd = np.std(dat_avg['max_dist_start'])
	mg = np.mean(max_dists)
	vg = np.std(max_dists)
	str_box = "Data Mean: " + str(round(md,2)) + "\nData STD: " + str(round(vd,2)) + "\nGenerated Mean: " +str(round(mg,2)) + "\nGenerated STD: " + str(round(vg,2))
	plt.annotate(str_box, xy=(0.05, 0.8), xycoords='axes fraction')
	# plt.xlim((0,10))
	plt.savefig("max_from_start_gan.png")


	print("KS Max From Start:", stats.ks_2samp(dat_avg['max_dist_start'], max_dists))

	return [stats.ks_2samp(dat_avg['max_dist_start'], max_dists)]

def crossings_calcs(gan_pd, dat_avg):
	newa = gan_pd[['x','y','user']]
	newa = np.asarray(newa)
	crossings_gan = []
	num_generate = min([len(gan_pd), len(dat_avg)])
	for plotid in tqdm(range(num_generate)):
	    c = newa[newa[:,2]==plotid]
	    crossings_gan.append(get_crossings(c))
	plt.figure(12)
	plt.hist(dat_avg['num_crossings'],bins=50, label="Real Data",alpha=0.5)
	plt.hist(crossings_gan, bins=50, label="GAN Generated Data",alpha=0.5)
	plt.title("Comparison of Crossings")
	plt.legend()
	# plt.xlim((0,10))
	md = np.mean(dat_avg['num_crossings'])
	vd = np.std(dat_avg['num_crossings'])
	mg = np.mean(crossings_gan)
	vg = np.std(crossings_gan)
	str_box = "Data Mean: " + str(round(md,2)) + "\nData STD: " + str(round(vd,2)) + "\nGenerated Mean: " +str(round(mg,2)) + "\nGenerated STD: " + str(round(vg,2))
	plt.annotate(str_box, xy=(0.05, 0.8), xycoords='axes fraction')
	plt.savefig("crossings_gan.png")


	print("KS Crossings:", stats.ks_2samp(dat_avg['num_crossings'], crossings_gan))

	return [stats.ks_2samp(dat_avg['num_crossings'], crossings_gan)]




if __name__ == "__main__":
	if(len(sys.argv) < 3):
		print("Please give a file containing x,y,t values and a file containing the pre-computed histogram values")
		sys.exit()


	filename_data = sys.argv[1]
	filename_avg = sys.argv[2]

	data_given = pd.read_csv(filename_data,header=None)
	data_given = np.asarray(data_given).astype(float)

	dat_avg = pd.read_csv(filename_avg)


	ks_metrics = []

	ks_metrics.append(speed_calcs(data_given, dat_avg))
	ks_metrics.append(acceleration_calcs(data_given, dat_avg))
	ks_metrics.append(angle_calcs(data_given, dat_avg))

	tdf = get_skmob(data_given, min([len(data_given), len(dat_avg)]))

	ks_metrics.append(straight_line_distance_calcs(tdf, dat_avg))
	ks_metrics.append(max_dist_calcs(tdf,dat_avg))

	pd_tdf = get_ganpd(data_given, min([len(data_given), len(dat_avg)]))

	#TODO: Distance to straight line


	ks_metrics.append(direction_changes_calcs(pd_tdf, dat_avg))
	ks_metrics.append(triangle_area_calcs(pd_tdf, dat_avg))
	ks_metrics.append(max_from_start_calcs(pd_tdf, dat_avg))
	ks_metrics.append(crossings_calcs(pd_tdf, dat_avg))


	print(ks_metrics)



'''[[KstestResult(statistic=0.6049012253063266, pvalue=0.0), KstestResult(statistic=0.9992498124531133, pvalue=0.0)], [KstestResult(statistic=0.9997499374843711, pvalue=0.0), KstestResult(statistic=0.9997499374843711, pvalue=0.0)], [KstestResult(statistic=0.34783695923980995, pvalue=5.53005669150203e-215), KstestResult(statistic=0.9414853713428357, pvalue=0.0)], [KstestResult(statistic=0.980557748812203, pvalue=0.0)], [KstestResult(statistic=0.9930921245936484, pvalue=0.0)], [KstestResult(statistic=0.9959989997499374, pvalue=0.0)], [KstestResult(statistic=0.9987496874218554, pvalue=0.0)], [KstestResult(statistic=0.9924981245311327, pvalue=0.0)], [KstestResult(statistic=0.5481370342585646, pvalue=0.0)]]
'''

