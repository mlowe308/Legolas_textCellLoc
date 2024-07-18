from core import *
import utils
import time as time
from pathlib import Path
import numpy as np

host_1 = "192.168.1.11"
host_2 = "192.168.1.14"

utils.restart_server(host=host_1)
utils.restart_server(host=host_2)

#After speaking with Gilad, this Jupyter cell was cleaned up for GP exploration
#Plotting to a file was added. All images are in subdir ./plots

import GPy

stage, depo_device, pH_device, conn1, conn2, config = load_from_config("config.yaml")

def clean():
    pH_device.move_to_loc("clean")
    pH_device.to_zpos("full_down")
    pH_device.to_zpos("full_up")

def blot():
    pH_device.move_to_loc("blot")
    pH_device.to_zpos("full_down")
    time.sleep(1)
    pH_device.to_zpos("full_up")

def measure(j,i):
    pH_device.move_to_cell(row=j, col=i)
    pH_device.to_zpos("full_down")
    k = pH_device.pH_measure(30)
    pH_device.to_zpos("full_up")
    return k

def depclean():
    depo_device.move_to_loc("clean")
    depo_device.to_zpos("full_down")
    depo_device.to_zpos("full_up")

def get_row_col(count):
    if count <= 5:
        row = 0
        col = count
    elif count <= 11:
        row = 1
        col = count - 6
    elif count <= 17:
        row = 2
        col = count - 12
    elif count <= 23:
        row = 3
        col = count - 18
    return row, col

def Bayesian_exploration_pH():
    # Uses GPy for Gaussian Process regression as surrogate function.

    # Set up file
    filename1="./plots/foo_measurements.txt"  #contains acid-base ratio and pH measurements
    f1=open(filename1,"w")
    f1.write("count     ratio      pH\n")

    # X_grid is the list of all possible acid-base ratios that can be investigated.
    # This is the array that will be indexed with next_sample_index
    X_grid = np.linspace(0.1,10,40)[:,None]  #no. pts has to be greater than loop iterations
    #Dsize = X_grid.shape # X_grid has number rows x 1 column
    #print(f'x grid: {X_grid}, Dsize : {Dsize}')

    # set up variables
    ratio = [.1] # acid-base ratio of initial sample to study, 0 is no acid
    sample_index = 0 # index of ratio in list of ratios
    count = 0 #counter for moving to appropriate well

    print('count value:',count)
    print('Acid/Base ratio measured:',ratio)

    # deposit first ratio in well and collect pH
    pH = BO_get_data(ratio,count)
    print('count value:',count)
    print('Acid/Base ratio measured:',ratio)
    #print('pH Value:',pH)
    f1.write("%5d, %8.2f, %8.2f\n" % (count, ratio[0], pH) )

    measured = np.atleast_1d(sample_index) # indices of ratios that have been measured
    full_indices = np.linspace(0, X_grid.shape[0]-1, X_grid.shape[0]) # indices of all ratios to be investigated
    unmeasured = np.setdiff1d(full_indices, measured).astype(int)  # indices of ratios that are still to be measured
    X_samples = np.atleast_1d(ratio)[:,None] # X_samples is the acid-base ratios already studied
    Y_samples = np.atleast_1d(pH)[:,None] # Y_samples are the corresponding pH for the measured ratios.

    # iteration loop for active learning (GP with exploration)
    for iterations in range(11):

        # Regression. Amplitude and length are built into RBF. GP fits to (or learns from) data.
        k = GPy.kern.RBF(1)
        m = GPy.models.GPRegression(X_samples, Y_samples, k)

        #blockPrint() # blocks printing statements to avoid printing GPy's optimization statements.
        m.optimize_restarts(5, robust=True);
        #try 5 different GP runs and initialize each differently. Take best result.
        #enablePrint() # restarts the internal printing statements

        mean_full, variance_full = m.predict(X_grid, full_cov = False) # Prediction. full_cov default if False?
        mean, variance = m.predict(X_grid[unmeasured]) # Prediction just for unmeasured ratios
        #print('Variance_full is\n', variance_full)

        # Active Learning by exploration
        alpha_full = variance_full # variance for all ratios. Use variance (uncertainty) in Gaussian Process
                                   # to guide next sample
        alpha = variance # variance for unmeasured ratios
        sample_index = unmeasured[ np.argmax(alpha) ] # index of next ratio in X_grid
        ratio = X_grid[sample_index,:] # next ratio which is a one-element array

        print('The next ratio to investigate is', ratio)

        # plot GP variance for all ratios
        plt.figure(figsize=(7,2))
        if iterations>=3:
            plt.ylim(2,7)  #Can set y-axis limits
        plot_gp(X_grid, mean_full, variance_full, iterations, training_points=(X_samples,Y_samples))

        # plot
        num_subplots = 3

        plt.figure(figsize = (10,2))
        plt.subplot(1,num_subplots,2)
        plt.plot(X_grid, alpha_full)  # plot the acquisition function for all ratios
        plt.plot([ratio, ratio],[np.min(alpha_full), np.max(alpha_full)],'m') # indicate the next ratio to be investigated
        plt.title(f'Acquisition func {iterations}')

        plt.subplot(1,num_subplots,1)
        plt.plot(X_grid, 2*np.sqrt(alpha_full))
        plt.title(f'2x Standard deviation {iterations}')

        plt.subplot(1,num_subplots,3)
        plt.plot(X_grid, mean_full, "-")  # ???Plot GP mean. How to find max, min of GP to fix vert axes
        plt.title(f'mean {iterations}')
        plt.ylim(2, 7)
        plt.savefig(f"./plots/foosubplots {iterations}.png", facecolor='white')
        plt.show()

        #plt.close()

        count+=1           #move to next well

        print('count value:',count)
        print('Acid/Base ratio measured:',ratio)

        # collect data
        pH = BO_get_data(ratio, count) # run the next experiment
        print('counter value:',count)
        print('Acid/Base ratio measured:',ratio)
        #print('pH Value:',pH)
        f1.write("%5d, %8.2f, %8.2f\n" % (count, ratio[0], pH) )

        measured = np.append(measured, sample_index) # add experiment ratio to the set of measured
        unmeasured = np.setdiff1d(full_indices, measured).astype(int)
        X_samples = np.append(X_samples, ratio)[:,None]
        Y_samples = np.append(Y_samples, pH)[:,None]

    f1.close()

    return m

def fill(acid_vol,base_vol,count):
    #r = count % 6
    #c = count // 6
    if count <= 5:          # 1st row [0] - 6 samples
        r = 0
        c = count
    elif count <= 11:        # 2nd row [1]  - 6 samples ...
        r = 1
        c = count-6
    elif count <= 17:        # 3rd row [2] - 6 samples ...
        r = 2
        c = count - 12
    elif count <= 23:        # 4th row [3] - 6 sample ...
        r = 3
        c = count - 18

    print(f'row: {r}, col: {c}')

    while acid_vol > 0.5:
        depo_device.acquire(0.5, location="acid")
        depo_device.deposition(0.5, row=r, col=c)
        acid_vol -= 0.5
    depo_device.acquire(acid_vol, location="acid")
    depo_device.deposition(acid_vol, row=r, col=c)

    while base_vol > 0.5:
        depo_device.acquire(0.5, location="base")
        depo_device.deposition(0.5, row=r, col=c)
        base_vol -= 0.5
    depo_device.acquire(base_vol, location="base")
    depo_device.deposition(base_vol, row=r, col=c)

    return r,c


def BO_get_data(ratio, count):   #modified to generate fake data
    acid_vol,base_vol = ratio_conversion(ratio)
    print('acid_vol, base_vol:', acid_vol, base_vol)
    r,c = fill(acid_vol,base_vol,count)

    clean()
    blot()
    voltage = measure(r,c)
    pH = line[0]*voltage+line[1]

    print(f"pH: {pH}")
    return pH

def ratio_conversion(ratio):
    acid_vol = float(2.0*ratio[0]/(1+ratio[0]))
    base_vol = float(2.0 - acid_vol)
    return acid_vol,base_vol

def plot_gp(X, m, C, iterations, training_points=None):   #C is covariance, 1D ndarray
    # Plot results of Gaussian Process analysis.
    # Plot 95% confidence interval, alpha is opacity
    ##plt.fill_between(X[:,0], m[:,0] - 1.96*np.sqrt(np.diag(C)), m[:,0] + 1.96*np.sqrt(np.diag(C)), alpha=0.5)
    plt.fill_between(X[:,0], m[:,0] - 1.96*np.sqrt(C[:,0]), m[:,0] + 1.96*np.sqrt(C[:,0]), alpha=0.5)

    #plt.title('GP model for pH')
    plt.title(f"GP model {iterations} for pH")
    plt.plot(X, m, "-")  # Plot GP mean
    plt.xlabel("x"), plt.ylabel("f")

    if training_points is not None:  # Plot training points if included
        X_, Y_ = training_points
        plt.plot(X_, Y_, "kx", mew=2)
        plt.savefig(f"./plots/foo{iterations}.png", facecolor='white')
        plt.show(block=False)
        plt.pause(1)


pH_device.to_zpos("full_up")

stage.home()

#pH SENSOR CALIBRATION

y = [7, 4, 7, 4, 7, 4]
x = []

for i in range(6):
    stage.home()
    clean()
    blot()
    if i % 2 == 0:
        m7 = measure(3,1)
        x.append(m7)
    else:
        m4 = measure(3,0)
        x.append(m4)

#PLOTTING FIT

line = np.polyfit(x,y,1)
print(f'slope: {line[0]}, intercept: {line[1]}')
trend = np.poly1d(line)
plt.plot(x,y, 'ro', x, trend(x), '-k')
plt.show



m = Bayesian_exploration_pH()

plt.show()

print(f"pH 7 remeasured: {line[0]*measure(3,1)+line[1]}")
print(f"pH 4 remeasured: {line[0]*measure(3,0)+line[1]}")


#Surrogate model regression results in ability to interpolate to predict value of mean and variance
#at an unmeasured x.
mean, variance = m.predict(np.atleast_1d([1.2])[:,None]) # Prediction just for unmeasured ratios
print('mean,variance:', mean, variance)
print(m)
#over a range of 2 in x, there would be a change in y of Sqrt[55] (roughly)