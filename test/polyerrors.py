import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# BEST DEGREE == 5

# plot errors per degree, show mean error for each degree, one for each cam

filename = "/threed_reconstruct/poly_degree_errors/poly_results.csv"
results = pd.read_csv(filename, index_col=False)

degree_range = [min(results['degree'].values), max(results['degree'].values)]
print('degree range:', degree_range[0], degree_range[1])

mind = int(degree_range[0])
maxd = int(degree_range[1])

for i in range(mind, maxd+1):
    deg_i = results[results["degree"].isin([i])]
    cam1 = deg_i[deg_i["Cam"].isin([1])]
    mean = [np.mean(cam1['RMSE'])] * 15

    save_name = "C:\\Users\\Blanca\\Documents\\CTRTracking\\threed_reconstruct\\poly_degree_errors\\rmse\\" + "cam1_" + str(i) + '.png'
    # graph
    plt.title('RMSE results for polynomial with degree='+ str(i))
    plt.plot(cam1['Image_idx'], mean, label=str(mean[0]))
    plt.plot(cam1['Image_idx'], cam1['RMSE'], label='RMSE per image', color='red')
    plt.legend()
    plt.savefig(save_name)
    plt.show()