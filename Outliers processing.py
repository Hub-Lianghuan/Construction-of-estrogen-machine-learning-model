
import numpy as np
from sklearn.neighbors import LocalOutlierFactor as LOF
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest
from scipy.spatial import distance
from sklearn.neighbors import NearestNeighbors
import matplotlib.pyplot as plt
from pyod.models.hbos import HBOS

data = pd.read_excel(r"C:\Users\MarsTao\Desktop\model_wu\修改4种重金属（含因子、浓度、风险）.xlsx",sheet_name='Sheet1')
x = []
for i in range(data.shape[0]):
    list = data.iloc[i,3:-2].to_list()
    x.append(list)

clf = LOF(n_neighbors=20, contamination=0.1)


y_pred = clf.fit_predict(x)  
X_scores = clf.negative_outlier_factor_  
scores = np.array(X_scores).reshape((data.shape[0],1))
scaler_lof = StandardScaler()
lof_scaled = scaler_lof.fit_transform(scores)

# T_F = np.array(y_pred).reshape((data.shape[0],1))
# result_lof = np.concatenate((scores,T_F),axis=1)


model = IsolationForest(contamination=0.1)
model.fit(x)


outliers = model.predict(x)

IS_scores = np.array(model.decision_function(x)).reshape(-1,1)*(-1)
scaler_is = StandardScaler()
is_scaled = scaler_lof.fit_transform(IS_scores)

def calculate_mahalanobis_score(data):
   
    mean = np.mean(data, axis=0)
    cov_matrix = np.cov(data, rowvar=False)

 
    inv_cov_matrix = np.linalg.inv(cov_matrix)

    
    mahalanobis_scores = []
    for point in data:
        diff = point - mean
        raw_distance = np.dot(np.dot(diff, inv_cov_matrix), diff.T)
        mahalanobis_distance = np.sqrt(np.maximum(0, raw_distance))
        mahalanobis_scores.append(mahalanobis_distance)
    return mahalanobis_scores


MCD_scores = calculate_mahalanobis_score(x)
MCD_scores=np.array(MCD_scores).reshape(-1,1)
scaler_MCD = StandardScaler()
mcd_scaled = scaler_lof.fit_transform(MCD_scores)


knn = NearestNeighbors(n_neighbors=5)

knn.fit(x)

distances, indices = knn.kneighbors(x)

avg_distances = distances.mean(axis=1)
knn_scores = np.array(avg_distances).reshape(-1,1)
scaler_knn = StandardScaler()
knn_scaled = scaler_lof.fit_transform(knn_scores)




clf = HBOS(n_bins=10, alpha=0.1, tol=0.5, contamination=0.1)


clf.fit(x)



df = pd.DataFrame(
    np.array(x),
    # columns=['Feature1', 'Feature2', 'Feature3', 'Feature4', 'Feature5']
)

# for column in df.columns:
#     plt.hist(df[column], bins=30, edgecolor='black')
#     plt.title(f'{column} Histogram')
#     plt.xlabel('Feature Values')
#     plt.ylabel('Frequency')
#     plt.show()


y_test_pred = clf.decision_function(x)  # outlier scores
hbos_scores = np.array(y_test_pred).reshape(-1,1)
scaler_hbos = StandardScaler()
hbos_scaled = scaler_lof.fit_transform(hbos_scores)

result_ = np.concatenate((lof_scaled,knn_scaled,is_scaled,hbos_scaled,mcd_scaled),axis=1)
total=np.mean(result_,axis=1).reshape(-1,1)
result = np.concatenate((result_,total),axis=1)


header = ['lof_score','knn_score','if_score','hbos_score','mcd_score','total_score']

header_str = ','.join(header)


with open(r"C:\Users\MarsTao\Desktop\model_wu\valid.csv", 'w') as f:
    f.write(header_str)
    f.write('\n')
with open(r"C:\Users\MarsTao\Desktop\model_wu\valid.csv", 'w') as f:
    np.savetxt(f, result, delimiter=',')
