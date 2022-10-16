import numpy as np
from datascience import *
import math as m
import qgrid as q
import pandas as pd
import re
import scipy.stats
import matplotlib.pyplot as plt
plt.rcParams["figure.figsize"] = (10,10)
import matplotlib.image as mpimg
from sklearn.decomposition import PCA, IncrementalPCA
import sklearn.manifold
import seaborn as sns
import scipy.signal
from scipy import ndimage as ndi
import skimage
from skimage import measure
from skimage.metrics import structural_similarity as ssim
import argparse
import imutils
import cv2
from skimage import feature
import tensorflow
import more_itertools as mit
import pandas as pd
import numpy as np
import os
import keras
import random
import cv2
import math
import seaborn as sns
from sklearn.cluster import AgglomerativeClustering
from sklearn.cluster import DBSCAN
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from functools import reduce
import matplotlib.pyplot as plt

from tensorflow.keras.layers import Dense,GlobalAveragePooling2D,Convolution2D,BatchNormalization
from tensorflow.keras.layers import Flatten,MaxPooling2D,Dropout

from tensorflow.keras.applications import DenseNet121
from tensorflow.keras.applications.densenet import preprocess_input

from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.image import ImageDataGenerator,img_to_array

from tensorflow.keras.models import Model

from tensorflow.keras.optimizers import Adam

from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau

import warnings
warnings.filterwarnings("ignore")

df = pd.read_csv("Data_Entry_2017_v2020.csv")
df['Distinct Diseases']=df['Finding Labels'].str.findall('[\w ]+')
patient_profile = df[['Patient ID', 'Follow-up #', 'Patient Age', 'Patient Gender', 'Distinct Diseases', 'View Position']]
grouped = df.groupby('Patient ID')

ls = []
for name, group in grouped:
    num = group['Follow-up #']
    dis = group['Distinct Diseases']
    ls += [dict(zip(num, dis))]

ids = patient_profile['Patient ID'].unique()

clinical_history = pd.DataFrame({'Patient ID': ids, 'Clinical History': ls})

def merge_lists(x, unique=True):
    if unique:
        return list(set([j for i in x for j in i]))
    return [j for i in x for j in i]

def retrieve_images(patient_id, show=False):
    this_df = df[df['Patient ID'] == patient_id]
    img_list = this_df['Image Index']
    for img_name in img_list:
        img = mpimg.imread("images/" + img_name)
        if show:
            imgplot = plt.imshow(img, cmap = plt.cm.gray)
            plt.show()
    return list(img_list)

def retrieve_record(patient_id):
    record = clinical_history.iloc[patient_id-1,:]['Clinical History']
    ls = []
    for visit in record:
        string = '|'.join(record[visit])
        ls += [string]
    patient_data = patient_profile[patient_profile['Patient ID'] == patient_id]
    return pd.DataFrame({'Age':patient_data['Patient Age'],'Gender':patient_data['Patient Gender'], 'View Position':patient_data['View Position'],'Visit Number':list(record.keys()), 'Clinical Histroy':ls})

def edges(images_id, show=False):
    img = mpimg.imread('images/' + images_id)
    try:
        edge = feature.canny(img[:,:,0], sigma=0.95)
    except:
        edge = feature.canny(img, sigma=0.95)
    if show:
        imshow(edges1, cmap=plt.cm.gray)
        plt.show()
    return edge

def patients_with_condition(condition):
    to_hist = patient_profile[['Patient ID','Distinct Diseases']].groupby('Patient ID').agg(merge_lists)
    with_condition = patient_profile[patient_profile["Distinct Diseases"].apply(lambda x: condition in x)]
    return with_condition

def find_similar_patients(patient_id, no_finding_control=False, small=True):
    record = retrieve_record(patient_id)
    age = list(record['Age'])[0]
    gender = list(record['Gender'])[0]
    view_positon = list(record['View Position'])[0]
    if no_finding_control:
        df = patients_with_condition('No Finding')
    else:
        df = patient_profile
    gendered = df[df['Patient Gender'] == gender]
    viewed = gendered[gendered['View Position'] == view_positon]
    aged = viewed[viewed['Patient Age'] <= age+5]
    final = aged[aged['Patient Age'] >= age-5]
    if final.shape[0] == 0:
        print("No matches found!")
        return None
    if small:
        return final[final['Patient ID'] < 1336]
    return final

def focus(float_arr, row=0, col=0):
    ls = [m.isclose(x, 0, abs_tol=0.01) for x in float_arr]
    boos = [i for i, x in enumerate(ls) if x]
    consecs = [list(group) for group in mit.consecutive_groups(boos)][0:-1]
    long_list = max(consecs, key=lambda y: len(y))
    if col:
        return [[col, long_list[0]],[col, long_list[-1]]]
    else:
        return [[long_list[0], row],[long_list[-1], row]]

def grid_overlay(img, step_size=1):
    df = pd.DataFrame(img)
    chords = []
    for col in np.arange(150,901,step_size):
        this_col = df[list(df.keys())[col]].astype(float)
        step_fn = scipy.signal.savgol_filter(this_col, 51, 1)
        try:
            chord = focus(step_fn, col=col)
        except:
            continue
        chords += [chord]
    for row in np.arange(200,800,step_size):
        this_row = df.iloc[[row]].T
        step_fn = scipy.signal.savgol_filter(np.ravel(this_row), 51, 1)
        try:
            chord = focus(step_fn, row=row)
        except:
            continue
        chords += [chord]
    return chords

def points_from_lines(lines):
    flat = [item for sublist in lines for item in sublist]
    clustering = DBSCAN(eps=20).fit(flat)
    x, y = np.array(flat).T
    plt.scatter(x, y, c=clustering.labels_)
    plt.gca().invert_yaxis()
    coords=pd.DataFrame({"x":x,"y":y,"label":clustering.labels_})
    return coords

def find_centroid(df):
    length = df.shape[0]
    sum_x = np.sum(df['x'])
    sum_y = np.sum(df['y'])
    return sum_x/length, sum_y/length

def centroid_deviance(df, centroid):
    x = list(df['x'])
    y = list(df['y'])
    distances = []
    for i in range(len(x)):
        point = np.array((x[i], y[i]))
        distance = np.linalg.norm(point - centroid)
        distances += [distance]
    return np.mean(distances), max(distances)

def make_centroid_df(df):
    labels = np.unique(df['label'])
    centroids = []
    deviances = []
    roundnesses = []
    max_deviance = []
    magnitudes = list(df.groupby("label").count()['x'])
    for label in labels:
        this_label = df[df['label']==label]
        if this_label.shape[0] == 0:
            roundnesses += [0]
            centroids += [[0,0]]
            deviances += [0]
            max_deviance += [0]
            continue
        roundnesses += [roundness(label, df)]
        centroid = find_centroid(this_label[['x','y']])
        deviance, maxd = centroid_deviance(this_label, centroid)
        centroids += [centroid]
        deviances += [deviance]
        max_deviance += [maxd]
    mid_df = pd.DataFrame({"label":labels,"centroid":centroids, "magnitude":magnitudes, "deviance":deviances, "max_deviance":max_deviance, "roundness":roundnesses})
    mid_df[['x_centroid','y_centroid']] = mid_df['centroid'].apply(pd.Series)
    return mid_df[['label','x_centroid','y_centroid','deviance','max_deviance','roundness','magnitude']]

def convex_hull_graham(points):
    '''
    Returns points on convex hull in CCW order according to Graham's scan algorithm.
    By Tom Switzer <thomas.switzer@gmail.com>.
    '''
    TURN_LEFT, TURN_RIGHT, TURN_NONE = (1, -1, 0)

    def cmp(a, b):
        return (a > b) - (a < b)

    def turn(p, q, r):
        return cmp((q[0] - p[0])*(r[1] - p[1]) - (r[0] - p[0])*(q[1] - p[1]), 0)

    def _keep_left(hull, r):
        while len(hull) > 1 and turn(hull[-2], hull[-1], r) != TURN_LEFT:
            hull.pop()
        if not len(hull) or hull[-1] != r:
            hull.append(r)
        return hull

    points = sorted(points)
    l = reduce(_keep_left, points, [])
    u = reduce(_keep_left, reversed(points), [])
    return l.extend(u[i] for i in range(1, len(u) - 1)) or l

def perimeter(hull):
    if hull == []:
        return 0
    length = 0
    last_point = np.array(hull[0])
    for point in hull:
        point = np.array(point)
        distance = np.linalg.norm(last_point - point)
        length += distance
        last_point = point
    return length + np.linalg.norm(last_point - np.array(hull[0]))

def area(hull):
    sep_hull = list(map(list, zip(*hull)))
    if sep_hull == []:
        return 0
    x = sep_hull[0]
    y = sep_hull[1]
    return 0.5*np.abs(np.dot(x,np.roll(y,1))-np.dot(y,np.roll(x,1)))

def roundness(label, coords):
    hull = convex_hull_graham(coords[coords['label']==label][['x','y']].values.tolist())
    a = area(hull)
    p = perimeter(hull)
    if a == 0 or p == 0:
        return 0
    return (4 * m.pi * a / (p ** 2))

def master_train(img, show=True):
    lines_=grid_overlay(img, step_size=1)
    flat_=[item for sublist in lines_ for item in sublist]
    clustering_=DBSCAN(eps=20).fit(flat_)
    x_, y_ = np.array(flat_).T
    coords_=pd.DataFrame({"x":x_,"y":y_,"label":clustering_.labels_})
    centroids_=make_centroid_df(coords_)
    max_roundness = max(centroids_['roundness'])
    ix_max_roundness = centroids_.index[centroids_.roundness == max_roundness]
    row = centroids_.loc[list(ix_max_roundness)[0], :]
    if show:
        fig, ax = plt.subplots()
        ax.scatter(x_, y_, c=clustering_.labels_)
        ax.scatter(centroids_.x_centroid, centroids_.y_centroid , c='r')

        for i, txt in enumerate(centroids_.label):
            ax.annotate(txt, (centroids_.x_centroid[i], centroids_.y_centroid[i]))

        plt.gca().invert_yaxis()

        drawing_uncolored_circle = plt.Circle((row['x_centroid'], row['y_centroid']), row.max_deviance,fill = False,color='r',lw=4)
        ax.add_artist(drawing_uncolored_circle)
    return row, centroids_
