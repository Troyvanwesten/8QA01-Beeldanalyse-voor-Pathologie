import numpy as np
import cv2
import math
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from scipy.spatial.distance import cdist
from scipy.stats.stats import mode
from sklearn.neighbors import NearestCentroid
from sklearn.utils import shuffle
from sklearn.cluster import KMeans
from sklearn.neighbors import KNeighborsClassifier
from tqdm import tqdm
import xlsxwriter as ex

def read_files():
    file = open("all_labels.csv")
    file_read = file.readlines()
    file.close()
    
    lesion_list = []
    
    for lines in file_read[1:]:
        lines = tuple(lines.rstrip().split(","))
        lesion_list.append(("{}_segmentation.png".format(lines[0]),"{}.jpg".format(lines[0])))
    
    return lesion_list

"""READ IMAGES AND CONVERT THEM"""
def img_conversion(mask_file,lesion_file):
    mask = cv2.imread(mask_file, cv2.IMREAD_GRAYSCALE)
    lesion = cv2.imread(lesion_file, cv2.COLOR_BGR2RGB)
    lesion = cv2.bitwise_and(lesion, lesion, mask=mask)
    
    height, width = mask.shape[:2]
    centre = (width // 2, height // 2)
    
    thresh = cv2.threshold(mask, 0, 255, cv2.THRESH_BINARY)[1]    
    contours = cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)[0]
    
    angle = cv2.fitEllipse(contours[0])[2] - 90
    moment = cv2.getRotationMatrix2D(centre, angle, 1.0)
    
    mask = cv2.warpAffine(mask, moment, (width, height))
    lesion = cv2.warpAffine(lesion, moment, (width, height))
    
    return (mask, lesion)

"""EVALUATE LESIONS"""
def border_evaluation(mask):       
    border_blanc = np.zeros((mask.shape[0], mask.shape[1], 3), np.uint8)
    
    thresh = cv2.threshold(mask, 0, 255, cv2.THRESH_BINARY)[1]    
    contours = cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)[0]
        
    border = cv2.drawContours(border_blanc,contours, 0, (255, 255, 255), 1)
    
    length_border = np.sum(border == 255)
    
    return length_border

def color_cluster_evaluation(lesion, mask, cluster_aantal = 5, HSV = False):
    area = np.sum(mask == 255)
    if HSV == True:
        lesion = cv2.cvtColor(lesion, cv2.COLOR_RGB2HSV)
    
    w, h, d = tuple(lesion.shape) # Sla afmetingen op
    image_array = np.reshape(lesion, (w * h, d)) # Zet alle pixels onder elkaar
    image_array_sample = shuffle(image_array, random_state=0)[:10000] # Alleen de eerste random 10000
    # pixels worden meegenomen in het bepalen van de clusters zodat het niet een uur duurt
    
    kmeans = KMeans(n_clusters=cluster_aantal, random_state=0).fit(image_array_sample)
    # Maak clusters
    
    centroids = kmeans.cluster_centers_
    # Zwaartepunten of midden van clusters

    D = cdist(centroids, centroids, metric='seuclidean') # Afstand tussen alle midden clusters
    totaal = 0
    for i in range(cluster_aantal):
        for j in range(cluster_aantal):
            if i<j:
                totaal = totaal + D[i, j] # Gemiddelde afstand berekenen
                
    return (totaal/((cluster_aantal*(cluster_aantal-1))/2), area)

def colour_evaluation(lesion, mask):
    mask_inv = 255 - mask
    area = np.sum(mask == 255)
    colour_score = 0
    
    colours = {'light brown low':(255*0.588, 255*0.2, 255*0),
              'light brown high':(255*0.94, 255*0.588, 255*392),
              'dark brown low':(255*0.243, 255*0, 255*0),
              'dark borwn high':(255*56, 255*0.392, 255*392),
              'white low':(255*0.8, 255*0.8, 255*0.8),
              'white high':(255, 255, 255),
              'red low':(255*0.588, 255*0, 255*0),
              'red high':(255, 255*0.19, 255*0.19),
              'blue gray low':(255*0, 255*0.392, 255*0.490),
              'blue gray high':(255*0.588, 255*0.588, 255*0.588),
              'black low':(255*0, 255*0, 255*0),
              'black high':(255*0.243, 255*0.243, 255*0.243)}
    
    for i in range(0,len(colours),2):
        mask_colour = cv2.inRange(lesion, colours.get(list(colours.keys())[i]), colours.get(list(colours.keys())[i+1]))
    
        if list(colours.keys())[i] == list(colours.keys())[-2] and list(colours.keys())[i+1] == list(colours.keys())[-1]:
            mask_colour = mask_colour - mask_inv
        
        if (np.sum(mask_colour == 255) / area) >= 0.05:
            colour_score += 1
            
    return (area,colour_score)
    
def symmetry_evaluation(mask):
    height, width = mask.shape[:2]
    moment = cv2.moments(mask)
    
    centre_blob_x = int(moment["m10"] / moment["m00"])
    centre_blob_y = int(moment["m01"] / moment["m00"])

    superior = mask[0:centre_blob_y, 0:width]
    inferior = mask[centre_blob_y:height, 0:width]
    inferior = cv2.flip(inferior, 0)
    
    left = mask[0:height, 0:centre_blob_x]
    left = cv2.flip(left, 1)
    right = mask[0:height, centre_blob_x:width]
    
    if superior.shape[0] > inferior.shape[0]:
        inferior = cv2.copyMakeBorder(inferior, superior.shape[0]-inferior.shape[0], None, None, None, 0, None, None)                     
        horizontal_result = superior - inferior
        
    if superior.shape[0] < inferior.shape[0]:
        superior = cv2.copyMakeBorder(superior, inferior.shape[0]-superior.shape[0], None, None, None, 0, None, None)  
        horizontal_result = inferior - superior
    
    if left.shape[1] > right.shape[1]:
        right = cv2.copyMakeBorder(right, None, None, None, left.shape[1]-right.shape[1], 0, None, None)
        vertical_result = left - right
        
    if left.shape[1] < right.shape[1]:
        left = cv2.copyMakeBorder(left, None, None, None, right.shape[1]-left.shape[1], 0, None, None)
        vertical_result = right - left
        
    if left.shape[1] == right.shape[1]:  
        vertical_result = right - left
    
    if superior.shape[0] == inferior.shape[0]:
        horizontal_result = superior - inferior
        
        
    horizontal_result = np.sum(horizontal_result == 255)
    vertical_result = np.sum(vertical_result == 255)
    
    return (horizontal_result,vertical_result)


"""RETURN RESULTS OF LESIONS"""
def return_results():
    #OPEN DOCUMENT TO INSERT RESULTS
    document = ex.Workbook("all_results.xlsx")
    worksheet = document.add_worksheet()
    worksheet.write('A1','Index')
    worksheet.write('B1','Lng_bor')
    worksheet.write('C1','Area')
    worksheet.write('D1','Hor_overl')
    worksheet.write('E1','Vrt_overl')
    worksheet.write('F1','Clr_score_5clusters')
    worksheet.write('G1','Clr_score_intervallen')
    data = read_files()
    row = 1
    column = 0
    for fileset in tqdm(data):
        index = fileset[0][:-4]
        
        mask_file = "all_masks\{}".format(fileset[1])
        lesion_file = "all_laesies\{}".format(fileset[0])
        
        mask_file = "all_masks\{}".format(fileset[0])
        lesion_file = "all_laesies\{}".format(fileset[1])
        
        lesion = img_conversion(mask_file, lesion_file)[1]
        mask = img_conversion(mask_file, lesion_file)[0]
        
        lng_bor = border_evaluation(mask)
        area = color_cluster_evaluation(lesion, mask)[1]
        clrc_score = color_cluster_evaluation(lesion, mask)[0]
        clr_score = colour_evaluation(lesion, mask)[1]
        hor_overl = symmetry_evaluation(mask)[0]
        vrt_overl = symmetry_evaluation(mask)[1]

        worksheet.write(row, column, index)
        worksheet.write(row, column+1, lng_bor)
        worksheet.write(row, column+2, area)
        worksheet.write(row, column+3, hor_overl)
        worksheet.write(row, column+4, vrt_overl)
        worksheet.write(row, column+5, clrc_score)
        worksheet.write(row, column+6, clr_score)
      
        # incrementing the value of row by one 
        # with each iteratons. 
        row += 1
    document.close()
        
return_results()
