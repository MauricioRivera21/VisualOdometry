import numpy as np
import cv2 as cv 
import os
from utils import *

from matplotlib import pyplot as plt
#from tqdm import tqdm

class VisualOdometry(object):
    """
    Esta clase implementa un algoritmo de Monocular Visual Odometry aplicando
    una correspondencia 2D-2D para la actualización del pose de la cámara. El trabajo
    ha sido adaptado únicamente para el KITTI dataset 

    Input: Dir_path (string): Dirección relativa o global de la carpeta del dataset
           FEATURE_PROCESSING (int): Flag para activar el procesamiento adicional 
                                     sobre los features
    """
    def __init__(self,dir_path, mask_perct, FEATURE_PROCESSING = 1):
        self.images       = self._read_images(os.path.join(dir_path,"image_l"))
        self.gt_poses     = self._read_gtposes(os.path.join(dir_path,"poses.txt")) 
        self.K,self.P     = self._read_calib_matrix(os.path.join(dir_path,"calib.txt")) 
        
        self.detector     = cv.ORB_create(5000)
        self.mask_perct   = mask_perct

        self.index_params = dict(algorithm=6,
                            table_number=6,
                            key_size=12,
                            multi_probe_level=2)
        self.search_params = dict(checks = 50)
        self.matcher      =  cv.FlannBasedMatcher(self.index_params, self.search_params)
        
        self.FEATURE_PROCESSING = FEATURE_PROCESSING
    
    @staticmethod
    def _read_images(dir_image):
        """
        _read_images: Obtiene las images del dataset KITTI en grayscale

        Output: list_img (list of 50 imgs (1336,652))
        """
        list_img = []
        for img_path in os.listdir(dir_image):
            img = cv.imread(os.path.join(dir_image,img_path), cv.IMREAD_GRAYSCALE)
            list_img.append(img)
        return list_img

    @staticmethod
    def _read_calib_matrix(dir_calib):
        """
        _read_calib_matrix: Lee los .txt del dataset y extrae los parámetros
                            de la cámara izquierda 
        Output: K (np.array [3,3])
                P (np.array [4,4])
        """
        params = np.loadtxt(dir_calib, dtype=np.float64)
        P = np.reshape(params[0], (3, 4))
        K = P[0:3, 0:3]

        return K,P

    @staticmethod
    def _read_gtposes(dir_gtposes):
        """
        _read_gtposes: Recupera todas las poses verdaderas o ground
                       truth de la cámara para comparar resultados
        Output: poses [list of 50 np.array(4,4)]: Se usa el formato
                de transformación homogenea
        """
        pose_arrays = np.loadtxt(dir_gtposes, dtype=np.float64)
        poses = []
        for T in pose_arrays:
            T = T.reshape(3,4)
            T = np.vstack((T,[0,0,0,1]))
            poses.append(T)
        return poses
    
    def _generate_pose_matrix(self,R,t):
        """
        _generate_pose_matrix: Junta la matriz de rotación R y el vector
                               de traslación t en una transformación 
                               homogenea T
        Output: T [np.array(4,4)]
        """
        T = np.eye(4, dtype=np.float64)
        T[:3,:3] = R
        T[:3,3]  = np.squeeze(t,axis= 1)
        return T

    def getAbsoluteScale(self, img_id): 
        """
        getAbsoluteScale: Retorno el factor de escala aplicado en 
                          la traslación producto que la correspondencia
                          2D-2D no puede obtener la escala de la vista
                          real 
        Output: scale (float)
        """
        gt_0 = self.gt_poses[img_id-1][:3,3]
        gt = self.gt_poses[img_id][:3,3]
        return np.linalg.norm(gt_0 - gt)
	
    def extract_features(self,img_id, var_error, umbral):
        """
        extract_features: Detecta los features de un frame por ORB
                          y genera su descriptor. Esto se aplica a
                          dos vistas (k-1,k)

        Output: k1, k2 (cv2.Keypoints)
                des1,des2 (cv2.Descriptors)
        """
        img_prev = self.images[img_id-1]
        img_cur  = self.images[img_id]

        if self.FEATURE_PROCESSING:
            img_prev, umbral = proyect_feature_processing(img_prev, var_error, umbral)
            img_cur, umbral = proyect_feature_processing(img_cur, var_error, umbral)

        mask = np.zeros((img_cur.shape[0],img_cur.shape[1]), np.uint8)
        cv.rectangle(mask,(0,0),(img_cur.shape[1],self.mask_perct*img_cur.shape[0]//10),255,-1)

        kp1 = self.detector.detect(img_prev,mask)     
        kp2 = self.detector.detect(img_cur,mask)     

        kp1, des1 = self.detector.compute(img_prev, kp1, mask)
        kp2, des2 = self.detector.compute(img_cur , kp2,mask)

        #print(des2)

        return kp1,kp2,des1,des2, umbral

    def process_Frame(self,kp1,kp2,des1,des2, ratio_distance = 0.75):
        """
        process_Frame: Procesa los keypoints y descriptores para aplicar
                       el matching por KNN utilizando la implementación
                       FLANN de cv2. Estos ya son los 2D points que pueden
                       usarse en la gran mayoria de algoritmos de VO.
        Output: q1,q2 [cv2.Keypoints]
        """

        matches = self.matcher.knnMatch(des1, des2, k=2)
        good = []

        if len(matches) < 1500 or not matches:                     #matches --> tuple
            print("no se encontraron buenos matches")
            return False,False
        
        #print(matches[0])
        #print(matches[1])
        for m,n in matches:
            if m.distance < ratio_distance*n.distance:
                good.append(m)

        q1 = np.float32([ kp1[m.queryIdx].pt for m in good ])
        q2 = np.float32([ kp2[m.trainIdx].pt for m in good ])
        
        return q1,q2

    def estimate_motion_2D(self,img_id, var_error, umbral):
        """
        estimate_motion_2D: Aplica la correspondenci 2D-2D donde 
                            se genera la transformación que lleva
                            el pose(k-1) a pose(k) por la matriz esencial.
        Output: T [np.array(4,4)]
                R [mp.array(3,3)]
                t [np.array(3,1)]
        """
        kp1,kp2,des1,des2, umbral = self.extract_features(img_id, var_error, umbral)
        q1 , q2         = self.process_Frame(kp1,kp2,des1,des2)
        if isinstance(q1,bool):
           T = np.eye(4)
           R = np.eye(3)
           t = np.zeros((3,1))
           return T,R,t, umbral

        E, mask         = cv.findEssentialMat(q1, q2, self.K, method=cv.RANSAC, prob=0.999, threshold=1.0)
        _,R,t,mask      = cv.recoverPose(E,q1,q2,self.K)
        T = self._generate_pose_matrix(R,t)

        return T,R,t, umbral

    
def main():

    PROCESS = 1
    path      = 'KITTI_sequence_2'
    vo        = VisualOdometry(path,FEATURE_PROCESSING = PROCESS, mask_perct=10)
    play_video(path,vo,SHOW_FEATURES = 1, PROCESS= PROCESS)
    gt_results = [] 
    vo_results = []

    var_error = 0.0
    umbral = 190

    """
    2D-2D Correspondence
    """

    for i, gt_pose in enumerate(vo.gt_poses):
        print(f"Prcdocesando Frame ",i)
        if i == 0:
            cur_pose = gt_pose
            cur_t    = cur_pose[:3,3]
            cur_R    = cur_pose[:3,:3]
        else:
            _,R,t, umbral = vo.estimate_motion_2D(i, var_error, umbral)
            scale      = vo.getAbsoluteScale(i)
            if (scale > 0.1):
                transf = vo._generate_pose_matrix(R,scale*t)
                cur_pose  = np.matmul(cur_pose,np.linalg.inv(transf))
        gt_results.append((gt_pose[0,3],gt_pose[2,3]))
        vo_results.append((cur_pose[0,3],cur_pose[2,3]))
        var_error = variacion_error(gt_results, vo_results, i)
        #error.append((abs(gt_pose[0,3]-cur_pose[0,3]) , abs(gt_pose[2,3] - cur_pose[2,3])))
        

    plot_error(gt_results,vo_results,path)
    #print(error)

if __name__ == "__main__":
    main()