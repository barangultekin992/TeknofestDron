import numpy as np
import scipy 
#devamlı hata modeli ortalaması 0 olacak şeklinde tahminlerde bulunan parazit modeli.
def Q_continuous(dim,Q,dt):
    if dim == 1:
        return np.array([Q * dt])
    if dim == 2:
        return np.array([[dt**3 /3,dt ** 2/2],
                          [dt ** 2/2, dt]]) * Q
    if dim == 3:
        return np.array([[(dt**5)/20., (dt**4)/8., (dt**3)/6.],
             [ (dt**4)/8., (dt**3)/3., (dt**2)/2.],
             [ (dt**3)/6., (dt**2)/2.,        dt]]) * Q
    if dim == 4:
        return np.array([[(dt**7)/252., (dt**6)/72., (dt**5)/30., (dt**4)/24.],
             [(dt**6)/72.,  (dt**5)/20., (dt**4)/8.,  (dt**3)/6.],
             [(dt**5)/30.,  (dt**4)/8.,  (dt**3)/3.,  (dt**2)/2.],
             [(dt**4)/24.,  (dt**3)/6.,  (dt**2)/2.,   dt]]) * Q
    else:
        raise ValueError("boyut 2 ile 4 arasinda olmali")

#ayrı ayrı parazit(hata) modeli. Bu modelde dt zamanı içinde ivmenin değişmediği varsayılır. 
# akat dt ler arasında değişebilmektedir.
def Q_discrete(dim,Q,dt):
    if dim == 2:
        v = np.array([dt**2/2,dt])
        return np.outer(v,v) * Q
    if dim == 3:
        v = np.array([dt**2/2,dt,1])
        return np.outer(v , v) * Q
    if dim == 4:
        v = np.array([dt**3/6,dt**2/2,dt,1])
        return np.outer(v,  v) * Q
    else:
        raise ValueError("boyut 2 ile 4 arasinda olmali")

#Unscented Kalman Filter(kokusuz kalman filtresi) kokusuz burada matematiksel bir anlam ifade etmiyor. 
#Sadece filtrenin mucidi tarafından yayılan bir şaka
class UKF():
    def __init__(self,dim_x,dim_z,fx,hx,x=None,P=None,R=1.,Q=1.,residual_fnc = None,mean_fnc = None,dt=1.):
        
        if not isinstance(x, np.ndarray):
            self.x = np.zeros((dim_x,))
        else:
            self.x = x
        if not isinstance(P, np.ndarray):
            self.P = np.eye(dim_x) * 1e-3
        else:
            self.P = P
        if not isinstance(R, np.ndarray):
            self.R = np.eye(dim_z) * R
        else:
            self.R = R
        if not isinstance(Q, np.ndarray):
            self.Q = np.eye(dim_x) * Q
        else:
            self.Q = Q
        
        self.dt = dt  
        print(f"UKF initialized with dt={self.dt}, dim_x={dim_x}, dim_z={dim_z}, x={x}, P={P}, R={R}, Q={Q}")
        
        self.fx = fx
        self.hx=hx
        self.mean_fnc = mean_fnc
        self.residual_fnc = residual_fnc
        if self.residual_fnc is None:
            self.residual_fnc = np.subtract

        self.dim_x = dim_x
        self.dim_z = dim_z

        self.K = np.zeros((self.dim_x,self.dim_z)) #Kalman kazanımı
        self.y = np.zeros((dim_z))

#sigma noktaları için ağırlıkları hesaplamak için en çok kullanılan 
#yol. mucidinin ismini almış
    
    def Van_der_merve_weights(self,alpha,beta,kappa):
        self.lambda_ = alpha ** 2 * (self.dim_x + kappa) - self.dim_x 
        self.Wm = np.full(2*self.dim_x + 1,1/ (2 * (self.dim_x + self.lambda_)))
        self.Wc = np.full(2*self.dim_x + 1,1/ (2 * (self.dim_x + self.lambda_)))
        self.Wm[0] = self.lambda_/(self.lambda_ + self.dim_x) 
        self.Wc[0] = self.lambda_/(self.lambda_ + self.dim_x) + beta + 1 - alpha ** 2
        return self.Wm,self.Wc,self.lambda_
    
    def sigma_points(self,mean,cov,lambda_):
        sigmas = np.empty((2*self.dim_x + 1, self.dim_x))
        print(f"cholesky^2{(lambda_ + self.dim_x) * cov}")
        try:
            sqrt = scipy.linalg.cholesky((lambda_ + self.dim_x) * cov + 1e-6 * np.eye(self.dim_x))
        except scipy.linalg.LinAlgError:
            # Fallback to SVD if Cholesky fails
            U, s, Vt = np.linalg.svd((lambda_ + self.dim_x) * cov + 1e-6 * np.eye(self.dim_x))
            sqrt = U @ np.diag(np.sqrt(s)) @ Vt
            
        sigmas[0] = mean
        print(sqrt)
        for i in range(0,self.dim_x):
            sigmas[i+1] = mean + sqrt[i, :]  # Fixed: use proper row indexing
            sigmas[i + 1 + self.dim_x] = mean - sqrt[i, :]  # Fixed: use proper row indexing
        return sigmas
    
    def unscented_transform(self,sigmas,Wm,Wc,noise=None,mean_fnc=None,residual_fnc=None):
        if mean_fnc is None:
            X = np.dot(Wm,sigmas)
            print(f"X = {X}")
        else:
            X = mean_fnc(Wm,sigmas)
        kmax,n = sigmas.shape
        print("geçti2")    
        if residual_fnc == None or residual_fnc == np.subtract:
            y=sigmas - X[np.newaxis,:]
            print("geçti3")
            # Fixed: Proper covariance calculation with weights
            P = np.zeros((n, n))
            for i in range(kmax):
                diff = sigmas[i] - X
                P += Wc[i] * np.outer(diff, diff)
            print("geçti4")
            
        else:
            P=np.zeros((n,n))
            print("geçti5")
            for i in range(kmax):
                print("geçti6")
                y = residual_fnc(sigmas[i],X)
                print(f"y={y}")
                P += Wc[i] * np.outer(y,y)
        if noise is not None:
            print(f"noise={noise}")
            P+=noise
        print(f"unscented_transform: X={X}, P={P}")
        return X,P
    
    def predict(self, points, *args):
        self.sigmas_fn = np.empty_like(points)
        for i in range(len(points)):
            self.sigmas_fn[i] = self.fx(points[i], *args)
        print("geçti")
        # Fixed: Add process noise to transformed sigma points
        self.sigmas_fn_noisy = self.sigmas_fn.copy()
        if hasattr(self, 'Q') and self.Q is not None:
            for i in range(len(self.sigmas_fn)):
                noise = np.random.multivariate_normal(np.zeros(self.dim_x), self.Q)
                self.sigmas_fn_noisy[i] += noise
                
        self.x_p, self.p_p = self.unscented_transform(
            self.sigmas_fn_noisy, self.Wm, self.Wc, None, self.mean_fnc
        )
        print(f"predicted state: {self.x_p}, predicted covariance: {self.p_p}")
        self.x = self.x_p.copy()  # Tahmin edilen durum
        self.P = self.p_p.copy()  # Tahmin edilen kovaryans
        print("UKF predict step completed.")
        print(f"UKF predict: x_p={self.x_p}, P_p={self.p_p}")
    
    def cross_multiply(self,x,z,sigmas_fx,sigmas_hx):
        if len(sigmas_fx) != len(sigmas_hx):
            raise ValueError("cross_multiply: len(sigmas_fx) != len(sigmas_hx)")
        # Fixed: Use proper dimensions
        Pxz = np.zeros((self.dim_x, self.dim_z))
        for i in range(len(sigmas_fx)):
            dx = self.residual_fnc(sigmas_fx[i] , x)
            dy = self.residual_fnc(sigmas_hx[i] , z)
            Pxz += self.Wc[i] * np.outer(dx,dy)
        return Pxz

    def update(self,z,*args):        
        sigmas_hx = np.empty((len(self.sigmas_fn), self.dim_z))
        for i in range(len(self.sigmas_fn)):
            sigmas_hx[i] = self.hx(self.sigmas_fn[i],*args)

        self.x_h,self.S = self.unscented_transform(sigmas_hx,self.Wm,self.Wc,self.R,self.mean_fnc,self.residual_fnc)
        # Fixed: Add numerical stability check
        try:
            self.SI = np.linalg.inv(self.S)
        except np.linalg.LinAlgError:
            # Add small regularization if matrix is singular
            self.SI = np.linalg.inv(self.S + 1e-6 * np.eye(self.dim_z))
            
        self.y = self.residual_fnc(z,self.x_h)
        self.K = np.dot(self.cross_multiply(self.x_p,self.x_h,self.sigmas_fn,sigmas_hx),self.SI) 
        self.x = self.x_p + np.dot(self.K,self.y)
        self.P = self.p_p - np.dot(np.dot(self.K,self.S),self.K.T)

    def compute_mahalonobis(self):
        """
        mahalonobis uzakligi 3 demek ölçümün tahminden 3 standart sapma uzakta olduğunu gösterir
        """
        self._mahalanobis = np.sqrt(float(np.dot(np.dot(self.y.T, self.SI), self.y)))
        return self._mahalanobis
    def get_mahalanobis(self):
        if not hasattr(self, '_mahalanobis'):
            self.compute_mahalonobis()
        return self._mahalanobis
    
    #verileri ayarlamak ve görmek için methodlar
    def get_state(self):    return self.x
    def get_covariance(self):   return self.P

    def set_measurement_noise(self, R): self.R = R
    def set_process_noise(self, Q): self.Q = Q
    def set_dt(self, dt):   self.dt = dt

    def __str__(self):
        return f"UKF(dim_x={self.dim_x}, dim_z={self.dim_z}, fx={self.fx}, hx={self.hx}, x={self.x}, P={self.P}, R={self.R}, Q={self.Q})"
    def __repr__(self):
        return f"UKF(dim_x={self.dim_x}, dim_z={self.dim_z}, fx={self.fx}, hx={self.hx}, x={self.x}, P={self.P}, R={self.R}, Q={self.Q})"
    