# five_point_algorithm.m
import numpy as np
from scipy.linalg import qr,lu
from numpy.linalg import inv
from utility.error import InvalidDenomError

def qr_null(A, tol=None):
    Q, R, P = qr(A.T, mode='full', pivoting=True)
    tol = np.finfo(R.dtype).eps if tol is None else tol
    rnk = min(A.shape) - np.abs(np.diag(R))[::-1].searchsorted(tol)
    return Q[:, rnk:].conj()

   

def five_point_algorithm(pts1=None,pts2=None,K1=None,K2=None,*args,**kwargs):
    #FIVE_POINT_ALGORITHM Given five points matches between two images, and the
# intrinsic parameters of each camera. Estimate the essential matrix E, the 
# rotation matrix R and translation vector t, between both images. This 
# algorithm is based on the method described by David Nister in "An 
# Efficient Solution to the Five-Point Relative Pose Problem"
# DOI: http://dx.doi.org/10.1109/TPAMI.2004.17
    
    # E = FIVE_POINT_ALGORITHM(pts1, pts2, K1, K2) returns in E all the valid
# Essential matrix solutions for the five point correspondence. If you
# don't need the R and t, use this version as it avoids computing
# unnecessary results.
    
    # [E_all, R_all, t_all, Eo_all] = FIVE_POINT_ALGORITHM(pts1, pts2, K1, K2) 
# also returns in R_all and t_all all the rotation matrices and translation
# vectors of camera 2 for the different essential matrices, such that a 3D
# point in camera 1 reference frame can be transformed into the camera 2
# reference frame through p_2 = R{n}*p_1 + t{n}. Eo_all is the essential
# matrix before the imposing the structure U*diag([1 1 0])*V'. It should
# help get a better feeling on the accuracy of the solution. All these
# return values a nx1 cell arrays.
    
    
    # Arguments:
# pts1, pts2 - assumed to have dimension 2x5 and of equal size. 
# K1, K2 - 3x3 intrinsic parameters of cameras 1 and 2 respectively
    
    # Know Issues:
# - R and t computation is done assuming perfect point correspondence.
    
    # TODO:
# - Extract R and t without perfect point correspondence
# - Augment example cases.
# - Implement the variant with 5 points over 3 images
# - Handle more than 5 points
    
    # Author: Sergio Agostinho - sergio(dot)r(dot)agostinho(at)gmail(dot)com 
# Date: Feb 2015
# Last modified: Mar 2015
# Version: 0.9
# Repo: https://github.com/SergioRAgostinho/five_point_algorithm
# Feel free to provide feedback or contribute.
    
    if pts1.shape != (3, 5) or pts2.shape != (3,5):
        raise ValueError('five_point_algorithm:wrong_dimensions','pts1 and pts2 must be of size 2x5')
    
    if K1.shape != (3,3) or K2.shape != (3,3):
        raise ValueError('five_point_algorithm:wrong_dimensions','K1 and K2 must be of size 3x3')

    #q1= K1.I.dot( np.vstack((pts1, np.ones((1,N)))) )
    #q2= K2.I.dot( np.vstack((pts2, np.ones((1,N)))) )

    K1_I = inv(K1)
    K2_I = inv(K2)
 
    K2_TI = inv(K2.T)    
    q1= K1_I.dot(pts1)
    q2= K2_I.dot(pts2)

    #matrix q: 5*9   -Morgan

    q=np.array([np.multiply(q1[0,:],q2[0,:]),np.multiply(q1[1,:],q2[0,:]),\
                np.multiply(q1[2,:],q2[0,:]),np.multiply(q1[0,:],q2[1,:]), \
                np.multiply(q1[1,:],q2[1,:]),np.multiply(q1[2,:],q2[1,:]), \
                np.multiply(q1[0,:],q2[2,:]),np.multiply(q1[1,:],q2[2,:]), \
                np.multiply(q1[2,:],q2[2,:])]).T
    #according to the author, the null space step can be further optimized, 
#following the efficiency considerations in section 3.2.1
# Can be further expand it to N > 5 by extracting the four singular vectors
# corresponding to the four smallest singular values.
    nullSpace=qr_null(q)

    X=nullSpace[:,0]  
    Y=nullSpace[:,1]
    Z=nullSpace[:,2]
    W=nullSpace[:,3]

    # populating the equation system
    Xmat=X.reshape(3,3)   
    Ymat=Y.reshape(3,3)
    Zmat=Z.reshape(3,3)
    Wmat=W.reshape(3,3)

    X_ = K2_TI.dot(Xmat).dot(K1_I)
    Y_ = K2_TI.dot(Ymat).dot(K1_I)
    Z_= K2_TI.dot(Zmat).dot(K1_I)
    W_= K2_TI.dot(Wmat).dot(K1_I)

    #det(F)
    detF=p2p1(p1p1(cat(X_[0,1],Y_[0,1],Z_[0,1],W_[0,1]),cat(X_[1,2],Y_[1,2],Z_[1,2],W_[1,2])) - \
            p1p1(cat(X_[0,2],Y_[0,2],Z_[0,2],W_[0,2]),cat(X_[1,1],Y_[1,1],Z_[1,1],W_[1,1])),
            cat(X_[2,0],Y_[2,0],Z_[2,0],W_[2,0])) + \
        p2p1(p1p1(cat(X_[0,2],Y_[0,2],Z_[0,2],W_[0,2]),cat(X_[1,0],Y_[1,0],Z_[1,0],W_[1,0])) - \
            p1p1(cat(X_[0,0],Y_[0,0],Z_[0,0],W_[0,0]),cat(X_[1,2],Y_[1,2],Z_[1,2],W_[1,2])), \
            cat(X_[2,1],Y_[2,1],Z_[2,1],W_[2,1])) + \
        p2p1(p1p1(cat(X_[0,0],Y_[0,0],Z_[0,0],W_[0,0]),cat(X_[1,1],Y_[1,1],Z_[1,1],W_[1,1])) - \
            p1p1(cat(X_[0,1],Y_[0,1],Z_[0,1],W_[0,1]),cat(X_[1,0],Y_[1,0],Z_[1,0],W_[1,0])), \
            cat(X_[2,2],Y_[2,2],Z_[2,2],W_[2,2]))

    #Flipped V
    EE_t11=p1p1(cat(Xmat[0,0],Ymat[0,0],Zmat[0,0],Wmat[0,0]),cat(Xmat[0,0],Ymat[0,0],Zmat[0,0],Wmat[0,0])) + \
        p1p1(cat(Xmat[0,1],Ymat[0,1],Zmat[0,1],Wmat[0,1]),cat(Xmat[0,1],Ymat[0,1],Zmat[0,1],Wmat[0,1])) + \
        p1p1(cat(Xmat[0,2],Ymat[0,2],Zmat[0,2],Wmat[0,2]),cat(Xmat[0,2],Ymat[0,2],Zmat[0,2],Wmat[0,2]))

    EE_t12=p1p1(cat(Xmat[0,0],Ymat[0,0],Zmat[0,0],Wmat[0,0]),cat(Xmat[1,0],Ymat[1,0],Zmat[1,0],Wmat[1,0])) + \
        p1p1(cat(Xmat[0,1],Ymat[0,1],Zmat[0,1],Wmat[0,1]),cat(Xmat[1,1],Ymat[1,1],Zmat[1,1],Wmat[1,1])) + \
        p1p1(cat(Xmat[0,2],Ymat[0,2],Zmat[0,2],Wmat[0,2]),cat(Xmat[1,2],Ymat[1,2],Zmat[1,2],Wmat[1,2]))

    EE_t13=p1p1(cat(Xmat[0,0],Ymat[0,0],Zmat[0,0],Wmat[0,0]),cat(Xmat[2,0],Ymat[2,0],Zmat[2,0],Wmat[2,0])) + \
        p1p1(cat(Xmat[0,1],Ymat[0,1],Zmat[0,1],Wmat[0,1]),cat(Xmat[2,1],Ymat[2,1],Zmat[2,1],Wmat[2,1])) + \
        p1p1(cat(Xmat[0,2],Ymat[0,2],Zmat[0,2],Wmat[0,2]),cat(Xmat[2,2],Ymat[2,2],Zmat[2,2],Wmat[2,2]))

    EE_t22=p1p1(cat(Xmat[1,0],Ymat[1,0],Zmat[1,0],Wmat[1,0]),cat(Xmat[1,0],Ymat[1,0],Zmat[1,0],Wmat[1,0])) + \
        p1p1(cat(Xmat[1,1],Ymat[1,1],Zmat[1,1],Wmat[1,1]),cat(Xmat[1,1],Ymat[1,1],Zmat[1,1],Wmat[1,1])) + \
        p1p1(cat(Xmat[1,2],Ymat[1,2],Zmat[1,2],Wmat[1,2]),cat(Xmat[1,2],Ymat[1,2],Zmat[1,2],Wmat[1,2]))

    EE_t23=p1p1(cat(Xmat[1,0],Ymat[1,0],Zmat[1,0],Wmat[1,0]),cat(Xmat[2,0],Ymat[2,0],Zmat[2,0],Wmat[2,0])) + \
        p1p1(cat(Xmat[1,1],Ymat[1,1],Zmat[1,1],Wmat[1,1]),cat(Xmat[2,1],Ymat[2,1],Zmat[2,1],Wmat[2,1])) + \
        p1p1(cat(Xmat[1,2],Ymat[1,2],Zmat[1,2],Wmat[1,2]),cat(Xmat[2,2],Ymat[2,2],Zmat[2,2],Wmat[2,2]))

    EE_t33=p1p1(cat(Xmat[2,0],Ymat[2,0],Zmat[2,0],Wmat[2,0]),cat(Xmat[2,0],Ymat[2,0],Zmat[2,0],Wmat[2,0])) +  \
        p1p1(cat(Xmat[2,1],Ymat[2,1],Zmat[2,1],Wmat[2,1]),cat(Xmat[2,1],Ymat[2,1],Zmat[2,1],Wmat[2,1])) + \
        p1p1(cat(Xmat[2,2],Ymat[2,2],Zmat[2,2],Wmat[2,2]),cat(Xmat[2,2],Ymat[2,2],Zmat[2,2],Wmat[2,2]))

    # Not used
# EE_t21 = EE_t12;
# EE_t31 = EE_t13;
# EE_t32 = EE_t23;
    
    A_11=EE_t11 - 0.5*(EE_t11 + EE_t22 + EE_t33)
    
    A_12= EE_t12
    A_13= EE_t13
    A_21= A_12
    A_22=EE_t22 - 0.5*(EE_t11 + EE_t22 + EE_t33)
    A_23= EE_t23
    A_31= A_13 
    A_32= A_23
    A_33=EE_t33 - 0.5*(EE_t11 + EE_t22 + EE_t33)

    AE_11=p2p1(A_11,cat(Xmat[0,0],Ymat[0,0],Zmat[0,0],Wmat[0,0])) + \
        p2p1(A_12,cat(Xmat[1,0],Ymat[1,0],Zmat[1,0],Wmat[1,0])) + \
        p2p1(A_13,cat(Xmat[2,0],Ymat[2,0],Zmat[2,0],Wmat[2,0]))

    AE_12=p2p1(A_11,cat(Xmat[0,1],Ymat[0,1],Zmat[0,1],Wmat[0,1])) + \
        p2p1(A_12,cat(Xmat[1,1],Ymat[1,1],Zmat[1,1],Wmat[1,1])) + \
        p2p1(A_13,cat(Xmat[2,1],Ymat[2,1],Zmat[2,1],Wmat[2,1]))

    AE_13=p2p1(A_11,cat(Xmat[0,2],Ymat[0,2],Zmat[0,2],Wmat[0,2])) + \
        p2p1(A_12,cat(Xmat[1,2],Ymat[1,2],Zmat[1,2],Wmat[1,2])) + \
        p2p1(A_13,cat(Xmat[2,2],Ymat[2,2],Zmat[2,2],Wmat[2,2]))

    AE_21=p2p1(A_21,cat(Xmat[0,0],Ymat[0,0],Zmat[0,0],Wmat[0,0])) + \
        p2p1(A_22,cat(Xmat[1,0],Ymat[1,0],Zmat[1,0],Wmat[1,0])) + \
        p2p1(A_23,cat(Xmat[2,0],Ymat[2,0],Zmat[2,0],Wmat[2,0]))

    AE_22=p2p1(A_21,cat(Xmat[0,1],Ymat[0,1],Zmat[0,1],Wmat[0,1])) + \
        p2p1(A_22,cat(Xmat[1,1],Ymat[1,1],Zmat[1,1],Wmat[1,1])) + \
        p2p1(A_23,cat(Xmat[2,1],Ymat[2,1],Zmat[2,1],Wmat[2,1]))

    AE_23=p2p1(A_21,cat(Xmat[0,2],Ymat[0,2],Zmat[0,2],Wmat[0,2])) + \
        p2p1(A_22,cat(Xmat[1,2],Ymat[1,2],Zmat[1,2],Wmat[1,2])) + \
        p2p1(A_23,cat(Xmat[2,2],Ymat[2,2],Zmat[2,2],Wmat[2,2]))

    AE_31=p2p1(A_31,cat(Xmat[0,0],Ymat[0,0],Zmat[0,0],Wmat[0,0])) + \
        p2p1(A_32,cat(Xmat[1,0],Ymat[1,0],Zmat[1,0],Wmat[1,0])) + \
        p2p1(A_33,cat(Xmat[2,0],Ymat[2,0],Zmat[2,0],Wmat[2,0]))

    AE_32=p2p1(A_31,cat(Xmat[0,1],Ymat[0,1],Zmat[0,1],Wmat[0,1])) + \
        p2p1(A_32,cat(Xmat[1,1],Ymat[1,1],Zmat[1,1],Wmat[1,1])) + \
        p2p1(A_33,cat(Xmat[2,1],Ymat[2,1],Zmat[2,1],Wmat[2,1]))

    AE_33=p2p1(A_31,cat(Xmat[0,2],Ymat[0,2],Zmat[0,2],Wmat[0,2])) + \
        p2p1(A_32,cat(Xmat[1,2],Ymat[1,2],Zmat[1,2],Wmat[1,2])) + \
        p2p1(A_33,cat(Xmat[2,2],Ymat[2,2],Zmat[2,2],Wmat[2,2]))
# five_point_algorithm.m:175
    # Group and permute the collumns of our polynomial vectors to prepare for
# the Gaussian Jordan elimination with partial pivoting
# Previously our 3rd order polynomial vector arrangement was like this 
# x^3 | y^3 | z^3 | x^2y | xy^2 | x^2z | xz^2 | y^2z | yz^2 | xyz
# x^2 | y^2 | z^2 | xy | xz | yz | x | y | z | 1
# and now we are going to need this
# x^3 | y^3 | x^2y | xy^2 | x^2z | x^2 | y^2z | y^2 | xyz | xy
# xz^2 | xz | x | yz^2 | yz | y | z^3 | z^2 | z | 1
    
    A=cat(detF,AE_11,AE_12,AE_13,AE_21,AE_22,AE_23,AE_31,AE_32,AE_33)

    A=A[:,[0,1,3,4,5,10,7,11,9,13,6,14,16,8,15,17,2,12,18,19]]
    # Gauss Jordan elimination (partial pivoting after)
    A_el=gj_elim_pp(A)
    # Subtraction and forming matrix B
    k_row=partial_subtrc(A_el[4,10:20],A_el[5,10:20])
    l_row=partial_subtrc(A_el[6,10:20],A_el[7,10:20])
    m_row=partial_subtrc(A_el[8,10:20],A_el[9,10:20])

    B_11=k_row[0:4]
    B_12=k_row[4:8]
    B_13=k_row[8:13]

    B_21=l_row[0:4]
    B_22=l_row[4:8]
    B_23=l_row[8:13]

    B_31=m_row[0:4]
    B_32=m_row[4:8]
    B_33=m_row[8:13]

    p_1=pz4pz3(B_23,B_12) - pz4pz3(B_13,B_22)
    p_2=pz4pz3(B_13,B_21) - pz4pz3(B_23,B_11)
    p_3=pz3pz3(B_11,B_22) - pz3pz3(B_12,B_21)

    n_row=pz7pz3(p_1,B_31) + pz7pz3(p_2,B_32) + pz6pz4(p_3,B_33)
    #Extracting roots from n_row using companion matrix eigen values
    #TODO: Need to verify it 
    # http://www.docin.com/p-772548201.html
    if(n_row[0] != 0.0):    
        n_row_scaled=n_row / n_row[0]
    
        companion_m = np.vstack((- n_row_scaled[1:], np.hstack((np.eye(9), np.zeros((9, 1))))))
    else:
        raise InvalidDenomError
        
#        companion_m = np.vstack((- n_row, np.hstack((np.eye(10), np.zeros((10, 1))))))

#    companion_m = np.vstack((- n_row, np.hstack((np.eye(10), np.zeros((10, 1))))))

    e_val, v = np.linalg.eig(companion_m)
    e_val_l = e_val.tolist()
    Eo_all=[]
    for val in e_val_l:
        if np.isreal(val):
            z= val.real
            p_z6= cat([z ** 6],[z ** 5],[z ** 4],[z ** 3],[z ** 2],[z],[1])
            p_z7= np.vstack((np.array([z ** 7]), p_z6))
            x= p_1.dot(p_z7)[0] / p_3.dot(p_z6)[0]
            y= p_2.dot(p_z7)[0] / p_3.dot(p_z6)[0]
            Eo= x*Xmat + y*Ymat + z*Zmat + Wmat
            Eo_all.append(Eo)

    
    return Eo_all
    
    

def cross_vec3(u=None,v=None,*args,**kwargs):

    #CROSS_VEC Function to compute the cross product of two 3D column vectors.
#The default MATLAB implementation is simply too slow.
    
    out=np.array(cat([mul(u[1],v[2]) - mul(u[2],v[1])], \
        [mul(u[2],v[0]) - mul(u[0],v[2])],\
        [mul(u[0],v[1]) - mul(u[1],v[0])]))

    return out
    
    
    

def pz6pz4(p1=None,p2=None,*args,**kwargs):

    #PZ4PZ3 Function responsible for np.multiplying a 6th order z polynomial p1
#   by a 4th order z polynomial p2
#   p1 - Is a row vector arranged like: z6 | z5 | z4 | z3 | z2 | z | 1
#   p2 - Is a row vector arranged like: z4 | z3 | z2 | z | 1
#   po - Is a row vector arranged like: 
#       z10 | z9 | z8 | z7 | z6 | z5 | z4 | z3 | z2 | z | 1
    
    po=np.array(cat(mul(p1[0],p2[0]),\
        mul(p1[1],p2[0]) + mul(p1[0],p2[1]),\
        mul(p1[2],p2[0]) + mul(p1[1],p2[1]) + mul(p1[0],p2[2]), \
        mul(p1[3],p2[0]) + mul(p1[2],p2[1]) + mul(p1[1],p2[2]) + mul(p1[0],p2[3]), \
        mul(p1[4],p2[0]) + mul(p1[3],p2[1]) + mul(p1[2],p2[2]) + mul(p1[1],p2[3]) + mul(p1[0],p2[4]),\
        mul(p1[5],p2[0]) + mul(p1[4],p2[1]) + mul(p1[3],p2[2]) + mul(p1[2],p2[3]) + mul(p1[1],p2[4]), \
        mul(p1[6],p2[0]) + mul(p1[5],p2[1]) + mul(p1[4],p2[2]) + mul(p1[3],p2[3]) + mul(p1[2],p2[4]), \
        mul(p1[6],p2[1]) + mul(p1[5],p2[2]) + mul(p1[4],p2[3]) + mul(p1[3],p2[4]), \
        mul(p1[6],p2[2]) + mul(p1[5],p2[3]) + mul(p1[4],p2[4]), \
        mul(p1[6],p2[3]) + mul(p1[5],p2[4]), \
        mul(p1[6],p2[4])))
    
    return po
    
    

def pz7pz3(p1=None,p2=None,*args,**kwargs):

    #PZ4PZ3 Function responsible for np.multiplying a 7th order z polynomial p1
#   by a 3rd order z polynomial p2
#   p1 - Is a row vector arranged like: z7 | z6 | z5 | z4 | z3 | z2 | z | 1
#   p2 - Is a row vector arranged like: z3 | z2 | z | 1
#   po - Is a row vector arranged like: 
#       z10 | z9 | z8 | z7 | z6 | z5 | z4 | z3 | z2 | z | 1
    
    po=np.array(cat(mul(p1[0],p2[0]),\
        mul(p1[1],p2[0]) + mul(p1[0],p2[1]),\
        mul(p1[2],p2[0]) + mul(p1[1],p2[1]) + mul(p1[0],p2[2]), \
        mul(p1[3],p2[0]) + mul(p1[2],p2[1]) + mul(p1[1],p2[2]) + mul(p1[0],p2[3]),\
        mul(p1[4],p2[0]) + mul(p1[3],p2[1]) + mul(p1[2],p2[2]) + mul(p1[1],p2[3]), \
        mul(p1[5],p2[0]) + mul(p1[4],p2[1]) + mul(p1[3],p2[2]) + mul(p1[2],p2[3]),\
        mul(p1[6],p2[0]) + mul(p1[5],p2[1]) + mul(p1[4],p2[2]) + mul(p1[3],p2[3]), \
        mul(p1[7],p2[0]) + mul(p1[6],p2[1]) + mul(p1[5],p2[2]) + mul(p1[4],p2[3]),\
        mul(p1[7],p2[1]) + mul(p1[6],p2[2]) + mul(p1[5],p2[3]), \
        mul(p1[7],p2[2]) + mul(p1[6],p2[3]),\
        mul(p1[7],p2[3])))
    
    return po
    
    

def pz4pz3(p1=None,p2=None,*args,**kwargs):

    #PZ4PZ3 Function responsible for np.multiplying a 4th order z polynomial p1
#   by a 3rd order z polynomial p2
#   p1 - Is a row vector arranged like: z4 | z3 | z2 | z | 1
#   p2 - Is a row vector arranged like: z3 | z2 | z | 1
#   po - Is a row vector arranged like: z7 | z6 | z5 | z4 | z3 | z2 | z | 1
    
    po=np.array(cat(mul(p1[0],p2[0]),\
        mul(p1[1],p2[0]) + mul(p1[0],p2[1]), \
        mul(p1[2],p2[0]) + mul(p1[1],p2[1]) + mul(p1[0],p2[2]),\
        mul(p1[3],p2[0]) + mul(p1[2],p2[1]) + mul(p1[1],p2[2]) + mul(p1[0],p2[3]),\
        mul(p1[4],p2[0]) + mul(p1[3],p2[1]) + mul(p1[2],p2[2]) + mul(p1[1],p2[3]),\
        mul(p1[4],p2[1]) + mul(p1[3],p2[2]) + mul(p1[2],p2[3]),\
        mul(p1[4],p2[2]) + mul(p1[3],p2[3]), \
        mul(p1[4],p2[3])))
    
    return po
    
    

def pz3pz3(p1=None,p2=None,*args,**kwargs):

    #PZ3PZ3 Function responsible for np.multiplying two 3rd order z polynomial
#   p1, p2 - Are row vector arranged like: z3 | z2 | z | 1
#   po - Is a row vector arranged like: z6 | z5 | z4 | z3 | z2 | z | 1
    
    po=np.array(cat(mul(p1[0],p2[0]),\
        mul(p1[0],p2[1]) + mul(p1[1],p2[0]),\
        mul(p1[0],p2[2]) + mul(p1[1],p2[1]) + mul(p1[2],p2[0]),\
        mul(p1[0],p2[3]) + mul(p1[1],p2[2]) + mul(p1[2],p2[1]) + mul(p1[3],p2[0]),\
        mul(p1[1],p2[3]) + mul(p1[2],p2[2]) + mul(p1[3],p2[1]),\
        mul(p1[2],p2[3]) + mul(p1[3],p2[2]),\
        mul(p1[3],p2[3])))
    
    return po
    
    
    

def partial_subtrc(p1=None,p2=None,*args,**kwargs):

    #PARTIAL_SUBTRC Given polinomials p1 and p2 substract them according to the
#following expression: p1 - z*p2
# p1, p2 - are row vectors with the following arrangement
#   xz^2 | xz | x | yz^2 | yz | y | z3 | z2 | z | 1
# po - is a row vector with the following arragement
#   xz3 | xz2 | xz | x | yz3 | yz2 | yz | y | z4 | z3 | z2 | z | 1
    
    po=np.array(cat(- p2[0],\
        p1[0] - p2[1],\
        p1[1] - p2[2],\
        p1[2],- p2[3],\
        p1[3] - p2[4],\
        p1[4] - p2[5],\
        p1[5],- p2[6],\
        p1[6] - p2[7],\
        p1[7] - p2[8],\
        p1[8] - p2[9],\
        p1[9]))
    return po
       

def gj_elim_pp(A=None,*args,**kwargs):

    #GJ_ELIM_PP Given Matriz A we perform partial pivoting as per specified in
    
    P, L, U = lu(A)

    B=np.zeros((10,20))

    B[0:3,:]=U[0:3,:]

    #Back substitution
    B[9,:]=U[9,:] / U[9,9]

    B[8,:]=(U[8,:] - mul(U[8,9],B[9,:])) / U[8,8]

    B[7,:]=(U[7,:] - mul(U[7,8],B[8,:]) - mul(U[7,9],B[9,:])) / U[7,7]

    B[6,:]=(U[6,:] - mul(U[6,7],B[7,:]) - mul(U[6,8],B[8,:]) - mul(U[6,9],B[9,:])) / U[6,6]

    B[5,:]=(U[5,:] - mul(U[5,6],B[6,:]) - mul(U[5,7],B[7,:]) - mul(U[5,8],B[8,:]) - mul(U[5,9],B[9,:])) / U[5,5]

    B[4,:]=(U[4,:] - mul(U[4,5],B[5,:]) - mul(U[4,6],B[6,:]) - mul(U[4,7],B[7,:]) - mul(U[4,8],B[8,:]) - mul(U[4,9],B[9,:])) / U[4,4]

    return B
    
    

def p1p1(p1=None,p2=None,*args,**kwargs):

    #P1P1 Given two first order polynomials with the structure [a_x,
#a_y, a_z, a_w] e returns the second order polynomial x,y,z with the
#structure: pout = [a_x2, a_y2, a_z2, a_xy, a_xz, a_yz, a_x, a_y, a_z, a]
    
    pout=np.array(cat(mul(p1[0],p2[0]),\
        mul(p1[1],p2[1]), \
        mul(p1[2],p2[2]), \
        mul(p1[0],p2[1]) + mul(p1[1],p2[0]),\
        mul(p1[0],p2[2]) + mul(p1[2],p2[0]), \
        mul(p1[1],p2[2]) + mul(p1[2],p2[1]), \
        mul(p1[0],p2[3]) + mul(p1[3],p2[0]), \
        mul(p1[1],p2[3]) + mul(p1[3],p2[1]), \
        mul(p1[2],p2[3]) + mul(p1[3],p2[2]), \
        mul(p1[3],p2[3])))

    
    return pout
    
    

def p2p1(p1=None,p2=None,*args,**kwargs):

    #P2P1 Given two polynomials, p1 of order 2 and p2 of order 1, with
#unknowns x, y and z, return their product. An order 3 polynomial with
#structure shown below
    #pout = [a_x3, a_y3, a_z3, a_x2y, a_xy2, a_x2z, a_xz2, a_y2z, a_yz2, a_xyz, a_x2, a_y2, a_z2, a_xy, a_xz, a_yz, a_x, a_y, a_z, a_1]
    pout=np.array(cat(p1[0]*p2[0],\
        mul(p1[1],p2[1]), \
        mul(p1[2],p2[2]),\
        mul(p1[0],p2[1]) + mul(p1[3],p2[0]),\
        mul(p1[1],p2[0]) + mul(p1[3],p2[1]), \
        mul(p1[0],p2[2]) + mul(p1[4],p2[0]), \
        mul(p1[2],p2[0]) + mul(p1[4],p2[2]), \
        mul(p1[1],p2[2]) + mul(p1[5],p2[1]), \
        mul(p1[2],p2[1]) + mul(p1[5],p2[2]), \
        mul(p1[3],p2[2]) + mul(p1[4],p2[1]) + mul(p1[5],p2[0]),\
        mul(p1[0],p2[3]) + mul(p1[6],p2[0]), \
        mul(p1[1],p2[3]) + mul(p1[7],p2[1]),\
        mul(p1[2],p2[3]) + mul(p1[8],p2[2]), \
        mul(p1[3],p2[3]) + mul(p1[6],p2[1]) + mul(p1[7],p2[0]),\
        mul(p1[4],p2[3]) + mul(p1[6],p2[2]) + mul(p1[8],p2[0]),\
        mul(p1[5],p2[3]) + mul(p1[7],p2[2]) + mul(p1[8],p2[1]), \
        mul(p1[6],p2[3]) + mul(p1[9],p2[0]),\
        mul(p1[7],p2[3]) + mul(p1[9],p2[1]),\
        mul(p1[8],p2[3]) + mul(p1[9],p2[2]),\
        mul(p1[9],p2[3])))

    
    return pout
    

def cat(*args):
    return np.array([*args])

def mul(p1, p2):
    return np.multiply(p1, p2)

'''
K1 = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
K2 = K1
pts1 = np.array([[1, 2, 3, 4, 5], [6, 7, 8, 9, 10], [1, 1, 1, 1, 1]])
pts2 = np.array([[3, 3, 4, 5, 6], [8, 9, 7, 6, 8], [1, 1, 1, 1, 1]])
Eo_all = five_point_algorithm(pts1, pts2, K1, K2)
print(Eo_all)
'''