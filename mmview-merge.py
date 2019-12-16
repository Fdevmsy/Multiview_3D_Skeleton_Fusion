import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
import math

class ImageProcessingTools():
  def __init__(self):
    self.j3d_1 = None
    self.j3d_2 = None
    self.plot_res = 0
          
  def save_image(self, image, name):
    plt.axis('off')
    imageio.imwrite(name, image)
    
  def plot(self, image):
    plt.axis('off')
    plt.imshow(image, cmap=plt.cm.gray)
    plt.show()
  
  def plot_3d(self, data, data2):
    fig = plt.figure()
    ax = plt.axes(projection='3d')
    data = np.array(data)
    x = data[:,0]
    y = data[:,1]
    z = data[:,2]
    
    try:
      data2.shape
      data2 = np.array(data2)
      x2 = data2[:,0]
      y2 = data2[:,1]
      z2 = data2[:,2]
    except:
      pass	
    
    ax.scatter3D(x, y, z, c='g', marker='o')
    try:
      data2.shape
      ax.scatter3D(x, y, z, c='r', marker='o')
      ax.scatter3D(x2, y2, z2, c='b', marker='o')
    except:
      pass
    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    ax.set_zlabel('Z Label')
    plt.pause(0.1)	
    return plt
  
  def rigid_transform_3D(self, A, B):
    """Return new A according to B"""
    assert (A.shape==B.shape)
    # centering
    centroid_A = np.mean(A, axis=0)
    centroid_B = np.mean(B, axis=0)

    AA = A - centroid_A
    BB = B - centroid_B

    H = np.transpose(AA) @ BB
    U, S, Vt = np.linalg.svd(H)

    R = Vt.T @ U.T

    # reflection 
    if np.linalg.det(R) < 0:
      print("Reflection detected")
      Vt[2,:] *= -1
      R = Vt.T @ U.T

    t = -R @ centroid_A.T + centroid_B.T
    
    A_new = (R @ A.T) + t
    A_new = A_new.T 
    
    return A_new
  
  def test_single_frame(self):
    # load data
    j3d = np.load('direction1_sample50.npy')		
    A = np.asmatrix(j3d[0,:,:])
    B = np.asmatrix(j3d[1,:,:])
    # get transform, convert A to B
    A_new = self.rigid_transform_3D(A, B)	
    self.plot_3d(A_new, B)
  
  
  def test_multi_frame(self):
    j3d_1 = self.j3d_1
    j3d_2 = self.j3d_2
    
    # shape (2266, 17, 3)	
    assert (j3d_1.shape == j3d_2.shape)	
    
    for frame in range(0, j3d_1.shape[0], 50):
      A = np.asmatrix(j3d_1[frame,:,:])
      B = np.asmatrix(j3d_2[frame,:,:])
      A_new = self.rigid_transform_3D(A, B)
      myplt = self.plot_3d(A_new, B)
    myplt.draw()
    
  def track_error(self, data):
    """compare the distance between frames"""
    # expected data shape (n, 17, 3)
    assert (data.shape[1]==17)
    assert (data.shape[2]==3)
    data_pre = data.copy()
    data_pre = data_pre.tolist()
    data_pre.insert(0, data[0].tolist())
    data_pre = np.array(data_pre)
    
    diff = data - data_pre[:-1]		
    diff_per_joint = np.sum(np.abs(diff)**2,axis=-1)**(1./2)
    # the return shape is (n, 17)
    return diff_per_joint
  
  def boardcast2D(self, mat):
    mat_new = np.zeros((mat.shape[0], mat.shape[1], 3))
    for i in range(3):
      mat_new[:,:,i] = mat
    return mat_new
    
  def boardcast1D(self, mat):
    mat_new = np.zeros((mat.shape[0], 3))
    for i in range(3):
      mat_new[:,i] = mat
    return mat_new
      
  def confidence_merge(self, mat1, mat2):	
    diff_per_joint_1 = self.track_error(mat1) + 1e-10
    diff_per_joint_2 = self.track_error(mat2) + 1e-10
    # notice, higher distance should get lower weight
    weight1 = diff_per_joint_2/(diff_per_joint_1+diff_per_joint_2)
    weight2 = diff_per_joint_1/(diff_per_joint_1+diff_per_joint_2)
    weight1 = self.boardcast2D(weight1)
    weight2 = self.boardcast2D(weight2)

    merged = mat1 * weight1 + mat2 * weight2
  
    print(merged)
    return merged
    
  def merge_from_last_merge(self, mat1, mat2):
    """ track error according to the last merged frame, in general case,
    mat1 is rotated A, mat2 is B """
    
    assert (mat1.shape == mat2.shape)
    ## init the T0 reference frame using the first frame of B 
    ref_merged = mat2[0]
    ## a list to hold the merged frames		
    merged_res = list()
    merged_res.append(ref_merged)
    ## track using a loop over all frames
    for i in range(mat1.shape[0]):
      # compute error compares to the ref for each frame
      diff1 = mat1[i]-ref_merged
      error1 = np.sum(np.abs(diff1)**2,axis=1)**(1./2)
      diff2 = mat2[i]-ref_merged
      error2 = np.sum(np.abs(diff2)**2,axis=1)**(1./2)			
      # compute the weight, higher error gets lower weight
      weight1 = error2/(error1+error2)
      weight2 = error1/(error1+error2)
      # broadcast (17,) to (17,3)
      weight1 = self.boardcast1D(weight1)
      weight2 = self.boardcast1D(weight2)
      # merge the 2 mat using weight and update the ref
      
      ref_merged = mat1[i] * weight1 + mat2[i] * weight2
      merged_res.append(ref_merged)
#			print(ref_merged.shape)
    
    merged_res = np.array(merged_res)
    return merged_res

  def test_merge(self):
    j3d_1 = self.j3d_1
    j3d_2 = self.j3d_2
    # shape (2266, 17, 3)	
    assert (j3d_1.shape == j3d_2.shape)	
    assert (j3d_1.shape[1] == 17)	
    A_new_list = list()
    
    ## rotate A to B
    for frame in range(0, j3d_1.shape[0]):
      A = np.asmatrix(j3d_1[frame,:,:])
      B = np.asmatrix(j3d_2[frame,:,:])
      A_new = self.rigid_transform_3D(A, B)
      A_new_list.append(A_new)
    
    A_rotated = np.array(A_new_list)
    merged = self.merge_from_last_merge(A_rotated, j3d_2)
#		merged = self.confidence_merge(A_rotated, j3d_2)
    print(merged.shape)
    np.save('merged.npy', merged)
    if self.plot_res == 1:
      for frame in range(0, merged.shape[0], 1):
        self.plot_3d(merged[frame], None)
      myplt.draw()	

if __name__ == "__main__":
  """ Input:
     		Paths of 2 j3d npy file, each of them is in shape (N, 17, 3)
        Fill the path at line 233 and line 234
      Output:
        A merged j3d saved in merged.npy in the same folder.
        The shape will be (N, 17, 3)
  """
  
  geo3d = ImageProcessingTools()
  # view 1 and view 2 
  # expected input shape: (N, 17, 3) 
  geo3d.j3d_1 = np.load('1/Discussion60457274_j3d.npy')
  geo3d.j3d_2 = np.load('1/Discussion54138969_j3d.npy')	

  print(geo3d.j3d_1.shape)
  print(geo3d.j3d_2.shape)
  # plot or not
  geo3d.plot_res = 0
  
  geo3d.test_merge()
  #	geo3d.test_multi_frame()
  #	geo3d.main()
