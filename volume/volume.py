"""Copyright 2024, OtoJig GmbH (Felix Repp)

   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

     http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License.
"""

"""
create a basic 3d Image class with numpy array and world_to_ijk matrix 
"""

import numpy as np

import neuclid as nc
import nrrd
import copy
import pdb

import os
from scipy import ndimage
import logging
from collections import OrderedDict
from copy import deepcopy


log = logging.getLogger("IRE")
import numpy
numpy.float = float
numpy.int = numpy.int_

class Volume():
    """
    a basic class for containing volume data and its relation to world coordinates
    world coordinates can be Left/Right, Posterior/Anterior, Superior/Inferior where the direction of the axis is specified
    LPS (Left, Posterior, Superior) is used in DICOM images and by the ITK toolkit and OSIRIX
    RAS (Right, Anterior, Superior) is similar to LPS with the first two axes flipped and used by 3D Slicer


    also provides loders for simpleitk pynrrd and pydicom
    """

    def __init__(self, data=None, ijk_to_world= nc.Matrix44(np.eye(4)), space = 'LPS'):
        self.data = data
        self._space = space
        if not space is None:
            self.space = space # to check syntax
        self.to_world = ijk_to_world
        self.vtk_image = None

    @property
    def to_world(self):
        return self._ijk_to_world


    @to_world.setter
    def to_world(self, value):
        self._ijk_to_world = nc.Matrix44(value)

    @property
    def to_ijk(self):
        return self.to_world.inverse
    
    @to_ijk.setter
    def to_ijk(self, value):
        self._ijk_to_world = nc.Matrix44(value).inverse

    @property
    def space(self):
        return self._space
     
    @space.setter       
    def space(self,  value):
        if not value is None:
            rl, pa, si = value
            assert rl in ['R','L'], 'First letter of space must be R or L'
            assert pa in ['P','A'], 'Second letter of space must be P or A'
            assert si in ['S','I'], 'Last letter of space must be S oi I'
        
        if value is None or value == self._space:
            self._space = value
        else: 
            self.switch_space(value)

    def switch_space(self, space):
        assert np.all(np.abs(np.diag(self.to_world.matrix33)) == np.linalg.norm(self.to_world.matrix33, axis=0))
        T = np.asarray(self.to_world)
        rl, pa, si = space
        rlo, pao, sio = self.space
        origin = T[:3,3]
        endpos = self.to_world(np.array(self.shape)-1)
        """
        if  rl != rlo:
            T[0,:]*=-1
            origin[0] *=-1# endpos[0]
        if  pa != pao:
            T[1,:]*=-1
            origin[1] *=-1# endpos[1]
        if  si == sio:
            T[2,:]*=-1
            origin[2] *=-1# endpos[2]
        """
        if  rl != rlo:
            self.data = self.data[::-1,:,:]
            origin[0] = -endpos[0]#T[0,:]*=-1
        if  pa != pao:
            self.data = self.data[:,::-1,:]
            origin[1] = -endpos[1]#
        if  si != sio:
            self.data = self.data[:,:,::-1]#T[2,:]*=-1
            origin[2] = -endpos[2]
        T=nc.Matrix44(T)
        T.translation = origin
        self.to_world = T
        self._space = space

    def make_diagonal(self):
        m = self.data
        a = np.where(~np.isclose(self.to_world.matrix33.T,0))[1]

        self.data = np.moveaxis(m,  a, range(3))
        tw = nc.Matrix44(np.eye(4))
        tw.matrix33 = self.to_world.matrix33[a]
        tw.translation = self.to_world.translation
        for i in range(3):
            if tw.matrix33[i,i] < 0:
                self.data = np.flip(self.data, i)
                tw.translation[i] = tw(np.array(self.shape)-1)[i]
                tw.matrix33[i,i] *= -1
        self.to_world = tw

    def from_itk(self, I_itk):
        import SimpleITK as sitk
        "assumes I_itk is in 'LPS space"

        data = sitk.GetArrayFromImage(I_itk).T
        T = np.eye(4)
        origin = np.array(I_itk.GetOrigin())
        T[:3,:3] = np.array(I_itk.GetDirection()).reshape(3,3).T * I_itk.GetSpacing()
        T[:3,3] = I_itk.GetOrigin()
        v = Volume(data,  ijk_to_world=nc.Matrix44(T), space = 'LPS')
        v.space = self.space
        self.__dict__.update(v.__dict__)

    def to_itk(self):
        import SimpleITK as sitk
        I = sitk.GetImageFromArray(self.data.T)
        assert np.all(np.abs(np.diag(self.to_world.matrix33)) == np.linalg.norm(self.to_world.matrix33, axis=0))
        I.SetOrigin(self.to_world[:3,3])
        I.SetDirection(np.eye(3).ravel())
        I.SetSpacing(np.diag(self.to_world.matrix33))
        return I


    def load_nrrd(self, filename, method='pynrrd'):
        """
        allowed methods are 'pynnrd' and sitk
        """
        if method == 'pynrrd':
            data, meta = nrrd.read(filename)
            #pdb.set_trace()
            if 'kinds' in meta.keys():

                kinds = np.array(meta['kinds'])
            else:
                 kinds = np.array(['domain','domain','domain'])

            T = np.eye(4)
            A = meta['space directions']
            T[:3,:3] = A[kinds == 'domain'].reshape(3,3) # needed to load for segmentation nrrd
            rl, pa, si = meta['space'].split('-')
            origin = meta['space origin']
            T[:3,3] = origin
            space = ''
            if  rl == 'left':
                space += 'L'
            elif not rl == 'right':
                raise Exception('space not known')
            else:
                space += 'R'
            if  pa == 'posterior':
                space += 'P'
            elif not pa =='anterior':
                raise Exception('space not known')
            else:
                space += 'A'
            if  si == 'inferior':
                space += 'I'
            elif not si == 'superior':
                raise Exception('space not known')
            else:
                space += 'S'
           
            v = Volume(data,  ijk_to_world=nc.Matrix44(T), space = space)
            old_space = self.space
          
            self.__dict__.update(v.__dict__)
            #pdb.set_trace()
            if kinds.shape[0] == 3: 
                self.data = data
                self.space = old_space
            else:
                volumes = []
                ax = int(np.where(np.array(meta['kinds'])=='list')[0])
                for i in range(data.shape[ax]):
                    if i == 0:
                        self.data = data.take(i,ax)
                        self.space  = old_space
                        volumes.append(self)
                    else:
                        v = volume_like(self, data = data.take(i,ax))
                        v.space = old_space
                        volumes.append(v)
                return volumes
        elif method == 'sitk':
            import SimpleITK as sitk
            I_itk = sitk.ReadImage(filename)
            self.from_itk(I_itk)

        else: 
            print('method not known')

    def write_nrrd(self, filename, udict={}):
        meta = OrderedDict()
        meta['type']= str(self.data.dtype)
     
        
        rl, pa, si = self.space
     
        space = ''
        if  rl == 'L':
            space += 'left'
        else:
            space += 'right'
        if  pa == 'P':
            space += '-posterior-'
        elif pa == 'A':
            space += '-anterior-'
        if  si == 'I':
            space += 'inferior'
        elif si == 'S':
            space += 'superior'
        meta['space'] = space
        meta['sizes'] = self.data.shape
        meta['space directions'] = self.to_world.matrix33.tolist()
        meta['kinds'] =  ['domain', 'domain', 'domain']
        meta['endian'] =  'little'
        meta['encoding']  =   'gzip'
        meta['space origin'] = self.to_world.translation
        meta.update(udict)

        return nrrd.write(filename, self.data, header = meta)

    @property
    def shape(self):
        if not self.data is None:
            return self.data.shape
        return None
    def __getitem__(self, value):
        if not isinstance(value, tuple):
            value = value,

        print(value)
        v = [slice(None),slice(None),slice(None)]
        origin = [0,0,0]
        steps = [1,1,1]
        for i, sl in enumerate(value):
            if not isinstance(sl,slice):
                sl = slice (sl,sl+1,None)
            """ start = sl.start
            stop = sl.stop
            step = sl.step
            """

            if sl.start is not None:
                if sl.start < 0:
                    origin[i] = - sl.start
                else:
                    origin[i] = sl.start

            #what happens when slice is bigger than volume?

            if not sl.step is None:
                steps [i] = 1

            v[i] = sl

        data = self.data[tuple(v)]

        ijk_to_world = self.to_world.copy()
        ijk_to_world.matrix33 = np.dot(ijk_to_world.matrix33, np.diag(1 / np.array(steps)))
        ijk_to_world.translation = self.to_world(origin)
        return Volume(data, ijk_to_world, self.space)

    def crop(self, min, max, world = True, copy=False):
        """if world is False  use ijk
            if copy is False use view
        """
        if world:
            imin_t, imax_t = self.to_ijk([min,max])
        
        else: 
            imin_t = min
            imax_t = max  


        imin = np.max([np.min([imin_t, imax_t,], axis = 0),[0,0,0]], axis = 0).round().astype(int)
        imax = np.min([np.max([imin_t, imax_t], axis = 0), self.data.shape], axis = 0).round().astype(int)

        origin = self.to_world(imin)
        data =  self.data[imin[0]:imax[0], imin[1]:imax[1], imin[2]:imax[2]]
        if copy:
            data = data.copy()
        T = self.to_world.copy()
        T[:3,3] = origin
        return Volume(data,  ijk_to_world=T, space = self.space)
    

    def crop_around(self, points, copy = False, margin = 0):
        ipoints = np.atleast_2d(self.to_ijk(points))
        min = np.min(ipoints,axis=0)
        max = np.max(ipoints,axis=0)
        m = np.abs(np.dot(np.asarray(self.to_ijk  [:3,:3]),[margin]*3))
        return self.crop(min-m, max+m, world = False, copy=copy)

    def zoom(self, zoom, order):
        ndata = ndimage.zoom(self.data,zoom)
        t = self.to_world.copy()
        t.matrix33/=zoom
        t.translation = self.to_world.translation+(1-1/zoom)*np.dot(self.to_world.matrix33,np.array([-1/2,-1/2,-1/2]))
        return Volume(ndata, t)


    def gaussian_filter(self, sigma, order=0, mode='reflect',cval=0.0, truncate=4.0):
        """
        maps scipy.ndimage.filters.gaussian_filter but uses world coordinates for the sigmas
        """
        assert np.all(np.abs(np.diag(self.to_world.matrix33)) == np.linalg.norm(self.to_world.matrix33, axis=0))
        if len(np.atleast_1d(sigma)) == 1:
            sigma = np.ones(3)*sigma
        sigma = np.dot(self.to_ijk.matrix33, sigma)
        data = ndimage.filters.gaussian_filter(self.data, sigma, order=order, mode=mode, cval=cval, truncate=truncate)
        v = volume_like(self, data = data)
        return v 


    def get_coordinate_arrays(self):
        i = np.arange(self.shape[0])
        j = np.arange(self.shape[1])
        k = np.arange(self.shape[2])
        if np.all(np.abs(np.diag(self.to_world.matrix33)) == np.linalg.norm(self.to_world.matrix33, axis=0)):
            spacing =np.diag(self.to_world.matrix33)
            offset = self.to_world.translation
            X = i *spacing[0]+offset[0]
            Y = j *spacing[1]+offset[1]
            Z = k *spacing[2]+offset[2]
        
            XX, YY, ZZ = np.meshgrid(X,Y,Z, indexing = 'ij')
        else:
            II, JJ, KK = np.meshgrid(i,j,k, indexing = 'ij')
            XX, YY, ZZ  = np.einsum('ij,jklm',self.to_world.matrix33,[II,JJ,KK]) + self.to_world.translation[:,np.newaxis,np.newaxis,np.newaxis]
        return XX, YY, ZZ 

    def distance_to(self, other, xyz=None, ret_xyz = False):
        if xyz is None:
            X,Y,Z = self.get_coordinate_arrays()
        else:
            X,Y,Z = xyz
        c = nc.Points3(np.array([X.ravel(),Y.ravel(),Z.ravel()]).T)
        d = c.distance_to(other)
        d = d.reshape(X.shape)
        d = volume_like(self, data = d)
        if ret_xyz:
            return d, (X,Y,Z)
        else: 
            return d

def load_flex_pydicom(flex_id, image_type, space = 'LPS'):
    from ire.data import myci
    dicom =  myci.load_image(flex_id, image_type)
    l = []
    if len(dicom)==0:
        return None
    for s in dicom:
        try:
            l.append(Volume(s.volume,  ijk_to_world = nc.Matrix44(s.ijk_to_xyz), space = 'LPS')  )
            l[-1].space = space
            print('loaded_image with shape {} '.format(l[-1].data.shape))
        except:
            print('skip')
    return l

def load_dicom(dicom_path, return_itk=False, space = 'LPS' , load_gentry = True, key = "ORIGINAL\PRIMARY\AXIAL"):
    """
    assumes dicom image is in LPS, converts it into the defined space
    """
    import SimpleITK as sitk
    reader = sitk.ImageSeriesReader()
    series = []
    itk_series = []
    for root, dirs, files in os.walk(dicom_path):
        for dire in dirs:
            
            folder = os.path.join(root, dire)
            for s in reader.GetGDCMSeriesIDs(folder):
                
                print(len(reader.GetGDCMSeriesFileNames( folder,s)))
                if len(reader.GetGDCMSeriesFileNames( folder,s))>1:

                    dicom_names =  reader.GetGDCMSeriesFileNames( folder,s)#))
                    reader2 = sitk.ImageFileReader()
                    reader2.SetFileName( dicom_names[0] )

                    reader2.LoadPrivateTagsOn();
                    reader2.ReadImageInformation()
                    #reader.GetMetaDataDictionaryArrayUpdate()
                    tilt = 0
                    if "0018|1120" in reader2.GetMetaDataKeys():
                        tilt = reader2.GetMetaData("0018|1120")
                        print("gantry " + tilt)
                        #print("gantry_slew " + reader2.GetMetaData("0018|1121"))
                        #reader.ForceOrthogonalDirectionOff()
                    print(reader2.GetMetaData("0008|0008"))
                    print(reader2.GetMetaData("0028|0030"))
                    print(reader2.GetMetaData("0018|0050"))
                    if float(tilt) > 0 and load_gentry == False:
                        continue
                    if key in reader2.GetMetaData("0008|0008"):
                        reader.SetFileNames(dicom_names)
                        #if len(reader.GetGDCMSeriesFileNames( folder,s))>m:
                        print ('load')
                        I_itk = reader.Execute()
                        v = Volume(space = space)
                        v.from_itk(I_itk)

                        series.append(v)

                        if return_itk:
                            itk_series.append(I_itk)
                    else: 
                        print('skip')
    if return_itk:
        return series, itk_series 
    else:
        return series



def sitk_load_flex(flex_id, image_type, return_itk=False, space = 'LPS', load_gentry = True, key = "ORIGINAL\PRIMARY\AXIAL"):
    import SimpleITK as sitk
    from ire.data import myci
    from ire.data import data_dir, imaging_data_base_dir
    p = data_dir / imaging_data_base_dir
    try:
        dicom_path = list(p.glob(flex_id + '/' + image_type + '_*/DICOM'))[0]
        if dicom_path.is_dir():
            return load_dicom(dicom_path, return_itk=return_itk, load_gentry=load_gentry, key=key)

        else:
            log.info(f'DICOM not loadable')

    except IndexError:
        log.error(f'DICOM folder for {flex_id} {image_type} missing.')



def cut_points_to_volume(points, volume, split = False):
    import SimpleITK as sitk
    pi = volume.to_ijk(points)

    pi = volume.to_ijk(points)
    m = np.logical_and(np.all(pi < volume.shape, axis = 1),np.all(pi >=0, axis = 1))

    if split:
        n = np.where(np.diff(m.astype('int'))==-1)[0]
        pcl= np.split(points,n+1)
        l = []
        for i, m in enumerate(np.split(m,n+1)):
           
            l.append(pcl[i][m])
 
        return l
   
    else:
        return points[m]

def volume_like(volume, data=None):
    if data is None:
        data = np.zeros_like(volume.data)
    else: 
        assert data.shape == volume.data.shape
    v = Volume(data)
    for key,value in volume.__dict__.items():
        if not key in ['data','vtk_image'] :
            v.__dict__[key] = deepcopy(value)
    v.vtk_image=None
    return v



def sitk_load_nrrd(filename,  return_itk=False, space = 'LPS'):
    import SimpleITK as sitk

    nr = sitk.ReadImage(filename)
   
    """rl, pa, si = nr.GetMetaData('NRRD_space').split('-')
                if  rl == 'left':
                    T[:,0]*=-1
                    origin[0]*=-1
                elif not 'right':
                    raise Exception('space not known')
                if  pa == 'posterior':
                    T[:,1]*=-1
                    origin[1]*=-1
                elif not 'anterior':
                    raise Exception('space not known')
            
                if  si == 'inferior':
                    T[:,2]*=-1
                    origin[1]*=-1
                elif not 'superior':
                    raise Exception('space not known')
            """
    v = Volume(space=space)
    v.from_itk(nr)
        
    
    if return_itk:
        return v, nr    
    else:
        return v

def transform_volume(transform,  volume , rotation_center = None, output_shape = None, spacing = None,  order =3, fill = 0.0, only_transform = False):
    """
    creates a transformed volume with the transformation defined by world coordinates.
    if no rotation_center is provided, the volume is rotated around its center that will be in the middle of the new volume. 
    If no output_shape is provided, it the mean of the original extend is used, taking into account the new spacing.
    When no spacing is provided, the mean spacing of the original image is used along all axes.
    voxels outside image will be filled with fill val
    
    """
    old_shape = np.array(volume.shape)
    if rotation_center is None:
        rotation_center = volume.to_world(old_shape/2)#.mean()
    if spacing is None:
        spacing =  np.abs(np.linalg.eigvals(volume.to_world.matrix33)).mean()
    if np.atleast_1d(3).shape == (1,):
        spacing = np.ones(3)*spacing
    
    if output_shape is None:
        old_extend = np.dot(volume.to_world.matrix33, np.array(volume.shape)).mean()
        output_shape = np.abs((old_extend/spacing)).round().astype('int')  ### is the abs neccessary?
       
    #if target_center is None:
    #    target_center = transform.inverse(rotation_center)
    ijklsource_to_target = nc.Matrix44(transform(volume.to_world))
    
    target_to_ijksource = ijklsource_to_target.inverse
    to_ijk = nc.Matrix44(np.eye(4))
    to_ijk.matrix33=np.diag(1/spacing)
    to_target = to_ijk.inverse
    M = target_to_ijksource(to_target)
    target_center_ijk = np.array(output_shape)/2
    offset = volume.to_ijk(rotation_center) - np.dot(M.matrix33,target_center_ijk ) 
    to_target.translation = (np.dot(to_target [:3,:3],-target_center_ijk))+transform(rotation_center)#np.array(i_c.shape)/2))
    M.translation = offset
    if only_transform:
        return(to_target)
    return Volume(ndimage.affine_transform(volume.data,M, 
                                            output_shape=output_shape,
                                            order=order, cval = fill),
                                 ijk_to_world=to_target, space=volume.space)

