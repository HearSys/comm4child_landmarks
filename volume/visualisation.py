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

import numpy as np
from matplotlib import pyplot as plt
from . import volume 
from mayavi import mlab
from tvtk.api import tvtk


def show_cochlea(vol, points = None,axis=0, di=0, crop_points_to_slice=True, imargs={}, plotargs={}):
    shift_offset=-0.5
    min_ = vol.to_world(np.array([0,0,0])+shift_offset)
    max_ = vol.to_world(np.array(vol.shape)-shift_offset)

    l = [0,1,2]
    l.remove(axis)
    i,j = l
    #sl = vol.data.take(indices=, axis=axis)
    k = vol.shape[axis]//2+di
    if axis == 0:
        sl = vol[k:k+1,:,:]
        sli = sl.data[0]
    if axis == 1:
        sl = vol[:,k:k+1,:]
        sli = sl.data[:,0]
    if axis == 2:
        sl = vol[:,:,k:k+1]
        sli = sl.data[:,:,0]
    plt.imshow(sli, extent=[min_[j], max_[j], max_[i], min_[i]], **imargs)
    #plt.plot(*i_c.ijk_to_world.inverse(xyzs)[:,1:].T)
    
    if not points is None:
        if crop_points_to_slice:
            
            l = volume.cut_points_to_volume(points, sl,True)
            for points in l:
                if len(points)>1:
                     plt.plot(points[:,j],points[:,i],'r-', **plotargs)
                else:
                    plt.plot(points[:,j],points[:,i],'ro', **plotargs)
        else:
            plt.plot(points[:,j],points[:,i],'r-',**plotargs)
    ax = plt.gca()
    ax.set_aspect('equal')


def sliceing(i, vol, helix, axis = 2, spacing =.3, order=3):

    max_phi = helix.angles(helix.points,from_world=True)[-1]
    points = helix.points_at_angles(np.linspace(0,max_phi,900))

    if axis == 2:
        center = helix.ccs.to_world([0,0,i*spacing])
        output_shape = [50,50,1]
    if axis == 1:
        center = helix.ccs.to_world([0,i*spacing,0])
        output_shape = [50,1,50]
    if axis == 0:
        center = helix.ccs.to_world([i*spacing,0,0])
        output_shape = [1,50,50]
    i_r = volume.transform_volume(helix.ccs.to_local,  vol , rotation_center =center, output_shape = output_shape, spacing = spacing, order = order)
    show_cochlea(i_r, points, axis =axis)


def add_vtk_dataset(volume):
    spacing = np.diag(volume.to_world.matrix33)
    assert np.all(np.abs(spacing) == np.linalg.norm(volume.to_world.matrix33, axis=0))
    i = tvtk.ImageData(spacing=spacing, origin=volume.to_world[:3,3])
    i.point_data.scalars = volume.data.T.ravel()
    i.point_data.scalars.name = 'scalars'
    i.dimensions = volume.data.shape
    volume.vtk_image = i

def contour3d(volume, contours, **kwargs):
    if volume.vtk_image is None:
        add_vtk_dataset(volume)
    return mlab.pipeline.contour_surface(volume.vtk_image, contours = contours, **kwargs)
    
    
def cutplane(volume, normal = None, origin = None):
    if volume.vtk_image is None:
        add_vtk_dataset(volume)
    
    sc = mlab.pipeline.scalar_cut_plane(volume.vtk_image)#ac.module_manager)
    #ac.visible=False
    pw = sc.implicit_plane.widgets[0]
    if not normal is None:
        pw.normal = normal
    if not origin is None:
        pw.origin = origin
    return sc,pw
    

