# -*- coding: utf-8 -*-

"""
neuclid is a combination of the word new (German: neu) and euclid, the famous inventor of modern euclidean geometry.

Provides friendly geometry objects and transformations.

We use pyrr and transformations.py under the hood with some facade code to make them friendly.
Neuclid follows the "column vectors on the right" and "row major storage" (C contiguous) conventions.
The translation components are in the right columnof the transformation matrix, i.e. M[:3, 3].

To interface with OpenGL, the transpose of a matrix can be used.

## Todo:

- [ ] split into modules (all in one __init__.py sucks)
- [ ] Plane3 ?
- [ ] CoordianateSystem3 ?
- [ ] fitting methods based on Marcels code
- [ ] when Slicer switches to py3, lets get also remove py2 support

"""

from __future__ import absolute_import, division, print_function
from numbers import Number
from multipledispatch import dispatch
import numpy as np

# locally import our vendored Pyrr stuff which we want to expose
from .Pyrr import pyrr
from .Pyrr.pyrr.objects.base import BaseObject, BaseVector, BaseVector3, NpProxy
from .Pyrr.pyrr import (
    Vector3,
    Vector4,
    Matrix33,
    # Quaternion,
)
import warnings

with warnings.catch_warnings():
    warnings.simplefilter("ignore", UserWarning)
    from .transformations import angle_between_vectors as _angle_between_vectors
    from .transformations import superimposition_matrix as _superimpositon_matrix
from .Pyrr.pyrr.objects.base import BaseVector4

BaseObject.__str__ = BaseObject.__repr__


def orthogonalize3(v1=None, v2=None, v3=None, normalize=True, seed=None):
    if normalize == False:
        raise NotImplemented  # todo, save length and add back again for each
    if seed is not None:
        np.random.seed(seed)

    if v1 is None:
        v1 = (np.random.randn(3) - 0.5) * 2
    v1 = Vector3(v1)
    assert v1.length > 0.0, "v1 has length 0.0"
    v1n = v1.normalized

    if v2 is None:
        v2 = (np.random.randn(3) - 0.5) * 2
    v2 = Vector3(v2)
    assert v2.length > 0.0, "v1 has length 0.0"
    v2n = v2.normalized
    # remove the component of v2n that points in the direction of v1n
    v2n = v2n - (v1n * np.dot(v1n, v2n))
    if v2n.length <= 0.0:
        raise Exception(
            "v1 and v2 are colinear. Set v2=None to repalce v2 by a random vector."
        )
    v2n.normalize()

    v3n = Vector3(np.cross(v1n, v2n))
    return v1n, v2n, v3n


class Point3(BaseVector3):
    _shape = (3,)
    _module = Vector3._module  # reuse length, normalized etc. from Vector3

    #: The X value of this Vector.
    x = NpProxy(0)
    #: The Y value of this Vector.
    y = NpProxy(1)
    #: The Z value of this Vector.
    z = NpProxy(2)
    #: The X,Y values of this Vector as a numpy.ndarray.
    xy = NpProxy([0, 1])
    #: The X,Y,Z values of this Vector as a numpy.ndarray.
    xyz = NpProxy([0, 1, 2])
    #: The X,Z values of this Vector as a numpy.ndarray.
    xz = NpProxy([0, 2])

    def __new__(cls, xyz=None, dtype=np.float):
        if xyz is not None:
            obj = np.asarray(xyz, dtype=dtype)
        else:
            obj = np.zeros(cls._shape, dtype=dtype)
        obj = obj.view(
            cls
        )  # necesarry to make this appear as a type Point3 instead of np.ndarray
        return super(Point3, cls).__new__(cls, obj)

    ########################
    # Vectors and Points
    @dispatch((BaseVector3, np.ndarray, list))
    def __add__(self, other):
        return type(self)(super(type(self), self).__add__(other))

    @dispatch((BaseVector3, np.ndarray, list))
    def __sub__(self, other):
        # can't do this dispatch because Point3 is not defined during the decorator is applied
        if isinstance(other, Point3):
            return Vector3(super(type(self), self).__sub__(other))
        else:
            return type(self)(super(type(self), self).__sub__(other))

    @dispatch((BaseVector3, np.ndarray, list))
    def __mul__(self, other):
        return type(self)(super(type(self), self).__mul__(other))

    @dispatch((BaseVector3, np.ndarray, list))
    def __truediv__(self, other):
        return type(self)(super(type(self), self).__truediv__(other))

    ########################
    # Number
    @dispatch((Number, np.number))
    def __add__(self, other):
        return type(self)(super(type(self), self).__add__(other))

    @dispatch((Number, np.number))
    def __sub__(self, other):
        return type(self)(super(type(self), self).__sub__(other))

    @dispatch((Number, np.number))
    def __mul__(self, other):
        return type(self)(super(type(self), self).__mul__(other))

    @dispatch((Number, np.number))
    def __truediv__(self, other):
        return type(self)(super(type(self), self).__truediv__(other))

    @dispatch((Number, np.number))
    def __div__(self, other):
        return type(self)(super(type(self), self).__div__(other))

    ########################
    # disable inherited stuff that is not defined for points
    def from_matrix44_translation(self):
        raise AttributeError("from_matrix44_translation not available for Point3")

    def normalize(self):
        raise AttributeError("Cannot normalize a point")

    @property
    def normalized(self):
        raise AttributeError("Cannot normalize a point")

    def normalise(self):
        raise AttributeError("Cannot normalize a point")

    @property
    def normalised(self):
        raise AttributeError("Cannot normalize a point")

    @property
    def magnitude(self):
        return self._module.length(self)

    @property
    def length(self):
        raise AttributeError("length is not defined for points")

    @length.setter
    def length(self, v):
        raise AttributeError("length is not defined for points")

    @property
    def squared_length(self):
        raise AttributeError("length is not defined for points")


# class Point4(Point3):


class LineSegment3(BaseObject):
    _shape = (2, 3)

    def __new__(cls, start=None, end=None, dtype=np.float):
        """A line from start to end."""
        if start is None and end is None:
            obj = pyrr.line.create_zeros()
        elif end is None:
            obj = start.copy()
        else:
            obj = pyrr.line.create_from_points(start, end)
        obj = obj.view(cls)
        return super(LineSegment3, cls).__new__(cls, obj)

    @property
    def start(self):
        "Returns a copy of the start"
        return Point3(pyrr.line.start(self))

    @property
    def end(self):
        return Point3(pyrr.line.end(self))

    def closest_point_to(self, p):
        return closest_point(self, p)

    @property
    def length(self):
        return (self.end - self.start).length

    @length.setter
    def length(self, value):
        self[1, :] = self.start + self.direction * value

    def connect(self, other):
        return connect(self, other)

    @property
    def v(self):
        "The vector pointing from start to end."
        return self.end - self.start

    @property
    def direction(self):
        return self.v.normalized


class Line3(LineSegment3):
    @property
    def length(self):
        "The length of a line is infinit per definition. "
        return np.Infinity


class Plane3(BaseVector4):

    # class Vector4(BaseVector4):
    _module = (
        Vector3._module
    )  # TODO : does this make  sense, or should it be plane or vector4 or so?
    _shape = (4,)

    #: The X value of the normal of this plane.
    x = NpProxy(0)
    #: The Y value of the normal of this plane.
    y = NpProxy(1)
    #: The Z value of the normal of this plane.
    z = NpProxy(2)
    #: The distance from the origin of this plane.
    d = NpProxy(3)
    #: The X,Y values of the normal of this plane as a numpy.ndarray.
    xy = NpProxy([0, 1])
    #: The X,Y,Z values of the normal of this plane as a numpy.ndarray.
    xyz = NpProxy([0, 1, 2])
    #: The X,Y,Z,D values of the normal of this plane as a numpy.ndarray.
    xyzd = NpProxy(slice(0, 4))
    #: The X,Z values of the normal of this plane as a numpy.ndarray.
    xz = NpProxy([0, 2])
    #: The X value of the normal and the distance of this plane as a numpy.ndarray.
    xd = NpProxy([0, 3])
    #: The Y value of the normal and the distance of this plane as a numpy.ndarray.
    yd = NpProxy([0, 3])
    #: The Z value of the normal and the distance of this plane as a numpy.ndarray.
    zd = NpProxy([0, 3])
    #: The X,Y values of the normal and the distance of this plane as a numpy.ndarray.
    xyd = NpProxy([0, 1, 3])
    #: The X,Z values of the normal of this plane as a numpy.ndarray.
    xzd = NpProxy([0, 2, 3])

    # normal = xyz #FIXME should this already be casted before (NpProxy)
    @property
    def normal(self):
        return Vector3(self[:3]).normalized

    @normal.setter  # should we normalize here?
    def normal(self, value):
        self[:3] = value

    ########################
    # Creation
    @classmethod
    def create(cls, normal=None, distance=0.0, dtype=None):
        # WARNING: it is not checked that normal is nomalized
        return cls(pyrr.plane.create(normal, distance, dtype))

    @classmethod
    def create_from_points(cls, point1, point2, point3, dtype=None):
        return cls(pyrr.plane.create_from_points(point1, point2, point3))

    @classmethod
    def create_from_position(cls, position, normal, dtype=None):
        return cls(pyrr.plane.create_from_position(position, normal, dtype))

    @classmethod
    def create_xy(cls, invert=False, distance=0.0, dtype=None):
        return cls(pyrr.plane.create_xy(invert, distance, dtype))

    @classmethod
    def create_xz(cls, invert=False, distance=0.0, dtype=None):
        return cls(pyrr.plane.create_xz(invert, distance, dtype))

    @classmethod
    def create_yz(cls, invert=False, distance=0.0, dtype=None):
        return cls(pyrr.plane.create_yz(invert, distance, dtype))

    def invert_normal(self):
        return Plane3(pyrr.plane.invert_normal(self))

    def position(self):
        return Point3(pyrr.plane.position(self))

    # def normal(self, plane):
    #   return Vector3(pyrr.plane(plane))

    @dispatch(Point3)
    def distance_to(self, point):
        return pyrr.geometric_tests.point_height_above_plane(point, self)

    @dispatch(Line3)
    def intersect_line(self, line):
        return pyrr.geometric_tests.ray_intersect_plane(
            [line.start, line.direction], self
        )

    @dispatch(LineSegment3)
    def intersect_line(self, line):
        if (
            self.distance_to(line.start) * self.distance_to(line.end) <= 0
        ):  # points are on opposite sites of plane, or one is on plane
            return pyrr.geometric_tests.ray_intersect_plane(
                [line.start, line.direction], self
            )
        else:
            return None

    @property
    def normalized(self):
        return self / np.linalg.norm(self[:3])

    def __new__(cls, value=None, dtype=None):
        if value is not None:
            obj = value
            if not isinstance(value, np.ndarray):
                obj = np.array(value, dtype=dtype)

            """# matrix44
            if obj.shape in ((4,4,)) or isinstance(obj, BaseMatrix44):
                obj = vector4.create_from_matrix44_translation(obj, dtype=dtype)
                """
        else:
            obj = np.zeros(cls._shape, dtype=dtype)
        obj = obj.view(cls)
        return super(Plane3, cls).__new__(cls, obj)

    ########################
    # Basic Operators
    @dispatch(BaseObject)
    def __add__(self, other):
        self._unsupported_type("add", other)

    @dispatch(BaseObject)
    def __sub__(self, other):
        self._unsupported_type("subtract", other)

    @dispatch(BaseObject)
    def __mul__(self, other):
        self._unsupported_type("multiply", other)

    @dispatch(BaseObject)
    def __truediv__(self, other):
        self._unsupported_type("divide", other)

    @dispatch(BaseObject)
    def __div__(self, other):
        self._unsupported_type("divide", other)

    @dispatch((BaseObject, Number, np.number))
    def __xor__(self, other):
        self._unsupported_type("XOR", other)

    @dispatch((BaseObject, Number, np.number))
    def __or__(self, other):
        self._unsupported_type("OR", other)

    @dispatch((BaseObject, Number, np.number))
    def __ne__(self, other):
        self._unsupported_type("NE", other)

    @dispatch((BaseObject, Number, np.number))
    def __eq__(self, other):
        self._unsupported_type("EQ", other)

    ######################## FIXME: does it make sense to do calculations like this with planes
    # Vectors
    @dispatch((BaseVector4, np.ndarray, list))
    def __add__(self, other):
        return Plane3(super(Plane3, self).__add__(other))

    @dispatch((BaseVector4, np.ndarray, list))
    def __sub__(self, other):
        return Plane3(super(Plane3, self).__sub__(other))

    @dispatch((BaseVector4, np.ndarray, list))
    def __mul__(self, other):
        return Plane3(super(Plane3, self).__mul__(other))

    @dispatch((BaseVector4, np.ndarray, list))
    def __truediv__(self, other):
        return Plane3(super(Plane3, self).__truediv__(other))

    @dispatch((BaseVector4, np.ndarray, list))
    def __div__(self, other):
        return Plane3(super(Plane3, self).__div__(other))

    # @dispatch(BaseVector)
    # def __xor__(self, other):
    #    return self.cross(Plane3(other))

    @dispatch((BaseVector4, np.ndarray, list))
    def __or__(self, other):
        return self.dot(Plane3(other))

    @dispatch((BaseVector4, np.ndarray, list))
    def __ne__(self, other):
        return bool(np.any(super(Plane3, self).__ne__(other)))

    @dispatch(
        (Vector4, np.ndarray, list)
    )  # FIXME: should this be Basevector4 instead of Vector4?
    def __eq__(self, other):
        return bool(np.all(super(Vector4, self).__eq__(other)))

    ########################
    # Number
    @dispatch((Number, np.number))
    def __add__(self, other):
        return type(self)(super(type(self), self).__add__(other))

    @dispatch((Number, np.number))
    def __sub__(self, other):
        return type(self)(super(type(self), self).__sub__(other))

    @dispatch((Number, np.number))
    def __mul__(self, other):
        return type(self)(super(type(self), self).__mul__(other))

    @dispatch((Number, np.number))
    def __truediv__(self, other):
        return type(self)(super(type(self), self).__truediv__(other))

    @dispatch((Number, np.number))
    def __div__(self, other):
        return type(self)(super(type(self), self).__div__(other))

    """
    ########################
    # Number
    @dispatch((Number, np.number))
    def __add__(self, other):
        return Plane3(super(Plane3, self).__add__(other))

    @dispatch((Number, np.number))
    def __sub__(self, other):
        return Plane3(super(Plane3, self).__sub__(other))

    @dispatch((Number, np.number))
    def __mul__(self, other):
        return Plane3(super(Plane3, self).__mul__(other))

    @dispatch((Number, np.number))
    def __truediv__(self, other):
        return Plane3(super(Plane3, self).__truediv__(other))

    @dispatch((Number, np.number))
    def __div__(self, other):
        return Plane3(super(Plane3, self).__div__(other))
        """

    ########################
    # Methods and Properties
    @property
    def inverse(self):
        """Returns the opposite of this vector."""
        return Vector4(-self)

    @property
    def vector3(self):
        """Returns a Vector3 and the W component as a tuple."""
        return (Vector3(self[:3]), self[3])


class Points3(np.ndarray):
    def __new__(cls, points, dtype=np.float):

        obj = np.asarray(points, dtype=dtype)
        obj = obj.view(
            cls
        )  # necesarry to make this appear as a type Point3 instead of np.ndarray
        assert len(obj.shape) == 2
        assert obj.shape[1] == 3
        return obj
        # return super(Points3, cls).__new__(cls, obj)

    def fit_plane(self):
        datamean = self.cms()
        u, dd, vv = np.linalg.svd(self - datamean, full_matrices=False)
        return Plane3.create_from_position(datamean, vv[-1])

    def fit_line(self):
        """"""
        datamean = self.cms()
        u, dd, vv = np.linalg.svd(self - datamean, full_matrices=False)
        length = np.linalg.norm(dd)
        return LineSegment3(
            datamean - length / 4 * vv[0], datamean + length / 4 * vv[0]
        )

    def cms(self):
        return Point3(self.mean(axis=0))

    @dispatch(Point3)
    def distance_to(self, point):
        return np.linalg.norm(self - np.atleast_2d(point), axis=1)

    @dispatch(Line3)
    def distance_to(self, line):
        rl = np.array(line.direction)  # line.end - line.start)
        rp = self - line.start
        dot = np.array(np.dot(rl, rp.T))
        closest_point = np.array(line.start)[np.newaxis, :] + (
            np.array(line.direction) * dot[:, np.newaxis]
        )
        connection = self - closest_point
        return np.linalg.norm(connection, axis=1)

    @dispatch(LineSegment3)
    def distance_to(self, line):
        rl = np.array(line.end - line.start)
        rp = self - line.start
        # squared_length = rl / np.linalg.norm(rl)
        dot = np.array(np.dot(line.direction, rp.T))
        closest_point = np.array(line.start)[np.newaxis, :] + (
            np.array(line.direction) * dot[:, np.newaxis]
        )
        closest_point[dot < 0] = line.start
        closest_point[dot > np.linalg.norm(rl)] = line.end
        connection = self - closest_point
        return np.linalg.norm(connection, axis=1)

    @dispatch(Plane3)
    def distance_to(self, plane):
        points4 = np.c_[self, np.ones(self.shape[0])]
        plane_copy = np.array(plane.normalized)  # .copy()
        # plane_copy[:3] /= np.linalg.norm(plane[:3])
        return np.dot(plane_copy, points4.T)

    @dispatch((BaseVector4, np.ndarray, list))
    def __add__(self, other):
        return Points3(super(Points3, self).__add__(np.asarray(other)))

    @dispatch((BaseVector4, np.ndarray, list))
    def __sub__(self, other):
        return Points3(super(Points3, self).__sub__(np.asarray(other)))

    @dispatch((BaseVector4, np.ndarray, list))
    def __mul__(self, other):
        return Points3(super(Points3, self).__mul__(np.asarray(other)))

    @dispatch((BaseVector4, np.ndarray, list))
    def __truediv__(self, other):
        return Points3(super(Points3, self).__truediv__(np.asarray(other)))

    @dispatch((BaseVector4, np.ndarray, list))
    def __div__(self, other):
        return Points3(super(Points3, self).__div__(np.asarray(other)))


# enhance Matrix44 by type preserving matmul (dot)
class Matrix44(pyrr.Matrix44):
    def __getitem__(self, item):
        return np.asarray(super(type(self), self).__getitem__(item))

    @dispatch(np.ndarray)
    def __call__(self, points):
        points2 = np.atleast_2d(
            points
        )  # assure points has the shape (n,3) where n is the number of points
        transformed = [pyrr.matrix44.apply_to_vector(self.T, p) for p in points2]
        if len(transformed) == 1:
            return transformed[
                0
            ]  # i a single [x, y, z] point was given, return a single point.
        else:
            return np.asanyarray(
                transformed
            )  #  Otherwise return in shape (n,3) for n points.

    @dispatch(list)
    def __call__(self, points):
        if isinstance(points[0], (list, Number)):
            # nested list of lists, so we assume something like [ [1,2,3], [4,5,6], ... ]
            return self(np.asarray(points))
        else:
            # flat list of Objects like Point3 or Vector3 etc.
            types = [type(i) for i in points]
            return [
                typ(pyrr.matrix44.apply_to_vector(self.T, p))
                for p, typ in zip(points, types)
            ]

    @dispatch(BaseVector)
    def __call__(self, other):
        return type(other)(pyrr.matrix44.apply_to_vector(self.T, other))

    @dispatch(pyrr.Matrix44)
    def __call__(self, other):
        return Matrix44(np.dot(self, other))

    @dispatch(Points3)
    def __call__(self, other):
        return Points3(
            np.dot(self, (np.r_[other.T, np.ones((1, other.shape[0]))]))[:3].T
        )

    @dispatch(LineSegment3)
    def __call__(self, other):
        other_start_transformed = pyrr.matrix44.apply_to_vector(self.T, other.start)
        other_end_transformed = pyrr.matrix44.apply_to_vector(self.T, other.end)
        return type(other)(start=other_start_transformed, end=other_end_transformed)

    # todo implement transformation chaining by calling m1(m2)
    @property
    def translation(self):
        if not np.allclose(self[3], [0, 0, 0, 1]):
            raise Exception(
                "Marrix44 is not an affine transformation, translation not defined"
            )
        return Point3(self[:3, 3])

    @translation.setter
    def translation(self, value):
        if not np.allclose(self[3], [0, 0, 0, 1]):
            raise Exception(
                "Marrix44 is not an affine transformation, translation not defined"
            )
        self[:3, 3] = value

    @property
    def matrix33(self):
        if not np.allclose(self[3], [0, 0, 0, 1]):
            raise Error(
                "Marrix44 is not an affine transformation, matrix33 not defined"
            )
        return np.asarray(self[:3, :3])

    @matrix33.setter
    def matrix33(self, value):
        if not np.allclose(self[3], [0, 0, 0, 1]):
            raise Error(
                "Marrix44 is not an affine transformation, translation not defined"
            )
        self[:3, :3] = value

    @classmethod
    def create_from_eulers(cls, eulers, dtype=None):
        return cls(pyrr.matrix44.create_from_eulers(eulers, dtype).T)

    @classmethod
    def create_from_axis_rotation(
        cls, axis, theta, center=Point3([0, 0, 0]), dtype=None
    ):
        rotation = cls(pyrr.matrix44.create_from_axis_rotation(axis, theta, dtype).T)
        if np.allclose(center, [0, 0, 0]):
            return rotation
        else:
            translation = cls.create_from_translation(center)
            return translation(rotation(translation.inverse))

    @classmethod
    def create_from_quaternion(cls, quat, dtype=None):
        return cls(pyrr.matrix44.create_from_quaternion(quat, dtype).T)

    @classmethod
    def create_from_inverse_of_quaternion(cls, quat, dtype=None):
        return cls(pyrr.matrix44.create_from_inverse_of_quaternion(quat, dtype).T)

    @classmethod
    def create_from_translation(cls, vec, dtype=None):
        return cls(pyrr.matrix44.create_from_translation(vec, dtype).T)

    @classmethod
    def create_from_x_rotation(cls, theta, dtype=None):
        return cls(pyrr.matrix44.create_from_x_rotation(theta, dtype))  # should be .T

    @classmethod
    def create_from_y_rotation(cls, theta, dtype=None):
        return cls(pyrr.matrix44.create_from_y_rotation(theta, dtype))  # should be .T

    @classmethod
    def create_from_z_rotation(cls, theta, dtype=None):
        return cls(pyrr.matrix44.create_from_z_rotation(theta, dtype))  # should be .T

    @classmethod
    def apply_to_vector(cls, mat, vec):
        return cls(mat)(vec)

    def decompose(self):
        return pyrr.matrix44.decompose(self.T)

    @classmethod
    def create_perspective_projection(cls, *args):
        raise NotImplementedError("create_perspective_projection not implemented")

    @classmethod
    def create_perspective_projection_from_bounds(cls, *args):
        raise NotImplementedError(
            "create_perspective_projection_from_bounds not implemented"
        )

    @classmethod
    def create_perspective_projection_matrix_from_bounds(cls, *args):
        raise NotImplementedError(
            "create_perspective_projection_matrix_from_bounds not implemented"
        )

    @classmethod
    def create_orthogonal_projection(cls, *args):
        raise NotImplementedError("create_orthogonal_projection not implemented")

    @classmethod
    def create_orthogonal_projection_matrix(cls, *args):
        raise NotImplementedErrorr(
            "create_orthogonal_projection_matrix not implemented"
        )

    @classmethod
    def create_look_at(cls, *args):
        raise NotImplementedError("create_look_a not implemented")


@dispatch(Point3, Point3)
def connect(p1, p2):
    return LineSegment3(p1, p2)


@dispatch(LineSegment3, Point3)
def connect(l, p):
    return LineSegment3(
        pyrr.geometric_tests.point_closest_point_on_line_segment(p, l), p
    )


@dispatch(Point3, LineSegment3)
def connect(p, l):
    return LineSegment3(
        p, pyrr.geometric_tests.point_closest_point_on_line_segment(p, l)
    )


@dispatch(Line3, Point3)
def connect(l, p):
    return LineSegment3(pyrr.geometric_tests.point_closest_point_on_line(p, l), p)


@dispatch(Point3, Line3)
def connect(p, l):
    return LineSegment3(p, pyrr.geometric_tests.point_closest_point_on_line(p, l))


@dispatch(Point3, Plane3)
def connect(point, plane):
    return LineSegment3(
        point, pyrr.geometric_tests.point_closest_point_on_plane(point, plane)
    )


@dispatch(Plane3, Point3)
def connect(plane, point):
    return LineSegment3(
        pyrr.geometric_tests.point_closest_point_on_plane(point, plane), point
    )


@dispatch(LineSegment3, LineSegment3)
def connect(la,lb): #two_lines
    """
    inspired from https://stackoverflow.com/questions/2824478/shortest-distance-between-two-line-segments
    """
    
    cross = np.cross(la.direction, lb.direction)
    denom =  np.linalg.norm(cross) ** 2
    c0 = lb.start - la.start
    if not denom: #lines parallel
        d0 = np.dot(la.direction, c0) #projection of difference along line
        
        if  not(isinstance(la,Line3) or isinstance(lb,Line3)): #both line segments
            
        
            d1 = np.dot(la.direction, lb.end - la.start)
            # Is segment B before A?
            if d0 <= 0 >= d1:
                if np.absolute(d0) < np.absolute(d1):
                    return LineSegment3(la.start, lb.start)
                return LineSegment3(la.start, lb.end)
            
            # Is segment B after A?
            elif d0 >= la.length <= d1:
                
                if np.absolute(d0) < np.absolute(d1):
                    return LineSegment3(la.end, lb.start)
                return LineSegment3(la.end, lb.end)
            
        c = d0*la.direction-c0
        #d = np.linalg.norm(c)
        return LineSegment3(la.start, la.start+c)
        # Lines criss-cross: Calculate the projected closest points
        
    detA = np.linalg.det([c0, lb.direction, cross])
    detB = np.linalg.det([c0, la.direction, cross])

    t0 = detA / denom
    t1 = detB / denom

    pA = la.start + (la.direction * t0)  # Projected closest point on segment A
    pB = lb.start + (lb.direction * t1)  # Projected closest point on segment B
    
    if  not (isinstance(la,Line3) and isinstance(lb,Line3)):
        if not isinstance(la,Line3):
            if t0 <0: 
                pA = la.start
            elif t0>la.length:
                pA = la.end
        
        if not isinstance(lb,Line3):
            if t1 <0: 
                pB = lb.start
            elif t1>lb.length:
                pB = lb.end
            
        
        if not isinstance(la,Line3) and (t0<0 or t0>la.length):
            dot = np.dot(lb.direction, (pA - lb.start))
            if not isinstance(lb,Line3):
                if  dot < 0:
                    dot = 0
                elif dot > lb.length:
                    dot = lb.length
            pB = lb.start + (lb.direction * dot)
        
        if not isinstance(lb,Line3) and (t1<0 or t1>lb.length):
            dot = np.dot(la.direction, (pB - la.start))
            if not isinstance(la,Line3):
                if  dot < 0:
                    dot = 0
                elif dot > la.length:
                    dot = la.length
            pA = la.start + (la.direction * dot)

    
    
    return LineSegment3(pA,pB)

    


def closest_point(starting_obj, target_obj):
    """Return the point on `starting_obj` that is closest to `target_obj`."""
    return Point3(connect(starting_obj, target_obj).start)


@dispatch(LineSegment3, LineSegment3)
def angle(l1, l2, directed=True):
    # when directed == False, the angle smaller than pi/2 is chosen
    return _angle_between_vectors(l1.direction, l2.direction, directed=directed)


@dispatch(Vector3, Vector3)
def angle(v1, v2, directed=True):
    # when directed == False, the angle smaller than pi/2 is chosen
    return _angle_between_vectors(v1, v2, directed=directed)


def superimpose_points(source_points, target_points, scale=False, usesvd=True):
    M = _superimpositon_matrix(
        np.array(source_points).T, np.array(target_points).T, scale=scale, usesvd=usesvd
    )
    return Matrix44(M)


# todo: add dispatched function for angle between Plane3 and vector/line/...
# todo: add dispatched function for angle between two planes
