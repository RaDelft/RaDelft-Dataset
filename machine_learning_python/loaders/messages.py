from __future__ import print_function

import os
import json
import copy
import threading

from PIL import Image
import numpy as np
import numpy.lib.recfunctions

# ---------------------------------------
# Build ROS objects from dictionary
# ---------------------------------------

def get_ROS_class_by_name(name):
    """ map ROS message type string to class
        e.g. 'geometry_msgs/TransformStamped[]' to
            geometry_msgs.msg.TransformStamped
    """
    import importlib

    if name.endswith('[]'):
        # ignore list specification
        name = name[:-2]

    modname, msgname = name.split('/')

    module = importlib.import_module(modname + '.msg')
    c = getattr(module, msgname)
    return c


def build_ROS_obj(name, d):
    c = get_ROS_class_by_name(name)
    x = c()
    fill_ROS_obj(x, d)
    return x

def fill_ROS_obj(obj, d):
    types_per_slot = dict(zip(obj.__slots__, obj._slot_types))

    for k, v in d.items():
        if not hasattr(obj, k):
            #print('Warning: skipping non-existent field "%s"' % k)
            continue
        elif isinstance(v, dict):
            fill_ROS_obj(getattr(obj,k), v)
        elif isinstance(v, list):
            iname = types_per_slot[k]
            v = [build_ROS_obj(iname, item) for item in v]
            setattr(obj, k, v)
        elif isinstance(v, unicode):
            v = v.encode('ascii', 'ignore')
            setattr(obj, k, v)
        else:
            setattr(obj, k, v)

# ---------------------------------------
# Helper functions
# ---------------------------------------

def update_defaults(data, defaults):
    """ Update the fields in data with the default values in defaults.
        Also return the updated dictionary.
    """
    for k, v in defaults.items():
        data.setdefault(k, v)
    return data

def set_header_in_data_dict(data, ts_ns=None, frame_id=None, seq=None):
    """ Helper function to update the common header fields in a data dict. """
    data.setdefault('header', DEFAULT_HEADER_DATA)
    if not ts_ns is None:
        secs = int(ts_ns // 1000000000)
        nsecs = int(ts_ns % 1000000000)
        data['header']['stamp']['secs'] = secs
        data['header']['stamp']['nsecs'] = nsecs
    if not frame_id is None:
        data['header']['frame_id'] = frame_id
    if not seq is None:
        data['header']['seq'] = seq

# ---------------------------------------
# Portable message objects to describe ROS messages
# with a dictionary and possibly related raw data,
#  e.g. an image stored on disk
# ---------------------------------------

DEFAULT_HEADER_DATA = {'stamp': {'secs': 0, 'nsecs': 0}, 'frame_id': None, 'seq': None}

class GenericMessage(object):
    DEFAULT_MSG_TYPE = None
    DEFAULT_DATA = {'_timestamp_ns': None}

    def __init__(self, msg_type=None, data=None, base_path=None):
        self.msg_type = (msg_type if msg_type else self.DEFAULT_MSG_TYPE)
        self.base_path = base_path
        self.raw_data = None
        self.data = copy.deepcopy(self.DEFAULT_DATA)
        if not data is None:
            self.set_data(data)

    def set_data(self, data):
        # update the argument data dict, and fill any missing
        # values with any existing values in self.data
        self.data = update_defaults(data, defaults=self.data)

    def get_timestamp_ns(self):
        return self.data['_timestamp_ns']

    def set_timestamp_ns(self, ts):
        self.data['_timestamp_ns'] = ts

    def get_hash_id(self):
        ts = self.data['_timestamp_ns']
        ts = int(ts)
        secs = ts // int(1e9)
        nsecs = ts % int(1e9)
        secs = int(secs)
        nsecs = int(nsecs)
        hash_id = '%012d.%09d' % (secs, nsecs)
        return hash_id

    def to_ROS_obj(self):
        """ build ROS instance from object. Python most be run
            in a ROS environment, providing rospy and ROS message modules.
        """
        return build_ROS_obj(self.msg_type, self.data)

    def __getitem__(self, prop):
        # shorthand
        return self.data[prop]

    def __repr__(self):
        return '<%s: %s>' % (self.__class__.__name__, self.msg_type)

# ---------------------------------------
# Messages with custom raw data handlers
# ---------------------------------------

class GenericDictMessage(GenericMessage):
    DEFAULT_MSG_TYPE = None
    DEFAULT_DATA = {'_timestamp_ns': None, '_data_filepath': None}

    def get_abs_filepath(self):
        raw_rel_path = self.data['_data_filepath']
        raw_abs_filepath = os.path.join(self.base_path, raw_rel_path)
        return raw_abs_filepath

    def save_dict(self, dictionary, raw_rel_path=None, overwrite=False):
        if raw_rel_path is None:
            raw_rel_path = self.data['_data_filepath']
        assert(not raw_rel_path is None)

        raw_abs_filepath = os.path.join(self.base_path, raw_rel_path)
        if overwrite or not os.path.exists(raw_abs_filepath):
            x = threading.Thread(target=self._save, args=(dictionary, raw_abs_filepath))
            x.start()
        self.data['_data_filepath'] = raw_rel_path

    def load_dict(self):
        with open(self.get_abs_filepath(), 'r') as file:
            return json.load(file)

    @staticmethod
    def _save(dictionary, raw_abs_filepath):
        with open(raw_abs_filepath, 'w') as file:
            json.dump(dictionary, file, indent=2)


class ImageMessage (GenericMessage):
    DEFAULT_MSG_TYPE = 'sensor_msgs/Image'
    DEFAULT_DATA = {
        '_timestamp_ns': None,
        '_data_filepath': None,
        'header': DEFAULT_HEADER_DATA,
        'width': None,
        'height': None,
        'encoding': None,
    }

    def set_data(self, data):
        self.data = update_defaults(data, defaults=self.data)

        self.width = self.data['width']
        self.height = self.data['height']
        self.encoding = self.data['encoding']

    def get_abs_filepath(self):
        raw_rel_path = self.data['_data_filepath']
        raw_abs_filepath = os.path.join(self.base_path, raw_rel_path)
        return raw_abs_filepath

    def save_PIL(self, img, raw_rel_path=None, overwrite=False):
        if raw_rel_path is None:
            raw_rel_path = self.data['_data_filepath']
        assert(not raw_rel_path is None)

        raw_abs_filepath = os.path.join(self.base_path, raw_rel_path)
        if overwrite or not os.path.exists(raw_abs_filepath):
            x = threading.Thread(target=self._save, args=(img, raw_abs_filepath))
            x.start()
        self.data['_data_filepath'] = raw_rel_path

    def save_raw(self, raw_data, raw_rel_path=None, overwrite=False):
        if raw_rel_path is None:
            raw_rel_path = self.data['_data_filepath']
        assert(not raw_rel_path is None)

        raw_abs_filepath = os.path.join(self.base_path, raw_rel_path)
        if overwrite or not os.path.exists(raw_abs_filepath):
            with open(raw_abs_filepath, 'wb') as outfile:
                outfile.write(raw_data)
        self.data['_data_filepath'] = raw_rel_path

    def load_PIL(self):
        raw_filepath = self.get_abs_filepath()
        img = Image.open(raw_filepath)
        assert(img.width == self.width)
        assert(img.height == self.height)

        return img
        
    def load_array(self):
        imdata = np.asarray(self.load_PIL())
        # cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB) # FIXME: test this alternative
        return imdata

    @staticmethod
    def _save(img, raw_abs_filepath):
        img.save(raw_abs_filepath)

class DisparityImageMessage (ImageMessage):
    DEFAULT_MSG_TYPE = 'stereo_msgs/DisparityImage'
    # DEFAULT_DATA is the same as parent class

    def set_data(self, data):
        self.data = update_defaults(data, defaults=self.data)
        
        self.width = self.data['image']['width']
        self.height = self.data['image']['height']
        self.encoding = self.data['image']['encoding']


class CameraInfoMessage (GenericMessage):
    DEFAULT_MSG_TYPE = 'sensor_msgs/CameraInfo'
    DEFAULT_DATA = {
        '_timestamp_ns': None,
        'header': DEFAULT_HEADER_DATA,
        'D': None,
        'K': None,
        'R': None,
        'P': None,
    }

    def get_K(self):
        """ 3x3 projecion/camera matrix of UNRECTIFIED image
            http://docs.ros.org/en/melodic/api/sensor_msgs/html/msg/CameraInfo.html
        """
        mat = np.array(self.data['K'])
        mat.shape = (3,3)
        return mat

    def get_P(self):
        """ 3x4 projecion/camera matrix of RECTIFIED image """
        mat = np.array(self.data['P'])
        mat.shape = (3,4)
        return mat

    def get_R(self):
        """ stereo rectification matrix """
        mat = np.array(self.data['R'])
        mat.shape = (3,3)
        return mat

    def get_D(self):
        """ 5 distortion parameters (k1,k2,t1,t2,k3) """
        vec = np.array(self.data['D'])
        return vec


class VirtualField:
    def __init__(self):
        pass

    def compute_field(self, cloud, msg, transform):
        return NotImplementedError

    def get_required_fields(self):
        return NotImplementedError


class PointCloud2Message (GenericMessage):
    DEFAULT_MSG_TYPE = 'sensor_msgs/PointCloud2'
    DEFAULT_DATA = {
        '_timestamp_ns': None,
        '_data_filepath': None,
        'header': DEFAULT_HEADER_DATA,
        'width': None,
        'height': None,
        'fields': [],
        'point_step': None,
        'row_step': None,
        'is_dense': None,
    }

    def __init__(self, msg_type=None, data=None, base_path=None):
        super(PointCloud2Message, self).__init__(msg_type, data, base_path)
        self.virtual_fields = dict()

    def set_data(self, data):
        self.data = update_defaults(data, defaults=self.data)
        
        self.width = self.data['width']
        self.height = self.data['height']
        self.is_dense = self.data['is_dense']
        self.point_step = self.data['point_step']
        self.row_step = self.data['row_step']

    def has_fields(self, fields):
        my_fields = [field['name'] for field in self.data['fields']]
        my_virtual_fields = self.virtual_fields.keys()
        for field in fields:
            if field not in my_fields and field not in my_virtual_fields:
                return False

        return True

    def add_virtual_field(self, name, virtual_field):
        assert self.has_fields(virtual_field.get_required_fields())
        self.virtual_fields[name] = virtual_field

    def load_cloud(self, fields=None, transform=None):
        raw_filepath = self.data['_data_filepath']
        assert(raw_filepath)

        raw_filepath = os.path.join(self.base_path, raw_filepath)
        cloud = np.load(raw_filepath)

        #assert(cloud.shape[0] == self.width)
        #assert(1 == self.height)

        if not cloud.dtype.names is None:
            # the pointcloud is stored as a structured array,
            # convert it to a standard numpy array

            # TODO: we should use the structured array to first extract the fields
            # before converting it to a regular unstructured field
            cloud = np.lib.recfunctions.structured_to_unstructured(cloud)

        if transform is not None:
            # FIXME: deprecate this, transformations should not be done in the loading routine
            orig_cloud = cloud.copy()
            points = np.array([[0.0], [0.0], [0.0], [1.0]]).repeat(cloud.shape[0], 1)
            field_indices = self._get_field_indices(["x", "y", "z"])
            points[:3, :] = cloud[:, field_indices].T
            cloud[:, field_indices] = np.matmul(transform, points).T[:, :3]
        else:
            orig_cloud = cloud

        if fields is not None:
            is_virtual = np.array([field in self.virtual_fields.keys() for field in fields])
            real_fields = [field for field, is_virt in zip(fields, is_virtual) if not is_virt]
            field_indices = self._get_field_indices(real_fields)
            cloud_subselect = np.zeros([cloud.shape[0], len(fields)])
            cloud_subselect[:, ~is_virtual] = cloud[:, field_indices]
            for virt_index in is_virtual.nonzero()[0]:
                virt_field = self.virtual_fields[fields[virt_index]]
                required_field_indices = self._get_field_indices(virt_field.get_required_fields())
                cloud_subselect[:, virt_index] = virt_field.compute_field(orig_cloud[:, required_field_indices], self, transform)
            cloud = cloud_subselect

        return cloud

    def _get_field_indices(self, fields):
        fields_to_index = {field['name']: index for index, field in enumerate(self.data['fields'])}
        selected_fields = [fields_to_index[field] for field in fields]
        return selected_fields

    def save_cloud(self, cloud, raw_rel_path=None, overwrite=False):
        if raw_rel_path is None:
            raw_rel_path = self.data['_data_filepath']
        assert(not raw_rel_path is None)

        raw_abs_filepath = os.path.join(self.base_path, raw_rel_path)
        if overwrite or not os.path.exists(raw_abs_filepath):
            x = threading.Thread(target=np.save, args=(raw_abs_filepath, cloud))
            x.start()
            #np.save(raw_abs_filepath, cloud)
        self.data['_data_filepath'] = raw_rel_path
        

class TFMessage (GenericMessage):
    DEFAULT_MSG_TYPE = 'sensor_msgs/PointCloud2'
    DEFAULT_DATA = {
        '_timestamp_ns': None,
        'header': DEFAULT_HEADER_DATA,
        'transforms': [],
    }

    def __build_cache(self):
        if not hasattr(self, '__cached_transforms'):
            self.__cached_transforms = dict([
                (d['frame_id'],np.array(d['tform']))
                for d in self.data['transforms'] if not d['tform'] == []
            ])
            base_frame_id = self.data['header']['frame_id']
            self.__cached_transforms[base_frame_id] = np.eye(4)

    def list_frames(self):
        frame_ids = [t['frame_id'] for t in self.data['transforms']]
        base_frame_id = self.data['header']['frame_id']
        if not base_frame_id in frame_ids:
            frame_ids = [base_frame_id] + frame_ids
        return frame_ids


    def get_transform(self, frame_target, frame_orig):
        if frame_target == frame_orig:
            return np.eye(4)

        self. __build_cache()
        base_frame_id = self.data['header']['frame_id']

        if frame_target == base_frame_id:
            return self.__cached_transforms[frame_orig]

        if frame_orig == base_frame_id:
            # invert
            return np.linalg.inv(self.__cached_transforms[frame_target])

        base_to_frame_a = self.__cached_transforms[frame_target]
        base_to_frame_b = self.__cached_transforms[frame_orig]

        # NOTE: solve(X,Y) is more accurate than inv(X).dot(Y)
        #return np.linalg.inv(base_to_frame_a).dot(base_to_frame_b)
        return np.linalg.solve(base_to_frame_a, base_to_frame_b)

    def has_transform(self, frame_target, frame_orig):
        if frame_target == frame_orig:
            return True

        self. __build_cache()
        base_frame_id = self.data['header']['frame_id']

        if frame_target == base_frame_id:
            return frame_orig in self.__cached_transforms

        if frame_orig == base_frame_id:
            return frame_target in self.__cached_transforms

        return (frame_target in self.__cached_transforms) and (frame_orig in self.__cached_transforms)

    def apply_transform(self, target_frame, orig_frame, orig_points):
        mat = self.get_transform(target_frame, orig_frame)
        target_points = (mat[:3,:3].dot(orig_points.T).T + mat[:3,3][None])
        return target_points

# Vision messages:
# - BoundingBox2D
# - Detection2D
# - Detection2DArray

class BoundingBox2D (GenericMessage):
    DEFAULT_MSG_TYPE = 'vision_msgs/BoundingBox2D'
    DEFAULT_DATA = {
        '_timestamp_ns': None,
        'center': {'x': 0., 'y': 0., 'theta': 0.},
        'size_x': 0.,
        'size_y': 0.,
    }

    @property
    def center(self):
        cx = self.data['center']['x']
        cy = self.data['center']['y']
        return cx, cy

    @center.setter
    def center(self, cxcy):
        cx, cy = cxcy
        self.data['center']['x'] = cx
        self.data['center']['y'] = cy

    @property
    def size(self):
        w = self.data['size_x']
        h = self.data['size_y']
        return w, h

    @size.setter
    def size(self, wh):
        w, h = wh
        self.data['size_x'] = w
        self.data['size_y'] = h

    def get_cxcywh(self):
        cx, cy = self.center
        w, h = self.size
        return cx, cy, w, h

    def set_cxcywh(self, cx, cy, w, h):
        self.center = (cx, cy)
        self.size = (w, h)

    def get_x1y1x2y2(self):
        cx, cy, w, h = self.get_cxcywh()
        x1 = cx - w // 2
        y1 = cy - h // 2
        x2 = x1 + w
        y2 = y1 + h
        return x1, y1, x2, y2

    def set_x1y1x2y2(self, x1, y1, x2, y2):
        w = x2 - x1
        h = y2 - y1
        cx = x1 + w // 2
        cy = y1 + h // 2
        self.center = (cx, cy)
        self.size = (w, h)


class Detection2D (GenericMessage):
    DEFAULT_MSG_TYPE = 'vision_msgs/Detection2D'
    DEFAULT_DATA = {
        '_timestamp_ns': None,
        'results': [],
        'bbox': {},
    }

    @property
    def bbox(self):
        return BoundingBox2D(data=self.data['bbox'], base_path=self.base_path)

    @bbox.setter
    def bbox(self, bbox2d):
        assert(isinstance(bbox2d, BoundingBox2D))
        self.data['bbox'] = bbox2d.data

    @property
    def results(self):
        results = self.data['results']
        class_scores = dict([(r['id'], r['score']) for r in results])
        return class_scores

    @results.setter
    def results(self, class_scores):
        results = [{'id': id, 'score': score} for id, score in class_scores.items()]
        self.data['results'] = results


class Detection2DArray (GenericMessage):
    DEFAULT_MSG_TYPE = 'vision_msgs/Detection2DArray'
    DEFAULT_DATA = {
        '_timestamp_ns': None,
        'header': DEFAULT_HEADER_DATA,
        'detections': [],
    }

    def __len__(self):
        return len(self.data['detections'])

    def __getitem__(self, index):
        det_data = self.data['detections'][index]
        return Detection2D(data=det_data, base_path=self.base_path)

    def __setitem__(self, index, detection2d):
        assert(isinstance(detection2d, Detection2D))
        self.data['detections'][index] = detection2d.data

    def append(self, detection2d):
        assert(isinstance(detection2d, Detection2D))
        self.data['detections'].append(detection2d.data)


# Vision messages:
# - BoundingBox3D
# - Detection3D
# - Detection3DArray

class BoundingBox3D (GenericMessage):
    DEFAULT_MSG_TYPE = 'vision_msgs/BoundingBox3D'
    DEFAULT_DATA = {
        '_timestamp_ns': None,
        'center': {
            'position': {'x': 0., 'y': 0., 'z': 0.},
            'orientation': {'x': 0., 'y': 0., 'z': 0., 'w': 1.},
        },
        'size': {'x': 0., 'y': 0., 'z': 0.},
    }

    @property
    def center(self):
        pos = self.data['center']['position']
        return pos['x'], pos['y'], pos['z']

    @center.setter
    def center(self, xyz):
        x, y, z = xyz
        self.data['center']['position'] = {'x': x, 'y': y, 'z': z}

    @property
    def orientation(self):
        quat = self.data['center']['orientation']
        return quat['x'], quat['y'], quat['z'], quat['w']

    @orientation.setter
    def orientation(self, xyzw):
        x, y, z, w = xyzw
        self.data['center']['orientation'] = {'x': x, 'y': y, 'z': z, 'w': w}

    @property
    def size(self):
        sz = self.data['size']
        return sz['x'], sz['y'], sz['z']

    @size.setter
    def size(self, xyz):
        x, y, z = xyz
        self.data['size'] = {'x': x, 'y': y, 'z': z}

class Detection3D (GenericMessage):
    DEFAULT_MSG_TYPE = 'vision_msgs/Detection3D'
    DEFAULT_DATA = {
        '_timestamp_ns': None,
        'results': [],
        'bbox': {},
    }

    @property
    def bbox(self):
        return BoundingBox3D(data=self.data['bbox'], base_path=self.base_path)

    @bbox.setter
    def bbox(self, bbox3d):
        assert(isinstance(bbox3d, BoundingBox3D))
        self.data['bbox'] = bbox3d.data

    @property
    def results(self):
        results = self.data['results']
        class_scores = dict([(r['id'], r['score']) for r in results])
        return class_scores

    @results.setter
    def results(self, class_scores):
        results = [{'id': id, 'score': score} for id, score in class_scores.items()]
        self.data['results'] = results


class Detection3DArray (GenericMessage):
    DEFAULT_MSG_TYPE = 'vision_msgs/Detection3DArray'
    DEFAULT_DATA = {
        '_timestamp_ns': None,
        'header': DEFAULT_HEADER_DATA,
        'detections': [],
    }

    def __iter__(self):
        return (Detection3D(data=data, base_path=self.base_path) for data in self.data['detections'])

    def __len__(self):
        return len(self.data['detections'])

    def __getitem__(self, index):
        det_data = self.data['detections'][index]
        return Detection3D(data=det_data, base_path=self.base_path)

    def __setitem__(self, index, detection3d):
        assert(isinstance(detection3d, Detection3D))
        self.data['detections'][index] = detection3d.data

    def append(self, detection3d):
        assert(isinstance(detection3d, Detection3D))
        self.data['detections'].append(detection3d.data)


# ---------------------------------------
# Convert ROS message to Portable message
# ---------------------------------------

MESSAGE_CLASS_GENERIC = GenericMessage
MESSAGE_CLASSES = {
    'sensor_msgs/Image': ImageMessage,
    'sensor_msgs/CompressedImage': ImageMessage,
    'stereo_msgs/DisparityImage': DisparityImageMessage,
    'sensor_msgs/PointCloud2': PointCloud2Message,
    'msgs_radar/RadarScan': PointCloud2Message,
    'msgs_radar/RadarScanExtended': PointCloud2Message,
    'sensor_msgs/CameraInfo': CameraInfoMessage,
    'tf2_msgs/TFMessage': TFMessage,
    'annotation_message': GenericDictMessage,
    'vision_msgs/BoundingBox2D': BoundingBox2D,
    'vision_msgs/Detection2D': Detection2D,
    'vision_msgs/Detection2DArray': Detection2DArray,
    'vision_msgs/BoundingBox3D': BoundingBox3D,
    'vision_msgs/Detection3D': Detection3D,
    'vision_msgs/Detection3DArray': Detection3DArray,
}

def get_msg_class(msg_type, data=None, base_path=None):
    MsgClass = MESSAGE_CLASSES.get(msg_type, MESSAGE_CLASS_GENERIC)
    return MsgClass

