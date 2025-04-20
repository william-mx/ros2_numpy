import rclpy
from rclpy.clock import Clock
from rclpy.node import Node
import numpy as np
import cv2
import struct
from cv_bridge import CvBridge
from sensor_msgs.msg import Image, CompressedImage, Imu, PointCloud2, LaserScan, MagneticField, PointField
from nav_msgs.msg import Path
from geometry_msgs.msg import Pose, PoseStamped, Quaternion
from ackermann_msgs.msg import AckermannDriveStamped
from std_msgs.msg import MultiArrayLayout, MultiArrayDimension, Float32MultiArray
from std_msgs.msg import Header
import math

def quaternion_to_yaw(quaternion):
    """
    Converts a ROS Quaternion message to a yaw angle (rotation around Z-axis).

    Parameters
    ----------
    quaternion : geometry_msgs.msg.Quaternion
        The quaternion to convert.

    Returns
    -------
    float:
        The yaw angle in radians.
    """
    t3 = +2.0 * (quaternion.w * quaternion.z + quaternion.x * quaternion.y)
    t4 = +1.0 - 2.0 * (quaternion.y * quaternion.y + quaternion.z * quaternion.z)
    yaw = math.atan2(t3, t4)
    return yaw

def yaw_to_quaternion(yaw_angle):
    """
    Converts a yaw angle (rotation around the Z-axis) to a ROS Quaternion message.

    In the context of driving, this yaw angle represents the vehicle's heading direction.
    A yaw angle of 0 radians corresponds to the vehicle pointing in the +X direction.

    Parameters
    ----------
    yaw_angle : float
        The yaw angle in radians. This angle represents the vehicle's heading
        direction, measured counterclockwise from the +X axis.

    Returns
    -------
    geometry_msgs.msg.Quaternion
        A ROS Quaternion message representing the rotation around the Z-axis
        that corresponds to the given yaw angle. The x and y components of the
        quaternion will be 0.0, as this rotation is solely around the z-axis.
    """
    quaternion = Quaternion()
    quaternion.x = 0.0  # Rotation around Z-axis, so no rotation around X
    quaternion.y = 0.0  # Rotation around Z-axis, so no rotation around Y
    quaternion.z = np.sin(yaw_angle / 2.0)
    quaternion.w = np.cos(yaw_angle / 2.0)
    return quaternion

def get_timestamp_unix(msg):
    try:
        timestamp_ros = msg.header.stamp
        timestamp_unix = timestamp_ros.to_sec()
        return timestamp_unix
    except AttributeError:
        return None
    
# --- Image ---
def image_to_np(msg):
    image = np.frombuffer(msg.data, dtype=np.uint8).reshape(msg.height, msg.width, 3)
    return image, get_timestamp_unix(msg)

def np_to_image(image, timestamp=None):
    bridge = CvBridge()
    ros_image = bridge.cv2_to_imgmsg(image, encoding="bgr8")
    ros_image.header.stamp = timestamp.to_msg() if timestamp else Clock().now().to_msg()
    return ros_image

# --- CompressedImage ---
def compressedimage_to_np(msg):
    image = cv2.imdecode(np.frombuffer(msg.data, np.uint8), cv2.IMREAD_COLOR)
    return image, get_timestamp_unix(msg)

def np_to_compressedimage(image, timestamp=None):
    ros_image = CompressedImage()
    ros_image.header.stamp = timestamp.to_msg() if timestamp else Clock().now().to_msg()
    ros_image.format = "jpeg"
    ros_image.data = np.array(cv2.imencode('.jpg', image)[1]).tobytes()
    return ros_image

# --- MultiArray ---
def multiarray_to_np(msg):
    pass

def np_to_multiarray(array, dtype=np.float32):
    pass

# --- Imu ---
def imu_to_np(msg):
    pass

def np_to_imu(array):
    pass

# --- PointCloud2 ---
def pointcloud_to_np(msg):
    pass

def np_to_pointcloud(points, frame_id='base_link', timestamp=None):
    """
    Create a ROS2 PointCloud2 message from a set of 2D or 3D points.
    """
    points = np.array(points)
    
    if points.shape[1] not in [2, 3]:
        raise ValueError("Points should have 2 or 3 columns representing [x, y] or [x, y, z].")
    
    if points.shape[1] == 2:
        points = np.hstack((points, np.zeros((points.shape[0], 1))))

    pc_msg = PointCloud2()
    pc_msg.header.stamp = timestamp.to_msg() if timestamp is not None else Clock().now().to_msg()
    pc_msg.header.frame_id = frame_id

    pc_msg.fields = [
        PointField(name='x', offset=0, datatype=PointField.FLOAT32, count=1),
        PointField(name='y', offset=4, datatype=PointField.FLOAT32, count=1),
        PointField(name='z', offset=8, datatype=PointField.FLOAT32, count=1)
    ]
    pc_msg.is_bigendian = False
    pc_msg.point_step = 12
    pc_msg.is_dense = True

    point_data = bytearray()
    for point in points:
        point_data.extend(struct.pack('fff', point[0], point[1], point[2]))

    pc_msg.data = bytes(point_data)
    pc_msg.row_step = pc_msg.point_step * points.shape[0]
    pc_msg.height = 1
    pc_msg.width = points.shape[0]

    return pc_msg

# --- LaserScan ---
def scan_to_np(msg):
    pass

def np_to_scan(array, frame_id="laser_frame", angle_min=0.0, angle_max=6.28318530718, angle_increment=0.01, time_increment=0.0, scan_time=0.0, range_min=0.0, range_max=10.0):
    pass


# --- AckermannDriveStamped ---
def from_ackermann(msg):
    speed = msg.drive.speed = speed
    steering_angle = msg.drive.steering_angle
    return speed, steering_angle, get_timestamp_unix(msg)

def to_ackermann(speed, steering_angle, timestamp=None):
    msg = AckermannDriveStamped()
    msg.header.stamp = timestamp.to_msg() if timestamp else Clock().now().to_msg()
    msg.drive.speed = speed
    msg.drive.steering_angle = steering_angle
    return msg

# --- Pose ---
def pose_to_np(msg):
    pass

def np_to_pose(point, yaw_angle, frame_id='base_link', timestamp=None):
    """
    Converts a NumPy array representing a 3D point and a yaw angle to a ROS PoseStamped message.

    The yaw angle represents the orientation of the pose around the Z-axis.

    Parameters
    ----------
    point : numpy.ndarray
        A NumPy array of shape (3,) representing the x, y, and z coordinates of the pose.
    yaw_angle : float
        The yaw angle in radians, representing the rotation around the Z-axis.
    frame_id : str, optional
        The frame ID for the PoseStamped message header. Defaults to 'base_link'.
    timestamp : rclpy.time.Time, optional
        The timestamp for the PoseStamped message header. If None, the current time is used.
        Defaults to None.

    Returns
    -------
    geometry_msgs.msg.PoseStamped
        A ROS PoseStamped message representing the 3D point and orientation.
    """
    pose_stamped = PoseStamped()

    # Handle timestamp
    pose_stamped.header.stamp = timestamp.to_msg() if timestamp else Clock().now().to_msg()

    # Set frame ID
    pose_stamped.header.frame_id = frame_id

    # Set position
    pose_stamped.pose.position.x = float(point[0])
    pose_stamped.pose.position.y = float(point[1])
    pose_stamped.pose.position.z = float(point[2]) if len(point) > 2 else 0.0

    # Set orientation using the helper function
    pose_stamped.pose.orientation = yaw_to_quaternion(yaw_angle)

    return pose_stamped

# --- Path ---
def pose_to_np(msg):
    """
    Converts a ROS Pose message to a NumPy array and extracts the yaw angle.

    Parameters
    ----------
    msg : geometry_msgs.msg.Pose
        The input ROS Pose message.

    Returns
    -------
    tuple (numpy.ndarray, float):
        A tuple containing:
        - A NumPy array of shape (3,) representing the x, y, and z coordinates of the pose.
        - The yaw angle (rotation around the Z-axis) in radians.
    """
    # Extract position
    point = np.array([msg.position.x, msg.position.y, msg.position.z])

    # Extract orientation and convert to yaw angle
    quaternion = msg.orientation
    yaw_angle = quaternion_to_yaw(quaternion)  # Use the helper function

    return point, yaw_angle

def np_to_path(waypoints, frame_id='base_link', timestamp=None):

    # Ensure waypoints is a numpy array.
    waypoints = np.array(waypoints)

    # If a single waypoint is provided, reshape it.
    if waypoints.ndim == 1:
        waypoints = waypoints.reshape(1, -1)

    if waypoints.shape[1] != 3:
        raise ValueError("Waypoints should have 3 columns: x, y, and theta.")

    path_msg = Path()
    path_msg.header.stamp = timestamp.to_msg() if timestamp else Clock().now().to_msg()
    path_msg.header.frame_id = frame_id

    for point in waypoints:
        pose = np_to_pose(point[:2], point[2])
        pose.header.frame_id = frame_id
        path_msg.poses.append(pose)

    return path_msg

# --- MagneticField ---
def magneticfield_to_np(msg):
    pass

def np_to_magneticfield(array, frame_id="sensor_frame"):
    pass

class ConversionPublisherNode(Node):
    def __init__(self):
        super().__init__('conversion_publisher_node')
        self.bridge = CvBridge()
        self.time_source = self.get_clock()  # Use the node's clock
        self.timestamp = self.time_source.now()  # Create a timestamp # rclpy.time.Time
        # bultin_interfaces.msg._time.Time

        # --- Publishers ---
        self.image_pub = self.create_publisher(Image, '/camera/color/image_raw', 10)
        self.compressed_image_pub = self.create_publisher(CompressedImage, '/camera/color/image_jpeg', 10)
        self.multiarray_pub = self.create_publisher(Float32MultiArray, 'multiarray_topic', 10)
        self.pointcloud_pub = self.create_publisher(PointCloud2, '/scan', 10)
        self.ackermann_pub = self.create_publisher(AckermannDriveStamped, '/rc/ackermann_cmd', 10)
        self.pose_pub = self.create_publisher(PoseStamped, 'pose', 10)
        self.path_pub = self.create_publisher(Path, 'path', 10)

        self.timer = self.create_timer(1.0, self.publish_data)  # Publish every second

    def publish_data(self):
        """Publishes example data for each conversion."""
        self.get_logger().info('Publishing data...')

        # --- Image ---
        image_np = np.zeros((100, 100, 3), dtype=np.uint8)  # Example image data
        image_msg = np_to_image(image_np, timestamp=self.timestamp)
        self.image_pub.publish(image_msg)

        # --- CompressedImage ---
        compressed_image_msg = np_to_compressedimage(image_np, timestamp=self.timestamp)
        self.compressed_image_pub.publish(compressed_image_msg)

        # # --- MultiArray ---
        # array_np = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float32)  # Example array
        # multiarray_msg = np_to_multiarray(array_np)
        # self.multiarray_pub.publish(multiarray_msg)

        # --- PointCloud2 ---
        points_np = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float32)  # Example points
        pointcloud_msg = np_to_pointcloud(points_np, frame_id="base_link", timestamp=self.timestamp)
        self.pointcloud_pub.publish(pointcloud_msg)

        # --- AckermannDriveStamped ---
        speed = 1.0
        steering_angle = 0.2
        ackermann_msg = to_ackermann(speed, steering_angle, timestamp=self.timestamp)
        self.ackermann_pub.publish(ackermann_msg)

        # --- Pose ---
        point_np = np.array([1.0, 2.0])
        yaw_angle = np.pi/4
        pose_stamped_msg = np_to_pose(point_np, yaw_angle, frame_id="base_link", timestamp=self.timestamp)
        self.pose_pub.publish(pose_stamped_msg)

        # --- Path ---
        waypoints_np = np.array([[1.0, 1.0, np.pi/6], [2.0, 2.0, np.pi/4], [3.0, 3.0, np.pi/2]])  # Example waypoints
        path_msg = np_to_path(waypoints_np, frame_id="base_link", timestamp=self.timestamp)
        self.path_pub.publish(path_msg)
        self.get_logger().info('Published all messages')


def main(args=None):
    rclpy.init(args=args)
    publisher_node = ConversionPublisherNode()
    rclpy.spin(publisher_node)
    publisher_node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()