import rclpy
from rclpy import qos
from rclpy.executors import MultiThreadedExecutor
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup
from rclpy.node import Node
from .topic_service import Publisher, Subscriber, Client, WallTimer

import time
import logging
class RCLPY_Handler:
    def __init__(self, node : str):
        rclpy.init()
        self.node = Node(node)
        self.executor = MultiThreadedExecutor()
        self.executor.add_node(self.node)
        self.connected = False
        self.__create_QoS()
        self.__logger_setup()

    def __create_QoS(self):
        self.qos = qos.QoSProfile(
            reliability=qos.ReliabilityPolicy.BEST_EFFORT, 
            durability=qos.DurabilityPolicy.VOLATILE, 
            depth=1
        )

    def __logger_setup(self):
        date_timestamp = time.strftime('%Y_%m_%d-%H_%M_%S')
        logging.basicConfig(
            level=logging.DEBUG,
            filename=f'logs/logs_autonav_{date_timestamp}',
            format='%(asctime)s %(message)s'
        )
        console = logging.StreamHandler()
        console.setLevel(logging.INFO)
        self.logger = logging.getLogger('autonav')
        self.logger.addHandler(console)

    def log(self, msg : str):
        self.node.get_logger().info(msg)
        self.logger.info(msg)

    def log_error(self, msg : str):
        self.node.get_logger().error(msg)
        self.logger.error(msg)

    def connect(self):
        self.connected = True
        self.log("rclpy connected!")
        rclpy.spin(self.node)

    def disconnect(self):
        if self.connected:
            self.connected = False
            self.log("Shutting down rclpy ...")
            self.node.destroy_node()
            rclpy.shutdown()

    def create_topic_publisher(self, topic: Publisher):
        topic.set_publisher(self.node.create_publisher(topic.get_type(), topic.get_name(), 10))

    def create_timer(self, timer: WallTimer):
        timer = self.node.create_timer(timer.get_period(), timer.run, callback_group=MutuallyExclusiveCallbackGroup())
        
    def publish_topic(self, topic: Publisher, data):
        try:
            topic.get_publisher().publish(data)
        except Exception as e:
            self.log_error("Failed to publish to topic: " + topic.get_name())
            self.log_error(f"ERROR: {e}")

    def create_topic_subscriber(self, topic: Subscriber, function=None):
        callback_group = MutuallyExclusiveCallbackGroup()
        try:
            if function == None:
                function = topic.set_data
            topic.set_subscription(self.node.create_subscription(topic.get_type(), topic.get_name(), function, self.qos, callback_group=callback_group))
        except Exception as e:
            self.log_error("Failed to subscribe to topic: " + topic.get_name())
            self.log_error(f"ERROR: {e}")

    def edit_topic_subscriber(self, topic: Subscriber, function=None):
        if function == None:
            function = topic.set_data
        self.node.destroy_subscription(topic.get_subscription())
        self.create_topic_subscriber(topic, function)

    def create_service_client(self, topic: Client):
        topic.set_client(self.node.create_client(topic.get_type(), topic.get_name()))

    def send_service_request(self, service: Client, data, timeout=30):
        try:
            srv = service.get_name()
            client = service.get_client()

            self.log(f"waiting for ROS service: {srv}")
            client.wait_for_service(timeout_sec=timeout)
            self.log(f"ROS service is up: {srv}")
            call_srv = client.call_async(data)
            return call_srv.result()
        except Exception as e:
            self.log_error("Failed to request service: " + srv)
            self.log_error(f"ERROR: {e}")

    def get_time(self):
        return self.node.get_clock().now().to_msg()