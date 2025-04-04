import json
import threading
import subprocess
import numpy as np
from scipy.spatial.transform import Rotation 
from geometry_msgs.msg import PoseStamped
from builtin_interfaces.msg import Time

from ..types import Point

class GzTopicParser:
    def __init__(self, topic: str = "/world/map/model/iris/joint_state"):
        """
        Initializes the GzTopicParser with the specified topic.
        """
        self.topic = topic
        self.position = np.zeros(3)  # Initialize as [0.0, 0.0, 0.0]
        self.quaternion = np.array([0.0, 0.0, 0.0, 1.0])  # Initialize as [0.0, 0.0, 0.0, 1.0]
        self.process = None
        self.thread = None
        self.running = True
        self.thread = threading.Thread(target=self._run_process, daemon=True)
        self.thread.start()

    def _parse_json_output(self, json_line):
        """
        Parses a JSON string for position and orientation.
        Updates the position and quaternion variables.
        """
        try:
            data = json.loads(json_line)
            pose = data.get("pose", {})
            position = pose.get("position", {})
            orientation = pose.get("orientation", {})

            # Update instance variables with NumPy arrays
            self.position = np.array([position.get("x", 0.0), position.get("y", 0.0), position.get("z", 0.0)])
            self.quaternion = np.array([orientation.get("x", 0.0), orientation.get("y", 0.0), orientation.get("z", 0.0), orientation.get("w", 1.0)])
        except json.JSONDecodeError as e:
            print(f"Error decoding JSON: {e}")

    def _run_process(self):
        """
        Runs the gz topic command and continuously reads the output.
        """
        command = ["gz", "topic", "--echo", "--json-output", "--topic", self.topic]
        self.process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

        while self.running:
            line = self.process.stdout.readline()
            if line.strip():
                self._parse_json_output(line)

        self.process.terminate()

    def __del__(self):
        """
        Stops the subprocess and processing thread.
        """
        if self.running:
            self.running = False
            self.thread.join()

    def get_pose(self) -> PoseStamped:
        data = PoseStamped()
        data.header.stamp = Time()
        data.header.frame_id = "map"
        data.pose.position.x = self.position[0]
        data.pose.position.y = self.position[1]
        data.pose.position.z = self.position[2]
        data.pose.orientation.x = self.quaternion[0]
        data.pose.orientation.y = self.quaternion[1]
        data.pose.orientation.z = self.quaternion[2]
        data.pose.orientation.w = self.quaternion[3]
        return data


# Example usage
if __name__ == "__main__":
    parser = GzTopicParser()
    try:
        while True:
            print(f"Pose: {parser.get_pose()}")
    except KeyboardInterrupt:
        print("\nStopping parser.")
        quit()
