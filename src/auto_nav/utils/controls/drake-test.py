# A simple physics sim setup for a Drake quadrotor model.
# Allows for sending control inputs and visualizing state outputs in the Drake visualizer.
import numpy as np
from pydrake.all import (
    DiagramBuilder,
    Simulator,
    LeafSystem,
    SceneGraph,
    StartMeshcat,
    MeshcatVisualizer
)
from pydrake.examples import QuadrotorPlant, QuadrotorGeometry


class DummyController(LeafSystem):
    """Template controller: outputs [Fz, tau_x, tau_y, tau_z]."""

    def __init__(self, n_states: int):
        super().__init__()
        self.DeclareVectorInputPort("state", n_states)
        self.DeclareVectorOutputPort("control", 4, self.CalcControl)

    def CalcControl(self, context, output):
        state = self.get_input_port(0).Eval(context)
        output.SetFromVector([0.5, 0, 0, 0])


def main():
    builder = DiagramBuilder()

    # Plant
    plant = builder.AddSystem(QuadrotorPlant())

    # Controller
    controller = builder.AddSystem(DummyController(plant.num_continuous_states()))
    builder.Connect(plant.get_output_port(0), controller.get_input_port(0))
    builder.Connect(controller.get_output_port(0), plant.get_input_port(0))

    # SceneGraph and visualization
    scene_graph = builder.AddSystem(SceneGraph())
    QuadrotorGeometry.AddToBuilder(builder, plant.get_output_port(0), scene_graph)

    meshcat = StartMeshcat()
    meshcat.Delete()
    meshcat.ResetRenderMode()
    MeshcatVisualizer.AddToBuilder(builder, scene_graph, meshcat)

    # Build and simulate
    diagram = builder.Build()
    simulator = Simulator(diagram)
    simulator.set_target_realtime_rate(1.0)

    context = simulator.get_mutable_context()
    # initialize state (12-dim vector for QuadrotorPlant)
    init_state = np.zeros(12)
    init_state[2] = 1.0  # start at z=1
    context.SetContinuousState(init_state)

    input("Meshcat started in browser. Press Enter to simulate...")

    simulator.Initialize()
    simulator.AdvanceTo(0.1)


if __name__ == "__main__":
    main()