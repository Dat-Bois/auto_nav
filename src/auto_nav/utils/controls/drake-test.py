# A simple physics sim setup for a Drake quadrotor model.
# Allows for sending control inputs and visualizing state outputs in the Drake visualizer.
import numpy as np
from pydrake.all import (
    DiagramBuilder,
    Simulator,
    LeafSystem,
    System,
    SceneGraph,
    StartMeshcat,
    MeshcatVisualizer
)
from pydrake.examples import QuadrotorPlant, QuadrotorGeometry, StabilizingLQRController

class StateWithAccelSystem(LeafSystem):
    def __init__(self, plant : System):
        super().__init__()
        self.plant = plant
        self.DeclareVectorInputPort("plant_state", plant.num_continuous_states())
        self.DeclareVectorOutputPort("state_with_accel", 18, self.CalcStateWithAccel)

    def CalcStateWithAccel(self, context, output):
        x = self.get_input_port(0).Eval(context)
        tmp_context = self.plant.CreateDefaultContext()
        tmp_context.SetContinuousState(x)
        xdot = self.plant.AllocateTimeDerivatives()
        self.plant.CalcTimeDerivatives(tmp_context, xdot)
        xd = xdot.CopyToVector()  # shape (12,)
        state18 = np.concatenate([x, xd[6:]])  
        output.SetFromVector(state18)

class DummyController(LeafSystem):
    '''Template controller: 
        Inputs: [x,y,z,r,p,y,vx,vy,vz,rd,pd,yd,ax,ay,az,rdd,pdd,ydd] (state)
        Outputs: [F1, F2, F3, F4] (forces for each rotor?)
        F1: +X
        F2: +Y
        F3: -X
        F4: -Y
        '''

    def __init__(self, n_states: int):
        super().__init__()
        self.DeclareVectorInputPort("state", n_states)
        self.DeclareVectorOutputPort("control", 4, self.CalcControl)

    def CalcControl(self, context, output):
        state = self.get_input_port(0).Eval(context)
        print("Current state:", state)
        output.SetFromVector([2, 2, 2, 2])


def main():
    builder = DiagramBuilder()

    # Plant
    plant = builder.AddSystem(QuadrotorPlant())
    state_with_accel = builder.AddSystem(StateWithAccelSystem(plant)) # by default, plant state doesnt have accel

    # Controller
    ctr = DummyController(18)
    # ctr = StabilizingLQRController(plant, [2,2,1])
    controller = builder.AddSystem(ctr)
    builder.Connect(plant.get_output_port(0), state_with_accel.get_input_port(0))
    builder.Connect(state_with_accel.get_output_port(0), controller.get_input_port(0))
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
    init_state = np.zeros(12)
    init_state[2] = 1.0  # start at z=1
    context.SetContinuousState(init_state)

    input("Meshcat started in browser. Press Enter to simulate...")

    simulator.Initialize()
    simulator.AdvanceTo(0.2)

if __name__ == "__main__":
    main()