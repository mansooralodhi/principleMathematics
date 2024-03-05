from constantNode import ConstantNodes
from variable import VariableNodes
from operationNode import OperationNode
from utils import node_details
import numpy as np


const2 = ConstantNodes((1,)).create_using(np.asarray([[2.0, 3.0], [3., 4.0]]))
var3 = VariableNodes((1,)).create_using(np.asarray([[3., 4.0], [2.0, 3.0]]))
opMul = OperationNode((1,)).create_using(np.array(6.0), '__mul__', const2, var3)
result = var3 * const2
node_details(result)