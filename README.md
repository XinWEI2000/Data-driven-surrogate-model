# Data-driven-surrogate-model
Using the surrogate model to do the prediction of the FEM results of lattice supporting structure in additive manufacturing. I mainly use the MLP and GPR.

Metal additive manufacturing (AM) of complex parts with overhangs typically requires the use of sacrificial support structures to hold the part.[1] This supporting structure can improve the residual stress distribution of the structure. The lattice structure is a novel application for supporting with very low-volume fractions.[1] This study takes a new step to optimize the lattice structure with the surrogate model. By iterating all possible structure schemes, the surrogate model can recommend the structure with the lowest volume while satisfying all thresholds for the indices. The simulation model is built in Ansys to get the Finite Element Analysis (FEA) report. Those data are used in the data-driven analysis with a surrogate model, which includes Gaussian Process Regression (GPR) and Multi-layer perceptron (MLP). Experiment results revealed that the surrogate model can precisely predict the lattice structure model’s feature and the optimization from the surrogate model can satisfy the thresholds for various physical features. The surrogate model can reduce the predicting time compared with the topology optimization. This approach can be a quicker optimizing method for other types of supporting structures.

In the research, the overhang structure is simplified into typical form as shown in Figure 1. The body-center lattice is the only research object for the lattice-supporting structure, as shown in figure 2. The variable for the experiment is the column diameter of each lattice structure. To simplify the model, the available column diameter can only be 2mm or 3mm. By changing the column diameter of each lattice structure, the deformation, elastic strain, and normal stress distribution can be changed to satisfy the threshold of each index. The paper assumes that multiple surrogate models can be trained to predict all indices for different lattice structure schemes precisely. The structure with the least volume can be recommended after iterating all schemes which satisfies all index thresholds.

![image](https://github.com/XinWEI2000/Data-driven-surrogate-model/assets/119705502/d034ce35-ec28-42be-b7a0-e612c0c8eedd)

Figure 1. Typical Overhang Structure

![image](https://github.com/XinWEI2000/Data-driven-surrogate-model/assets/119705502/cbd0d2da-4f10-454c-88dd-e66bb084b230)

Figure 2. Body-center Lattice Structure

In this project, the python scripts MLPSS.py and GPRSS.py are the surrogate model for predicting the FEM results of different lattice layout supporting structure. The dataset used in the script is from FEM analysis in ANSYS.
