### A Cross-Feature Mutual Learning Framework Integrating Multiple Features for Brain Disorder Diagnosis
### **⚡️We released dynamic attention visualization code❗️**

​	 To promote the development of this field, particularly to advance interpretability techniques in neuroimaging, we have decided to open-source our dynamic visualization solution. 

- Input: Attention results from multiple epochs

- Function:

  - K-fold cross validation averages can be calculated and displayed separately, and exported to PDF and PNG for display.

  - Provide the ranking of important brain areas in text form.

  - Provide a custom parameter interface.

  - Three types of diagrams (Here's the demo on the SZ dataset)

    - Average attention heatmap showing the evolution of attention weights across training epochs  and independent components.

    <img src="https://github.com/AbnerAI/CFML/blob/main/Average_Attention_Heatmap.jpg" alt="image-20250710221335851" style="zoom:20%;" />

    - Temporal dynamics of the top 5 most important IC components throughout training epochs. Lines show mean attention weights with shaded areas representing standard deviation across folds, highlighting the stability and convergence patterns of critical brain regions

    <img src="https://github.com/AbnerAI/CFML/blob/main/Top_x_IC_Components.jpg" alt="image-20250710221533847" style="zoom:16%;" />

    - Standard deviation heatmap illustrating the variability of attention weights across different folds

    <img src="https://github.com/AbnerAI/CFML/blob/main/Attention_standard_deviation.jpg" alt="image-20250710221621265" style="zoom:20%;" />

### ⚙️ Setup Environment

##### Hardware Configuration:

- GPU: NVIDIA GeForce RTX     2080Ti (11GB VRAM)
- CUDA: 10.0
- cuDNN: 7.4
- GPU Driver: 410.48

##### Software Environment:

- Python: 3.7
- Operating System: Ubuntu     18.04 LTS
- TensorFlow: 1.13.1 with     Keras 2.2.4

##### Setup Environment

```
conda create -n cfml
conda activate cfml
pip install -r requirements.txt
```

### 👨🏻‍💻Training and Evaluation
We use Keras for training and sklearn.metrics for evaluation. The code execution command is as follows: 
```
python main_tc_fnc.py
```

