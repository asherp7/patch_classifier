# CNN patch classifier for segmentation of liver tumors

### Workflow:
1. Generate patches h5 file using the generate_patches.py script
2. Train model using training_script.py
3. Predict on test set ct scans using prediction_script.py
4. Analyze results using analyze_script.py