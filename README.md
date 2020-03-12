# CNN patch classifier for segmentation of liver tumors

### Workflow:
1. Combine data using combine_data.py script. Data should be organized in 3 sub-folders:
   - ct_scans
   - liver_seg
   - tumors
2. Generate patches h5 file using the generate_patches.py script
3. Train model using training_script.py
4. Predict on test set ct scans using prediction_script.py
5. Analyze results using analyze_script.py