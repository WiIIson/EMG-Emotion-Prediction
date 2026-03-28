# EMG Emotion Prediction

Predicting emotional state from EMG data.

Developed by:
- William Conley
- Adrian Fudge
- Oluwateniola "Teni" Adegbite
- Dilan Mian
- Alannis Davis

## Notes From Developing the Model
- Made the inidial model, got ~40% accuracy
- Added more linear layers, as well as dropout and batch normalization, got ~60%
- Replaced RELU activation with GELU, more stable for sensor data. Accuracy did not noticibly change, but had less variation.
- Realized the model was suffering from data imbalance (50% of the data is neutral). Removed neutral entries from the dataset and got ~45%.