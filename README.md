# Datat-Science-Intern-task
### Assessment task for the Data Science Intern position at Tooploox.

### Repository
Repository contains 3 Jupyter notebooks with analysis of Cifar10 images:

1. PART I - Shallow classifier

Here, I train the initial shallow classifier on the visual features extracted using "standard" image processing methods. 

2. PART II - CNN Visualisation

Here, I load and visualise CNN codes (from inception v3 max pool 0:3).

3. PART III - Final learning

I train the final classifier on the CNN codes.

There are also 5 Python classes used during a project:

1. downloadANDload.py

For downloading, extracting and loading Cifar10 images and features.

2. feature_extraction.py

For extracting visual features using image processing methods.

3. plot_functions.py

For useful plotting functions.

additionally, 

4. cifar-10_experiment.py

For setting up CNN extraction.

5. classify_image.py

For running feature extraction from the Inception v3 pretrained model. Official Tensorflow tutorial script.

########################################################################################

### Results

Maximum accuracy achieved on the test data: 0.91
compare it to http://rodrigob.github.io/are_we_there_yet/build/classification_datasets_results.html

#########################################################################################

### Author

AlekSkorupa, Jul, 2017
