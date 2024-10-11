# %%
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score
from collections import Counter
from imblearn.over_sampling import SMOTE
from sklearn.svm import SVC
import time 

# %% [markdown]
# # Problem 1: Digit Recognizer

# %%
# Loading the dataset
dataset = np.load("dataset.npz")
X, y = dataset["X"], dataset["y"]

# %% [markdown]
# ## Data exploration

# %%
# Print a random sample to visualise it
print(y[5])

# %%
# Here we can see clearly that it is an 8
plt.imshow(X[5].reshape(20,20), vmin=0, vmax=255.0, cmap="gray")
plt.show()

# %%
# Visualizing the distribution of classes in the dataset

label_counts = Counter(y)
labels = list(label_counts.keys())
counts = list(label_counts.values())
sorted_labels_counts = sorted(zip(labels, counts), key=lambda x: x[0])
sorted_labels, sorted_counts = zip(*sorted_labels_counts)


plt.figure(figsize=(10, 6))
plt.bar(sorted_labels, sorted_counts, color='blue')
plt.xlabel('Labels')
plt.ylabel('Number of Samples')
plt.title('Distribution of Labels in Dataset')
plt.xticks(sorted_labels)  
plt.show()

print("Highest count of classes: ", max(label_counts.values()))
print("Lowest count of classes: ", min(label_counts.values()))


# %% [markdown]
# Above we see that the classes has very different amount of samples. Class 14 has only 74, and class 3 has as as much as 950. This means that there is a imbalance in the dataset that we need to account for. Our approach will be to oversample our data so that we remove the imbalance.
# 
# The reason we chose to oversample is because this improves performance on minority classes.
# We know that oversampling can lead to overfitting since we have more of the "same" data. Also with oversampling it increases computational cost for us to train and tune since it has more data to go through. But we think it is worth it, and we are choosing to use k-fold cross validation when we are tuning to account for overfitting risks.

# %% [markdown]
# ## Data pre-processing

# %%
# For reproducability we set a seed
seed = 0
np.random.seed(seed)

# %%
#Split the data into test and (training and validation)
X_train_val, X_test, y_train_val, y_test = train_test_split(X,y, test_size=0.3, random_state=seed, shuffle=True, stratify=y)
# We use stratify so that the distribution is equal in the test and the training + validation sets.


# We normalize so that whatever classifier we choose all features will contribute equally. (This is not needed for random forest, but it is for others.)
# Normalize the data to range from 0 to 1
X_train_val = X_train_val / 255.0
X_test = X_test / 255.0

# Oversample our data with Smote
smote = SMOTE(random_state=seed)
X_train_val, y_train_val = smote.fit_resample(X_train_val, y_train_val) 
# We only oversample the training and validation! Why? 
# Because if we oversampled the test-data we would make the test data unrealistic and not represntative of real world data.

# %%
# Visualizing the distribution of classes in the dataset now after oversampling
label_counts = Counter(y_train_val)
labels = list(label_counts.keys())
counts = list(label_counts.values())
sorted_labels_counts = sorted(zip(labels, counts), key=lambda x: x[0])
sorted_labels, sorted_counts = zip(*sorted_labels_counts)


plt.figure(figsize=(10, 6))
plt.bar(sorted_labels, sorted_counts, color='blue')
plt.xlabel('Labels')
plt.ylabel('Number of Samples')
plt.title('Distribution of Labels in Dataset')
plt.xticks(sorted_labels)  
plt.show()

print("Highest count of classes: ", max(label_counts.values()))
print("Lowest count of classes: ", min(label_counts.values()))

# %% [markdown]
# After oversampling we see that the distribution has no imbalance.

# %% [markdown]
# # Random Forest classifier

# %%
rf = RandomForestClassifier(random_state=seed)

# Defining the hyperparameters we want to tune:
param_grid = {
    'n_estimators': [10,15,20,40,100],  # Number of trees in the forest
    'max_depth': [None, 10,15,20,30],  # Maximum depth of the trees
    'max_features': ['sqrt','log2'], # Number of features to consider for best split
    'criterion': ['entropy','gini'] # How to compute the information gain when splitting
}   

# param_grid = {
#     'n_estimators': [100],  # Number of trees in the forest
#     'max_depth': [20],  # Maximum depth of the trees
#     'max_features': ['sqrt'], # Number of features to consider for best split
#     'criterion': ['entropy'] # How to compute the information gain when splitting
# }   


grid_search_rf = GridSearchCV(estimator=rf, param_grid=param_grid, 
                           scoring='accuracy', 
                           cv=5, n_jobs=-1, verbose=0)
# Here we can use accruacy when tuning since we removed the imbalance

grid_search_rf.fit(X_train_val, y_train_val)

print("Best Hyperparameters:", grid_search_rf.best_params_)
print("Best Cross-Validation accuracy:", grid_search_rf.best_score_)

# %% [markdown]
# # SVM
# 

# %%
svm = SVC(random_state = seed)

param_grid = {
    'C': [0.1, 1, 10, 100],  # Regularization parameter
    'kernel': ['linear', 'rbf', 'poly'],  # Kernel type to be used in the algorithm
    'gamma': ['scale', 'auto'],  # Kernel coefficient
    'degree': [2, 3, 4]  # Degree of the polynomial kernel function ('poly')
}

# param_grid = {
#     'C': [10],  # Regularization parameter
#     'kernel': ['rbf'],  # Kernel type to be used in the algorithm
#     'gamma': ['scale'],  # Kernel coefficient
#     'degree': [2]  # Degree of the polynomial kernel function ('poly')
# }

grid_search_SVM = GridSearchCV(estimator=svm, param_grid=param_grid, 
                           scoring='accuracy',          
                           cv=5, n_jobs=-1, verbose=0)
# Here we can use accruacy when tuning since we removed the imbalance


grid_search_SVM.fit(X_train_val, y_train_val)

print("Best Hyperparameters:", grid_search_SVM.best_params_)
print("Best Cross-Validation accuracy:", grid_search_SVM.best_score_)

# %% [markdown]
# ### Choosing the best classifier:
# Here our code chooses the classifier that had the best cross-validation accuracy.

# %%
# Evaluate the best model on the test set
if grid_search_SVM.best_score_ >= grid_search_rf.best_score_:
    best_model = grid_search_SVM.best_estimator_ # Get the SVM with the best parameters
    print("SVM: ")
else:
    best_model = grid_search_rf.best_estimator_
    print("Random forest: ")

# Here we use classification report so that we can see how well the classifier performs on each class
print(classification_report(y_test, best_model.predict(X_test)))

# %% [markdown]
# ### Looking at the confusion matrix:

# %%
import numpy as np

class_names = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'A', 'B', 'C', 'D', 'E', 'F', 'Empty']

cm = confusion_matrix(y_test, best_model.predict(X_test))
# Normalize the confusion matrix by dividing each value by the sum of its respective row
cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

# Plot confusion matrix using matplotlib with normalized values for colors
plt.figure(figsize=(8, 6))
plt.imshow(cm_normalized, interpolation='nearest', cmap=plt.cm.Reds)
plt.title('Confusion Matrix (Normalized by Row)')
plt.colorbar()

tick_marks = range(len(class_names))
plt.xticks(tick_marks, class_names, rotation=45)
plt.yticks(tick_marks, class_names)

# Annotating the matrix with numbers (showing the actual counts, not normalized values)
for i in range(len(cm)):
    for j in range(len(cm[0])):
        plt.text(j, i, format(cm[i, j], 'd'),
                 horizontalalignment="center",
                 color="white" if cm_normalized[i, j] > cm_normalized.max() / 2. else "black")

plt.tight_layout()
plt.ylabel('True label')
plt.xlabel('Predicted label')
plt.show()

# %% [markdown]
# ### Looking at images that the classifier gets wrong:

# %%
# Get predictions from the best model
y_pred = best_model.predict(X_test)

# Find indices where the predictions and true labels don't match
misclassified_indices = np.where(y_test != y_pred)[0]

# Sort the misclassified indices by the true class
misclassified_indices_sorted = misclassified_indices[np.argsort(y_test[misclassified_indices])]

# Define the number of images per row
images_per_row = 15

# Calculate the total number of misclassified images
total_misclassified = len(misclassified_indices_sorted)

# Calculate the number of rows needed
num_rows = (total_misclassified + images_per_row - 1) // images_per_row

# Plot misclassified images, sorted by true class
plt.figure(figsize=(images_per_row * 2, num_rows * 2))

for i, index in enumerate(misclassified_indices_sorted):
    plt.subplot(num_rows, images_per_row, i + 1)
    plt.imshow(X_test[index].reshape(20, 20), cmap='gray')  # Assuming images are 20x20
    plt.title(f"True: {class_names[y_test[index]]}, Pred: {class_names[y_pred[index]]}")
    plt.axis('off')

plt.tight_layout()
plt.show()


# %% [markdown]
# ### Precision-recall curve for each class:

# %%
from sklearn.metrics import precision_recall_curve
from sklearn.preprocessing import label_binarize

if "SVC" in  (str(best_model)):
    print("Best model SVM")
    best_model = SVC(probability=True,C=grid_search_SVM.best_params_.get("C"),kernel=grid_search_SVM.best_params_.get("kernel"),gamma=grid_search_SVM.best_params_.get("gamma"),degree= grid_search_SVM.best_params_.get("degree"))
    # Needed to add probaility=True if the best model is SVM because it is needed for the plot.
else:
    print("Best model RF")
    



best_model.fit(X_train_val, y_train_val)
param_grid = {
    'C': [0.1],  # Regularization parameter
    'kernel': ['linear'],  # Kernel type to be used in the algorithm
    'gamma': ['scale'],  # Kernel coefficient
    'degree': [2]  # Degree of the polynomial kernel function ('poly')
}
# Binarize the labels for each class (necessary for multi-class precision-recall curve)
y_test_bin = label_binarize(y_test, classes=range(len(class_names)))
y_pred_proba = best_model.predict_proba(X_test)

plt.figure(figsize=(10, 8))

# Plot Precision-Recall curve for each class
for i, class_name in enumerate(class_names):
    precision, recall, _ = precision_recall_curve(y_test_bin[:, i], y_pred_proba[:, i])
    plt.plot(recall, precision, label=f'Class {class_name}')

plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve for Each Class')
plt.legend(loc='best')
plt.show()


# %% [markdown]
# # Problem 2:  Dimensionality Reduction

# %%
from sklearn.decomposition import PCA
pca = PCA(random_state=seed).fit(X_train_val)
plt.plot(np.cumsum(pca.explained_variance_ratio_))
plt.xlabel('Number of Components')
plt.ylabel('Cumulative Explained Variance')
plt.show()

# %%
from sklearn.decomposition import PCA

# List of n_components values to test
n_components_list = [10, 20, 50, 100, 200]

# Define a structure to store the results
results = []


if "SVC" in  (str(best_model)):
    print("Best model SVM")
    # Define the hyperparameters to tune
    param_grid = {
        'C': [1, 10, 100],  # Regularization parameter
        'kernel': ['rbf', 'poly'],  # Kernel type to be used in the algorithm
        'gamma': ['scale', 'auto'],  # Kernel coefficient
        'degree': [2,3]  # Degree of the polynomial kernel function ('poly')
    }
    # Here we tune less parameters because we don't want it to take too long of time
    model = SVC(random_state=seed)
else:
    print("Best model RF")
    param_grid = {
    'n_estimators': [10,20,40],  # Number of trees in the forest
    'max_depth': [10,15,20],  # Maximum depth of the trees
    'max_features': ['sqrt','log2'], # Number of features to consider for best split
    'criterion': ['entropy','gini'] # How to compute the information gain when splitting
    }
    model = RandomForestClassifier(random_state=seed)



for dimensions in n_components_list:
    start_time = time.time()

    # Perform PCA
    pca = PCA(n_components=dimensions, random_state=seed)
    X_train_val_pca = pca.fit_transform(X_train_val)

    # Calculate total variance explained
    total_variance_explained = pca.explained_variance_ratio_.sum()
    

    # Set up GridSearchCV with SVM
    grid_search = GridSearchCV(estimator=model, param_grid=param_grid, 
                               scoring='accuracy', 
                               cv=5, n_jobs=-1, verbose=0)
    
    # Fit the grid search to the data
    grid_search.fit(X_train_val_pca, y_train_val)
    
    # Get the best hyperparameters and cross-validation accuracy
    best_hyperparams = grid_search.best_params_
    best_cv_accuracy = grid_search.best_score_

    end_time = time.time()
    execution_time = end_time - start_time
    
    # Store the results in the results list
    results.append({
        'n_components': dimensions,
        'total_variance_explained': total_variance_explained,
        'best_hyperparameters': best_hyperparams,
        'best_cv_accuracy': best_cv_accuracy,
        'execution_time': execution_time
    })

# %% [markdown]
# ## Results when PCA is set to n_components = 10:

# %%
for key, val in results[0].items():
        print(f"{key}: {val}")    

# %% [markdown]
# ## Results when PCA is set to n_components = 20:

# %%
for key, val in results[1].items():
        print(f"{key}: {val}")   

# %% [markdown]
# ## Results when PCA is set to n_components = 50:

# %%
for key, val in results[2].items():
        print(f"{key}: {val}")  

# %% [markdown]
# ## Results when PCA is set to n_components = 100:

# %%
for key, val in results[3].items():
        print(f"{key}: {val}")  

# %% [markdown]
# ## Results when PCA is set to n_components = 200:

# %%
for key, val in results[4].items():
        print(f"{key}: {val}")  

# %% [markdown]
# ### Checking which n_components was the best:
# 
# The best result was with n_components set to 100 with cross-validation accuracy of 0.9737284387439187.
# 
# And this was with these hyperparameters: {'C': 100, 'degree': 3, 'gamma': 'scale', 'kernel': 'poly'}.
# 
# This had a execution time of 160 seconds compared to ... # TODO
# 
# Testing this on testdata:

# %%
pca = PCA(n_components=100, random_state=seed)
X_train_val_pca = pca.fit_transform(X_train_val)
X_test_pca = pca.transform(X_test)

svm = SVC(random_state = seed,C=100, degree=3, gamma='scale', kernel='poly') #SVM since this was the best in problem 1

svm.fit(X_train_val_pca,y_train_val)

print(classification_report(y_test, svm.predict(X_test_pca)))

# %% [markdown]
# ## Looking at the confusion matrix for this:

# %%
import numpy as np

class_names = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'A', 'B', 'C', 'D', 'E', 'F', 'Empty']

cm = confusion_matrix(y_test, svm.predict(X_test_pca))
# Normalize the confusion matrix by dividing each value by the sum of its respective row
cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

# Plot confusion matrix using matplotlib with normalized values for colors
plt.figure(figsize=(8, 6))
plt.imshow(cm_normalized, interpolation='nearest', cmap=plt.cm.Reds)
plt.title('Confusion Matrix (Normalized by Row)')
plt.colorbar()

tick_marks = range(len(class_names))
plt.xticks(tick_marks, class_names, rotation=45)
plt.yticks(tick_marks, class_names)

# Annotating the matrix with numbers (showing the actual counts, not normalized values)
for i in range(len(cm)):
    for j in range(len(cm[0])):
        plt.text(j, i, format(cm[i, j], 'd'),
                 horizontalalignment="center",
                 color="white" if cm_normalized[i, j] > cm_normalized.max() / 2. else "black")

plt.tight_layout()
plt.ylabel('True label')
plt.xlabel('Predicted label')
plt.show()

# %% [markdown]
# # Problem 3: Detecting Out-of-Distribution Images
# 

# %% [markdown]
# Different approaches to find the out of dist images:
# 
# 
# Approach 1: Based on the smallest difference between the top two probabilities.
# Approach 2: Based on entropy, which measures the overall uncertainty of the probability distribution.
# 
# 
# We counted which of these approaches had the most out of distribution images in them and went with that one.
# 
# If we could find all of the images we would count 85, this way we can kinda find a accuracy for finding these!

# %%
# Loading the dataset
dataset = np.load("corrupt_dataset.npz")
X = dataset["X"]

# Normalise the data
X = X / 255.0

x = 109

# Define a threshold
threshold = 0.4

# # Apply threshold to binarize the image data
# X = (X >= threshold).astype(int)

# X_train_val = (X_train_val >= threshold).astype(int)

# Visualize the binarized image for sample x
plt.imshow(X[x].reshape(20, 20), vmin=0, vmax=1, cmap="gray")
plt.show()

# %%
# I want to use SVM with PCA set to 100, because this gave me the best results:
# svm = SVC(random_state = seed, probability=True, C=10,  degree=3 , gamma='scale', kernel='poly' )

svm = SVC(probability=True, random_state=seed, C=100, degree=3, gamma='scale', kernel='poly')

pca = PCA(n_components=100, random_state=seed)
X_train_val_pca = pca.fit_transform(X_train_val)
X_corrupt_pca = pca.transform(X)

svm.fit(X_train_val_pca, y_train_val)

# X_train_val = (X_train_val >= threshold).astype(int)


predicts = svm.predict_proba(X_corrupt_pca)

# best_hyperparameters: {'C': 10, 'degree': 2, 'gamma': 'scale', 'kernel': 'rbf'}

# plt.imshow(X[x].reshape(20,20), vmin=0, vmax=1, cmap="gray")
# plt.show()

# X_test = (X_test >= threshold).astype(int)

# print(classification_report(y_test, svm.predict(X_test)))

# %% [markdown]
# ### Showing all of the pictures:

# %%
num_classes = predicts.shape[1]  # Number of classes
images_per_class = 55 # Number of images to show per class
image_size = (20, 20)  # Assuming 20x20 images as in your example

fig, axes = plt.subplots(num_classes, images_per_class, figsize=(images_per_class * 2, num_classes * 2))

# Step 3: Loop over each class and find the most confident images
for class_idx in range(num_classes):
    # Sort the images by probability for the current class
    most_confident_indices = np.argsort(-predicts[:, class_idx])[:images_per_class]
    
    for img_idx, ax in enumerate(axes[class_idx]):
        image = X[most_confident_indices[img_idx]].reshape(image_size)
        ax.imshow(image, vmin=0, vmax=1, cmap="gray")
        ax.axis('off')  # Turn off axis labels

    # Add a class label on the left
    axes[class_idx, 0].set_ylabel(f'Class {class_idx}', fontsize=12, labelpad=10)

# Show the figure
plt.tight_layout()
plt.show()


# %% [markdown]
# ### Above we can see all of the images in the dataset, we see that a lot of them look like clothes
# 

# %% [markdown]
# # Finding the out of distribution images:

# %%
# predicts_proba = best_svm.predict_proba(X)
predicts_proba = svm.predict_proba(X_corrupt_pca)

# %% [markdown]
# ### Results for calculating the difference between the highest and second-highest probabilities:
# 39/85!
# 

# %%

# Step 1: Calculate the difference between the highest and second-highest probabilities
sorted_probs = np.sort(predicts_proba, axis=1)  # Sort the probabilities for each image
prob_diffs = sorted_probs[:, -1] - sorted_probs[:, -2]  # Difference between the two largest probabilities

# Step 2: Sort by uncertainty (smallest difference = most uncertain)
most_uncertain_indices = np.argsort(prob_diffs)  # Sorted from smallest to largest difference

# Step 3: Plot the most uncertain images, from highest uncertainty to lowest
num_uncertain = 85  # Number of most uncertain images to display
images_per_row = 10  # Number of images per row in the plot
num_rows = (num_uncertain + images_per_row - 1) // images_per_row  # Calculate number of rows

fig, axes = plt.subplots(num_rows, images_per_row, figsize=(images_per_row * 2, num_rows * 2))

for i, ax in enumerate(axes.flat):
    if i < num_uncertain:
        image_idx = most_uncertain_indices[i]  # Start with the most uncertain
        image = X[image_idx].reshape(20, 20)  # Assuming the images are 20x20
        ax.imshow(image, vmin=0, vmax=1, cmap="gray")
        ax.set_title(f"Prob diff: {prob_diffs[image_idx]:.2f}")
    ax.set_xticks([])  # Remove x-axis ticks
    ax.set_yticks([])  # Remove y-axis ticks

plt.tight_layout()
plt.show()


# %% [markdown]
# ### Result for calculating the entropy for each image:
# 43/85

# %%
from scipy.stats import entropy

# Step 1: Calculate entropy for each image
entropies = np.apply_along_axis(entropy, 1, predicts_proba)  # Apply entropy along axis 1

# Step 2: Sort by entropy (highest entropy = most uncertain)
most_uncertain_indices = np.argsort(entropies)[::-1]  # Sort in descending order of entropy

# Step 3: Plot the most uncertain images based on entropy
num_uncertain = 85 # Number of most uncertain images to display
images_per_row = 10  # Number of images per row in the plot
num_rows = (num_uncertain + images_per_row - 1) // images_per_row  # Calculate number of rows

fig, axes = plt.subplots(num_rows, images_per_row, figsize=(images_per_row * 2, num_rows * 2))

for i, ax in enumerate(axes.flat):
    if i < num_uncertain:
        image_idx = most_uncertain_indices[i]
        image = X[image_idx].reshape(20, 20)  # Assuming the images are 20x20
        ax.imshow(image, vmin=0, vmax=1, cmap="gray")
        ax.set_title(f"Entropy: {entropies[image_idx]:.2f}")
    ax.set_xticks([])  # Remove x-axis ticks
    ax.set_yticks([])  # Remove y-axis ticks

plt.tight_layout()
plt.show()


# %% [markdown]
# ### Result for combining both approaches for each image:
# ??

# %%
from scipy.stats import entropy

# Step 1: Calculate entropy for each image
entropies = np.apply_along_axis(entropy, 1, predicts_proba)

# Step 2: Combine probability differences and entropy
threshold_diff = 0.1  # You can tune this
threshold_entropy = 0.4  # You can tune this

# Flag images with small probability difference or high entropy
flagged_indices = np.where((prob_diffs < threshold_diff) | (entropies > threshold_entropy))[0]

# Step 3: Plot flagged images
num_flagged = len(flagged_indices)
fig, axes = plt.subplots((num_flagged + 9) // 10, 10, figsize=(20, 2 * (num_flagged + 9) // 10))

for i, ax in enumerate(axes.flat):
    if i < num_flagged:
        image_idx = flagged_indices[i]
        image = X[image_idx].reshape(20, 20)
        ax.imshow(image, cmap='gray', vmin=0, vmax=1)
        ax.set_title(f"Entropy: {entropies[image_idx]:.2f}\nDiff: {prob_diffs[image_idx]:.2f}")
    ax.axis('off')

plt.tight_layout()
plt.show()


# %% [markdown]
# # Experimenting
# 53/85

# %%
from scipy.stats import entropy
import numpy as np
import matplotlib.pyplot as plt

# Assuming you have the necessary data such as predicts_proba, prob_diffs, and X (the image data)

# Step 1: Calculate entropy for each image
entropies = np.apply_along_axis(entropy, 1, predicts_proba)

# Step 2: Combine probability differences and entropy
threshold_diff = 0.1  # You can tune this
threshold_entropy = 0.4  # You can tune this

# Flag images with small probability difference or high entropy
flagged_indices = np.where((prob_diffs < threshold_diff) | (entropies > threshold_entropy))[0]

# Step 3: Sort flagged images by the difference between entropy and probability difference
entropy_diff = entropies[flagged_indices] - prob_diffs[flagged_indices]
sorted_indices = np.argsort(entropy_diff)[::-1]  # Sort in descending order

# Step 4: Limit to the first 85 flagged images
num_to_display = min(85, len(sorted_indices))  # Display up to 85 images

# Step 5: Plot the first 85 flagged and sorted images
fig, axes = plt.subplots((num_to_display + 9) // 10, 10, figsize=(20, 2 * (num_to_display + 9) // 10))

for i, ax in enumerate(axes.flat):
    if i < num_to_display:
        image_idx = flagged_indices[sorted_indices[i]]
        image = X[image_idx].reshape(20, 20)
        ax.imshow(image, cmap='gray', vmin=0, vmax=1)
        ax.set_title(f"Entropy: {entropies[image_idx]:.2f}\nDiff: {prob_diffs[image_idx]:.2f}")
    ax.axis('off')

plt.tight_layout()
plt.show()



