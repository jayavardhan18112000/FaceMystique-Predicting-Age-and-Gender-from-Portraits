## Overview
This project aims to predict the gender and approximate age of individuals based on facial images using a dataset containing age and gender labels.

## Dataset
The project utilizes the UTKFace dataset, a collection of grayscale facial images, each annotated with age and gender labels. You can find this dataset [here](https://www.kaggle.com/jangedoo/utkface-new).

## Exploratory Data Analysis (EDA)
In this section, we perform an exploratory data analysis to gain insights into the dataset.

### Basic Statistics
- **DataFrame Shape:** (23407, 3)
- **Data Types:** The DataFrame contains image paths, age labels, and gender labels.
- **Missing Values:** There are no missing values in the dataset.

### Age Distribution
- We visualize the age distribution using a histogram with a kernel density estimate (KDE) overlay, separated by gender.
  - **Result:**The graph shows that the count for male is higher than gender female for ages 0 to 60, while the count for gender female is higher than gender male for ages 60 to 120.  The distribution is skewed to the right, with a longer tail on the right side of the graph.

![Age Distribution by Gender](histplot.png)

### Gender Distribution
- We visualize the gender distribution using a countplot.
  - **Result:** The dataset contains a relatively balanced distribution of gender, with slightly more number of males than females.

![Gender Distribution](countplot.png)

### Age vs. Gender
- We explore the relationship between age and gender using a boxplot.
  - **Result:** The boxplot reveals that the median age for females is slightly lower than for males, indicating a subtle age difference between genders.

![Age vs. Gender](boxplot.png)

### Age Distribution by Gender (Violin Plot)
- We use a violin plot to visualize the age distribution by gender.
  - **Result:** The violin plot provides a detailed view of the age distribution by gender, showing that the majority of individuals in both genders are in their 20s and 30s.

![Age Distribution by Gender (Violin Plot)](violin plot.png)

### Pairplot
- We create a pairplot to visualize relationships between variables, with gender as a hue.
  - **Result:** The pairplot reveals potential correlations between age and gender with respect to other variables in the dataset, providing valuable insights into the data's structure.

![Pairplot by Gender](pairplot.png)

## Requirements
Ensure you have the following dependencies installed:
- Pandas
- Seaborn
- Matplotlib

