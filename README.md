[![Review Assignment Due Date](https://classroom.github.com/assets/deadline-readme-button-22041afd0340ce965d47ae6ef1cefeee28c7c493a6346c4f15d667ab976d596c.svg)](https://classroom.github.com/a/GMtiwQXP)
# Cyber Security Data Analysis
## Project Overview
This dataset contains network information features to determine anomaly and normal behavior.

We will use to predict whether there is anomaly or not. This is a classification task.


## Objectives
- Practice the end-to-end machine learning process on classification problem

## 📝 Instructions
### 0.Setup
- Accept the assignment on GitHub Classroom. (looks like you've already done that)
  - This repository is private; only you and the instructors can see it. Please don't change the visibility of the repository in the settings.
- Clone the repository to your computer.
  - You can use GitHub Desktop, the command line, or VSCode to do that.
  - You can view the ["Cloning a Repository - GitHub Docs](https://docs.github.com/en/repositories/creating-and-managing-repositories/cloning-a-repository?tool=webui) for more information.
- Open the assignment root folder in VSCode.
  - If you get a notification asking you to install the recommended extensions, go ahead and do that.
- using the terminal, install the dependencies for this project:
  - if you're using `pip`, run `pip install -r requirements.txt`
- Open the `python-exercises.ipynb` Notebook in VSCode
- Make sure you have the right kernel selected for the notebook. Use this [guide](https://it4063c.github.io/guides/FAQ/vscode-jupyter) if you need help.

### 1. Complete the exercises
The notebook itself will guide you through the exercises. You'll find instructions regarding each exercise in the notebook.
You'll also find some tips and links to documentations that will help you complete the exercises.

- Make sure you commit often, you'll find this emoji 🚩, signifying suggested spots where you can make a commit of your code.
- Don't forget to push your code to GitHub when you're done.
- If you're doing this assignment, over multiple sessions, make sure you always push your code to GitHub before you close your computer to avoid losing your work.

### 2. Finalize and Submit your Work
- **Once you've completed all the exercises, come back to this file** and complete the reflection section and the self-evaluation section at the end of this file.
- Submit your work by submitting a link to your repository on Canvas.

---------------
## 💭 Reflection and Self Assessment

**I learned:** (repeat as needed)
- How to handle and debug TypeError and AttributeError related to data types and object misnaming.

- The importance of checking DataFrame column types before performing transformations like scaling.

- How to properly use StandardScaler to normalize data for machine learning preparation.

- How function name conflicts can interfere with variable assignments and break the code.

**I struggled with:** (repeat as needed)
- Resolving an error caused by a DataFrame accidentally being overwritten as a function.

- Understanding the exact cause of the error message related to float() and non-numeric columns.

- Identifying and removing non-numeric columns before applying normalization..

**I need the instructor to help me with:** (repeat as needed)
- Best practices to avoid naming conflicts between functions and variables in large notebooks.

- Clarification on when to use normalization versus standardization depending on the type of data and model..

**How long did it take you to complete this assignment? and reflect on that**
[ 2 ] hours.

**If I were to do this assignment again, I would:** (repeat as needed)
- Be more careful with variable naming to avoid overwriting objects.

- Check data types and missing values earlier in the pipeline.

- Document each step of the process in the notebook to reduce confusion during troubleshooting..

**💯 Self Grade:** For this assignment, based on my work and my reflections I should get [ 20 ] out of 20.

--------------------
## 📚 References and Citations
**I used the following links, books, and other resources in my work:** (repeat as needed)
- scikit-learn StandardScaler documentation

- Pandas get_dummies() documentation

- StackOverflow posts related to TypeError: float() argument must be a string or a real number while using StandardScaler .
  
**I received help from the following ppeole:** (repeat as needed)
- None .

---
## Copyrights and License
IT4063C Data Technologies Analytics by [Yahya Gilany](https://yahyagilany.io). is licensed under [CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/)
