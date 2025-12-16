# AEs-for-ranking-phasefield-feasibility
Autoencoders for ranking the chemical feasibility of novel phasefields, with respect to the dataset on which the models are trained. 

## Installation Instructions (Windows, Anaconda/Miniconda)

1. **Install Anaconda**
- If Anaconda or Miniconda is not already installed, it can be installed from the university programs or using the [link to installation](https://www.anaconda.com/products/individual).

2. **Open the Anaconda Prompt**
- On windows, search for and open the anaconda prompt window.

3. **Create a virtual environment with the required packages**
- Type or copy the following line into the command prompt to create a virtual environment:
```conda create -n myenv -c anaconda git jupyter pandas numpy scikit-learn matplotlib```
- If required, myenv can be changed to whichever name you prefer for the environment
  
4. **Activate the environment**
- Type or copy the following line into the command prompt to activate the environment:
     '''conda activate myenv'''
- If you changed the name of myenv in **step 3**, use this instead of myenv in the above line.

5. **Install TensorFlow**
- If your workstation has an NVIDIA GPU with CUDA installed, install TensorFlow with the following:
     '''conda install tensorflow'''
- If you only have CPUs, install TensorFlow with this:
     '''conda install tensorflow-cpu'''

6. **Using GitHub**
- Create a GitHub account if needed.
- Ensure you have access to the lrcfmd github repositories

7. **Clone the repository**
- In the command promt, navigate the directory where you wish to copy this repository. Eg.
     '''cd C:\Users\**YOUR USER**\Documents\phasefield_ranking'''
- cd changes directory, dir lists subdirectories within the current directory, mkdir creates a new directory in the directory you are in.
- Clone the repository using the following in the command prompt:
     '''git clone  '''
- You will then need to login to GitHub to complete the download.

8. **Navigate into the working directory**
- Change into the working directory with the following line:
     '''cd  '''

9. **Unzip the vectors**
- Large vectors are required for the ranking, which have been compressed to save memory.
- Use the following compand to expand the files:
     '''Expand-Archive -Path archive.zip -DestinationPath foldername'''

10. **Moving datasets**
- If using your own dataset for the training of the autoencoders, move them into the DATA folder.
- This can be done either using the computers GUI and moving the dataset in Files, or by using the following command:      '''Copy-Item C:\Users\**YOUR USER**\path\to\dataset C:\Users\**YOUR USER**\path\to\*****\DATA''

12. **Launching programme**
- In the ***** folder, launch jupyter notebook with the following:
   '''jupyter notebook'''

13. **Running the programme**
- Follow the instructions within the notebook to run your own ranking.
- It contains an example script, ****, that uses an pre-assembled dataset to perform a ranking. It may be best to try using this first to ensure it is working, and that the inputs are all understood.
    - A blank script can be found under ****. 
