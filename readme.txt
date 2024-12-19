Name: Kirana Irfano
CWID: 10886395
Programming Language: python

Code Structure: 
The code is written in python. It opens a csv file to train the data, preprocesses the data, and performs feature selection. The model is then trained and uses cross validation. Finally, the model is used on the test set and various performance metrics are used. At the very bottom of the code, there is code written so that if users have the right metric values in their own csv file, the program will predict the classification of the data provided by the user using the model.

Instructions:
To run this program, several packages need to be installed into your environment. I am currently using the Python 3.10.11 version, and the following libraries are required to be installed:
- pandas
- numpy
- seaborn
- scikit-learn
- matplotlib

The easiest way to ensure the code will run in your environment, a virtual environment should be used. 
To create a virtual environment in VS Code, type in your terminal: `python -m venv env`
And to activate it (Windows): `.\env\Scripts\activate`
Once the virtual environment has been made and activated, the command `pip install -r requirements.txt` can be run to install all the required packages. 
If there are better ways to install these packages or create a virtual environment please feel free to do so. 

The program currently uses two csv files. the first is the breast-cancer.csv file which should be located in this folder. If the breast-cancer.csv file is not in the same folder, you may need to change the path in the code (line 13).
The second file is used as user input. When prompted, the user can type a csv file with data containing an unknown diagnosis. The user can type in the file name and when they click Enter, the diagnosis is outputted based off the prediction of the model. 
The first line of the csv file should match the user_input.csv example file included. To test if this works, you can use the user_input.csv file that is found in this folder by typing 'user_input.csv' when prompted. (Note: this is different than shown in the demo, in the demo it does not take user input, it just runs straight)

To view the project report, please open "Advanced ML Final Report.pdf".
To view the demo, please open "demo.mp4".