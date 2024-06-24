#Assignment 2

##Instructions to Install dependencies

# Assignment 2

Link to software - https://github.com/climberwb/ML/tree/main/A2
Link to Overleaf - https://www.overleaf.com/read/vdmnvrtgqwvp#b7cc03

1. git clone git@github.com:climberwb/ML.git
2. cd A2
3. git clone https://github.com/jlm429/pyperch.git            (NOTE: make sure you are in your-path/ML/A2 when doing this)
4. git clone https://github.com/hiive/mlrose/                 (NOTE: make sure you are in your-path/ML/A2 when doing this)
5. conda env create -f environment.yml
6. run (jupyter notebook) either command line or vscode 
-----------------------------------------------------------------------------------------------
## Instructions to run the code

part1: 

1. Gather data - run notebooks in:
	* ML/A2/A2P1DataCollection_4peaks.ipynb
	* ML/A2/A2P1DataCollection_maxkcolor.ipynb

	These scripts run Runners across many seeds for the 2 problems above and store them in csv files

2. Gather plots - run notebooks in:
	* ML/A2/A2P1Graphs_4peaks.ipynb 
	* ML/A2/A2P1Graphs_maxkcolor.ipynb

part2 NN:

1. Gather plots  + Data 
	* ML/A2/A2NNPart2.ipynb 

   
WARNING: If running this on a Linux machine there might be CLRF line endings from windows that were added. If you run into compatibility issues try converting the files being run into LF format. 
    
