# Overview
This project highlights the work of our Senior Capstone Project for the Data Science major at UC San Diego. Under the mentorship of esteemed researchers Yusu Wang and Gal Mishne from the Halicioglu Data Science Institute at UC San Diego, our project delves into the intricate world of graph representations within highway networks, exploring new techniques for traffic speed prediction. Through the process of constructing diverse graphs, integrating a variety of node features and edge types, we aim to explore new ways to represent the dynamics of highway traffic networks. We then utilize two Spatial-Temporal Graph Attention Network (ST-GAT) variants, ST-GAT Single Edge and ST-GAT Edge Type, to predict traffic speeds over the next 15, 30, and 45 minutes from a given time period.

A description of each graph is provided below (more details in report). Feel free to take a look at our experiments and results using the Google Colab link provided for each respective graph!
## Graph Descriptions and Results
| Graph  | Node Features | Edge Types Included | Edge Types Learned | Evaluations |
|---|---|---|---|---|
| ```Graph1_SingleEdge``` | Speed | 1 | Not Learned | [Colab](https://colab.research.google.com/drive/1XK0Dd5cXaE4yifseLFVUiwlk0aZIHcEa?usp=sharing) |
| ```Graph2_SingleEdge``` | Speed | 1,2 | Not Learned | [Colab](https://colab.research.google.com/drive/11eEAzZlGl7gyDrr5nwx7tIy7IJKwgBRp?usp=sharing) |
| ```Graph3_SingleEdge``` | Speed | 1,2,3 | Not Learned | [Colab](https://colab.research.google.com/drive/16CTs787T_riPqUhrr1jTvDhxy6s9Y-mI?usp=sharing) |
| ```Graph4_SingleEdge``` | Speed, Lanes, Day of Week, Hour of Day | 1 | Not Learned | [Colab](https://colab.research.google.com/drive/1KBgXRU87pbs2cG5n8KjQlBuiSYEcOWg-?usp=sharing) |
| ```Graph5_SingleEdge``` | Speed, Lanes, Day of Week, Hour of Day | 1,2 | Not Learned | [Colab](https://colab.research.google.com/drive/10bfLNOjDnF-FO15Q44KmeSHtezl6HAK1?usp=sharing) |
| ```Graph6_SingleEdge``` | Speed, Lanes, Day of Week, Hour of Day | 1,2,3 | Not Learned | [Colab](https://colab.research.google.com/drive/1J7NE5TULHKIYu_wSzyyL0olOkTsAxc4k?usp=sharing) |
| ```Graph1_EdgeType``` | Speed | 1 | Learned | [Colab](https://colab.research.google.com/drive/16akY-0YMSd7yz5kDbTVfqKH4zR8ffEj_?usp=sharing) |
| ```Graph2_EdgeType``` | Speed | 1,2 | Learned | [Colab](https://colab.research.google.com/drive/1vrO3xDQljO8g71jWHr4XXLgiJidjemSs?usp=sharing) |
| ```Graph3_EdgeType``` | Speed | 1,2,3 | Learned | [Colab](https://colab.research.google.com/drive/1Waemv1-oZY3bzFK_qOTaGjkSIEMVxivZ?usp=sharing) |
| ```Graph4_EdgeType``` | Speed, Lanes, Day of Week, Hour of Day | 1 | Learned | [Colab](https://colab.research.google.com/drive/1WLRpkIO02nM4jTMqgvuzROAD6lh8Rftd?usp=sharing) |
| ```Graph5_EdgeType``` | Speed, Lanes, Day of Week, Hour of Day | 1,2 | Learned | [Colab](https://colab.research.google.com/drive/1IjPj5u85C8WgpK7wU38miUu0uof_-yxr?usp=sharing) |
| ```Graph6_EdgeType``` | Speed, Lanes, Day of Week, Hour of Day | 1,2,3 | Learned | [Colab](https://colab.research.google.com/drive/10diC8lZd3s_V-nJ5rccwY-q3fEV3rb2u?usp=sharing) |

# Project Structure
The repository is organized as follow:
```
Traffic-Speed-Prediction/
├── data/
│   ├── sensor_speeds/
│   │   ├── SD_15/
│   │   ├── SD_18/
│   │   └── SD_I805/
│   ├── create_datasets.py
│   ├── non_conn.csv
│   ├── sd.geojson
│   ├── sensor_conn.csv
│   ├── sensor_dist.csv
│   ├── sensor_maps.ipynb
│   ├── sensor_speed.csv
│   ├── vds_info.csv
│   └── vds_info_w_lanes.csv
├── graphs/
│   ├── Graph1_EdgeType.py
│   ├── Graph1_SingleEdge.py
│   ├── Graph2_EdgeType.py
│   ├── Graph2_SingleEdge.py
│   ├── Graph3_EdgeType.py
│   ├── Graph3_SingleEdge.py
│   ├── Graph4_EdgeType.py
│   ├── Graph4_SingleEdge.py
│   ├── Graph5_EdgeType.py
│   ├── Graph5_SingleEdge.py
│   ├── Graph6_EdgeType.py
│   └── Graph6_SingleEdge.py
├── results/
├── README.md
├── requirements.txt
├── run.py
└── showcase_poster.pdf
```

# Usage
Below are the steps for replicating our project on your local machine. We recommend using a GPU for faster training of the models. 
1. Clone this repository on your local machine
2. Open your terminal
3. Change (```cd```) into the directory to the cloned repository
4. Type  ``` pip install -r requirements.txt```. This contains all the necessary packages for running the code.
5. (Optional) Although the datasets are already prepared to be read in, you can create the datasets yourself. To do so, ```cd``` into the data folder. Then type ```python create_datasets.py``` in your terminal. Your datasets are now ready.
6. Return to the home directory of the repository. Use run.py to execute the code. Type ```python run.py {specify graph to evaluate}``` in your terminal. Where it says {specify graph to evaluate}, replace this with one of the graphs (name must be exactly the same as it is in the table containing graph descriptions) and only the specified graph will be evaluated on. If no graph is specified and you just type ```python run.py```, all of the graphs will be evaluated on.

## Requirements
1) Python 3
2) Libraries listed in requirements.txt

# Contributors
1) Ricky Miura
2) Gita Anand
3) Sheena Patel
4) Mentor: Gal Mishne
5) Mentor: Yusu Wang
