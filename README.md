# Overview
Under the mentorship of esteemed researchers Yusu Wang and Gal Mishne from the Halicioglu Data Science Institute at UC San Diego, our project delves into the intricate world of graph representations within highway networks, exploring new techniques for traffic speed prediction. Through the process of constructing diverse graphs, integrating a variety of node features and edge types, we aim to explore new ways to represent the dynamics of highway traffic networks. We then utilize a Spatial-Temporal Graph Attention Network (ST-GAT), training it on these graphs and predicting traffic speed over the enxt 15, 30, and 45 minutes from a given time period.

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
| ```Graph1_EdgeType``` | Speed | 1 | Learned | [Colab](https://colab.research.google.com/drive/1XK0Dd5cXaE4yifseLFVUiwlk0aZIHcEa?usp=sharing) |
| ```Graph2_EdgeType``` | Speed | 1,2 | Learned | [Colab](https://colab.research.google.com/drive/11eEAzZlGl7gyDrr5nwx7tIy7IJKwgBRp?usp=sharing) |
| ```Graph3_EdgeType``` | Speed | 1,2,3 | Learned | [Colab](https://colab.research.google.com/drive/16CTs787T_riPqUhrr1jTvDhxy6s9Y-mI?usp=sharing) |
| ```Graph4_EdgeType``` | Speed, Lanes, Day of Week, Hour of Day | 1 | Learned | [Colab](https://colab.research.google.com/drive/1KBgXRU87pbs2cG5n8KjQlBuiSYEcOWg-?usp=sharing) |
| ```Graph5_EdgeType``` | Speed, Lanes, Day of Week, Hour of Day | 1,2 | Learned | [Colab](https://colab.research.google.com/drive/10bfLNOjDnF-FO15Q44KmeSHtezl6HAK1?usp=sharing) |
| ```Graph6_EdgeType``` | Speed, Lanes, Day of Week, Hour of Day | 1,2,3 | Learned | [Colab](https://colab.research.google.com/drive/1J7NE5TULHKIYu_wSzyyL0olOkTsAxc4k?usp=sharing) |

# Project Structure
The repository is organized as follow:
```
Traffic-Speed-Prediction/
├─ data
  ├─ GATv2.py
├─ GCN.py
├─ GIN.py
├─ README.md
├─ graphGPS.py
├─ requirements.txt
├─ run.py
```
