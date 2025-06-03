from math import atan2, cos, radians, sin, sqrt
import pandas as pd
import networkx as nx
from mis import MISInstance

class DataLoader:
    def distance_from_coordinates(self, a_1 : tuple[float, float],a_2 : tuple[float, float]) -> float:
        # Radius of the Earth in kilometers
        R = 6371.0
        # Convert latitude and longitude from degrees to radians
        lat1, lon1 = radians(a_1[0]), radians(a_1[1])
        lat2, lon2 = radians(a_2[0]), radians(a_2[1])

        # Differences in coordinates
        dlat = lat2 - lat1
        dlon = lon2 - lon1

        # Haversine formula
        a = sin(dlat / 2)**2 + cos(lat1) * cos(lat2) * sin(dlon / 2)**2
        c = 2 * atan2(sqrt(a), sqrt(1 - a))

        # Distance in kilometers
        distance_km = R * c
        return distance_km

    def load_from_csv_coordinates(self, file_path : str):
        df = pd.read_csv(file_path, sep=';')
        self.coordinates_dataset = [(float(c.split(',')[0]),float(c.split(',')[1])) for c in df['coordonnees']]
        
    def build_mis_instance_from_coordinates(self, antenna_range : float, antennas : set[int] = None) -> MISInstance:
        if self.coordinates_dataset is None:
            raise ValueError("Coordinates dataset is not loaded. Please load the dataset using load_from_csv_coordinates method.")
        
        if antennas is None:
            antennas = set(range(len(self.coordinates_dataset)))
        
        graph = nx.Graph()
        for i in antennas:
            graph.add_node(i)
            
        for i in antennas:
            for j in antennas:
                if i < j and self.distance_from_coordinates(self.coordinates_dataset[i], self.coordinates_dataset[j]) <= antenna_range:
                    graph.add_edge(i, j)
                    
        return MISInstance(graph)
