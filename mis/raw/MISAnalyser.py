"""This module provides helpers to analyze MIS experiments data by gathering
the relevant features in a dataframe.
"""

from __future__ import annotations

import re
import warnings
from enum import Enum
from typing import Optional, Tuple, Union

import networkx as nx
import numpy as np
import polars as pl
import seaborn as sns
from matplotlib import pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.lines import Line2D
from pulser_simulation.simresults import CoherentResults, NoisyResults

from qoolbox.bitstring import Bitstring
from qoolbox.cost import BitstringCost, create_mis_cost
from qoolbox.graph.mis import is_independent, is_maximal_independent
from qoolbox.interpret.postprocess import maximalize
from qoolbox import MIS_graph
# Define the hex color values in the order provided by PASQAL
colors = [
    #"#173035"
    #,  # Soft Green
    "#E1F6E9",  # Bright Green
    "#00C887",  # Mint Green
    "#173035",
    #"#397378",  # Metal Blue
    #"#92DCE5",  # Neon Blue
    #"#867BFA",  # Purple
    #"#FF986E",  # Orange
]
# Create a linear colormap from the colors
PASQAL_cmap = LinearSegmentedColormap.from_list("custom_palette", colors)
PASQAL_cmap_r = LinearSegmentedColormap.from_list("custom_palette", colors[::-1])
# Register this colormap with seaborn


ResType = Union[CoherentResults, NoisyResults, dict]

# The following constants are used to index columns of polars dataframes.
_BITSTRING = "bitstring"
_PROBABILITY = "probability"
_WEIGHT = "weight"
_COST = "cost"
_INDEPENDENCE_TYPE = "independence_type"
_DATA_TYPE = "data_type"
_DISTANCE = "distance_to_MIS"


class IndepencendenceType(str, Enum):
    """The different independence types for a subset of vertices of a graph."""
    NOT = "not"
    INDEPENDENT = "independent"
    MAXIMAL = "maximal"
    MAXIMUM = "maximum"


_independence_order = [
    IndepencendenceType.MAXIMUM.value,
    IndepencendenceType.MAXIMAL.value,
    IndepencendenceType.INDEPENDENT.value,
    IndepencendenceType.NOT.value,
]

class MISAnalyser:
    """
    Analyzes and manages multiple independent set (MIS) datasets on graphs with options for postprocessing, 
    probability thresholding, cutoff parameters, and custom cost functions.

    Attributes:
        graph (list[nx.Graph]): List of graphs, with one for each dataset.
        labels (list[str]): Labels for each dataset.
        probability_threshold (list[Optional[float]]): Threshold probabilities for each dataset.
        probability_cutoff (list[Optional[float]]): Cutoff probabilities for each dataset.
        postprocess (list[bool]): Flags for whether to apply postprocessing to each dataset.
        mis_size (list[int]): List of MIS sizes for each dataset.
        cost_function (list[BitstringCost]): Cost functions applied to each graph.
        df (list[pd.DataFrame]): List of dataframes containing processed results for each dataset.
    """

    def __init__(
        self, 
        graph: Union[nx.Graph, list[nx.Graph]], 
        data: Union[ResType, list[ResType]], 
        labels: Union[list[str],str],  # Optional labels for each result
        probability_threshold: Optional[Union[float, list[float], None]] = None,
        probability_cutoff: Optional[Union[float, list[float], None]] = None,
        postprocess: Optional[Union[bool, list[bool], None]] = None,
        cost_function: Optional[BitstringCost] = None,
        mis_size: Optional[Union[int, list[int], None]] = None, 
        mis_compute: Optional[bool] = False
    ):
        """
        Initializes MISAnalyser with multiple dataset options and configuration parameters.

        Args:
            graph (Union[nx.Graph, list[nx.Graph]]): Single graph or list of graphs for each dataset.
            data (Union[ResType, list[ResType]]): Results data or list of results.
            labels (Optional[list[str]]): Labels for each dataset, if applicable.
            probability_threshold (Optional[Union[float, list[float], None]]): Probability thresholds for filtering data.
            probability_cutoff (Optional[Union[float, list[float], None]]): Probability cutoffs for filtering data.
            postprocess (Optional[Union[bool, list[bool], None]]): Whether to apply postprocessing, specified as a 
                single boolean or list.
            cost_function (Optional[BitstringCost]): Cost function to be applied; can be a single instance or list.
            mis_size (Optional[Union[int, list[int], None]]): Size of the MIS, either specified or computed.
            mis_compute (Optional[bool]): If True, calculates MIS size for each graph if not provided.
        
        Raises:
            ValueError: For mismatched lengths of data and labels or for incorrect parameter types.
            TypeError: If inputs do not match the expected types.
        """
        if isinstance(data, list):
            lendata=len(data)
            if not all(isinstance(d, ResType) for d in data):
                raise ValueError("All elements in 'data' must be of type ResType.")
        elif isinstance(data, ResType):
            lendata=1
            data=[data]
        else:
            raise ValueError("'data' must be of type ResType.")

        if isinstance(labels, list):
            lenlabels=len(labels)
            if not all(isinstance(l, str) for l in labels):
                raise ValueError("All elements in 'labels' must be of type str.")
            self.labels=labels
        elif isinstance(labels, str):
            lenlabels=1
            self.labels=[labels]
        else:
            raise ValueError("'labels' must be of type str.")
        if lendata!=lenlabels:
            raise ValueError("'data' and 'labels' must have the same length.")
      

        if isinstance(graph, nx.Graph):
            self.graph = [graph] * lendata  # Use the same graph for all datasets
        elif isinstance(graph, list):
            if len(graph) != lendata:
                raise ValueError("Length of graph list must match the number of datasets.")
            self.graph = graph
        else:
            raise TypeError("graph should be a `nx.graph', list of `nx.graph', or None.")

        if mis_size == None:
            if mis_compute==True:
                self.mis_size=[len(MIS_graph(x).graph["opt-set"]) for x in self.graph]
            else:
                self.mis_size = [0] * lendata 
        elif isinstance(mis_size, int):
            self.mis_size = [mis_size] * lendata  # Apply the same boolean to all datasets
        elif isinstance(mis_size, list):
            if len(mis_size) != lendata:
                raise ValueError("Length of mis_size list must match the number of datasets.")
            if not all(isinstance(x, int) for x in mis_size):
                raise ValueError("All elements in 'mis_size' must be of type int.")
            self.mis_size = mis_size
        else:
            raise TypeError("mis_size should be a `int', list of `int' or None")

        if postprocess is None or postprocess == []:
            self.postprocess = [False] * lendata  # Default to False for all datasets if None or empty
        elif isinstance(postprocess, bool):
            self.postprocess = [postprocess] * lendata  # Apply the same boolean to all datasets
        elif isinstance(postprocess, list):
            if len(postprocess) != lendata:
                raise ValueError("Length of postprocess list must match the number of datasets.")
            self.postprocess = postprocess
        else:
            raise TypeError("postprocess should be a boolean, list of booleans, or None.")

        # Process probability_threshold
        if probability_threshold is None or probability_threshold == []:
            self.probability_threshold = [None] * lendata
        elif isinstance(probability_threshold, (float, int)):
            self.probability_threshold = [probability_threshold] * lendata
        elif isinstance(probability_threshold, list):
            if len(probability_threshold) != lendata:
                raise ValueError("Length of probability_threshold list must match the number of datasets.")
            self.probability_threshold = probability_threshold
        else:
            raise TypeError("probability_threshold should be a float, list of floats, or None.")

        # Process probability_cutoff
        if probability_cutoff is None or probability_cutoff == []:
            self.probability_cutoff = [None] * lendata
        elif isinstance(probability_cutoff, (float, int)):
            self.probability_cutoff = [probability_cutoff] * lendata
        elif isinstance(probability_cutoff, list):
            if len(probability_cutoff) != lendata:
                raise ValueError("Length of probability_cutoff list must match the number of datasets.")
            self.probability_cutoff = probability_cutoff
        else:
            raise TypeError("probability_cutoff should be a float, list of floats, or None.")

        if isinstance(self.graph,list):
            self.cost_function = [cost_function or create_mis_cost(x, 1.2) for x in self.graph]
        else: 
            self.cost_function = cost_function or create_mis_cost(self.graph, 1.2)

        self.df = [self.create_mis_dataframe(result, graph, label, p_threshold, p_cutoff, postproc, cost_function, mis_size) 
                   for result, graph, label, p_threshold, p_cutoff, postproc, cost_function, mis_size 
                   in zip(
                       data,
                       self.graph,
                       self.labels,
                       self.probability_threshold,
                       self.probability_cutoff,
                       self.postprocess,
                       self.cost_function,
                       self.mis_size
                       )] 

    def create_mis_dataframe(
        self, 
        result: ResType, 
        graph: Union[nx.Graph, list[nx.Graph]],
        label: Optional[str] = None, 
        p_threshold: Optional[float] = None, 
        p_cutoff: Optional[float] = None, 
        postproc: bool = False,
        cost_function: Optional[BitstringCost] = None,
        mis_size: int = 0,
    ) -> pl.DataFrame:
        """
    Create a polars.DataFrame for studying the Maximum Independent Set (MIS) problem on a graph.
    
    This function processes a set of bitstrings, analyzes their probabilities, and appends relevant 
    analysis data such as costs, independence types, and distances to the MIS. The resulting DataFrame
    is suitable for further analysis of the MIS problem for a given graph.

    Args:
        result (ResType): The input result which can be in the form of either a `CoherentResults`, 
                          `NoisyResults`, or a mapping of bitstrings or strings to probabilities.
                          This data is used to construct a distribution of bitstrings and their probabilities.
        
        label (Optional[str], default=None): A label for the data, which will be added as a column to 
                                              the resulting DataFrame. If not provided, no label will 
                                              be added.

        p_threshold (Optional[float], default=None): A probability threshold. Any bitstring with a 
                                                     probability lower than this value will be excluded
                                                     from the final DataFrame. If not provided, no threshold 
                                                     filtering is applied.

        p_cutoff (Optional[float], default=None): A cumulative probability cutoff. This controls the
                                                   percentage of the total probability mass to retain. 
                                                   Bitstrings whose cumulative probability is less than 
                                                   this cutoff will be excluded. If not provided, no cutoff
                                                   filtering is applied.

        postproc (bool, default=False): A flag that determines whether postprocessing should be applied to
                                        the bitstrings in the dataset. If `True`, non-independent bitstrings
                                        will be processed to make them maximal. If `False`, no postprocessing 
                                        is done.

         Returns:
            pl.DataFrame: A Polars DataFrame containing the analyzed bitstrings. The columns include:
                      - `bitstring`: The bitstring representation.
                      - `probability`: The probability associated with the bitstring.
                      - `weight`: The weight of the bitstring (number of 1's).
                      - `cost`: The cost associated with the bitstring (if any).
                      - `independence_type`: The independence type of the bitstring, one of 
                        "independent", "maximal", "maximum", or "not".
                      - `distance_to_MIS`: The distance from the MIS, only present if the `mis_size` 
                        is provided.

        """
        df = self.bitstring_distribution(result) 
        df = self.limit_dataframe(df,  pthreshold=p_threshold, pcutoff=p_cutoff)
        if postproc:
            df = self.postprocess_bitstring_distribution(df, graph)
        if mis_size>0:
            df = self.add_mis_analysis(df, graph, cost_function=cost_function, mis_size=mis_size)
        if label is not None:
            df = df.with_columns(pl.lit(label).alias(_DATA_TYPE))  

        return df
    
    def add_df(
        self, 
        new_data: ResType, 
        new_graph: nx.Graph, 
        new_label: str,
        new_probability_threshold: Optional[float] = None, 
        new_probability_cutoff: Optional[float] = None, 
        new_postprocess: bool = False,
        new_mis_size: Optional[int] = None
    ):
        """
        Adds a new dataframe to the instance by updating each attribute accordingly.

        Args:
            new_data (ResType): The new dataset to add.
            new_graph (nx.Graph): The graph associated with the new dataset.
            new_label (str): The label for the new dataset.
            new_probability_threshold (Optional[float]): Probability threshold for the new data.
            new_probability_cutoff (Optional[float]): Probability cutoff for the new data.
            new_postprocess (bool): Whether to apply postprocessing for the new data.
            new_mis_size (Optional[int]): MIS size for the new dataset.
        """
        # Validate the new label
        if not isinstance(new_label, str):
            raise ValueError("The new label must be a string.")
        
        # Append the new elements to each attribute
        self.graph.append(new_graph)
        self.labels.append(new_label)
        
        # Handle probability_threshold and probability_cutoff as lists or single values
        self.probability_threshold.append(new_probability_threshold)
        self.probability_cutoff.append(new_probability_cutoff)
        
        # Append postprocess and mis_size with new values, using defaults as necessary
        self.postprocess.append(new_postprocess)
        self.mis_size.append(new_mis_size if new_mis_size is not None else 0)
        
        # Generate the new DataFrame and append it to the list of dataframes
        new_df = self.create_mis_dataframe(
            new_data,
            new_graph,
            new_label,
            new_probability_threshold,
            new_probability_cutoff,
            new_postprocess,
            new_mis_size or 0
        )
        self.df.append(new_df)

    def remove_df(self, label: str):
        """
        Removes a dataset and its associated information by the given label.

        Args:
            label (str): The label of the dataset to remove.
        """
        # Find the index of the label to remove
        try:
            index = self.labels.index(label)
        except ValueError:
            raise ValueError(f"Label '{label}' not found in the dataset labels.")
        
        # Remove the entry at the found index for each attribute
        self.graph.pop(index)
        self.labels.pop(index)
        self.probability_threshold.pop(index)
        self.probability_cutoff.pop(index)
        self.postprocess.pop(index)
        self.mis_size.pop(index)
        self.df.pop(index)

        print(f"Dataset with label '{label}' removed successfully.")

    @staticmethod
    def merge(df: list[pl.DataFrame]) ->  pl.DataFrame:
        """
        Merges a list of dataframes into a single dataframe.
        Args:
            df (list of pl.DataFrame): A list of Polars dataframes to merge.
        Returns:
            pl.DataFrame: A single dataframe created by concatenating the input dataframes.
        Raises:
            ValueError: If the input is not a list of dataframes.
        """
        if isinstance(df,list):
            return pl.concat(df)
        else:
            raise ValueError("'df' must be a list of dataframes to merge.")
    
    def merge_df(self):
        """
        Merges 'self.df' into a single dataframe.

        This method updates the 'self.df' attribute by merging it with itself using 
        the `merge` method.

        Returns:
            None
        """

        self.df = self.merge(self.df)
        
    @staticmethod
    def split(df: pl.DataFrame, column: str) ->  list[pl.DataFrame]:
        """
        Splits a dataframe into a list of dataframes based on a specified column.

        Args:
            df (pl.DataFrame): The dataframe to split.
            column (str): The column by which to split the dataframe.

        Returns:
            list of pl.DataFrame: A list of dataframes, each corresponding to a partition
                                based on the unique values in the specified column.

        Raises:
            ValueError: If the input is a list of dataframes or the specified column does not exist.
        """
        if isinstance(df,list):
            raise ValueError("`df` must be a single dataframe.")
        if column not in df.columns:
            raise ValueError(f"Column '{column}' does not exist in the dataframe.")
        else:
            return list(df.partition_by(column))  


    def split_df(self, column: str):
        """
        Splits 'self.df' into a list of dataframes based on the specified column.

        This method updates the 'self.df' attribute by splitting it using the `split` method.

        Args:
            column (str): The column by which to split the dataframe.

        Returns:
            None
        """
        self.df = self.split(self.df,column=column) 

    @staticmethod
    def bitstring_distribution(result: ResType,
    ) -> pl.DataFrame:
        """
        Converts a set of bitstring samples from various formats into a Polars dataframe 
        with "bitstring", "probability", and "weight" columns.

        Args:
            result (ResType): Result in the form of a pulser Result object (either coherent or noisy),
                            or any mapping between bitstrings/strings and counts/probabilities.

        Returns:
            pl.DataFrame: A dataframe containing bitstrings, probabilities, and weights.
        """
        if isinstance(result, CoherentResults):
            result = result[-1].sampling_dist
        elif isinstance(result, NoisyResults):
            result = result.results[-1]
        tot = sum(result.values())
        bitstring = list(result.keys())
        if isinstance(bitstring[0], Bitstring):
            bitstring = [b.as_str() for b in bitstring]
        if isinstance(bitstring[0], Bitstring):
            weight = [b.weight() for b in bitstring]
            bitstring = [b.as_str() for b in bitstring]
        else:
            weight = [Bitstring.from_str(b).weight() for b in bitstring]
        prob = [p / tot for p in result.values()]

        df = pl.DataFrame(
            {
                _BITSTRING: pl.Series(bitstring, dtype=pl.Utf8),
                _PROBABILITY: pl.Series(prob, dtype=pl.Float64),
                _WEIGHT: pl.Series(weight, dtype=pl.Int64),
            }
        )
        return df

    @staticmethod
    def bitstring_distribution_to_dict(df: pl.DataFrame) -> dict[Bitstring, float]:
        """
        Converts a bitstring distribution dataframe into a dictionary of bitstrings and their probabilities.

        Args:
            df (pl.DataFrame): A Polars dataframe containing "bitstring" and "probability" columns.

        Returns:
            dict[Bitstring, float]: A dictionary where keys are Bitstrings and values are the associated probabilities.
        """
        distr: dict[Bitstring, float] = {}
        for i in range(len(df)):
            b = Bitstring.from_str(df[i][_BITSTRING].item())
            if b in distr:
                distr[b] += df[i][_PROBABILITY].item()
            else:
                distr[b] = df[i][_PROBABILITY].item()
        return distr

    @staticmethod
    def add_mis_analysis(df: pl.DataFrame,
        graph: Optional[nx.Graph],
        cost_function: Optional[BitstringCost] = None,
        mis_size: Optional[int] = None,
    ) -> pl.DataFrame:
        """
        Adds mis analysis columns:
            - cost
            - independence type
        to a bistring distribution dataframe with existing columns
            - bitstring
            - probability
        If 'cost(x)' is already present in the dataframe then new
        columnames are prepended with counter number (eg 'cost(x+1)').

        Args:
            df (pl.Dataframe): dataframe to analyse and extend
            graph (nx.Graph): graph to determing independence of bitstrings
            cost_function (Callable or None): bitstring cost to use to generate contents of cost
                column (default is to use create_mis_cost with penalty 1.2)
            mis_size (int): size of the mis of the graph if known, used to determine
                contents of independence type. If not given then will look for
                answer in graph dict, if this fails then only possible
                independence types are "not", "independent" and "maximal" (i.e.
                not "maximum)

        Returns:
            polars dataframe with analysis columns included
        """
        bitstrings = df[_BITSTRING]
        if cost_function is None:
            # NOTE(lvignoli): the 1.2 has been chosen empirically by Finzgar et al.
            # in doi.org/10.48550/arXiv.2305.13365.
            cost_function = create_mis_cost(graph, 1.2)
        col_names = "".join(df.columns)
        ncosts = len(re.findall(_COST + "(\d*)", col_names))
        if ncosts == 0:
            colname_append = ""
        else:
            colname_append = str(ncosts + 1)
        cost: list[float] = []
        independence_type: list[IndepencendenceType] = []
        distance: list[int] = []
        for i in range(len(bitstrings)):
            b = Bitstring.from_str(bitstrings[i])
            weight = b.weight()
            cost.append(cost_function(b))
            independence = is_independent(b, graph)
            is_maximal = is_maximal_independent(b, graph)
            is_maximum = is_maximal and weight == mis_size and weight > 0
            if not independence:
                independence_type.append(IndepencendenceType.NOT)
                distance.append(-1)
            else:
                distance.append(int(mis_size - b.weight()))
                if not is_maximal:
                    independence_type.append(IndepencendenceType.INDEPENDENT)
                elif not is_maximum:
                    independence_type.append(IndepencendenceType.MAXIMAL)
                else:
                    independence_type.append(IndepencendenceType.MAXIMUM)

        series_to_add = [
            pl.Series(name=_COST + colname_append, values=cost, dtype=pl.Float64),
            pl.Series(
                name=_INDEPENDENCE_TYPE + colname_append, values=independence_type, dtype=pl.Utf8
            ),
        ]
        if mis_size > 0:
            series_to_add.append(
                pl.Series(name=_DISTANCE + colname_append, values=distance, dtype=pl.Int32)
            )
        df = df.with_columns(*series_to_add)

        return df

    @staticmethod
    def limit_dataframe(
        df: pl.DataFrame,
        pthreshold: Optional[float] = None,
        pcutoff: Optional[float] = None,
        sortby: str = _PROBABILITY,
        descending: bool = True,
    ) -> pl.DataFrame:
        """
        Limits a dataframe to the top (1 - pcutoff) samples and/or to those samples
        with probabilities above pthreshold. In the case where there are several
        cases of the limiting value the remaining probability is split proportionally
        between the cases so that there is no ordering/sampling bias (beyond that
        imposed deliberately by the sortby argument).

        Args:
            df: dataframe to limit
            pthreshold: discard samples with probability below this if provided
            pcutoff: cumulative probability of samples to discard if provided
            sortby: (default probability) column to sort by before discarding
                bottom pcutoff samples
            descending: (default True) whether to sort in descending order,
                must be True if sortby is probability

        Returns:
            dataframe with (1 - pcutoff) samples included, ordered by sortby
        """
        sorted_df = df.sort(by=sortby, descending=descending)
        # remove bitstrings with probability below pthreshold
        if pthreshold is not None:
            sorted_df = sorted_df.filter(pl.col(_PROBABILITY) > pthreshold)
        # remove bottom bitstrings with cumulative probability of pcutoff
        if pcutoff is not None:
            total_prob = sorted_df.select(pl.sum(_PROBABILITY)).item()
            if (1 - total_prob) > pcutoff:
                warnings.warn(
                    f"Attempting to limit a dataframe to the top {(1 - pcutoff)}"
                    f"samples but only {total_prob} found"
                )
                return sorted_df
            if (pcutoff - (1 - total_prob)) < 1e-12:
                return sorted_df
            # find the rows which span the frequency cutoff point and
            # split the remaning frequency proportionally between them
            total_prob = df.select(pl.sum(_PROBABILITY)).item()
            cumulative_prob = sorted_df[_PROBABILITY].cumsum()
            limit_ind = list(cumulative_prob <= (1 - pcutoff)).index(False)
            limit_val = sorted_df[sortby][limit_ind]
            limit_val_inds = np.argwhere(np.isclose(sorted_df[sortby], limit_val)).flatten()
            limit_probs = sorted_df[_PROBABILITY][limit_val_inds]
            existing_prob = 0 if limit_val_inds[0] == 0 else cumulative_prob[: limit_val_inds[0]][-1]
            limit_probs *= (1 - pcutoff - existing_prob) / limit_probs.sum()
            probs = np.append(
                sorted_df[_PROBABILITY][: limit_val_inds[0]].to_numpy(), limit_probs.to_numpy()
            )
            sorted_df = sorted_df[: limit_val_inds[-1] + 1].with_columns(
                pl.Series(probs).alias(_PROBABILITY)
            )
        return sorted_df

    @staticmethod
    def postprocess_bitstring_distribution(
        df: pl.DataFrame,
        graph: nx.Graph,
    ) -> pl.DataFrame:
        """
        Postproccesses the bitstrings in a bitstring_distribution dataframe
        to apply vertex reduction to non independent bitstrings where necessary and
        then apply vertex addition to all bitstrings until they are maximal.
        Returned dataframe will be a bitstring_distribution dataframe with columns
            - bitstring: the postprocessed bitstring
            - probability: the probability of the bitstring (and those which were
                postprocessed into it)
            - weight: the cardinal of the set, ie the number of 1's in the bitstring
        but any additional columns (eg cost or independence_type will not be present
        and must be re-generated)

        Args:
            df: dataframe to postprocess
            graph: graph to use to determine independence maximality of bitstrings

        """
        dist = MISAnalyser.bitstring_distribution_to_dict(df)
        dist_postprocessed: dict[Bitstring, float] = {}
        for b, f in dist.items():
            b = maximalize(b, graph)
            if b in dist_postprocessed.keys():
                dist_postprocessed[b] += f
                continue
            dist_postprocessed[b] = f
        return MISAnalyser.bitstring_distribution(dist_postprocessed)

    def _get_dataframe_by_label(self, label: str) -> 'pl.DataFrame':
        """
        Retrieves a dataframe by its label.
        Args:
            label (str): The label associated with the desired dataframe.
        Returns:
            pl.DataFrame: The dataframe matching the given label.
        Raises:
            ValueError: If the label does not match any dataframe.
        """
        try:
            index = self.labels.index(label)
            return self.df[index]
        except ValueError:
            raise ValueError(f"No dataframe found with label '{label}'")
        
    def get_variance(self, column_name: str) -> list[float]:
        """
        Calculate the variance of a specified column for each dataframe in the class.
        Args:
            column_name (str): The name of the column for which to calculate the variance.
        Returns:
            list[float]: A list containing the variance for the specified column in each dataframe.
                         If the column does not exist in a dataframe, None is returned for that dataframe.
        """

        variances = []
        for df in self.df:
            if column_name in df.columns:
                variance = df[column_name].var()
                variances.append(variance)
            else:
                variances.append(None)
        return variances
    
    def get_distance_variance(self) -> list[float]:
        """
        Calculate the variance of a specified column for each dataframe in the class.
        Args:
            column_name (str): The name of the column for which to calculate the variance.
        Returns:
            list[float]: A list containing the variance for the specified column in each dataframe.
                         If the column does not exist in a dataframe, None is returned for that dataframe.
        """

        variances = []
        for df in self.df:
            df=df.to_pandas()
            df=df.groupby(_DISTANCE, as_index=False).agg({_PROBABILITY: "sum"})
            df = df[df[_DISTANCE]>=0]
            mean=(df[_DISTANCE]*df[_PROBABILITY]).sum()
            variance=((df[_DISTANCE]-mean)**2*df[_PROBABILITY]).sum()
            variances.append(variance)
        return variances
    
    def get_distance_mean(self) -> list[float]:
        """
        Calculate the variance of a specified column for each dataframe in the class.
        Args:
            column_name (str): The name of the column for which to calculate the variance.
        Returns:
            list[float]: A list containing the variance for the specified column in each dataframe.
                         If the column does not exist in a dataframe, None is returned for that dataframe.
        """

        avg = []
        for df in self.df:
            df=df.to_pandas()
            df=df.groupby(_DISTANCE, as_index=False).agg({_PROBABILITY: "sum"})
            df = df[df[_DISTANCE]>=0]
            mean=(df[_DISTANCE]*df[_PROBABILITY]).sum()
            avg.append(mean)
        return avg
        
    def get_cost_mean(self) -> list[float]:
        """
        Calculate the variance of a specified column for each dataframe in the class.
        Args:
            column_name (str): The name of the column for which to calculate the variance.
        Returns:
            list[float]: A list containing the variance for the specified column in each dataframe.
                         If the column does not exist in a dataframe, None is returned for that dataframe.
        """

        avg = []
        for i,df in enumerate(self.df):
            df=df.to_pandas()
            df=df.groupby(_COST, as_index=False).agg({_PROBABILITY: "sum"})
            mean=((self.mis_size[i]+df[_COST])/self.mis_size[i]*df[_PROBABILITY]).sum()
            avg.append(mean)
        return avg
    
    def get_gap_mean(self) -> list[float]:
        """
        Calculate the variance of a specified column for each dataframe in the class.
        Args:
            column_name (str): The name of the column for which to calculate the variance.
        Returns:
            list[float]: A list containing the variance for the specified column in each dataframe.
                         If the column does not exist in a dataframe, None is returned for that dataframe.
        """

        avg = []
        for i,df in enumerate(self.df):
            df=df.to_pandas()
            df=df.groupby(_DISTANCE, as_index=False).agg({_PROBABILITY: "sum"})
            df = df[df[_DISTANCE]>=0]
            mean=(df[_DISTANCE]*df[_PROBABILITY]/self.mis_size[i]).sum()
            avg.append(mean)
        return avg
    
    def get_gap_variance(self) -> list[float]:
        """
        Calculate the variance of a specified column for each dataframe in the class.
        Args:
            column_name (str): The name of the column for which to calculate the variance.
        Returns:
            list[float]: A list containing the variance for the specified column in each dataframe.
                         If the column does not exist in a dataframe, None is returned for that dataframe.
        """

        variances = []
        for i,df in enumerate(self.df):
            df=df.to_pandas()
            df=df.groupby(_DISTANCE, as_index=False).agg({_PROBABILITY: "sum"})
            df = df[df[_DISTANCE]>=0]
            mean=(df[_DISTANCE]*df[_PROBABILITY]).sum()
            variance=(((df[_DISTANCE]-mean)/self.mis_size[i])**2*df[_PROBABILITY]).sum()
            variances.append(variance)
        return variances
    
    
    def get_distance_distributions(self) -> list[np.ndarray]:
        """
        Extract the distribution of distances for each dataframe in the class.
        Returns:
            list[np.ndarray]: A list containing arrays of distances for each dataframe.
                            If the _DISTANCE column does not exist in a dataframe, 
                            an empty array is returned for that dataframe.
        """
        distributions = []
        for df in self.df:
            df = df.to_pandas()
            df=df.groupby(_DISTANCE, as_index=False).agg({_PROBABILITY: "sum"})
            distributions.append(df)
        return distributions

        
    def get_MIS_probability(self, label: Optional[str] = None) -> float:
        """
        Calculates the probability of the maximal independent set (MIS) for a specific dataframe or all dataframes.

        Args:
            label (Optional[str]): Label of the dataframe to process. If None, processes all dataframes.

        Returns:
            float: The total probability of the maximal independent set.
        """
        if label:
            df = self._get_dataframe_by_label(label)
            return float(df.filter(pl.col(_INDEPENDENCE_TYPE) == "maximum")[_PROBABILITY].sum())
        else:
            return [float(df.filter(pl.col(_INDEPENDENCE_TYPE) == "maximum")[_PROBABILITY].sum()) 
                for df in self.df]

    def get_MIS_minus_k_probability(self,k:int, label: Optional[str] = None) -> float:
        """
        Calculates the probability of the maximal independent set (MIS)minus k  for a specific dataframe or all dataframes.

        Args:
            label (Optional[str]): Label of the dataframe to process. If None, processes all dataframes.

        Returns:
            float: The total probability of the maximal independent set.
        """
        if label:
            df = self._get_dataframe_by_label(label)
            return float(df.filter(pl.col(_DISTANCE) == k)[_PROBABILITY].sum())
        else:
            return [float(df.filter(pl.col(_DISTANCE) == k)[_PROBABILITY].sum()) 
                for df in self.df]
        

    def get_IS_probability(self, label: Optional[str] = None) -> float:
        """
        Calculates the probability of the independent set (IS) for a specific dataframe or all dataframes.

        Args:
            label (Optional[str]): Label of the dataframe to process. If None, processes all dataframes.

        Returns:
            float: The total probability of the independent set.
        """
        if label:
            df = self._get_dataframe_by_label(label)
            return float(df.filter(pl.col(_INDEPENDENCE_TYPE) != "not")[_PROBABILITY].sum())
        else:
            return [float(df.filter(pl.col(_INDEPENDENCE_TYPE) != "not")[_PROBABILITY].sum()) 
                for df in self.df]

    def get_mIS_probability(self, label: Optional[str] = None) -> float:
        """
        Calculates the probability of the maximal independent set (mIS) for a specific dataframe or all dataframes.

        Args:
            label (Optional[str]): Label of the dataframe to process. If None, processes all dataframes.

        Returns:
            float: The total probability of the maximal independent set (mIS).
        """
        if label:
            df = self._get_dataframe_by_label(label)
            return float(df.filter(pl.col(_INDEPENDENCE_TYPE) == "maximal")[_PROBABILITY].sum())
        else:
            return [float(df.filter(pl.col(_INDEPENDENCE_TYPE) == "maximal")[_PROBABILITY].sum()) 
                for df in self.df]

    def get_most_probable_indept_set(self, label: Optional[str] = None) -> Tuple[str, float]:
        """
        Retrieves the most probable independent set from the dataframe for a specific label or across all dataframes.
        Args:
            label (Optional[str]): Label of the dataframe to process. If None, processes all dataframes.
        Returns:
            Tuple[str, float]: The bitstring of the most probable independent set and its probability.
        """
        if label:
            # Retrieve the dataframe for the specific label
            df = self._get_dataframe_by_label(label)
            # Filter and sort to get the most probable independent set
            most_probable_row = df.filter(pl.col(_INDEPENDENCE_TYPE) != "not").sort(_PROBABILITY, descending=True)
            return [(most_probable_row[_BITSTRING][0], most_probable_row[_PROBABILITY][0])]
        else:
            # Process all dataframes and retrieve the most probable independent set for each
            most_probable_list = []
            for df in self.df:
                most_probable_row = df.filter(pl.col(_INDEPENDENCE_TYPE) != "not").sort(_PROBABILITY, descending=True)
                most_probable_list.append((most_probable_row[_BITSTRING][0], most_probable_row[_PROBABILITY][0]))
            return most_probable_list

    def plot_bitstring_distribution(
            self,
            labels: Union[list[str],str]=None,
            context: str = "notebook",
            probability_threshold: Optional[float] = None,
            probability_cutoff: Optional[float] = None,
            sort_by: Optional[str] = _PROBABILITY,
        ) -> plt.Figure:
        """Plot the distribution of bitstrings as a barplot from a dataframe
        for one or multiple dataframes

        Bars are coloured by the independence type of the bitstrings.

        Args:
            dfs (pl.DataFrame): mis dataframes including bitstring, probability and independence_type columns
            labels (Optional list[str]): Labels to differentiate different dataframes if multiple are provided
            context (str): plotting context to be used for seaborne
            probability_threshold (float or None): probability threshold below which bitstrings are discarded
                if provided (default 1e-4)
            probability_cutoff (float or None): cumulative probability of least probable bitstrings to discard
                before processing if provided (default None)

        Returns
            seaborne catplot
        """
        if labels is not None:
            if isinstance(labels,list) and  all(isinstance(x, str) for x in labels):
                df=[self._get_dataframe_by_label(x) for x in labels]
                df=self.merge(df)
            elif isinstance(labels,str):
                df=[self._get_dataframe_by_label(labels)]
                df=self.merge(df)
            else:
                raise ValueError("labels must be of type str.")
        else:
            labels=self.labels
            df=self.merge(self.df)
        df=self.limit_dataframe(df, pthreshold=probability_threshold, pcutoff=probability_cutoff)
        df=df.to_pandas()
        # Combine the DataFrames
        # Pivot the Combined DataFrame to prepare for plotting
        pivot_df = df.pivot_table(
            index=[_BITSTRING, _INDEPENDENCE_TYPE],
            columns=_DATA_TYPE,
            values=_PROBABILITY,
            fill_value=0,
        ).reset_index()

        # Melt the DataFrame for use in catplot
        melted_df = pivot_df.melt(
            id_vars=[_BITSTRING, _INDEPENDENCE_TYPE], var_name=_DATA_TYPE, value_name=_PROBABILITY
        )

        if sort_by==_PROBABILITY:
            melted_df = melted_df.sort_values(by=sort_by, ascending=False)
        elif sort_by==sort_byE:
            melted_df[sort_by] = pd.Categorical(melted_df[sort_by], categories=labels, ordered=True)
            melted_df = melted_df.sort_values(by=sort_by).reset_index(drop=True)

        # Create a mapping of bitstring to independence_type
        bitstring_to_independence = (
            melted_df[[_BITSTRING, _INDEPENDENCE_TYPE]]
            .drop_duplicates()
            .set_index(_BITSTRING)
            .to_dict()[_INDEPENDENCE_TYPE]
        )

        with sns.plotting_context(context):
            independence_colors = {
                IndepencendenceType.MAXIMUM.value: "green",
                IndepencendenceType.MAXIMAL.value: "orange",
                IndepencendenceType.INDEPENDENT.value: "grey",
                IndepencendenceType.NOT.value: "red",
            }
            sns.set_theme(style="whitegrid")

            # Create a bar plot with 'hue' as the Dataset column to show side-by-side bars
            g = sns.catplot(
                data=melted_df,
                x=_BITSTRING,
                y=_PROBABILITY,
                hue=_DATA_TYPE,
                kind="bar",
                height=6,
                aspect=1.5,  # This puts the bars side by side
                palette=PASQAL_cmap(np.linspace(0, 1, len(labels)))
            )

        g.set_xticklabels(rotation=90)

        # Color the x-tick labels based on the unique independence_type for each bitstring
        for label in g.ax.get_xticklabels():
            bitstring = label.get_text()
            independence_type = bitstring_to_independence.get(
                bitstring, IndepencendenceType.NOT.value
            )
            label.set_color(independence_colors.get(independence_type, "black"))

        # Create custom lines for the legend
        custom_lines = [
            Line2D([0], [0], color=independence_colors[x], lw=4) for x in _independence_order
        ]

        # Add legend with custom labels
        g.ax.legend(
            custom_lines,
            _independence_order,
            bbox_to_anchor=(1.225, 0.05),
            frameon=True,
            title=_INDEPENDENCE_TYPE,
        )
        g.set_axis_labels("Bitstring", "Probability")
        return g


    def plot_distance_to_mis(
            self,
            labels: Optional[list[str],str]=None,
            context: str = "notebook",
            probability_threshold: Optional[float] = None,
            probability_cutoff: Optional[float] = None,
            plot_only_IS: Optional[float] = True,
            sort_by: Optional[str]= _PROBABILITY,
        ) -> plt.Figure:
        
        """Plot the probability as a function of distance from MIS for multiple datasets with side-by-side bars.

        Args:
            dfs (list[pd.DataFrame]): List of dataframes containing `_DISTANCE` and `_PROBABILITY`.
            labels (list[str]): Labels to differentiate the datasets.
            context (str): Seaborn plotting context (e.g., "notebook", "talk").
            probability_threshold (float): Minimum probability threshold for plotting.

        Returns:
            plt.Figure: The plot figure.
        """
        if labels is not None:
            if isinstance(labels,list) and  all(isinstance(x, str) for x in labels):
                df=[self._get_dataframe_by_label(x) for x in labels]
                df=self.merge(df)
            elif isinstance(labels,str):
                df=[self._get_dataframe_by_label(labels)]
                df=self.merge(df)
            else:
                raise ValueError("labels must be of type str.")
        else:
            labels=self.labels
            df=self.merge(self.df)
        df = self.limit_dataframe(
                    df, pthreshold=probability_threshold, pcutoff=probability_cutoff
                )
        df = df.to_pandas()

        if plot_only_IS:
            df= df[df[_DISTANCE] >= 0].groupby([_DISTANCE, _DATA_TYPE], as_index=False).agg({_PROBABILITY: "sum"})
        else:
            df= df.groupby([_DISTANCE, _DATA_TYPE], as_index=False).agg({_PROBABILITY: "sum"})
        
        pivot_df = (
            df.pivot(index=_DISTANCE, columns=_DATA_TYPE, values=_PROBABILITY)
            .fillna(0)
            .reset_index()
        )

        # Melt the pivoted dataframe to make it suitable for Seaborn
        melted_df = pivot_df.melt(id_vars=_DISTANCE, var_name=_DATA_TYPE, value_name=_PROBABILITY)

        if sort_by==_PROBABILITY:
            melted_df = melted_df.sort_values(by=sort_by, ascending=False)
        elif sort_by==sort_by:
            melted_df[sort_by] = pd.Categorical(melted_df[sort_by], categories=labels, ordered=True)
            melted_df = melted_df.sort_values(by=sort_by).reset_index(drop=True)
        
        # Plotting
        with sns.plotting_context(context):
            sns.set_theme(style="whitegrid")
            #sns.set_palette(sns.color_palette(PASQAL_cmap(np.linspace(0, 1, len(self.labels)))))
            
            # Create bar plot with `_DISTANCE` on x-axis and `_PROBABILITY` on y-axis
            g = sns.catplot(
                data=melted_df,
                x=_DISTANCE,
                y=_PROBABILITY,
                hue=_DATA_TYPE,
                kind="bar",
                height=6,
                aspect=1.5,
                dodge=True,
                palette=sns.color_palette(PASQAL_cmap(np.linspace(0, 1, len(labels))))
            )

        g.set_axis_labels("Distance from MIS", "Probability")
        g.legend.set_title("Dataset")

        return g
    
    def plot_cost(
            self,
            labels: Optional[list[str],str]=None,
            context: str = "notebook",
            probability_threshold: Optional[float] = None,
            probability_cutoff: Optional[float] = None,
            plot_only_IS: Optional[float] = True,
            sort_by: Optional[str]= _PROBABILITY,
        ) -> plt.Figure:
        
        """Plot the probability as a function of distance from MIS for multiple datasets with side-by-side bars.

        Args:
            dfs (list[pd.DataFrame]): List of dataframes containing `_DISTANCE` and `_PROBABILITY`.
            labels (list[str]): Labels to differentiate the datasets.
            context (str): Seaborn plotting context (e.g., "notebook", "talk").
            probability_threshold (float): Minimum probability threshold for plotting.

        Returns:
            plt.Figure: The plot figure.
        """
        if labels is not None:
            if isinstance(labels,list) and  all(isinstance(x, str) for x in labels):
                df=[self._get_dataframe_by_label(x) for x in labels]
                df=self.merge(df)
            elif isinstance(labels,str):
                df=[self._get_dataframe_by_label(labels)]
                df=self.merge(df)
            else:
                raise ValueError("labels must be of type str.")
        else:
            labels=self.labels
            df=self.merge(self.df)
        df = self.limit_dataframe(
                    df, pthreshold=probability_threshold, pcutoff=probability_cutoff
                )
        df = df.to_pandas()

        if plot_only_IS:
            df= df[df[_INDEPENDENCE_TYPE]!="not"].groupby([_COST, _DATA_TYPE], as_index=False).agg({_PROBABILITY: "sum"})
        else:
            df= df.groupby([_COST, _DATA_TYPE], as_index=False).agg({_PROBABILITY: "sum"})
        
        pivot_df = (
            df.pivot(index=_COST, columns=_DATA_TYPE, values=_PROBABILITY)
            .fillna(0)
            .reset_index()
        )

        # Melt the pivoted dataframe to make it suitable for Seaborn
        melted_df = pivot_df.melt(id_vars=_COST, var_name=_DATA_TYPE, value_name=_PROBABILITY)

        if sort_by==_PROBABILITY:
            melted_df = melted_df.sort_values(by=sort_by, ascending=False)
        elif sort_by==sort_by:
            melted_df[sort_by] = pd.Categorical(melted_df[sort_by], categories=labels, ordered=True)
            melted_df = melted_df.sort_values(by=sort_by).reset_index(drop=True)
        
        # Plotting
        with sns.plotting_context(context):
            sns.set_theme(style="whitegrid")
            #sns.set_palette(sns.color_palette(PASQAL_cmap(np.linspace(0, 1, len(self.labels)))))
            
            # Create bar plot with `_DISTANCE` on x-axis and `_PROBABILITY` on y-axis
            g = sns.catplot(
                data=melted_df,
                x=_COST,
                y=_PROBABILITY,
                hue=_DATA_TYPE,
                kind="bar",
                height=6,
                aspect=1.5,
                dodge=True,
                palette=sns.color_palette(PASQAL_cmap(np.linspace(0, 1, len(labels))))
            )

        g.set_axis_labels("Cost", "Probability")
        g.legend.set_title("Dataset")

        return g


    def plot_distance_distribution(
            self,
            labels: Optional[list[str],str]=None,
            context: str = "notebook",
            probability_threshold: Optional[float] = None,
            probability_cutoff: Optional[float] = None,
            plot_only_IS: Optional[float] = True,
            sort_by: Optional[str]= _PROBABILITY,
        ) -> plt.Figure:
        
        """Plot the probability as a function of distance from MIS for multiple datasets with side-by-side bars.

        Args:
            dfs (list[pd.DataFrame]): List of dataframes containing `_DISTANCE` and `_PROBABILITY`.
            labels (list[str]): Labels to differentiate the datasets.
            context (str): Seaborn plotting context (e.g., "notebook", "talk").
            probability_threshold (float): Minimum probability threshold for plotting.

        Returns:
            plt.Figure: The plot figure.
        """
        if labels is not None:
            if isinstance(labels,list) and  all(isinstance(x, str) for x in labels):
                df=[self._get_dataframe_by_label(x) for x in labels]
                df=self.merge(df)
            elif isinstance(labels,str):
                df=[self._get_dataframe_by_label(labels)]
                df=self.merge(df)
            else:
                raise ValueError("labels must be of type str.")
        else:
            labels=self.labels
            df=self.merge(self.df)
        df = self.limit_dataframe(
                    df, pthreshold=probability_threshold, pcutoff=probability_cutoff
                )
        df = df.to_pandas()

        if plot_only_IS:
            df= df[df[_INDEPENDENCE_TYPE]!="not"].groupby([_COST, _DATA_TYPE], as_index=False).agg({_PROBABILITY: "sum"})
        else:
            df= df.groupby([_COST, _DATA_TYPE], as_index=False).agg({_PROBABILITY: "sum"})
        
        pivot_df = (
            df.pivot(index=_COST, columns=_DATA_TYPE, values=_PROBABILITY)
            .fillna(0)
            .reset_index()
        )

        # Melt the pivoted dataframe to make it suitable for Seaborn
        melted_df = pivot_df.melt(id_vars=_COST, var_name=_DATA_TYPE, value_name=_PROBABILITY)

        if sort_by==_PROBABILITY:
            melted_df = melted_df.sort_values(by=sort_by, ascending=False)
        elif sort_by==sort_by:
            melted_df[sort_by] = pd.Categorical(melted_df[sort_by], categories=labels, ordered=True)
            melted_df = melted_df.sort_values(by=sort_by).reset_index(drop=True)
        
        # Plotting
        with sns.plotting_context(context):
            sns.set_theme(style="whitegrid")
            #sns.set_palette(sns.color_palette(PASQAL_cmap(np.linspace(0, 1, len(self.labels)))))
            
            # Create bar plot with `_DISTANCE` on x-axis and `_PROBABILITY` on y-axis
            g = sns.catplot(
                data=melted_df,
                x=_COST,
                y=_PROBABILITY,
                hue=_DATA_TYPE,
                kind="bar",
                height=6,
                aspect=1.5,
                dodge=True,
                palette=sns.color_palette(PASQAL_cmap(np.linspace(0, 1, len(labels))))
            )

        g.set_axis_labels("Cost", "Probability")
        g.legend.set_title("Dataset")

        return g
