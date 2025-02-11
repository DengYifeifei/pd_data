import os
import sys
import json
import pandas as pd
import subprocess
from config import VERSION

class TrialProcessor:
    def __init__(self, version):
        self.version = version
        self.processed_trials = []
        self.setup_directories()

    def setup_directories(self):
        os.makedirs(f'data/processed/{self.version}/practice_data', exist_ok=True)
        os.makedirs(f'data/processed/{self.version}/trial_data', exist_ok=True)

    def process_file(self, filepath):
        print(f"Processing file: {filepath}")
        try:
            with open(filepath) as f:
                data = json.load(f)
        except FileNotFoundError:
            print(f"Error: File not found {filepath}")
            return None

        # Assume data has "practice_data" and "trial_data"
        wid = os.path.basename(filepath).replace('.json', '')

        # Save practice data
        practice_data_path = f'data/processed/{self.version}/practice_data/{wid}.json'
        with open(practice_data_path, 'w') as f:
            json.dump(data["practice_data"], f, indent=4)
        
        # Process trial data
        if isinstance(data["main_data"], list):
            trial_data = pd.DataFrame(data["main_data"])
            #print("trial_data", trial_data)
            trial_data = self.process_trial(trial_data, wid)  # Assuming this method updates and returns processed trial data
            if isinstance(trial_data, pd.DataFrame):
                trial_data = trial_data.to_dict(orient='records')  # Convert to a list of dicts for JSON saving
            return trial_data
        return None


        

    def save_data(self, data, path):
        if isinstance(data, pd.DataFrame):
            data = data.to_dict(orient='records')  # Convert DataFrame to list of dicts
        with open(path, 'w') as f:
            json.dump(data, f, indent=4)  # Saving the data as JSON

    
    # def calculate_best_reward(self,graph, rewards, start):
    #     def dfs(node, current_reward, include_reward=True):
    #         if include_reward:
    #             current_reward += rewards[node]
    #         if node >= len(graph) or not graph[node]:  # Leaf node
    #             return current_reward
    #         return max(dfs(child, current_reward) for child in graph[node])
    #     return dfs(start, 0, include_reward=False)

    # def calculate_average_reward(self, graph, rewards, start):
    #     def dfs(node, current_reward, include_reward=True):
    #         if include_reward:
    #             current_reward += rewards[node]
    #         if node >= len(graph) or not graph[node]:  # Leaf node
    #             return current_reward, 1  # Return current reward and count this path as 1
    #         total_reward, total_paths = 0, 0
    #         for child in graph[node]:
    #             child_reward, child_paths = dfs(child, current_reward)
    #             total_reward += child_reward
    #             total_paths += child_paths
    #         return total_reward, total_paths

    #     total_reward, total_paths = dfs(start, 0, include_reward=False)
    #     best_reward = self.calculate_best_reward(graph, rewards, start)
    #     average_reward = best_reward - ((total_reward - best_reward) / (total_paths-1) )
    #     return average_reward

    # def calculate_diff_2nd(self, events, rewards, graph):
    #     default_result = (None, None, None)
    #     # Iterate through events
    #     for event in events:
    #         if event['event'] == 'select' and 'selected' in event:
    #             current_state = event['selected']
    #             # Ensure current state is within bounds and has connected states
    #             if current_state < len(graph) and graph[current_state]:
    #                 connected_states = graph[current_state]
    #                 if len(connected_states) >= 2:
    #                     first_connected_state = connected_states[0]
    #                     second_connected_state = connected_states[1]
    #                     # Check the validity of the connected state indices
    #                     if (first_connected_state is not None and second_connected_state is not None and
    #                         first_connected_state < len(rewards) and second_connected_state < len(rewards)):
    #                         diff_2 = rewards[second_connected_state] - rewards[first_connected_state]
    #                         return first_connected_state, second_connected_state, abs(diff_2)
    #                     elif len(connected_states) == 1:
    #                         return connected_states[0], None, None
    #     return default_result


    # def calculate_diff_1st(self, events, rewards, graph):
    #     # for event in events:
    #         null_indices = [index for index, reward in enumerate(rewards) if reward is None]
        
    #         for null_index in null_indices:
    #             if null_index < len(graph):  # Check if the index is within the graph's range
    #                 connected_states = graph[null_index]
    #             if len(connected_states) >= 2:
    #                 first_connected_state = connected_states[0]
    #                 second_connected_state = connected_states[1]
    #                     # Check the validity of the connected state indices
    #                 if (first_connected_state is not None and second_connected_state is not None and
    #                     first_connected_state < len(rewards) and second_connected_state < len(rewards)):
    #                     diff_1 = rewards[second_connected_state] - rewards[first_connected_state]
    #                     return second_connected_state,first_connected_state, abs(diff_1)

    # def calculate_paths(self, graph, start_node):
    #     # Recursive function to traverse the binary tree
    #     def traverse(node, current_path, all_paths):
    #         if node is None or node >= len(graph) or not graph[node]:
    #             # Reached a leaf node, add current path to all paths
    #             all_paths.append(current_path)
    #             return
    #         # Visit left and right children
    #         for child in graph[node]:
    #             traverse(child, current_path + [child], all_paths)
        
    #     all_paths = []
    #     traverse(start_node, [start_node], all_paths)
    #     return all_paths

    # def round_up(self, n):
    #     try:
    #         # Convert n to float before rounding
    #         return round(float(n) * 2) / 2
    #     except ValueError:
    #         # Handle the case where conversion to float fails
    #         return None

    # def calculate_paths(self, graph, start_node):
    #     # Recursive function to traverse the binary tree
    #     def traverse(node, current_path, all_paths):
    #         if node is None or node >= len(graph) or not graph[node]:
    #             # Reached a leaf node, add current path to all paths
    #             all_paths.append(current_path)
    #             return
    #         # Visit children
    #         for child in graph[node]:
    #             traverse(child, current_path + [child], all_paths)
        
    #     all_paths = []
    #     traverse(start_node, [start_node], all_paths)
    #     return all_paths

    # def calculate_reward(self, paths, rewards):
    #     path_rewards = []
    #     for path in paths:
    #         cumulative_reward = sum(rewards[node] for node in path if rewards[node] is not None)
    #         path_rewards.append(cumulative_reward)
    #     return path_rewards

    # def categorize_path(self, graph, start, rewards):
    #     paths = self.calculate_paths(graph, start)
    #     path_rewards = self.calculate_reward(paths, rewards)
        
    #     # Function to check if two paths share the same immediate child of the start node
    #     def same_side(start_node, path1, path2):
    #         if len(path1) > 1 and len(path2) > 1:
    #             return path1[1] == path2[1]
    #         return False
        
    #     if len(path_rewards) == 3:
    #         max_reward = max(path_rewards)
    #         max_index = path_rewards.index(max_reward)
            
    #         remaining_rewards = [r for r in path_rewards if r != max_reward]
    #         second_best_reward = max(remaining_rewards)
    #         second_best_index = path_rewards.index(second_best_reward)
            
    #         min_reward = min(path_rewards)
    #         min_index = path_rewards.index(min_reward)
            
    #         if same_side(start, paths[max_index], paths[second_best_index]):
    #             return (1, "best_second")
    #         elif same_side(start, paths[max_index], paths[min_index]):
    #             return (1, "best_min")
    #         else:
    #             return (1, "best_alone")
        
    #     elif len(path_rewards) == 4:
    #         max_reward = max(path_rewards)
    #         max_index = path_rewards.index(max_reward)
            
    #         remaining_rewards = [r for r in path_rewards if r != max_reward]
    #         second_best_reward = max(remaining_rewards)
    #         second_best_index = path_rewards.index(second_best_reward)
            
    #         third_best_reward = sorted(remaining_rewards)[-2]
    #         third_best_index = path_rewards.index(third_best_reward)
            
    #         min_reward = min(path_rewards)
    #         min_index = path_rewards.index(min_reward)
            
    #         if same_side(start, paths[max_index], paths[second_best_index]):
    #             return (2, "best_second")
    #         elif same_side(start, paths[max_index], paths[min_index]):
    #             return (2, "best_min")
    # #         elif same_side(start, paths[max_index], paths[third_best_index]):
    # #             return (2, "best_third")
    #         else:
    #             return (2, "best_third")

    #     return (None, "undefined")

    # def find_connected_nodes(self, graph, start):
    #     connected_nodes = []  # Use a list to store connected nodes
    #     non_connected_nodes = list(range(len(graph)))  # Initialize all nodes as potentially non-connected

    #     def dfs(node):
    #         if node not in connected_nodes:
    #             connected_nodes.append(node)
    #             non_connected_nodes.remove(node)  # Mark node as connected by removing from non-connected list
    #             for child in graph[node]:
    #                 if child not in connected_nodes:
    #                     dfs(child)

    #     dfs(start)  # Start DFS from the starting node

    #     return connected_nodes, non_connected_nodes



    # def accuracy_first(self, events, graph, rewards, start):
    #     # Calculate all possible paths from the start node
    #     all_paths = self.calculate_paths(graph, start)
    #     # Calculate rewards for each path
    #     path_rewards = self.calculate_reward(all_paths, rewards)
    #     # Find the best path based on the maximum reward
    #     best_path_index = path_rewards.index(max(path_rewards))
    #     best_path = all_paths[best_path_index]

    #     # Collect all visited states from the events
    #     visited_nodes = [event['state'] for event in events if 'state' in event]

    #     # Ensure there are enough nodes in visited_nodes and best_path to perform this comparison
    #     if len(visited_nodes) > 1 and len(best_path) > 1:
    #         # Check if the first node visited (beyond start) matches the first node of the best path (beyond start)
    #         accuracy = 1 if visited_nodes[1] == best_path[1] else 0
    #     else:
    #         accuracy = 0

    #     return accuracy


    def process_trial(self, data, wid):
        trial_index = 0
        processed_trials = []

        for index, row in data.iterrows():
            trial = row['trial']
            print("trial:", trial)
            trial_index += 1
            events = row['events']
            print("events:", events)
            #graph = trial['graph']
            #rewards = trial['rewards']
            #start = trial['start']

            # Calculate additional data
            # best_reward = self.calculate_best_reward(graph, rewards, start)
            # average_reward = self.calculate_average_reward(graph, rewards, start)
            # current_reward = sum(rewards[node] for node in set(event['state'] for event in events if 'state' in event) if rewards[node] is not None)
            # connected_nodes, non_connected_nodes = self.find_connected_nodes(graph, start)
            # df, trial_type = self.categorize_path(graph, start, rewards)
            # difficulty = self.round_up(self.calculate_average_reward(graph, rewards, start))
        
            # visited_nodes = set(event['state'] for event in events if 'state' in event)
            # current_reward = sum(rewards[node] for node in visited_nodes if rewards[node] is not None)
            # node_c, node_d,diff_2 = self.calculate_diff_2nd(events, rewards, graph)
            # node_a, node_b,diff_1 = self.calculate_diff_1st(events, rewards, graph)
            
            # accuracy = 1 if current_reward == best_reward else 0
            # accuracy_1 = self.accuracy_first(events, graph, rewards, start)

            # Calculate response times
            # response = next(entry['response'] for entry in events if entry['event'] == 'response')
            # stimulus_time = next(entry['time'] for entry in events if entry['event'] == 'start sound')
            # response_time = next(entry['time'] for entry in events if entry['event'] == 'response')
            # RT = None
            # if response:
            #     RT = stimulus_time-response_time
             

            

            # Assemble processed trial data
            processed_trial = {
                **trial,  # Include existing trial data

                'events': events,  
                'trial_index': trial_index,
                'wid': wid,
                #'RT': RT ,
            }
            
            processed_trials.append(processed_trial)

        return processed_trials



