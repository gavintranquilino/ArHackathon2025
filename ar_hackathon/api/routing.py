"""
Amazon Robotics Hackathon - Routing API

This module defines the routing API for the Amazon Robotics Hackathon.
Students will implement the route_package function in this module.

*****IMPORTANT*****
Team name: Gavin Tranquilino   
Email address: gtranqui@uwaterloo.ca    
*******************
"""

# gavintranquilino.com

from typing import Optional, Dict, List, Set
import heapq
from ar_hackathon.models.game_state import GameState
from ar_hackathon.models.package import Package

# keep track of where packages have been to avoid loops 
package_visited_history: Dict[str, Set[str]] = {}

def route_package(state: GameState, package: Package) -> Optional[str]:
    """
    Determine the next FC to route a package to.
    
    This is the function that students will implement. The game engine will call
    this function for each package at each time step to determine where to route it.
    
    Args:
        state: GameState object containing the current state of the network
        package: Package object containing information about the package
        
    Returns:
        next_fc_id: ID of the next FC to route the package to, or None to stay at current FC
    """
    try:
        current_fc = package.current_fc
        destination_fc = package.destination_fc
        
        # already at destination just stay there and cleanup
        if current_fc == destination_fc:
            if package.id in package_visited_history:
                del package_visited_history[package.id]
            return None
        
        # use dijkstra to find the shortest path 
        next_fc = find_shortest_path_next_hop(state, current_fc, destination_fc, package.id)
        return next_fc
        
    except Exception as e:
        # something broke FC21 idk so just use simple routing as backup
        return fallback_greedy_routing(state, package.current_fc, package.id)


def find_shortest_path_next_hop(state: GameState, start_fc: str, destination_fc: str, package_id: str = None) -> Optional[str]:
    if start_fc == destination_fc:
        return None
    
    # check where this package has been before to avoid cycles
    visited_fcs = set()
    if package_id and package_id in package_visited_history:
        visited_fcs = package_visited_history[package_id]
    
    # count how many packages are using each connection right now  
    connection_traffic = {}
    for pkg in state.active_packages:
        if pkg.in_transit and pkg.transit_destination:
            connection_key = (pkg.current_fc, pkg.transit_destination)
            connection_traffic[connection_key] = connection_traffic.get(connection_key, 0) + 1
    
    # build the graph of all fulfillment centers
    graph = {}
    all_fcs = set()
    
    # add all the fulfillment centers to our graph
    for fc in state.fulfillment_centers:
        graph[fc.id] = []
        all_fcs.add(fc.id)
    
    # add connections but make them smarter based on congestion and stuff
    for connection in state.connections:
        # make sure both endpoints actually exist 
        if connection.from_fc not in all_fcs or connection.to_fc not in all_fcs:
            continue
            
        # only use connections that have bandwidth available
        if connection.available_bandwidth is None or connection.available_bandwidth > 0:
            weight = connection.weight
            
            # dont want packages going in circles so penalize revisited places
            if package_id and connection.to_fc in visited_fcs:
                weight *= 1.5  # make it a bit more expensive but not impossible
            
            # if bandwidth is limited make congested routes more expensive
            if connection.bandwidth is not None and connection.available_bandwidth is not None:
                utilization = 1.0 - (connection.available_bandwidth / connection.bandwidth)
                if utilization > 0.7:  # pretty congested 
                    weight *= (1.0 + utilization * 0.5)  # up to 50% penalty
                elif utilization > 0.3:  # somewhat busy
                    weight *= (1.0 + utilization * 0.2)  # small penalty
            
            # also avoid routes that lots of packages are using right now
            connection_key = (connection.from_fc, connection.to_fc)
            current_traffic = connection_traffic.get(connection_key, 0)
            if current_traffic > 0:
                # each package using this route makes it slightly worse
                traffic_penalty = 1.0 + (current_traffic * 0.1)  # 10% per package
                weight *= traffic_penalty
            
            graph[connection.from_fc].append((connection.to_fc, weight))
    
    # make sure start and end points actually exist in our graph
    if start_fc not in all_fcs or destination_fc not in all_fcs:
        return fallback_greedy_routing(state, start_fc, package_id)
    
    # run dijkstras algorithm to find shortest path
    distances = {fc_id: float('inf') for fc_id in all_fcs}
    distances[start_fc] = 0
    previous = {}
    pq = [(0, start_fc)]
    dijkstra_visited = set()
    
    while pq:
        current_dist, current_fc = heapq.heappop(pq)
        
        if current_fc in dijkstra_visited:
            continue
            
        dijkstra_visited.add(current_fc)
        
        if current_fc == destination_fc:
            break
            
        for neighbor, weight in graph.get(current_fc, []):
            if neighbor in dijkstra_visited:
                continue
                
            distance = current_dist + weight
            
            if distance < distances[neighbor]:
                distances[neighbor] = distance
                previous[neighbor] = current_fc
                heapq.heappush(pq, (distance, neighbor))
    
    # work backwards from destination to find the first step
    if destination_fc not in previous and destination_fc != start_fc:
        # couldnt find a path so use backup method
        return fallback_greedy_routing(state, start_fc, package_id)
    
    # trace back to find next step 
    if destination_fc == start_fc:
        return None
        
    # go backwards through the path to find first move
    current = destination_fc
    path = [current]
    
    while current in previous and previous[current] != start_fc:
        current = previous[current]
        path.append(current)
    
    # the next step is the last thing we added when going backwards
    if path:
        next_hop = path[-1]  # this is where we go next
        
        # remember where weve been for this package
        if package_id:
            if package_id not in package_visited_history:
                package_visited_history[package_id] = set()
            package_visited_history[package_id].add(start_fc)
        
        return next_hop
    
    return None


def fallback_greedy_routing(state: GameState, current_fc: str, package_id: str = None) -> Optional[str]:
    try:
        # check where this package has been before
        visited_fcs = set()
        if package_id and package_id in package_visited_history:
            visited_fcs = package_visited_history[package_id]
        
        best_next_fc = None
        best_weight = float('inf')
        
        for connection in state.connections:
            if connection.from_fc == current_fc:
                # only consider connections with bandwidth  
                if connection.available_bandwidth is None or connection.available_bandwidth > 0:
                    weight = connection.weight
                    
                    # really dont want to revisit places in fallback mode
                    if package_id and connection.to_fc in visited_fcs:
                        weight *= 2.0  # double the cost if weve been there
                    
                    # avoid really congested connections even more in fallback
                    if connection.bandwidth is not None and connection.available_bandwidth is not None:
                        utilization = 1.0 - (connection.available_bandwidth / connection.bandwidth)
                        if utilization > 0.8:  # really bad congestion
                            weight *= (1.0 + utilization * 0.8)  # big penalty 
                        elif utilization > 0.5:  # medium congestion
                            weight *= (1.0 + utilization * 0.4)  # some penalty
                    
                    if weight < best_weight:
                        best_weight = weight
                        best_next_fc = connection.to_fc
        
        # keep track of where we go 
        if package_id and best_next_fc:
            if package_id not in package_visited_history:
                package_visited_history[package_id] = set()
            package_visited_history[package_id].add(current_fc)
        
        return best_next_fc
    except Exception:
        # even the fallback broke just stay put
        return None
