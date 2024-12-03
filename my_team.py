import random
import contest.util as util

from contest.capture_agents import CaptureAgent
from contest.game import Directions
from contest.util import nearest_point
#################
# Team creation #
#################

def create_team(first_index, second_index, is_red,
    first='OffensiveAgent', second='DefensiveAgent', num_training=0):
    return [eval(first)(first_index), eval(second)(second_index)]

# ##########
# # Agents #
# ##########
class OffensiveAgent(CaptureAgent):
    """
    An offensive agent that uses expectimax with alpha-beta pruning to choose actions.
    """

    def __init__(self, index, time_for_computing=.1):
        super().__init__(index, time_for_computing)
        self.start = None
        self.depth = 3 # Define the lowest depth of the expectimax tree before evaluation

    def register_initial_state(self, game_state):
        self.start = game_state.get_agent_position(self.index)
        CaptureAgent.register_initial_state(self, game_state)


    def choose_action(self, game_state):
        """
        Returns the expectimax action using alpha-beta pruning
        """
        # Get visible enemy indices
        visible_enemies = [enemy for enemy in self.get_opponents(game_state) 
                           if game_state.get_agent_state(enemy).get_position() is not None]

        # If no enemies are visible, use simple evaluation to get best action from current state
        if len(visible_enemies) == 0:
            legal_actions = game_state.get_legal_actions(self.index)
            values = [self.evaluate(game_state, a) for a in legal_actions]
            max_value = max(values)
            best_actions = [a for a, v in zip(legal_actions, values) if v == max_value]
            return random.choice(best_actions)

        # Otherwise use expectimax with alpha-beta pruning
        alpha = float('-inf')
        beta = float('inf')
        action, _ = self.alpha_beta_expectimax(game_state, self.index, self.depth, alpha, beta, visible_enemies)
        return action

    def alpha_beta_expectimax(self, game_state, agent_index, depth, alpha, beta, enemies):
        """
        Returns (action, value) tuple for expectimax with alpha-beta pruning
        """
        # If we reach the lowest depth, evaluate the current state
        if depth == 0:
            return None, self.evaluate(game_state, Directions.STOP)
        # Compute next agent index
        next_agent = (agent_index + 1) % game_state.get_num_agents()
        # Skip teammate's turn
        team_indices = game_state.get_red_team_indices() if self.red else game_state.get_blue_team_indices()
        if next_agent in team_indices and next_agent != self.index:  # Call expectimax for next agent
            return self.alpha_beta_expectimax(game_state,  
            (next_agent + 1) % game_state.get_num_agents(), depth, alpha, beta, enemies)

        # Maximizing player (our agent only)
        if agent_index == self.index:
            # Get legal actions from current state
            legal_actions = game_state.get_legal_actions(self.index)
            if len(legal_actions) == 0:
                return None, self.evaluate(game_state, Directions.STOP)

            # Initialize best action and score
            max_score = float('-inf')
            best_action = Directions.STOP
            
            # Iterate over all legal actions, finding the one with the highest score
            for action in legal_actions:
                next_state = self.get_successor(game_state, action)
                # Current state score plus child score
                _, future_score = self.alpha_beta_expectimax(next_state, next_agent, depth - 1, alpha, beta, enemies)
                score = self.evaluate(game_state, action) + future_score
                
                # Update best score and action
                if score > max_score:
                    max_score = score
                    best_action = action
                    
                # Beta pruning
                alpha = max(alpha, max_score)
                if max_score > beta:
                    return best_action, max_score
            # Return best action and score
            return best_action, max_score

        # Minimizing player (enemy ghosts only)
        else:
            # If this enemy is not visible, skip their turn
            if agent_index not in enemies:
                return self.alpha_beta_expectimax(game_state, next_agent, depth, alpha, beta, enemies)
            
            # Get ghost position and legal actions
            ghost_pos = game_state.get_agent_state(agent_index).get_position()
            legal_ghost_actions = self.get_ghost_actions(game_state, ghost_pos)
            
            # If no legal actions, evaluate current state
            if len(legal_ghost_actions) == 0:
                return None, self.evaluate(game_state, Directions.STOP)
            
            # Calculate expected score
            expected_score = 0
            probability = 1.0 / len(legal_ghost_actions) # Uniform distribution over legal ghost actions

            # Iterate over all legal ghost actions and calculate expected score
            for action in legal_ghost_actions:
                next_state = game_state.generate_successor(agent_index, action)
                _, score = self.alpha_beta_expectimax(next_state, next_agent, depth, alpha, beta, enemies)
                expected_score += score * probability
            
            # Return any action and the expected score over all legal ghost actions
            return legal_ghost_actions[0], expected_score

    def get_ghost_actions(self, game_state, current_pos):
        """
        Returns a list of valid actions that a ghost can take from its current position.
        We created this method since game_state.get_legal_actions(ghost_index) was not working
        when receiving an enemy ghost index. 
        """
        # Get walls and boundaries of the game
        walls = game_state.get_walls()
        layout_width = walls.width
        layout_height = walls.height
        actions = []
        # Get current position
        x, y = int(current_pos[0]), int(current_pos[1])
        # Check each cardinal direction, accounting for borders
        if y+1 < layout_height and not walls[x][y+1]: actions.append('North')
        if y-1 >= 0 and not walls[x][y-1]: actions.append('South')
        if x+1 < layout_width and not walls[x+1][y]: actions.append('East')
        if x-1 >= 0 and not walls[x-1][y]: actions.append('West')

        return actions
    
    def get_features(self, game_state, action):
        """
        Returns a list of features for the current state and action that the offensive agent will evaluate the state on.
        """
        features = util.Counter()
        successor = self.get_successor(game_state, action)
        food_list = self.get_food(successor).as_list()
        my_state = successor.get_agent_state(self.index)
        my_pos = my_state.get_position()

        # Stop action penalty
        if action == Directions.STOP:
            features['stop'] = 1.0

        # Successor score 
        features['successor_score'] = -len(food_list)
        
        # Distance to closest food score
        if len(food_list) > 0:
            min_distance = min([self.get_maze_distance(my_pos, food) for food in food_list])
            features['distance_to_food'] = min_distance

        # Ghost avoidance and scared ghost hunting
        enemies = [successor.get_agent_state(i) for i in self.get_opponents(successor)]
        ghosts = [a for a in enemies if not a.is_pacman and a.get_position() is not None] # Get positions of ghosts that are not pacman
        scared_ghosts = [g for g in ghosts if g.scared_timer > 0] # get scared ghosts
        active_ghosts = [g for g in ghosts if g.scared_timer <= 0] # get active ghosts

        # Ghost avoidance score - only avoid ghosts that are close
        if len(active_ghosts) > 0:
            ghost_dists = [self.get_maze_distance(my_pos, g.get_position()) for g in active_ghosts]
            min_ghost_dist = min(ghost_dists)
            if min_ghost_dist <= 5:  # Only care about ghosts within 5 spaces
                features['ghost_distance'] = min_ghost_dist
                if min_ghost_dist == 1:  # Immediate danger
                    features['ghost_distance'] = -float('inf')

        # Scared ghost score
        if len(scared_ghosts) > 0:
            scared_dists = [self.get_maze_distance(my_pos, g.get_position()) for g in scared_ghosts]
            features['scared_ghost_distance'] = min(scared_dists)

        # Power capsule score
        capsules = self.get_capsules(successor)
        if len(capsules) > 0 and len(active_ghosts) > 0:
            ghost_dists = [self.get_maze_distance(my_pos, g.get_position()) for g in active_ghosts]
            min_ghost_dist = min(ghost_dists)
            min_capsule_dist = min([self.get_maze_distance(my_pos, cap) for cap in capsules])
            
            # Only prioritize capsules if both ghost and capsule are nearby and the capsule is closer than the ghost
            if min_ghost_dist <= 5 and min_capsule_dist <= min_ghost_dist:
                features['capsule_distance'] = min_capsule_dist * 2
            else:
                features['capsule_distance'] = min_capsule_dist

        # Return home when carrying food
        if my_state.num_carrying > 1:
            features['carrying_food'] = my_state.num_carrying
            dist_to_home = self.get_maze_distance(my_pos, self.start)
            
            features['distance_to_home'] = dist_to_home
            
            # Encourage return when carrying lots of food or close to home
            if my_state.num_carrying > 3 or dist_to_home <= 5:
                features['return_now'] = 1.0

        return features

    def get_weights(self, game_state, action):
        """
        Returns a dictionary of weights for each feature to prioritize them.
        """
        return {'successor_score': 100,'distance_to_food': -1,'ghost_distance': 2.0,
            'scared_ghost_distance': -1.5,'capsule_distance': -2.0,'carrying_food': 5.0,
            'distance_to_home': -2.0,'return_now': 100.0, 'stop': -100.0}

    def get_successor(self, game_state, action):
        """
        Finds the next successor which is a grid position (location tuple).
        """
        successor = game_state.generate_successor(self.index, action)
        pos = successor.get_agent_state(self.index).get_position()
        if pos != nearest_point(pos):
            # Only half a grid position was covered
            return successor.generate_successor(self.index, action)
        else:
            return successor
        
    def evaluate(self, game_state, action):
        """
        Computes a linear combination of features and feature weights
        """
        features = self.get_features(game_state, action)
        weights = self.get_weights(game_state, action)
        return features * weights


class DefensiveAgent(CaptureAgent):
    """
    A defensive agent that uses A* search to find the best path to the goal
    """

    def register_initial_state(self, game_state):
        CaptureAgent.register_initial_state(self, game_state)
        self.patrol_index = 0 # Index of the current patrol point
        self.border_points = self.get_border_points(game_state) # List of border points to patrol
    
        
    def get_border_points(self, game_state):
        """
        Get list of border positions to patrol when no invaders are visible.
        """
        border_x = game_state.data.layout.width // 2 # Get x-coordinate of center border
        if self.red:
            border_x = game_state.data.layout.width // 2 - 1
    
        # Get all border points that aren't walls
        border_points = [(border_x, y) for y in range(1, game_state.data.layout.height - 1) if not game_state.has_wall(border_x, y)]
        
        # Sort points by their proximity to food we're defending
        defending_food = self.get_food_you_are_defending(game_state).as_list()
        if defending_food:
            # For each border point, find its distance to the nearest food
            def distance_to_nearest_food(point):
                distances = [self.get_maze_distance(point, food_pos) for food_pos in defending_food]
                return min(distances)
            
            # Sort border points based on distance to nearest food
            border_points.sort(key=distance_to_nearest_food)
            
        return border_points
    
    def choose_action(self, game_state):
        """
        Combines A* pathfinding with feature-based evaluation for better decision making. 
        """
        # Get current agent state and position
        my_state = game_state.get_agent_state(self.index)
        my_pos = my_state.get_position()
        # Get enemy positions
        enemies = [game_state.get_agent_state(i) for i in self.get_opponents(game_state)]
        invaders = [a for a in enemies if a.is_pacman and a.get_position() is not None]
        # Get legal actions
        legal_actions = game_state.get_legal_actions(self.index)
        
        # If there are invaders, use both A* and features to choose action
        if len(invaders) > 0:
            positions = [inv.get_position() for inv in invaders]
            dists = [self.get_maze_distance(my_pos, pos) for pos in positions]
            closest_pos = positions[dists.index(min(dists))]
            
            # Get A* recommended action
            a_star_action = self.a_star_search(game_state, closest_pos)
            
            # If A* finds a clear path, use it
            if a_star_action != Directions.STOP:
                return a_star_action
            
            # Otherwise, fall back to feature evaluation
            values = [self.evaluate(game_state, a) for a in legal_actions]
            max_value = max(values)
            best_actions = [a for a, v in zip(legal_actions, values) if v == max_value]
            return random.choice(best_actions)
        
        # For patrol behavior, combine A* and feature evaluation
        else:
            # Get current patrol target
            current_target = self.border_points[self.patrol_index]
            
            # If we're at the current patrol point, update target
            if self.get_maze_distance(my_pos, current_target) <= 1:
                self.patrol_index = (self.patrol_index + 1) % len(self.border_points)
                current_target = self.border_points[self.patrol_index]
            
            # Try A* first
            a_star_action = self.a_star_search(game_state, current_target)
            
            # If A* finds a good path, evaluate it against other options
            if a_star_action != Directions.STOP:
                # Get feature-based values for all actions
                values = [self.evaluate(game_state, a) for a in legal_actions]
                a_star_value = self.evaluate(game_state, a_star_action)
                
                # Use A* action if it's nearly as good as the best feature-based action
                if a_star_value >= max(values) * 0.8:
                    return a_star_action
            
            # Fall back to feature evaluation
            values = [self.evaluate(game_state, a) for a in legal_actions]
            max_value = max(values)
            best_actions = [a for a, v in zip(legal_actions, values) if v == max_value]
            return random.choice(best_actions)

    def a_star_search(self, game_state, goal_pos):
        """
        A* search to find best path to target using legal game actions.
        """
        frontier = util.PriorityQueue()
        frontier.push((game_state, []), 0)  # Keep track of (game_state, path), priority in queue
        explored = set()
        
        # Keep going until there are no more states to explore
        while not frontier.is_empty():
            current_state, path = frontier.pop()
            current_pos = current_state.get_agent_state(self.index).get_position()
            
            # Return the first action in the path if we've reached the goal
            if current_pos == goal_pos:
                if len(path) > 0:
                    return path[0]
                else:
                    return Directions.STOP
                
            # Add current position to explored set if it hasn't been explored yet
            if current_pos not in explored:
                explored.add(current_pos)
                
                # Get legal actions from game state
                legal_actions = current_state.get_legal_actions(self.index)
                if Directions.STOP in legal_actions:
                    legal_actions.remove(Directions.STOP)
                
                # Generate successors using legal actions
                for action in legal_actions:
                    successor = self.get_successor(current_state, action)
                    next_pos = successor.get_agent_state(self.index).get_position()
                    
                    if next_pos not in explored:
                        new_path = path + [action]
                        g_n = len(new_path)  # g(n) = path length
                        h_n = self.get_maze_distance(next_pos, goal_pos)  # Use maze distance to goal heuristic
                        f_n = g_n + h_n
                        frontier.push((successor, new_path), f_n)
        
        return Directions.STOP
    
    def patrol_border(self, game_state):
        """
        Patrol the border when no invaders are visible
        """
        # Get current agent position
        my_pos = game_state.get_agent_state(self.index).get_position()
        
        # Get current patrol target
        current_target = self.border_points[self.patrol_index]

        # If we've reached our current patrol point, move to next patrol point
        if (self.get_maze_distance(my_pos, current_target) <= 1):
            self.patrol_index = (self.patrol_index + 1) % len(self.border_points)
            current_target = self.border_points[self.patrol_index]
            
        # Use A* to find the best path to the current patrol target
        return self.a_star_search(game_state, current_target)
    
    def get_features(self, game_state, action):
        """
        Returns a dictionary of features for the current state and action that the defensive agent will evaluate the state on.
        """
        features = util.Counter()
        successor = self.get_successor(game_state, action)
        my_state = successor.get_agent_state(self.index)
        my_pos = my_state.get_position()

        # Computes whether we're on defense (1) or offense (0)
        features['on_defense'] = 1
        if my_state.is_pacman: features['on_defense'] = 0

        # Computes distance to invaders we can see
        enemies = [successor.get_agent_state(i) for i in self.get_opponents(successor)]
        invaders = [a for a in enemies if a.is_pacman and a.get_position() is not None]
        features['num_invaders'] = len(invaders)
        
        # Distance to closest invader
        if len(invaders) > 0:
            dists = [self.get_maze_distance(my_pos, a.get_position()) for a in invaders]
            features['invader_distance'] = min(dists)
            # Score for invader distance to closest food being defended 
            defending_food = self.get_food_you_are_defending(successor).as_list()
            if len(defending_food) > 0:
                min_dist = float('inf')
                for invader in invaders: # Compute distance to closest food for each invader
                    for food in defending_food:
                        dist = self.get_maze_distance(invader.get_position(), food)
                        min_dist = min(min_dist, dist)
                invader_to_food = min_dist
                features['invader_to_food'] = invader_to_food

        else: # Distance to closest border point when no invaders
            dist_to_border = min([self.get_maze_distance(my_pos, point) for point in self.border_points])
            features['border_distance'] = dist_to_border

        # Penalty for stopping or reversing
        if action == Directions.STOP: 
            features['stop'] = 1
        rev = Directions.REVERSE[game_state.get_agent_state(self.index).configuration.direction]
        if action == rev: 
            features['reverse'] = 1

        # Add feature for being near power capsules we're defending
        capsules = self.get_capsules_you_are_defending(successor)
        if len(capsules) > 0:
            features['defending_capsule'] = min([self.get_maze_distance(my_pos, capsule) for capsule in capsules])
            
        return features

    def get_weights(self, game_state, action):
        """
        Returns a dictionary of weights for each feature to prioritize them.
        """
        # Get visible enemy positions
        enemies = [game_state.get_agent_state(i) for i in self.get_opponents(game_state)]
        invaders = [a for a in enemies if a.is_pacman and a.get_position() is not None]
        
        # Base weights
        weights = {
            'num_invaders': -1000, 'on_defense': 100,'invader_distance': -10,
            'stop': -100,'reverse': -2,'border_distance': -7,'defending_capsule': -5,
            'invader_to_food': 2  
        }
        
        # Adjust weights based on game state
        if len(invaders) > 0:
            # When there are invaders, prioritize tracking them down
            weights['invader_distance'] *= 2
            weights['border_distance'] *= 0.5
            weights['invader_to_food'] *= 2
            
            # If invader is very close to food, become more aggressive
            features = self.get_features(game_state, action)
            if 'invader_to_food' in features and features['invader_to_food'] <= 2:
                weights['invader_distance'] *= 2
                weights['stop'] *= 2
        else:
            # When patrolling, focus on border coverage
            weights['border_distance'] *= 2

        return weights
    
    def get_successor(self, game_state, action):
        """
        Finds the next successor which is a grid position (location tuple).
        """
        successor = game_state.generate_successor(self.index, action)
        pos = successor.get_agent_state(self.index).get_position()
        if pos != nearest_point(pos):
            # Only half a grid position was covered
            return successor.generate_successor(self.index, action)
        else:
            return successor
        
    def evaluate(self, game_state, action):
        """
        Computes a linear combination of features and feature weights
        """
        features = self.get_features(game_state, action)
        weights = self.get_weights(game_state, action)
        return features * weights