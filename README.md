# Capture the Flag Pacman Contest

We implemented two custom agents for the Capture the Flag Pacman contest. Our main strategy has one agent working on offense, and another one specializing in defense.

## Team Setup

- The **OffensiveAgent** specializes in collecting food on the opponent's side while avoiding the enemy ghosts.
- The **DefensiveAgent** focuses on protecting our food and capsules by patrolling strategic points and tracking down invaders.

## Offensive Agent

The OffensiveAgent is programmed to navigate through the opponent's territory, collecting food, and returning home safely. We implemented the decision making of the agent with expectimax, and used alpha-beta pruning to not explore unnecessary branches.
For the expectimax, we only consider 3 agents (our defensive agent's turn is skipped since it has a separate goal), the offensive agent as the maximizer and the enemy agents as the minimizers. We also only do beta pruning in the maximizer since in the minimizer we are computing the expected score over all actions. We also only considered the enemy's turn when they are in our agents' proximity.

**Features**
1. successor_score: Number of remaining food pellets (negative value to encourage eating). 
2. distance_to_food: Distance to nearest food pellet (encourages moving toward the food).
3. ghost_distance: Distance to nearest active ghost. Critical when ghost is within 5 spaces, becomes infinitely negative when ghost is adjacent.
4. return_now: Flag activated when carrying lots of food (>3) or close to home (≤5 spaces). Prioritizes returning to the home side when carrying a significant amount of food or agent is close to base.
5. distance_to_home: Distance back to friendly territory. Used when returning home. 
6. capsule_distance: Distance to nearest power capsule. Prioritized when ghosts are nearby and capsule is closer than ghost. 
7. scared_ghost_distance: Distance to nearest scared ghost (encourages chasing vulnerable ghosts).
8. stop: Penalize the ghost for stopping, to encourage it to always be moving.
9. carrying_food: Amount of food currently carrying.

## Defensive Agent
The DefensiveAgent is designed to defend the home territory, intercepting invaders, and patrolling key areas. We used  a combination of A*  and feature-based evaluation. It  switches between patrol and tracking based on the presence of invaders in our base.
When no invaders are detected, the agent patrols a series of border points. These points are chosen based on their proximity to the food we are defending . The agent uses A* to find the shortest path.
When invaders are detected, the agent switches to a pursuit mode. It calculates the optimal path to intercept the invaders setting them as the goal state and using A* search.

**Features**
1. border_distance: Distance to nearest border point when no invaders present, moves along key border points to intercept the invaders.
2. invader_distance: Actively tracks closest visible invader and tries to eat them.
3. defending_capsule: Guards power capsules to prevent the opponent from eating them.
4. num_invaders: Score that takes into account how many invaders are on our side of the map.
5. invader_to_food: Distance between closest invader and our nearest food pellet.
6. stop: Penalty for staying still, amplified if the invaders are close to food.
7. reverse: Penalty for reversing direction (we tried to prevent oscillating behavior).

## Issues
Currently the offensive agent sometimes gets stuck in a loop after some time in the game. Sometimes it goes back and forth in the enemy territory when not in danger, or sometimes it stays still on the border while carrying food without actually scoring the food. We tried to adjust weights to change the priorities among other things, but ultimately we were not able to find a good solution to these bugs.
As for the defensive agent, it also sometimes stops patrolling and goes back and forth in the middle of the map, and only reacts if a new pacman is visible.
