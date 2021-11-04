# The water environment

The 'map' folder contains 11 randomly generated maps. Each map consists of 12 colored balls moving at constant speed inside a box. Each ball could be red, green, blue, yellow, cyan, or magenta (there are 2 balls per color on each map). The agent, a white ball, can increase its speed on the four cardinal directions. We defined 10 tasks (i.e. reward machines) for this environment using the following events:

- Event 'a' means that the agent just touched a *red* ball.
- Event 'b' means that the agent just touched a *green* ball.
- Event 'c' means that the agent just touched a *blue* ball.
- Event 'd' means that the agent just touched a *yellow* ball.
- Event 'e' means that the agent just touched a *cyan* ball.
- Event 'f' means that the agent just touched a *magenta* ball.

The 'reward_machines' folder contains the 10 tasks for this environment: 

- Task 1: touch a red ball and then a green ball.
- Task 2: touch a blue ball and then a cyan ball.
- Task 3: touch a magenta ball and then a yellow ball.
- Task 4: perform task 1 and task 2.
- Task 5: perform task 2 and task 3.
- Task 6: perform task 1 and task 3.
- Task 7: perform task 1, 2, and 3.
- Task 8: (touch a red ball, then a green ball, and then a blue ball) and also (touch a yellow ball, then a cyan ball, and then a magenta ball).
- Task 9: touch a red ball, then a green ball, and then a blue ball while avoiding touching balls of any other color.
- Task 10: touch a yellow ball, then a cyan ball, and then a magenta ball while avoiding touching balls of any other color.

The 'tests' folder contains 11 testing scenarios. Each test runs the 10 tasks over one particular map. The 'options' folder contains the list of options to be used by our Hierarchical RL baselines when solving the water world environment.