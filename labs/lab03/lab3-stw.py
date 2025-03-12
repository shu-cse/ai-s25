# This is Stephen's solution to Lab 3
from agents import *
import numpy as np

loc_A, loc_B = (0, 0), (1, 0)
CLEAN_STATE = {(0,0) : "Clean",
               (1,0) : "Clean"}

# Part 1.1
def count_steps_until_clean(): # env, agent, end_state
    trivial_vacuum_env = TrivialVacuumEnvironment()
    random_agent = Agent(
        program=RandomAgentProgram(['Right','Left','Suck','NoOp']))
    trivial_vacuum_env.add_thing(random_agent)

    num_steps = 0
    while trivial_vacuum_env.status != CLEAN_STATE:
        trivial_vacuum_env.step()
        num_steps += 1
    return num_steps

print(f"It took {count_steps_until_clean()} steps")

tot_steps_list = []
for i in range(10000):
    tot_steps = count_steps_until_clean()
    tot_steps_list.append(tot_steps)
tot_steps_array = np.array(tot_steps_list)
print(f"Average number of steps = {tot_steps_array.mean()}")
print(f"Standard deviation = {tot_steps_array.std()}")

# Part 2.1 -- OPTIONAL
class NondeterministicVacuumEnvironment(Environment):

    def __init__(self, success_rate=0.9):
        super().__init__()
        self.status = {loc_A: random.choice(['Clean', 'Dirty']),
                       loc_B: random.choice(['Clean', 'Dirty'])}
        self.success_rate = success_rate

    def execute_action(self, agent, action):
        if action == 'Right':
            if random.random() <= self.success_rate:
                agent.location = loc_B
            agent.performance -= 1
        elif action == 'Left':
            if random.random() <= self.success_rate:
                agent.location = loc_A
            agent.performance -= 1
        elif action == 'Suck':
            if self.status[agent.location] == 'Dirty':
                agent.performance += 10
            if random.random() <= self.success_rate:
                self.status[agent.location] = 'Clean'

nondeterministic_vacuum_env = NondeterministicVacuumEnvironment(0.8)

# trivial_vacuum_env = TrivialVacuumEnvironment()
# random_agent = Agent(
#     program=RandomAgentProgram(['Right','Left','Suck','NoOp']))
# trivial_vacuum_env.add_thing(random_agent)

# num_steps = 0
# while trivial_vacuum_env.status != CLEAN_STATE:
#     trivial_vacuum_env.step()
#     num_steps += 1

# print(f"The environment status is {trivial_vacuum_env.status}")
# print(f"Is it clean? {trivial_vacuum_env.status == CLEAN_STATE}")
# print(f"It took {num_steps} steps")