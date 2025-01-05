import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import reefscape
import random
from reefscape import Robot


def quickestActionTime(actions):
    quickestTime = 999
    for action in actions:
        actionQuickestTime = action.average_time - (action.time_spread * 3)
        quickestTime = actionQuickestTime if actionQuickestTime < quickestTime else quickestTime
    return quickestTime


def test_robot(robot):
    autoTime = 15
    teleopTime = 135

    points = [0, 0, 0]

    while autoTime > 0:
        if quickestActionTime(robot.autoActions) > autoTime:
            break
        action = robot.autoSample()
        if (autoTime > action.get_time()):
            #            print(f"Doing Auto Action: {reefscape.find_key(reefscape.auto_actions, action)}")
            points[0] += action.points if random.randint(1, 10) < action.success_proportion * 10 else 0
            autoTime -= action.get_time()

    while teleopTime > 0:
        if quickestActionTime(robot.telopActions) > teleopTime:
            break
        doEndgame = np.random.choice([True, False],
                                     p=[((135 - teleopTime) / 135) * 0.8, (((135 - teleopTime) / 135) * -0.8) + 1])
        if doEndgame:
            endgame = robot.endgame()
            endgameTime = endgame.get_time()
            if endgameTime > teleopTime:
                #                print(f"Doing Endgame: {reefscape.find_key(reefscape.endgame_actions, endgame)}")
                points[2] += endgame.points if random.randint(1, 10) < endgame.success_proportion * 10 else 0
                break
        action = robot.teleopSample()
        actionTime = action.get_time()
        if (teleopTime > actionTime):
            #            print(f"Doing Teleop Action: {reefscape.find_key(reefscape.teleop_actions, action)}")
            if robot.pipeGround:
                while random.randint(1, 10) > action.success_proportion * 10:
                    teleopTime -= np.random.normal(4, 0.5)
                points[1] += action.points
                teleopTime -= actionTime
            else:
                points[1] += action.points if random.randint(1, 10) < action.success_proportion * 10 else 0
                teleopTime -= actionTime

    #    print(f"Final Points: {points}")
    return points


testbot = Robot(
    list(map(reefscape.auto_actions.get, ["Leave", "Score L1"])),
    list(map(reefscape.teleop_actions.get, ["Score L1", "Score L2"])),
    list(map(reefscape.endgame_actions.get, ["Park", "Shallow Climb"])),
    False
)

goodbot = Robot(
    list(map(reefscape.auto_actions.get, ["Leave", "Score L1", "Score L2", "Score L4"])),
    list(map(reefscape.teleop_actions.get, ["Score L1", "Score L2", "Score L3", "Score L4"])),
    list(map(reefscape.endgame_actions.get, ["Park"])),
    True
)

algaebot = Robot(
    list(map(reefscape.auto_actions.get, ["Leave", "Score L1"])),
    list(map(reefscape.teleop_actions.get, ["Score L1", "GroundNet", "GroundProcessor", "ReefNet", "ReefProcessor"])),
    list(map(reefscape.endgame_actions.get, ["Park"])),
    False
)

fig, axes = plt.subplots(1, 3, figsize=(15, 5))

print("running...")
results = [test_robot(goodbot) for _ in range(2000)]
for i, (data, ax) in enumerate(zip(list(map(list, zip(*results))), axes)):
    sns.histplot(data=data, kde=True, ax=ax)
print("done")

plt.tight_layout()
plt.savefig("mostrecentrun.png")
plt.show()
