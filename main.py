import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import reefscape
import random
from reefscape import Robot
import copy


def quickestActionTime(actions):
    quickestTime = 999
    for action in actions:
        actionQuickestTime = action.average_time - (action.time_spread * 2)
        quickestTime = actionQuickestTime if actionQuickestTime < quickestTime else quickestTime
    return quickestTime


def test_robot():
    robot = Robot(
        list(map(reefscape.auto_actions.get, ["Leave", "Score L1", "Score L2", "Score L4"])),
        list(map(reefscape.teleop_actions.get, ["Score L1", "Score L2", "Score L3", "Score L4"])),
        list(map(reefscape.endgame_actions.get, ["Park"])),
        True
    )

    reefscape.remainingReef = {
        "Score L1": 24,
        "Score L2": 12,
        "Score L3": 12,
        "Score L4": 12
    }

    reefscape.groundAlgae = 3
    reefscape.reefAlgae = 6

    autoTime = 15
    teleopTime = 135


    points = [0, 0, 0]

    while autoTime > 0:
        if quickestActionTime(robot.autoActions) > autoTime:
            break
        action = robot.autoSample()
        actionTime = action.get_time()
        print(f"Action: {actionTime} Auto: {autoTime}")
        if autoTime > actionTime:
            points[0] += action.points if random.randint(1, 10) < action.success_proportion * 10 else 0
            autoTime -= actionTime
        else:
            break

    while teleopTime > 0:
        if quickestActionTime(robot.telopActions) > teleopTime:
            break
        doEndgame = np.random.choice([True, False],
                                     p=[((135 - teleopTime) / 135) * 0.8, (((135 - teleopTime) / 135) * -0.8) + 1])
        if doEndgame:
            endgame = robot.endgame()
            endgameTime = endgame.get_time()
            if endgameTime > teleopTime:
                points[2] += endgame.points if random.randint(1, 10) < endgame.success_proportion * 10 else 0
                break
        action = robot.teleopSample()
        actionTime = action.get_time()
        if (teleopTime > actionTime):
            if robot.pipeGround:
                while random.randint(1, 10) > action.success_proportion * 10:
                    teleopTime -= np.random.normal(4, 0.5)
                points[1] += action.points
                teleopTime -= actionTime
            else:
                points[1] += action.points if random.randint(1, 10) < action.success_proportion * 10 else 0
                teleopTime -= actionTime
        else:
            break
    print(f"Points: {points}")
    return points


# testbot = RobotConfig(
#     list(map(reefscape.auto_actions.get, ["Leave", "Score L1"])),
#     list(map(reefscape.teleop_actions.get, ["Score L1", "Score L2"])),
#     list(map(reefscape.endgame_actions.get, ["Park", "Shallow Climb"])),
#     False
# )
#
# goodbot = RobotConfig(
#     list(map(reefscape.auto_actions.get, ["Leave", "Score L1", "Score L2", "Score L3", "Score L4"])),
#     list(map(reefscape.teleop_actions.get, ["Score L1", "Score L2", "Score L3", "Score L4"])),
#     list(map(reefscape.endgame_actions.get, ["Park"])),
#     True
# )
#
# algaebot = RobotConfig(
#     list(map(reefscape.auto_actions.get, ["Leave", "Score L1"])),
#     list(map(reefscape.teleop_actions.get, ["Score L1", "ReefNet", "ReefProcessor"])),
#     list(map(reefscape.endgame_actions.get, ["Park"])),
#     False
# )

fig, axes = plt.subplots(1, 3, figsize=(15, 5))

print("running...")
results = [test_robot() for _ in range(1000)]
for i, (data, ax) in enumerate(zip(list(map(list, zip(*results))), axes)):
    sns.histplot(data=data, kde=True, ax=ax)
print("done")

plt.tight_layout()
plt.savefig("mostrecentrun.png")
plt.show()
