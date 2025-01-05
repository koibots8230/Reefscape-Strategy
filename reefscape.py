import numpy as np

locations = [
    "Source Left",
    "Source Right",
    "Processor",
    "Reef A",
    "Reef B",
    "Reef C",
    "Reef D",
    "Reef E",
    "Reef F",
    "Barge",
    "Floor L",
    "Floor M",
    "Floor R"
]

# Times for swerve @ 3.0 m/s max & 3.0 m/s^2 max
times = np.array([
    [0, 3.02, 2.91, 2.24, 2.12, 2.53, 3.13, 4.69, 2.73, 3.94, 0.93, 2.01, 2.53],
    [3.02, 0, 3.86, 2.24, 2.73, 4.69, 3.13, 2.53, 2.12, 3.27, 2.53, 2.01, 0.93],
    [2.91, 3.86, 0, 2.77, 2.35, 2.09, 2.30, 2.80, 3.63, 3.01, 2.59, 2.93, 3.34],
    [2.24, 2.24, 2.77, 0, 1.46, 2.08, 2.95, 2.08, 1.46, 2.94, 1.91, 1.30, 1.91],
    [2.12, 2.73, 2.35, 1.46, 0, 1.46, 2.08, 2.95, 2.08, 3.11, 1.68, 1.78, 2.16],
    [2.53, 4.69, 2.09, 2.08, 1.46, 0, 1.46, 2.08, 2.95, 2.52, 2.18, 2.26, 2.64],
    [3.13, 3.13, 2.30, 2.95, 2.08, 1.46, 0, 1.46, 2.08, 2.15, 2.62, 2.76, 2.62],
    [4.69, 2.53, 2.80, 2.08, 2.95, 2.08, 1.46, 0, 1.46, 2.52, 2.64, 2.26, 2.18],
    [2.73, 2.12, 3.63, 1.46, 2.08, 2.95, 2.08, 1.46, 0, 3.11, 2.16, 1.78, 1.68],
    [3.94, 3.27, 3.01, 2.94, 3.11, 2.52, 2.15, 2.52, 3.11, 0, 3.53, 3.16, 2.98],
    [0.93, 2.53, 2.59, 1.91, 1.68, 2.18, 2.62, 2.64, 2.16, 3.53, 0, 1.20, 2.24],
    [2.01, 2.01, 2.93, 1.30, 1.78, 2.26, 2.76, 2.26, 1.78, 3.16, 1.20, 0, 1.20],
    [2.53, 0.93, 3.34, 1.91, 2.16, 2.64, 2.62, 2.18, 1.68, 2.98, 2.24, 1.20, 0]
])


class Action:
    def __init__(self, points: int, max_quantity: int, average_time: float, time_spread: float,
                 proportion_of_success: float = 1):
        self.points = points
        self.max_quantity = max_quantity
        self.average_time = average_time
        self.time_spread = time_spread
        self.success_proportion = proportion_of_success

    def get_time(self):
        return np.random.normal(self.average_time, self.time_spread)


remainingReef = {
    "Score L1": 24,
    "Score L2": 12,
    "Score L3": 12,
    "Score L4": 12
}

groundAlgae = 3
reefAlgae = 6

autoTime = 15
teleopTime = 135

auto_actions = {
    "Score L1": Action(3, 24, 5, 0.5, .90),
    "Score L2": Action(4, 12, 6, 1, .80),
    "Score L3": Action(6, 12, 6.5, 1.2, .80),
    "Score L4": Action(7, 12, 7, 1.5, .75),
    "GroundNet": Action(4, 3, 9, 2, .35),
    "GroundProcessor": Action(6, 3, 9, 2, .50),
    "ReefNet": Action(4, 6, 9, 1, .65),
    "ReefProcessor": Action(6, 6, 9, 1, .90),
    "Leave": Action(3, 1, 1, 0, .95),
}

preload_actions = {
    "Score L1": Action(3, 1, 3, 0.5, .90),
    "Score L2": Action(4, 1, 4, 1, .80),
    "Score L3": Action(4, 1, 4.5, 1.2, .80),
    "Score L4": Action(7, 1, 5, 1.5, .75),
}

teleop_actions = {
    "Score L1": Action(2, 24, 5, 0.75, .90),
    "Score L2": Action(3, 12, 6, 1.25, .80),
    "Score L3": Action(4, 12, 6.5, 1.45, .80),
    "Score L4": Action(5, 12, 7, 1.75, .75),
    "GroundNet": Action(4, 3, 9, 2, .60),
    "GroundProcessor": Action(6, 3, 9, 2, .85),
    "ReefNet": Action(4, 6, 9, 1, .65),
    "ReefProcessor": Action(6, 6, 9, 1, .90),
}

endgame_actions = {
    "Deep Climb": Action(12, 1, 12, 3.5, .70),
    "Shallow Climb": Action(6, 1, 10, 3.5, .75),
    "Park": Action(2, 1, 2.4, 2.5, .70),
}


def find_key(input_dict, value):
    for key, val in input_dict.items():
        if val == value: return key
    return "None"


def findBestAction(actionsList):
    highestPointsForTime = 0
    bestAction = None
    for action in actionsList:
        if action.points / action.average_time > highestPointsForTime:
            highestPointsForTime = action.points / action.average_time
            bestAction = action
    return bestAction


class Robot:
    def __init__(self, autoActions, teleopActions, endgameActions, hasPipeGroundIntake):
        self.autoActions = autoActions
        self.telopActions = teleopActions
        self.endgameActions = endgameActions
        self.scoredPreload = False
        self.pipeGround = hasPipeGroundIntake

    def autoSample(self, intelligent: bool = True):
        global groundAlgae
        global reefAlgae
        global remainingReef

        if auto_actions["Leave"] in self.autoActions:
            self.autoActions.remove(auto_actions["Leave"])
            return auto_actions["Leave"]
        else:
            if intelligent:
                sample = findBestAction(self.autoActions)
            else:
                sample = np.random.choice(self.autoActions)
            if "Score L" in find_key(auto_actions, sample):
                remainingReef[find_key(auto_actions, sample)] -= 1

                if remainingReef[find_key(auto_actions, sample)] == 0:
                    try:
                        self.autoActions.remove(sample)
                        self.telopActions.remove(teleop_actions[find_key(auto_actions, sample)])
                    except:
                        pass

                if not self.scoredPreload:
                    self.scoredPreload = True
                    return preload_actions[find_key(auto_actions, sample)]
                return sample
            else:
                if "Ground" in find_key(auto_actions, sample):
                    groundAlgae -= 1
                    if groundAlgae == 0:
                        try:
                            self.autoActions.remove(auto_actions["GroundNet"])
                            self.autoActions.remove(auto_actions["GroundProcessor"])
                        except:  # specify ValueError (nitpick)
                            pass
                else:
                    reefAlgae -= 1
                    if reefAlgae == 0:
                        try:
                            self.autoActions.remove(auto_actions["ReefNet"])
                            self.autoActions.remove(auto_actions["ReefProcessor"])
                        except:
                            pass
                return sample

    def teleopSample(self, intelligent: bool = True):
        global groundAlgae
        global reefAlgae
        global remainingReef

        if intelligent:
            sample = findBestAction(self.telopActions)
        else:
            sample = np.random.choice(self.telopActions)
        if "Score L" in find_key(auto_actions, sample):
            remainingReef[find_key(teleop_actions, sample)] -= 1

            if remainingReef[find_key(teleop_actions, sample)] == 0:
                self.telopActions.remove(sample)
            if not self.scoredPreload:
                self.scoredPreload = True
                return preload_actions[find_key(teleop_actions, sample)]
            return sample
        else:
            if "Ground" in find_key(auto_actions, sample):
                groundAlgae -= 1
                if groundAlgae == 0:
                    try:
                        self.telopActions.remove(teleop_actions["GroundNet"])
                        self.telopActions.remove(teleop_actions["GroundProcessor"])
                    except:
                        pass
            else:
                reefAlgae -= 1
                if reefAlgae == 0:
                    try:
                        self.telopActions.remove(teleop_actions["ReefNet"])
                        self.telopActions.remove(teleop_actions["ReefProcessor"])
                    except:
                        pass
            return sample

    def endgame(self):
        return findBestAction(self.endgameActions)