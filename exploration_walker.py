from collections import deque
import json
from player import KeyboardPlayerPyGame
import pygame
from vis_nav_game import Action


class AutoPlayerPyGame(KeyboardPlayerPyGame):

    def __init__(self, action_json):
        super().__init__()
        with open(action_json, "r") as actions_file:
            self.actions = json.load(actions_file)
        self.last_key = None
        self.act_index = 0
        self.act_queue = deque()

    def act(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                self.last_act = Action.QUIT
                return Action.QUIT
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_v:
                    print(f"Current act index: {self.act_index}")
                    if len(self.act_queue) == 0:
                        act1 = self.actions[self.act_index]["action"][0]
                        for action in self.actions[self.act_index]["action"][1:]:
                            self.act_queue.append(action)
                        self.act_index += 1
                    else:
                        act1 = self.act_queue.popleft()
                    self.last_act |= self.act_map[act1]
                    self.last_key = act1
                else:
                    self.show_target_images()
            if event.type == pygame.KEYUP:
                if event.key == pygame.K_v:
                    self.last_act ^= self.act_map[self.last_key]
        return self.last_act

    
    def reset(self):
        self.fpv = None
        self.last_act = Action.IDLE
        self.screen = None

        pygame.init()

        self.act_map = {
            "IDLE": Action.IDLE,
            "LEFT": Action.LEFT,
            "RIGHT": Action.RIGHT,
            "FORWARD": Action.FORWARD,
            "BACKWARD": Action.BACKWARD,
            "CHECKIN": Action.CHECKIN,
            "QUIT": Action.QUIT
        }


if __name__ == "__main__":
    import vis_nav_game as vng
    vng.play(the_player=AutoPlayerPyGame("exploration_data/data_info.json"))