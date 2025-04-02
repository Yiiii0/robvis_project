# import necessary libraries and modules
from vis_nav_game import Player, Action, Phase
import pygame
import cv2

import numpy as np
import os
import pickle
import shutil
from sklearn.cluster import KMeans
from sklearn.neighbors import BallTree
from tqdm import tqdm
from matplotlib import pyplot as plt
from natsort import natsorted

# Current Changes:
import networkx as nx
from scipy.spatial import distance
import joblib
from collections import Counter
from sklearn.cluster import MiniBatchKMeans

from map_approximator import execute_bulk_actions, render_traj

import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')


# Define a class for a player controlled by keyboard input using pygame
class KeyboardPlayerPyGame(Player):
    def __init__(self):
        # Initialize class variables
        self.fpv = None  # First-person view image
        self.last_act = Action.IDLE  # Last action taken by the player
        self.screen = None  # Pygame screen
        self.keymap = None  # Mapping of keyboard keys to actions
        super(KeyboardPlayerPyGame, self).__init__()
        # self.tree = None
        # Variables for reading exploration data
        self.save_dir = "data/Images/"
        if not os.path.exists(self.save_dir):
            print(f"Directory {self.save_dir} does not exist, please download exploration data.")

        # Initialize ORB detector (replacing SIFT)
        # ORB stands for Oriented FAST and Rotated BRIEF
        self.orb = cv2.ORB_create(nfeatures=1000)
        # Load pre-trained orb features and codebook
        self.orb_descriptors, self.codebook, self.database = None, None, None
        if os.path.exists("orb_descriptors.npy"):
            self.orb_descriptors = np.load("orb_descriptors.npy", allow_pickle=True)
        if os.path.exists("codebook_orb.pkl"):
            self.codebook = pickle.load(open("codebook_orb.pkl", "rb"))
        if os.path.exists("BoVW_database.pkl"):
            self.database = pickle.load(open("BoVW_database.pkl", "rb"))
        
        # G - graph
        self.G = None
        # Load knn graph
        if os.path.exists("knn_graph.pkl"):
            print("Loading precomputed graph...")
            with open("knn_graph.pkl", "rb") as f:
                self.G = pickle.load(f)
        # Initialize goal location
        self.goal = None
        self.pos_hist = None
        self.heading_hist = None
        

    def reset(self):
        # Reset the player state
        self.fpv = None
        self.last_act = Action.IDLE
        self.screen = None

        # Initialize pygame
        pygame.init()

        # Define key mappings for actions
        self.keymap = {
            pygame.K_LEFT: Action.LEFT,
            pygame.K_RIGHT: Action.RIGHT,
            pygame.K_UP: Action.FORWARD,
            pygame.K_DOWN: Action.BACKWARD,
            pygame.K_SPACE: Action.CHECKIN,
            pygame.K_ESCAPE: Action.QUIT
        }

    def act(self):
        """
        Handle player actions based on keyboard input
        """
        for event in pygame.event.get():
            #  Quit if user closes window or presses escape
            if event.type == pygame.QUIT:
                pygame.quit()
                self.last_act = Action.QUIT
                return Action.QUIT
            # Check if a key has been pressed
            if event.type == pygame.KEYDOWN:
                # Check if the pressed key is in the keymap
                if event.key in self.keymap:
                    # If yes, bitwise OR the current action with the new one
                    # This allows for multiple actions to be combined into a single action
                    self.last_act |= self.keymap[event.key]
                else:
                    # If a key is pressed that is not mapped to an action, then display target images
                    self.show_target_images()
            # Check if a key has been released
            if event.type == pygame.KEYUP:
                # Check if the released key is in the keymap
                if event.key in self.keymap:
                    # If yes, bitwise XOR the current action with the new one
                    # This allows for updating the accumulated actions to reflect the current sate of the keyboard inputs accurately
                    self.last_act ^= self.keymap[event.key]
        return self.last_act

    def show_target_images(self):
        """
        Display front, right, back, and left views of target location in 2x2 grid manner
        """
        targets = self.get_target_images()

        # Return if the target is not set yet
        if targets is None or len(targets) <= 0:
            return

        # Create a 2x2 grid of the 4 views of target location
        hor1 = cv2.hconcat(targets[:2])
        hor2 = cv2.hconcat(targets[2:])
        concat_img = cv2.vconcat([hor1, hor2])

        w, h = concat_img.shape[:2]
        
        color = (0, 0, 0)

        concat_img = cv2.line(concat_img, (int(h/2), 0), (int(h/2), w), color, 2)
        concat_img = cv2.line(concat_img, (0, int(w/2)), (h, int(w/2)), color, 2)

        w_offset = 25
        h_offset = 10
        font = cv2.FONT_HERSHEY_SIMPLEX
        line = cv2.LINE_AA
        size = 0.75
        stroke = 1

        cv2.putText(concat_img, 'Front View', (h_offset, w_offset), font, size, color, stroke, line)
        cv2.putText(concat_img, 'Right View', (int(h/2) + h_offset, w_offset), font, size, color, stroke, line)
        cv2.putText(concat_img, 'Back View', (h_offset, int(w/2) + w_offset), font, size, color, stroke, line)
        cv2.putText(concat_img, 'Left View', (int(h/2) + h_offset, int(w/2) + w_offset), font, size, color, stroke, line)

        cv2.imshow(f'KeyboardPlayer:target_images', concat_img)
        cv2.waitKey(1)

    def set_target_images(self, images):
        """
        Set target images
        """
        super(KeyboardPlayerPyGame, self).set_target_images(images)
        self.show_target_images()

    def display_img_from_id(self, id, window_name):
        """
        Display image from database based on its ID using OpenCV
        """
        path = self.save_dir + str(id) + ".jpg"
        if os.path.exists(path):
            img = cv2.imread(path)
            cv2.imshow(window_name, img)
            cv2.waitKey(1)
        else:
            print(f"Image with ID {id} does not exist")

    def compute_orb_features(self):
        """
        Compute ORB features for images in the data directory
        """
        files = natsorted([x for x in os.listdir(self.save_dir) if x.endswith('.jpg')])
        orb_descriptors = list()
        for img in tqdm(files, desc="Processing images"):
            img = cv2.imread(os.path.join(self.save_dir, img))
            # Pass the image to ORB detector and get keypoints + descriptions
            # We only need the descriptors
            # These descriptors represent local features extracted from the image.
            _, des = self.orb.detectAndCompute(img, None)
            
            # Handle case when no features are detected
            if des is not None:
                # ORB features are already binary and normalized
                # Extend the orb_descriptors list with descriptors of the current image
                orb_descriptors.extend(des)
        
        return np.asarray(orb_descriptors)
    
    def get_BoVW(self, img):
        """
        Compute Bag of Visual Words (BoVW) descriptor for a given image
        """
        # We use ORB in combination with BoVW as a feature extractor as it offers several benefits
        # 1. ORB features are rotation-invariant and resistant to noise
        # 2. ORB is significantly faster than SIFT
        # 3. BoVW creates a histogram of visual word occurrences, making it compact and efficient
        # 4. BoVW representation is easier to compute than VLAD while still providing good distinctiveness

        # Pass the image to ORB detector and get keypoints + descriptions
        _, des = self.orb.detectAndCompute(img, None)
        
        # Handle case when no features are detected
        if des is None:
            # Return zero vector if no features detected
            return np.zeros(self.codebook.n_clusters)
            
        # Predict which cluster each descriptor belongs to
        pred_labels = self.codebook.predict(des)
        
        # Create histogram of visual words
        # This is the main difference from VLAD - we're only counting occurrences, not residuals
        histogram = np.zeros(self.codebook.n_clusters)
        
        # Count occurrences of each visual word (cluster)
        counts = Counter(pred_labels)
        for cluster_id, count in counts.items():
            histogram[cluster_id] = count
            
        # Normalize the histogram to get a frequency distribution
        # This makes the representation invariant to the number of features detected
        if np.sum(histogram) > 0:
            histogram = histogram / np.sum(histogram)
            
        return histogram

    def get_neighbor(self, img):
        """
        Find the nearest neighbor in the database based on BoVW descriptor
        """
        # Get the BoVW feature of the image
        q_BoVW = self.get_BoVW(img).reshape(1, -1)
        # This function returns the index of the closest match of the provided BoVW feature from the database
        # The '1' indicates the we want 1 nearest neighbor
        _, index = self.tree.query(q_BoVW, 1)
        return index[0][0]

    def pre_nav_compute(self):
        """
        Create a KNN graph for graph based path finding
        Build BallTree for nearest neighbor search and find the goal ID
        """
        # Compute ORB features for images in the database
        if self.orb_descriptors is None:
            print("Computing ORB features...")
            self.orb_descriptors = self.compute_orb_features()
            np.save("orb_descriptors.npy", self.orb_descriptors)
        else:
            print("Loaded ORB features from orb_descriptors.npy")

        # KMeans clustering algorithm is used to create a visual vocabulary, also known as a codebook,
        # from the computed ORB descriptors.
        # Using similar parameters as before but optimized for ORB binary features
        
        if self.codebook is None:
            print("Computing codebook...")
            # Use fewer clusters for BoVW as it works better with a smaller vocabulary
            self.codebook = KMeans(n_clusters=128, init='k-means++', n_init=5, verbose=1).fit(self.orb_descriptors)
            """ Did not use minibatch kmeans since there was enough time for full kmeans
            self.codebook = MiniBatchKMeans(
                n_clusters=128,
                init='k-means++',
                batch_size=1000,  # 每批处理的样本数
                n_init=3,         # 减少重复运行次数
                max_iter=100,     # 减少最大迭代次数
                verbose=1
            ).fit(self.orb_descriptors)
             """
            pickle.dump(self.codebook, open("codebook_orb.pkl", "wb"))
        else:
            print("Loaded codebook from codebook_orb.pkl")
        
        # get BoVW embedding for each image in the exploration phase
        if self.database is None:
            self.database = []
            print("Computing BoVW embeddings...")
            exploration_observation = natsorted([x for x in os.listdir(self.save_dir) if x.endswith('.jpg')])
            for img in tqdm(exploration_observation, desc="Processing images"):
                img = cv2.imread(os.path.join(self.save_dir, img))
                BoVW = self.get_BoVW(img)
                self.database.append(BoVW)
            with open("BoVW_database.pkl", "wb") as f:
                pickle.dump(self.database, f)
            print("BoVW database is saved")
            
        if self.G is None:
            # Initialize graph
            self.G = nx.Graph()
            # Adding nodes to the graph based on the BoVWs
            for i in range(len(self.database)):
                self.G.add_node(i)
            
            # Computing distances 
            for i in tqdm(range(len(self.database)), desc="Computing distances..."):
                node_dist = []
                for j in range(len(self.database)):
                    if i!=j:
                        # For BoVW, we can use chi-square distance as it's better for histograms
                        # but Euclidean also works well
                        d = distance.euclidean(self.database[i], self.database[j]) 
                        node_dist.append((j, d))

                # Select the K-nearest neighbours 
                node_dist = sorted(node_dist, key=lambda x:x[1])[:10]

                # Adding weighted edges to the graph
                for neighbour, dist in node_dist:
                    self.G.add_edge(i, neighbour, weight=dist)
            
            # Save the graph to avoid recomputation during each run
            with open("knn_graph.pkl", "wb") as f:
                pickle.dump(self.G, f)
            print("Graph saved successfully!")
            
        # Build BallTree with BoVW features
        print("Building BallTree...")
        tree = BallTree(self.database, leaf_size=64)
        self.tree = tree 

        # Generate position and heading histories
        self.pos_hist, self.heading_hist = execute_bulk_actions(os.path.join("data", "data_info.json"))

    def pre_navigation(self):
        """
        Computations to perform before entering navigation and after exiting exploration
        """
        super(KeyboardPlayerPyGame, self).pre_navigation()
        self.pre_nav_compute()
        
    def display_next_best_view(self):
        """
        Display the next best view based on the current first-person view
        """
        # Get the neighbor of current FPV
        # In other words, get the image from the database that closely matches current FPV
        current_index = self.get_neighbor(self.fpv)

        try:
            # Find the shortest path from the current position to the goal
            path = nx.astar_path(self.G, current_index, self.goal, weight="weight")
        except nx.NetworkXNoPath:
            print("No valid path found to the goal! Improve your algorithm!")
            return
        
        if len(path)>1:
            # Obtain adjacent nodes that are potential paths
            possible_views = list(self.G.neighbors(current_index))

            # Score the best view (Prioritize the ones that lead to the goal)
            next_best_view = min(possible_views, key=lambda view: nx.shortest_path_length(self.G, view, self.goal, weight="weight", method="dijkstra"))

            # Display the next best view
            self.display_img_from_id(next_best_view, "Next Best view")
            print(f'Next View ID: {current_index} || Goal ID: {self.goal}')


    def see(self, fpv):
        """
        Set the first-person view input
        """
        # Return if fpv is not available
        if fpv is None or len(fpv.shape) < 3:
            return

        self.fpv = fpv

        # If the pygame screen has not been initialized, initialize it with the size of the fpv image
        # This allows subsequent rendering of the first-person view image onto the pygame screen
        if self.screen is None:
            h, w, _ = fpv.shape
            self.screen = pygame.display.set_mode((w, h))

        def convert_opencv_img_to_pygame(opencv_image):
            """
            Convert OpenCV images for Pygame.

            see https://blanktar.jp/blog/2016/01/pygame-draw-opencv-image.html
            """
            opencv_image = opencv_image[:, :, ::-1]  # BGR->RGB
            shape = opencv_image.shape[1::-1]  # (height,width,Number of colors) -> (width, height)
            pygame_image = pygame.image.frombuffer(opencv_image.tobytes(), shape, 'RGB')

            return pygame_image

        pygame.display.set_caption("KeyboardPlayer:fpv")

        # If game has started
        if self._state:
            # If in exploration stage
            if self._state[1] == Phase.EXPLORATION:
                # TODO: could you employ any technique to strategically perform exploration instead of random exploration
                # to improve performance (reach target location faster)?
                
                # Nothing to do here since exploration data has been provided
                pass
            
            # If in navigation stage
            elif self._state[1] == Phase.NAVIGATION:
                # TODO: could you do something else, something smarter than simply getting the image closest to the current FPV?
                
                if self.goal is None:
                    # Get the neighbor nearest to the front view of the target image and set it as goal
                    targets = self.get_target_images()
                    index = self.get_neighbor(targets[0])
                    self.goal = index
                    print(f'Goal ID: {self.goal}')
                                
                # Key the state of the keys
                keys = pygame.key.get_pressed()
                # If 'q' key is pressed, then display the next best view based on the current FPV
                if keys[pygame.K_q]:
                    self.display_next_best_view()
                if keys[pygame.K_m]:
                    current_index = self.get_neighbor(self.fpv)
                    shutil.copy(os.path.join(self.save_dir, f"{current_index}.jpg"), "current_view_match.jpg")    # Be able to tell if current match is good
                    fig_traj = render_traj(self.pos_hist, self.heading_hist, visible=(current_index, current_index + 1), figsize=(4, 4))
                    fig_traj.gca().scatter(*self.pos_hist[self.goal])
                    fig_traj.draw(fig_traj.canvas.get_renderer())
                    arr_fig = np.frombuffer(fig_traj.canvas.tostring_argb(), dtype=np.uint8)
                    arr_fig = arr_fig.reshape(fig_traj.canvas.get_width_height()[::-1] + (4,))
                    plt.close(fig_traj)
                    cv2.imshow("traj loc", arr_fig)
                    cv2.waitKey(1)

        # Display the first-person view image on the pygame screen
        rgb = convert_opencv_img_to_pygame(fpv)
        self.screen.blit(rgb, (0, 0))
        pygame.display.update()


if __name__ == "__main__":
    import matplotlib
    import vis_nav_game
    matplotlib.use("agg")
    # Start the game with the KeyboardPlayerPyGame player
    vis_nav_game.play(the_player=KeyboardPlayerPyGame())
