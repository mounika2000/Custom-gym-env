# HyperPilot: Custom environment (Group 8)
Develop a custom gymnasium environment that represents a realistic problem of interest.

### A small GIF of how the game works

<img src="https://github.com/mounika2000/Custom-gym-env/blob/main/final-rl.gif" width="75%" height="75%">

# HyperPilot

## Project Summary
HyperPilot is a custom Gym environment where players control a helicopter to navigate through 2D space, avoiding birds and collecting fuel. The objective is to keep the helicopter fueled while avoiding bombs, with visuals rendered using OpenCV.

HyperPilot stands out as a custom Gym environment that offers a unique and engaging challenge, blending strategy and quick reflexes. In this environment, players pilot a helicopter through a dynamic 2D space, confronting a constant flux of aerial obstacles like birds and bombs. The primary goal is to adeptly maneuver this helicopter, ensuring it remains fueled while deftly avoiding the looming threats.   
The use of OpenCV for rendering adds a layer of realism to the visuals, enhancing the immersive experience. This environment is not just about survival but also about mastering the delicate art of navigation and resource management. The player's success hinges on their ability to strategically collect fuel and dodge dangers, making HyperPilot an intriguing test of both tactical thinking and agility.

## State Space
- **Observation Shape:** (600, 800, 3) representing the RGB image of the environment.
- This value has been played around during training to find which model works best


## Action Space
- **Actions:** Discrete space with six actions:
  - 0: Move Right
  - 1: Move Left
  - 2: Move Down
  - 3: Move Up
  - 4: Do Nothing

## Rewards
- **Survival:** +1 reward for each time step without collision.
- **Collision:** -10 penalty for colliding with a bomb, where the episode ends.

## Reinforcement Learning Algorithm

In the HyperPilot project, we employed the Proximal Policy Optimization (PPO) algorithm for training the agent. This choice is evident from the usage of the `ray.rllib.algorithms.ppo` module. PPO is renowned for its efficiency and effectiveness in complex environments, particularly those with high-dimensional observation spaces, such as ChopperScape. Our implementation initializes the PPO algorithm with a custom convolutional neural network configuration, suitable for processing the image-based observations of the environment.

### Research Papers on PPO
As discussed in the work of Smith et al. (2020), the implementation of XYZ algorithm shows significant improvements in performance [1].

### References
[1] J. Smith, A. Johnson, and K. Lee, "Title of the Paper," in *Journal Name*, vol. X, no. Y, pp. Z-ZZ, 2020. [Link](URL_to_the_paper)



## Starting State
- The helicopter is initialized at a random position at the start of each episode.

## Episode End Conditions
- Collision with a bomb.
- Running out of fuel.

## Training Results  

MatplotLib plot of Reward over episodes  
<img src="https://github.com/mounika2000/Custom-gym-env/blob/main/plt.PNG" width="80%" height="80%">

Adding few screenshots of the TensorFlow dashboard  
<img src="https://github.com/mounika2000/Custom-gym-env/blob/main/tensorflow-board.png" width="75%" height="75%">
<img src="https://github.com/mounika2000/Custom-gym-env/blob/main/tb1.png" width="50%" height="50%">
<img src="https://github.com/mounika2000/Custom-gym-env/blob/main/tb2.png" width="50%" height="50%">
<img src="https://github.com/mounika2000/Custom-gym-env/blob/main/tb3.png" width="50%" height="50%">

The Pilot was able to gain a reward in the range of 85-95 when tried during various iterations:  

<img src="https://github.com/mounika2000/Custom-gym-env/blob/main/iteration-results.png" width="30%" height="30%">

## Conclusion

The custom environment game in this project shows promising signs of improvement in the agent's performance. Although the reward increase is modest, likely due to the limited scope of training with only 100 iterations, the results are encouraging. The constraint of resources prevented longer training sessions which could potentially yield better results. Nevertheless, the existing outcomes and screenshots suggest that the Ray RLlib agents have a significant potential for further improvement, indicating that with extended training, the agents could achieve higher rewards and more effectively navigate the custom environment.

Working on this project was an immensely fun and interesting experience. The process of developing and fine-tuning the custom environment, along with observing the agent's learning progress over time, was both challenging and rewarding.



