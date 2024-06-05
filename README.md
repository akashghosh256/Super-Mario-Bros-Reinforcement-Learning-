# Super Mario Bros RL Agent with DDQN

This project demonstrates the use of reinforcement learning to train an agent to play the classic game Super Mario Bros using Double Deep Q-Networks (DDQN).

![App Screenshot](https://github.com/akashghosh256/Super-Mario-Bros-Reinforcement-Learning-/blob/main/screenshots/gameplay.gif)

## Project Overview

In this project, we created a reinforcement learning agent that learns to play Super Mario Bros. The agent uses Double Deep Q-Networks (DDQN) to improve performance and stability during the learning process.

## Key Concepts

![App Screenshot](https://raw.githubusercontent.com/akashghosh256/Super-Mario-Bros-Reinforcement-Learning-/main/screenshots/types.webp)

### Reinforcement Learning (RL)
Reinforcement Learning is a type of machine learning where an agent learns to make decisions by performing actions and receiving rewards. The agent's goal is to maximize cumulative rewards over time.

![App Screenshot](https://github.com/akashghosh256/Ecommerce-App/blob/main/screenshots/filter.png)

### Deep Q-Networks (DQN)
Deep Q-Networks (DQN) use neural networks to approximate the Q-value function, which helps the agent decide the best actions to take in different states.

### Double DQN (DDQN)
Double DQN is an enhancement to DQN that reduces overestimation bias by using two separate networks: one for selecting actions and one for evaluating them.

### Replay Buffer
A replay buffer is a memory storage that holds past experiences. By sampling from this buffer, the agent can learn from a diverse set of experiences, stabilizing training.

### Wrappers
Wrappers are used to preprocess game observations and rewards, making it easier for the agent to learn effectively from the environment.


![App Screenshot](https://raw.githubusercontent.com/akashghosh256/Super-Mario-Bros-Reinforcement-Learning-/main/screenshots/cycle2.webp)

![App Screenshot](https://github.com/akashghosh256/Ecommerce-App/blob/main/screenshots/filter.png)

![App Screenshot](https://github.com/akashghosh256/Ecommerce-App/blob/main/screenshots/filter.png)

## Learning Outcomes

Through this project, I have gained:

1. **Understanding of Reinforcement Learning**: I learned how reinforcement learning algorithms work and how they can be applied to game environments.
2. **Deep Q-Networks and Double DQN**: I explored how DQNs work and how Double DQN improves upon them by reducing overestimation bias.
3. **Experience with Gym Environment**: I became familiar with the OpenAI Gym framework and the gym-super-mario-bros environment, which are essential for creating and managing game simulations.
4. **Neural Networks with PyTorch**: I enhanced my skills in building and training neural networks using PyTorch, a powerful deep learning library.
5. **Creating Custom Wrappers**: I learned how to create and apply custom wrappers to preprocess observations and rewards, improving the agent's learning efficiency.

## Conclusion

This project provided a comprehensive learning experience in applying advanced reinforcement learning techniques to a well-known game. By using Double DQN and various other techniques, the agent can learn to navigate and play Super Mario Bros efficiently.

## References

- [Gym Super Mario Bros](https://pypi.org/project/gym-super-mario-bros/)
- [PyTorch](https://pytorch.org/)
- [OpenAI Gym](https://gym.openai.com/)
- [Double DQN Paper](https://arxiv.org/abs/1509.06461)
