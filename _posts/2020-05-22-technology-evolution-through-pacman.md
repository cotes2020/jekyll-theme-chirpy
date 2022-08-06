---
title: Technology Evolution Through Pac-Man 
date: 2020-05-22 09:11
category: thoughts 
tags: [tech,data science,retro]
layout: post
---

We can mark the significant evolution of technology through 3 different milestones based on the game <a  target="_blank" href="https://en.wikipedia.org/wiki/Pac-Man">Pac-Man</a>. For those who don 't know [Pac-Man], it is the arcade game where you control Pac-Man, who must eat all the dots inside an enclosed maze while avoiding four colored ghosts. The game was first released in Japan by Namco in 1980 and later became a global phenomenon worldwide. For me, first seeing the game at my cousin 's house on the Atari 2600 somewhere between 1980 and 1981 triggered my passion for computers ever since.

![pacman and Tarek](/assets/images/retro/atari-pacman.jpg)

## milestone 1 - developing games by humans, and playing them by humans

In the 1980s, PacMan was released on all major home gaming consoles and microcomputers. People would play the game on their Atari 2600, Commodore, and many other devices. The four ghosts in the game, Blinky (red), Pinky (pink), Inky (cyan), and Clyde (orange), had some distinctive artificial intelligence capabilities that made the game of chasing or avoiding the ghosts fun and challenging. Blinky would directly go toward Pac-Man, while Pinky and Inky would try to position themselves in front of Pac-Man, usually by cornering him, and Clyde will switch between chasing Pac-Man and fleeing from him (<a  target="_blank" href="https://en.wikipedia.org/wiki/Pac-Man">wiki</a>). The game was so popular that it led to several sequels, variations (Ms. Pacman, Pac Land, Super Pac Land, to name a few), television series, and significant revenue-making franchises. The game inspired a lot of future gaming and more development of video games. Thanks to video games like Pac-Man in the 80s, gaming consoles and microcomputers in the 80s became everyday household items. But the game was mostly for fun. Its developer created the game, and the computer/game console users played the game.

{% include video id="wtv6SE6RnD8" provider="youtube" %}

## milestone 2: developing games by humans, and playing them by humans and machines

Computer technologies significantly evolved from the eighties to the millenniums. The last decade also saw the rise of artificial intelligence and machine learning algorithms. These are software that attempts to mimic human intelligence, and in some cases, exceeds human intelligence to solve problems. One particular company, <a target="_blank" href="https://en.wikipedia.org/wiki/DeepMind" >Deep Mind</a>, which Google later acquired, was working on implementing an AI algorithm using <a target="_blank" href="https://en.wikipedia.org/wiki/Artificial_neural_network" >artificial neural networks</a> and <a target="_blank" href="https://en.wikipedia.org/wiki/Reinforcement_learning" >reinforcement learning</a> that learn on its own by learning through experimentation (trial and error) and observation. It developed the \[Deep-Q Network algorithm\](https://en.wikipedia.org/wiki/Q-learning] which, in simple terms, takes input as raw pixels (what you see on the screen), a set of available actions (move up, down, etc.), and a game score (positive values means gain) then decides on the next best course of action by estimating future rewards. In other words, the algorithm would learn to develop an optimum gaming strategy by thinking of short versus long-term rewards. It keeps playing games repeatedly, learning to determine what generates a positive score or causes a loss. DQN learned to <a target="_blank" href="https://medium.com/the-physics-arxiv-blog/the-last-ai-breakthrough-deepmind-made-before-google-bought-it-for-400m-7952031ee5e1" >play many Atari games in 2014</a> by mimicking how humans would do:

  * Looking at the monitoring.
  * Moving the game characters around.
  * Understanding what is good and what is bad.

Shortly after, such artificial intelligence algorithms started playing PacMan and ultimately made history when an advanced version of the algorithm <a  target="_blank" href="https://deepmind.com/research/case-studies/alphago-the-story-so-far">AlphaGo became the strongest Go player in history</a>. The use of artificial intelligence in games ultimately made games like PacMan an activity that is not only for humans for machines as well! For instance, the AI algorithm AlphaStar became a StarCraft grandmaster (<a  target="_blank" href="https://www.youtube.com/watch?v=lPERfjRaZug">video</a>). A list of other games with AI vs. humans is available <a  target="_blank" href="https://en.wikipedia.org/wiki/Artificial_intelligence_in_video_games#Examples">here</a>. Suppose you are interested in getting deeper into AI in games. In that case, a book draft of "Artificial Intelligence and Games" by Georgios Yannakakis and Julian Togelius (2018) is available to download at their website

{% include video id="4MlZncshy1Q" provider="youtube" %}

## milestone 3 developing and playing games by machines and humans

Algorithms similar to DQN became hugely popular in human vs. machines game competitions. Now, programmers can use machines are playing all sorts of games, such as <a target="_blank" href="https://gym.openai.com/envs/#atari" >Atari games</a>, <a target="_blank" href="https://cheesyprogrammer.com/2018/03/01/deep-learning-training-an-ai-to-play-doom/" >DOOM</a>, and more. For instance, you can install <a target="_blank" href="https://github.com/openai/retro" >OpenAI Gym Retro</a> to let machines play on Atari, NEC, Nintendo, and Sega systems.

Today, <a  target="_blank" href="https://blogs.nvidia.com/blog/2020/05/22/gamegan-research-pacman-anniversary/">Nvidia released an artificial intelligence algorithm called GameGAN that did not play Pac-Man but created one</a>! By watching 50,000 Pac-Man games played, GameGAN learned that the game is a maze, has a set of pixels that makes the character Pac-Man, the ghosts, and the dots, and then generated implementation of the game without looking into the underlying game engine code of the game itself. The algorithm GameGAN learned how to create the game mechanics and designed the layout. It used a specific algorithm in AI called <a  target="_blank" href="https://blogs.nvidia.com/blog/2017/05/17/generative-adversarial-networks/">generative adversarial networks</a> that were first invented by Ian Goodfellow in 2014 while still a Ph.D. student at the time. Even if the technical schema of the work was not publicly disclosed, it is built on top of computer algorithms that are publicly being shared with the world by major companies like DeepMind, OpenAI, Google, and Facebook. Having algorithms that play the game and create the game itself is a new milestone in computer science engineering. The makers of GameGAN, Seung Wook Kim, Yuhao Zhou, Jonah Philion, Antonio Torralba, and Sanja Fidle, put a web page, <a  target="_blank" href="https://nv-tlabs.github.io/gameGAN/">"Learning to Simulate Dynamic Environments with GameGAN"</a> that provides more detail about their work and how they tackled the problem.

{% include video id="4OzJUNsPx60" provider="youtube" %}

Koichiro Tsutsumi from Namco that owns the right for Pac-Man, said: "We were blown away when we saw the results, in disbelief that AI could recreate the iconic PAC-MAN experience without a game engine. This research presents exciting possibilities to help game developers accelerate the creative process of developing new level layouts, characters, and even games." <a target="_blank" href="https://blogs.nvidia.com/blog/2020/05/22/gamegan-research-pacman-anniversary/" >blog post</a>.

NVidia promises to release a playable version of the game on their <a target="_blank" href="https://www.nvidia.com/en-us/research/ai-playground/" >Nvidia AI Playground</a> website. That will be interesting to check out!

Not only playing the game is now for humans and machines but developing the game is now by humans and machines!

<video controls src="https://blogs.nvidia.com/wp-content/uploads/2020/05/GameGAN-PAC-MAN\_V6\_BLOG.mp4"></video>

<a  target="_blank" href="https://blogs.nvidia.com/wp-content/uploads/2020/05/GameGAN-PAC-MAN_V6_BLOG.mp4">NVIDIA GameGan in action</a>

What a great way to celebrate Pac-man 's 40th anniversary! From a game that was created by humans to be played by humans, we are now at the frontier of games that are created by machines and played by machines!
