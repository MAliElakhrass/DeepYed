# IFT6756
<!--
*** Thanks for checking out the Best-README-Template. If you have a suggestion
*** that would make this better, please fork the repo and create a pull request
*** or simply open an issue with the tag "enhancement".
*** Thanks again! Now go create something AMAZING! :D
-->



<!-- PROJECT SHIELDS -->
<!--
*** I'm using markdown "reference style" links for readability.
*** Reference links are enclosed in brackets [ ] instead of parentheses ( ).
*** See the bottom of this document for the declaration of the reference variables
*** for contributors-url, forks-url, etc. This is an optional, concise syntax you may use.
*** https://www.markdownguide.org/basic-syntax/#reference-style-links
-->


<!-- PROJECT LOGO -->
<br />
<p align="center">
  <a href="https://github.com/MAliElakhrass/DeepYed">
    <img src="image/logo.png" alt="Logo" width="80" height="80">
  </a>

  <h3 align="center">Final project 6756</h3>

  <p align="center">
    An awesome README template to jumpstart your projects!
    <br />
    <a href="https://github.com/othneildrew/Best-README-Template"><strong>Explore the docs »</strong></a>
    <br />
    <br />
    <a href="https://github.com/othneildrew/Best-README-Template">View Demo</a>
    ·
    <a href="https://github.com/othneildrew/Best-README-Template/issues">Report Bug</a>
    ·
    <a href="https://github.com/othneildrew/Best-README-Template/issues">Request Feature</a>
  </p>
</p>



<!-- TABLE OF CONTENTS -->
<details open="open">
  <summary>Table of Contents</summary>
  <ol>
    <li>
      <a href="#about-the-project">About The Project</a>
      <ul>
        <li><a href="#built-with">Built With</a></li>
      </ul>
    </li>
    <li>
      <a href="#getting-started">Getting Started</a>
      <ul>
        <li><a href="#prerequisites">Prerequisites</a></li>
        <li><a href="#installation">Installation</a></li>
      </ul>
    </li>
    <li><a href="#heuristic-approach">Heuristic Approach</a></li>
    <li><a href="#neural-network">Neural Network</a></li>
    <li><a href="#roadmap">Roadmap</a></li>
    <li><a href="#contributing">Contributing</a></li>
    <li><a href="#license">License</a></li>
    <li><a href="#contact">Contact</a></li>
    <li><a href="#acknowledgements">Acknowledgements</a></li>
  </ol>
</details>



<!-- ABOUT THE PROJECT -->
## About The Project

[![Product Name Screen Shot][product-screenshot]](https://example.com)

This project is our take on building an automated chess player. We first started building an heuristic, a neural network and a reinforcement learning approach.
Here's why:
* Your time should be focused on creating something amazing. A project that solves a problem and helps others
* You shouldn't be doing the same tasks over and over like creating a README from scratch
* You should element DRY principles to the rest of your life :smile:

Of course, no one template will serve all projects since your needs may be different. So I'll be adding more in the near future. You may also suggest changes by forking this repo and creating a pull request or opening an issue. Thanks to all the people have have contributed to expanding this template!

A list of commonly used resources that I find helpful are listed in the acknowledgements.

### Built With

This section lists any the major frameworks that we built our project using. 
* [Pytorch](https://pytorch.org/)
* [Tensorflow](https://www.tensorflow.org/)
* [Python](https://www.python.org/)
* [CUDA](https://developer.nvidia.com/cuda-toolkit)
* [python-chess](https://python-chess.readthedocs.io/en/latest)



<!-- GETTING STARTED -->
## Getting Started

This is an example of how you may give instructions on setting up your project locally.
To get a local copy up and running follow these simple example steps.

### Prerequisites

This is an example of how to list things you need to use the software and how to install them.
* npm
  ```sh
  npm install npm@latest -g
  ```

### Installation

1. Get a free API Key at [https://example.com](https://example.com)
2. Clone the repo
   ```sh
   git clone https://github.com/your_username_/Project-Name.git
   ```
3. Install NPM packages
   ```sh
   npm install
   ```
4. Enter your API in `config.js`
   ```JS
   const API_KEY = 'ENTER YOUR API';
   ```

## Heuristic approach
We used the Negamax algorithm which is a variant of the minimax search.

_For more details on this algorithm, please refer to:  [Website](https://en.wikipedia.org/wiki/Negamax)_
 


<!-- USAGE EXAMPLES -->
## Neural Network
### Dataset
First, we used the CCRL 40/15 Dataset containing 1 233 013 games where Whites win for 34.4%, Black win for 25.2% and draws are 40.5% of the dataset.Use this space to show useful examples of how a project can be used.

### Preprocessing
We decided to go with a preprocessing inspired by the one used by the authors of DeepChess. Therefore, we first ignored all th draws since they apparently did not add any value. We just kept the wins and losses. We extracted 10 random moves per game while making sure these moves did not end in a capture from either sides.
Also, these moves were not one of the first five. Each move was represented by the state of the board with the actual move in it. The board was encoded into a 773 binary bit-string array called bitboard. 
This amount of bits is obtained by taking in consideration the two sides (White and Black), the 6 types of pieces (queen, king, pawn, bishop rook and knight),
the 64 squares on a board (8 x 8) and the five last bits are for the side to move (White's turn or Black's) and the castling rights. 
Indeed, the last 4 bits indicate if the Whites can castle kingside, if the Whites can castle queenside, if the Blacks can castle kingside and if the Blacks can castle queenside.

### Implementation 
Our neural network has two parts, the autoencoder part and a siamese network. The autoencoder consists of five fully connected layers 773-600-400-200-100. First, we added batch normalisation layers
and a Leaky Relu activation function because it got better than the official implmentation of DeepChess which did not have any regularization and the activation functions were ReLU. 
The learning rate used is 0.005 and it was multiplied by 0.98 after each epoch like indicated by the authors of DeepChess. The autoencoder was trained for 200 epochs. The results were not very good for this
first architecture. The autoencoder seemed to overfit after 7 epochs. Therefore, we tried another architecture which used DenseTied layers. BLA BLA BLA

The Siamese network had to take two inputs. One input would be a a move from a win and one move that ended up in a loss. Two bitboards are passed to the trained encoder of the autoencoder to extract important features.
The two obtained representations  have 100 features each. The architecture used is 400-200-100-2. The loss used to train this part is the binary cross entropy.

### Results
The results were not as expected. We first tried to implement a similar architecture to the one presented in the paper of DeepChess. The autoencoder was composed of an encoder of linear four layers
of 773-600-400-200. The decoder had the following architecture 200-400-600-773 to rebuild the input. All the activation functions used between each layer is a ReLU function and the last activation function is 
a Sigmoid function. The siamese network used had four linear layers with the following dimensions 200-400-200-100. For this part, ReLU was used as activation function and sigmoid for the last layer. The results were very bad
with an accuracy of 50%. Then we tried adding some batch normalisation between each layers and changed the activation function to Leaky ReLU and obtained an accuracy of 58% which was better but still very bad. We tried several other
architecture changing the learning rates, the number of features per layer, the error used but the score was still very bad.

_For more details about DeepChess, please refer to the [Paper](https://arxiv.org/pdf/1711.09667.pdf)_

_For more details about the CCRL 40/15 Dataset, please refer to this [Website](https://ccrl.chessdom.com/ccrl/4040/)_


<!-- ROADMAP -->
## Roadmap

See the [open issues](https://github.com/othneildrew/Best-README-Template/issues) for a list of proposed features (and known issues).



<!-- CONTRIBUTING -->
## Contributing

Contributions are what make the open source community such an amazing place to be learn, inspire, and create. Any contributions you make are **greatly appreciated**.

1. Fork the Project
2. Create your Feature Branch (`git checkout -b feature/AmazingFeature`)
3. Commit your Changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the Branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request



<!-- LICENSE -->
## License

Distributed under the MIT License. See `LICENSE` for more information.



<!-- CONTACT -->
## Contact

Your Name - [@your_twitter](https://twitter.com/your_username) - email@example.com

Project Link: [https://github.com/your_username/repo_name](https://github.com/your_username/repo_name)



<!-- ACKNOWLEDGEMENTS -->
## Acknowledgements
* [GitHub Emoji Cheat Sheet](https://www.webpagefx.com/tools/emoji-cheat-sheet)
* [Img Shields](https://shields.io)
* [Choose an Open Source License](https://choosealicense.com)
* [GitHub Pages](https://pages.github.com)
* [Animate.css](https://daneden.github.io/animate.css)
* [Loaders.css](https://connoratherton.com/loaders)
* [Slick Carousel](https://kenwheeler.github.io/slick)
* [Smooth Scroll](https://github.com/cferdinandi/smooth-scroll)
* [Sticky Kit](http://leafo.net/sticky-kit)
* [JVectorMap](http://jvectormap.com)
* [Font Awesome](https://fontawesome.com)





<!-- MARKDOWN LINKS & IMAGES -->
<!-- https://www.markdownguide.org/basic-syntax/#reference-style-links -->
[contributors-shield]: https://img.shields.io/github/contributors/othneildrew/Best-README-Template.svg?style=for-the-badge
[contributors-url]: https://github.com/othneildrew/Best-README-Template/graphs/contributors
[forks-shield]: https://img.shields.io/github/forks/othneildrew/Best-README-Template.svg?style=for-the-badge
[forks-url]: https://github.com/othneildrew/Best-README-Template/network/members
[stars-shield]: https://img.shields.io/github/stars/othneildrew/Best-README-Template.svg?style=for-the-badge
[stars-url]: https://github.com/othneildrew/Best-README-Template/stargazers
[issues-shield]: https://img.shields.io/github/issues/othneildrew/Best-README-Template.svg?style=for-the-badge
[issues-url]: https://github.com/othneildrew/Best-README-Template/issues
[license-shield]: https://img.shields.io/github/license/othneildrew/Best-README-Template.svg?style=for-the-badge
[license-url]: https://github.com/othneildrew/Best-README-Template/blob/master/LICENSE.txt
[linkedin-shield]: https://img.shields.io/badge/-LinkedIn-black.svg?style=for-the-badge&logo=linkedin&colorB=555
[linkedin-url]: https://linkedin.com/in/othneildrew
[product-screenshot]: images/screenshot.png
