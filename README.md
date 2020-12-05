# Metropolis-Hastings algorithm for optimization

### Problem
In telecommunications, 5G networks are the next generation of broadband cellular networks. Telecommunication companies are actively testing and starting to roll them out in different parts of the world. The task of this project is to deliver a roadmap to test this new network while optimizing the  cost of the maintenance of the new installations. 
We propose a Metropolis-Hastings based algorithm to find an approximate solution to this problem.

More information about the problem can be found in the folder `references`.

### Team
The project is accomplished by team `Aquarium` with members:
- Riccardo Cadei: [@riccardocadei](https://github.com/riccardocadei)
- Anita Dürr: [AnitaDurr](https://github.com/AnitaDurr)
- Loïc Busson: [@loicbusson](https://github.com/loicbusson)

### Environment
The project has been developed and test with `python3.6`.
The required library for running the models are `numpy`,`pandas` and`scipy`.
The library for visualization is `matplotlib`.


## Project structure

### Istances generator

`DatasetGenerator.py`: generators of instances of the problem

### Algorithm

`helpers.py`: useful functions to define the problem 
`markov_algos.py`: Metropolis-Hasting based algorithm implementation

### Notebook

`main.ipynb`: hyper-parameter tuning and answers to the problem

### Report

`references/report.pdf`: a 4-pages report of the complete solution.
