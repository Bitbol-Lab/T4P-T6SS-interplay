# T4P-T6SS-interplay

Supporting repository for: "Interactions between pili affect the outcome of bacterial competition driven by the type VI secretion system" (https://doi.org/10.1101/2023.10.25.564063).

## Getting started ##

Clone this repository on your local machine by running:

```bash
git clone git@github.com:Bitbol-Lab/T4P-T6SS-interplay.git
``` 
 

Executing the following line runs a working example:
```bash
python T4P_T6SS_interplay.py
``` 

## Requirements ##

In order to use the function `main`, Numba is required.

## Usage ##

`
main
`
simulates a 40^3 large body-centred cubic lattice with 50 prey and 50 predators, with matching pili, during 10 minutes, and yields the number of prey, predators and lysing prey over time. These parameters can be tuned.

Besides, in order to prevent the diffusion of aggregates as whole units, just replace line 245 "elif number_of_free_neighbors < 8:" by "elif False:" so that the code dedicateed to the diffusion of aggregates as whole units is never executed.

## Warning ##

Note that in the comments of the code we use the words particles, cells and bacteria in an interchangable fashion.
