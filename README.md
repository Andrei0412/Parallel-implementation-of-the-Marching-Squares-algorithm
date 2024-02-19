# Marching Square algorithm optimization through multithreading

## Project goals
* Improve the speed of the sequential implementation of the Marching Square algorithm using the Pthreads Library in C/C++. The desired outcome is to get the same output as the original implementation but with faster execution time.

## Algorithm details
* The Marching Square algorithm is a computer graphics algorithm that can be used for contouring. It can be used to draw altitude lines on topographical maps, temperature on temperature heat maps, pressure points on pressure field maps, etc.

## Constraints
* The images that need to be processed need to have the .ppm format.

## Usage
* The build rule of the Makefile can be used to generate an executable:
```bash
make build
```

* To run the program for a particular image use:
```bash
./tema1_par <input_file> <output_file> <nr_threads>
```

* **Note that this implementation resizes the original image if it exceeds 2048 pixels for either X or Y.**
* **The checker folder contains the sequential implementation of the algorithm.**
