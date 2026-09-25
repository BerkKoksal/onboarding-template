#pragma once

#include <cstddef>
#include <vector>


// Starter Grid for the 2D heat-diffusion problem.
//
// The evaluation harness uses operator() to set initial conditions and to read
// results; it never touches your internal storage. Keep this interface,
// everything else is yours.
class Grid {
private:
  std::size_t rows_;
  std::size_t cols_;
  std::vector<double> grid;

public:
  //Using this shape of constructor because it constructs at the right size, instead of resizing
  Grid(std::size_t rows, std::size_t cols) :
    rows_(rows),
    cols_(cols),
    grid(rows * cols){}

  double& operator()(std::size_t i, std::size_t j){
    return grid[i*cols_ + j];
  }
  double  operator()(std::size_t i, std::size_t j) const{
    return grid[i*cols_ +j];
  }

  //Just providing simple getter functions
  std::size_t rows() const{
    return rows_;
  }
  std::size_t cols() const{
    return cols_;
  }

};  

// Apply the five-point stencil over all interior points, copying the boundary
// values unchanged from old_grid to new_grid. Implement your solution here.

/*
I have a flat vector.
In memory this means that it is all contigious but the best way to acess this is to iterate through each column in a row
Becasue the memory adresses are closer together which increase our chances of hitting cache.
*/
void apply_stencil(const Grid& old_grid, Grid& new_grid){
  std::size_t rows = old_grid.rows();
  std::size_t cols = old_grid.cols();
  //Copy over the first row
  for(std::size_t j = 0; j < cols; j++){
    new_grid(0,j) = old_grid(0,j);
  }

  //Copy over the middle
  //Great candidate for paralellism because we are reading from old grid and writing to new one. No threads can mutate the same data
  #pragma omp parallel for schedule(static)
  for(std::size_t i = 1; i < rows - 1; i++){
    //copy beginning
    new_grid(i,0) = old_grid(i,0);
    //calcualte middle
    for(std::size_t j = 1; j < cols - 1; j++){
      new_grid(i, j) =
        0.5 * old_grid(i, j) +
        0.125 * (
            old_grid(i - 1, j) +
            old_grid(i + 1, j) +
            old_grid(i, j - 1) +
            old_grid(i, j + 1)
        );
    }
    new_grid(i,cols-1) = old_grid(i,cols-1);
    //copy end
  }
  //Copy over the last row
  for(std::size_t j = 0; j < cols; j++){
     new_grid(rows-1,j) = old_grid(rows-1,j);
  }
}