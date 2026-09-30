#pragma once

#include <cstddef>
#include <vector>
#include <new>


template <typename T, std::size_t Alignment>
class AlignedAllocator{
  public:
  using value_type = T;
  AlignedAllocator() noexcept = default;

  template <typename U>
  AlignedAllocator(const AlignedAllocator<U, Alignment>&) noexcept {}

  template <typename U>
  struct rebind{
    using other = AlignedAllocator<U, Alignment>;
  };

  T* allocate(std::size_t n){
    void* ptr = ::operator new(n * sizeof(T),std::align_val_t{Alignment});

    return static_cast<T*>(ptr);
  }

  void deallocate(T* ptr, std::size_t){
    ::operator delete(ptr,std::align_val_t{Alignment});
  }

  template <typename U>
  bool operator==(const AlignedAllocator<U, Alignment>&) const noexcept{
    return true;
  }

  template <typename U>
  bool operator!=(const AlignedAllocator<U, Alignment>&) const noexcept{
    return false;
  }
};


struct ConstGridView{
  const double* data;
  std::size_t rows;
  std::size_t cols;
  std::size_t stride;

  const double& operator()(std::size_t i, std::size_t j) const{
    return data[i*stride + j];
  }

  const double* row(std::size_t i) const{
    return data + i*stride;
  }
};


struct GridView{
  double* data;
  std::size_t rows;
  std::size_t cols;
  std::size_t stride;

  double& operator()(std::size_t i, std::size_t j){
    return data[i*stride + j];
  }

  double* row(std::size_t i){
    return data + i*stride;
  }
};


class Grid {
private:
  std::size_t rows_;
  std::size_t cols_;
  std::size_t stride_;
  std::size_t version_;
  std::vector<double, AlignedAllocator<double, 64>> grid;

public:
  Grid(std::size_t rows, std::size_t cols) :
    rows_(rows),
    cols_(cols),
    stride_(cols + 16),
    version_(0),
    grid(rows * stride_){}

  double& operator()(std::size_t i, std::size_t j){
    version_++;
    return grid[i*stride_ + j];
  }

  double operator()(std::size_t i, std::size_t j) const{
    return grid[i*stride_ + j];
  }

  ConstGridView view() const{
    return {
      grid.data(),
      rows_,
      cols_,
      stride_
    };
  }

  GridView view(){
    return {
      grid.data(),
      rows_,
      cols_,
      stride_
    };
  }

  std::size_t version() const{
    return version_;
  }

  std::size_t rows() const{
    return rows_;
  }

  std::size_t cols() const{
    return cols_;
  }

};


struct ActiveRegion{
  bool has_active = false;
  std::size_t top = 0;
  std::size_t bottom = 0;
  std::size_t left = 0;
  std::size_t right = 0;
};


struct StencilState{
  const Grid* grid_one = nullptr;
  const Grid* grid_two = nullptr;

  std::size_t grid_one_version = 0;
  std::size_t grid_two_version = 0;

  std::size_t rows = 0;
  std::size_t cols = 0;

  ActiveRegion active;
};


inline bool same_grid_pair(
  const StencilState& state,
  const Grid& old_grid,
  const Grid& new_grid
){
  if(state.rows != old_grid.rows() || state.cols != old_grid.cols()){
    return false;
  }

  if(state.grid_one == &old_grid && state.grid_two == &new_grid){
    return
      state.grid_one_version == old_grid.version() &&
      state.grid_two_version == new_grid.version();
  }

  if(state.grid_one == &new_grid && state.grid_two == &old_grid){
    return
      state.grid_one_version == new_grid.version() &&
      state.grid_two_version == old_grid.version();
  }

  return false;
}


inline ActiveRegion copy_and_find_active_region(
  ConstGridView old_grid,
  GridView new_grid
){
  ActiveRegion active;

  for(std::size_t i = 0; i < old_grid.rows; i++){
    for(std::size_t j = 0; j < old_grid.cols; j++){
      double value = old_grid(i,j);

      new_grid(i,j) = value;

      if(value != 0.0){
        if(!active.has_active){
          active.top = i;
          active.bottom = i;
          active.left = j;
          active.right = j;
          active.has_active = true;
        }
        else{
          if(i < active.top){
            active.top = i;
          }

          if(i > active.bottom){
            active.bottom = i;
          }

          if(j < active.left){
            active.left = j;
          }

          if(j > active.right){
            active.right = j;
          }
        }
      }
    }
  }

  return active;
}


inline void copy_boundaries(
  ConstGridView old_grid,
  GridView new_grid
){
  for(std::size_t j = 0; j < old_grid.cols; j++){
    new_grid(0,j) = old_grid(0,j);
  }

  for(std::size_t i = 1; i < old_grid.rows - 1; i++){
    new_grid(i,0) = old_grid(i,0);
    new_grid(i,old_grid.cols - 1) = old_grid(i,old_grid.cols - 1);
  }

  for(std::size_t j = 0; j < old_grid.cols; j++){
    new_grid(old_grid.rows - 1,j) = old_grid(old_grid.rows - 1,j);
  }
}


inline void expand_active_region(
  ActiveRegion& active,
  std::size_t rows,
  std::size_t cols
){
  if(active.top > 0){
    active.top--;
  }

  if(active.bottom + 1 < rows){
    active.bottom++;
  }

  if(active.left > 0){
    active.left--;
  }

  if(active.right + 1 < cols){
    active.right++;
  }
}


inline ActiveRegion clamp_to_interior(
  ActiveRegion active,
  std::size_t rows,
  std::size_t cols
){
  if(active.top < 1){
    active.top = 1;
  }

  if(active.bottom > rows - 2){
    active.bottom = rows - 2;
  }

  if(active.left < 1){
    active.left = 1;
  }

  if(active.right > cols - 2){
    active.right = cols - 2;
  }

  return active;
}


inline void apply_stencil_row(
  ConstGridView old_grid,
  GridView new_grid,
  std::size_t i,
  std::size_t col_begin,
  std::size_t col_end
){
  const double* previous = old_grid.row(i - 1);
  const double* current = old_grid.row(i);
  const double* next = old_grid.row(i + 1);

  double* output = new_grid.row(i);

  #pragma omp simd
  for(std::size_t j = col_begin; j <= col_end; j++){
    output[j] =
      0.5 * current[j] +
      0.125 * (
        previous[j] +
        next[j] +
        current[j - 1] +
        current[j + 1]
      );
  }
}


inline void apply_stencil_region(
  ConstGridView old_grid,
  GridView new_grid,
  const ActiveRegion& active
){
  #pragma omp parallel for schedule(static)
  for(std::size_t i = active.top; i <= active.bottom; i++){
    apply_stencil_row(
      old_grid,
      new_grid,
      i,
      active.left,
      active.right
    );
  }
}


inline void apply_stencil(const Grid& old_grid, Grid& new_grid){
  ConstGridView old_view = old_grid.view();
  GridView new_view = new_grid.view();

  static StencilState state;

  if(!same_grid_pair(state, old_grid, new_grid)){
    state.grid_one = &old_grid;
    state.grid_two = &new_grid;

    state.grid_one_version = old_grid.version();
    state.grid_two_version = new_grid.version();

    state.rows = old_grid.rows();
    state.cols = old_grid.cols();

    state.active = copy_and_find_active_region(
      old_view,
      new_view
    );
  }

  copy_boundaries(old_view, new_view);

  if(!state.active.has_active){
    return;
  }

  expand_active_region(
    state.active,
    old_view.rows,
    old_view.cols
  );

  ActiveRegion interior = clamp_to_interior(
    state.active,
    old_view.rows,
    old_view.cols
  );

  if(interior.top <= interior.bottom && interior.left <= interior.right){
    apply_stencil_region(
      old_view,
      new_view,
      interior
    );
  }
}