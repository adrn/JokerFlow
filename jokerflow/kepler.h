#ifndef _KEPLER_H_
#define _KEPLER_H_

#define KEPLER_MAX_ITER 2000
#define KEPLER_TOL      1.234e-14

template <typename T>
inline T kepler (const T& M, const T& e) {
  T E0 = M, E = M;
  for (int i = 0; i < KEPLER_MAX_ITER; ++i) {
    T g = E0 - e * sin(E0) - M, gp = 1.0 - e * cos(E0);
    E = E0 - g / gp;
    if (std::abs((E - E0) / E) <= T(KEPLER_TOL)) {
      return E;
    }
    E0 = E;
  }

  // If we get here, we didn't converge, but return the best estimate.
  return E;
}

template <typename Device, typename T>
struct KeplerFunctor {
  void operator()(const Device& d, int size, const T* M, const T* e, T* E);
};

#if GOOGLE_CUDA
// Partially specialize functor for GpuDevice.
template <typename Eigen::GpuDevice, typename T>
struct KeplerFunctor {
  void operator()(const Eigen::GpuDevice& d, int size, const T* M, const T* e, T* E);
};
#endif

#undef KEPLER_MAX_ITER
#undef KEPLER_TOL

#endif // _KEPLER_H_
